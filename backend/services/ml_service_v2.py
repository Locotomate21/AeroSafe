"""
Servicio de inferencia de riesgo aeronautico.

Nota de diseno sobre el modo mock: existe para que la API siga levantando
cuando falta el modelo (desarrollo, CI, primer arranque), pero NUNCA debe
confundirse con una prediccion real. Toda salida lleva 'model_status', y
la caida a mock se registra como ERROR, no como warning. Un sistema que
informa riesgo meteorologico no puede devolver reglas if/else disfrazadas
de modelo entrenado.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from core.config import settings
from features.build_features import FeaturePipelineError, build_features
from features.defaults import complete_raw_features
from models.models import RiskPrediction

logger = logging.getLogger(__name__)

# Niveles que el modelo de produccion puede predecir.
RISK_LEVELS = ["BAJO", "MODERADO", "ALTO"]

MODEL_STATUS_ML = "ml"
MODEL_STATUS_MOCK = "mock"


class MLServiceV2:
    """Carga el modelo de produccion y expone prediccion unitaria y batch."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        model=None,
        scaler=None,
        encoders=None,
    ):
        """
        Args:
            model_path: Ruta al .pkl. Si es None, se usa settings.MODEL_PATH.
            model: Modelo ya instanciado. Si se pasa, no se lee del disco
                   (lo usan los tests con un mock).
            scaler: StandardScaler ajustado, para inyeccion directa.
            encoders: dict de LabelEncoders ajustados, para inyeccion directa.
        """
        self.model = model
        self.scaler = scaler
        self.label_encoder = encoders
        self.feature_names: List[str] = []

        if model is not None:
            logger.info("MLServiceV2 inicializado con un modelo inyectado")
            return

        model_file = Path(model_path) if model_path else settings.get_model_path(
            settings.MODEL_PATH
        )

        try:
            if not model_file.exists():
                logger.error(
                    "Modelo no encontrado en %s. El servicio arranca en modo "
                    "MOCK: las predicciones NO provienen del modelo entrenado.",
                    model_file,
                )
                return

            self.model = joblib.load(model_file)
            logger.info("Modelo cargado desde %s", model_file)

            scaler_path = settings.get_model_path(settings.SCALER_PATH)
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Scaler cargado desde %s", scaler_path)

            encoder_path = settings.get_model_path(settings.ENCODER_PATH)
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
                logger.info("Encoders cargados desde %s", encoder_path)

            feature_names_path = settings.get_model_path(settings.FEATURE_NAMES_PATH)
            if feature_names_path.exists():
                # encoding explicito: el fichero contiene 'dia_año' y el
                # default del sistema difiere entre Windows (cp1252) y
                # Linux (utf-8), que es donde corre el contenedor.
                with open(feature_names_path, "r", encoding="utf-8") as f:
                    self.feature_names = [line.strip() for line in f if line.strip()]
                logger.info("Feature names cargados: %d", len(self.feature_names))

            # Sin scaler/encoders el modelo esta cargado pero es inutilizable:
            # mejor decirlo al arrancar que descubrirlo en la primera peticion.
            if self.scaler is None or self.label_encoder is None:
                logger.error(
                    "Modelo cargado pero faltan artefactos del pipeline "
                    "(scaler=%s, encoders=%s). No se podra inferir.",
                    self.scaler is not None,
                    self.label_encoder is not None,
                )

        except Exception as e:
            logger.exception("Error cargando el modelo: %s", e)
            self.model = None

    def is_loaded(self) -> bool:
        """True si hay un modelo cargado."""
        return self.model is not None

    def can_infer(self) -> bool:
        """True si se puede inferir de verdad con el modelo."""
        return self.model is not None

    def predict(self, payload: Dict[str, Any], *, db=None) -> Dict[str, Any]:
        """Predice el riesgo para un unico caso."""
        raw_df = pd.DataFrame([payload])
        result_df = self.predict_batch(
            raw_df,
            ciudad=payload.get("ciudad"),
            icao=payload.get("icao"),
            db=db,
        )
        return result_df.iloc[0].to_dict()

    def predict_batch(
        self,
        raw_df: pd.DataFrame,
        *,
        ciudad: Optional[str] = None,
        icao: Optional[str] = None,
        db=None,
    ) -> pd.DataFrame:
        """
        Predice el riesgo para multiples casos.

        Devuelve el DataFrame de entrada mas las columnas:
            riesgo, confianza, model_status, prob_<CLASE>...
        """
        if not self.can_infer():
            return self._predict_mock(
                raw_df, motivo="modelo no disponible en el servicio"
            )

        try:
            completed, imputados = complete_raw_features(raw_df, icao=icao)
            X = build_features(
                completed, scaler=self.scaler, encoders=self.label_encoder
            )

            preds = self.model.predict(X)
            probs = self.model.predict_proba(X)
            classes = [str(c) for c in self.model.classes_]

            output = raw_df.copy()
            output["riesgo"] = [str(p) for p in preds]
            output["confianza"] = probs.max(axis=1)
            output["model_status"] = MODEL_STATUS_ML

            # Probabilidades reales del modelo, una columna por clase.
            for i, cls in enumerate(classes):
                output[f"prob_{cls}"] = probs[:, i]

            if imputados:
                logger.info(
                    "Prediccion con %d features imputadas: %s",
                    len(imputados),
                    ", ".join(imputados),
                )
            output.attrs["imputed_features"] = imputados

            if db is not None:
                self._persistir(output, probs, classes, raw_df, ciudad, icao, db)

            return output

        except FeaturePipelineError as e:
            return self._predict_mock(raw_df, motivo=f"pipeline de features: {e}")
        except Exception as e:
            logger.exception("Error inesperado en prediccion: %s", e)
            return self._predict_mock(raw_df, motivo=f"error de inferencia: {e}")

    def _persistir(self, output, probs, classes, raw_df, ciudad, icao, db) -> None:
        """Guarda las predicciones en base de datos."""
        try:
            for i in range(len(output)):
                record = RiskPrediction(
                    ciudad=ciudad,
                    icao=icao,
                    riesgo=str(output["riesgo"].iloc[i]),
                    confianza=float(probs[i].max()),
                    probabilidades={
                        cls: float(p) for cls, p in zip(classes, probs[i])
                    },
                    temperatura=raw_df.iloc[i].get("temperatura"),
                    humedad=raw_df.iloc[i].get("humedad"),
                    viento=raw_df.iloc[i].get("viento"),
                    visibilidad=raw_df.iloc[i].get("visibilidad"),
                )
                db.add(record)
            db.commit()
            logger.info("%d predicciones guardadas en BD", len(output))
        except Exception as e:
            logger.error("Error guardando predicciones en BD: %s", e)
            db.rollback()

    def _predict_mock(self, raw_df: pd.DataFrame, *, motivo: str) -> pd.DataFrame:
        """
        Prediccion por reglas, para cuando el modelo no esta disponible.

        Se registra como ERROR a proposito: si esto aparece en produccion,
        la API esta devolviendo heuristicas, no el modelo.
        """
        logger.error(
            "PREDICCION MOCK (no es el modelo entrenado). Motivo: %s", motivo
        )

        output = raw_df.copy()

        def evaluar(row):
            score = 0
            visibilidad = row.get("visibilidad", 10000)
            viento = row.get("viento", 0)
            humedad = row.get("humedad", 50)

            if visibilidad < 1000:
                score += 3
            elif visibilidad < 5000:
                score += 2

            if viento > 40:
                score += 3
            elif viento > 25:
                score += 2
            elif viento > 15:
                score += 1

            if humedad > 85:
                score += 1

            if score >= 4:
                return "ALTO"
            if score >= 2:
                return "MODERADO"
            return "BAJO"

        output["riesgo"] = raw_df.apply(evaluar, axis=1)
        # Sin modelo no hay confianza que reportar. Cero es honesto;
        # un 0.85 inventado no lo es.
        output["confianza"] = 0.0
        output["model_status"] = MODEL_STATUS_MOCK
        output["mock_reason"] = motivo
        for cls in RISK_LEVELS:
            output[f"prob_{cls}"] = float("nan")

        return output


# Instancia global usada por las rutas y el batch.
try:
    ml_service_v2 = MLServiceV2()
    if ml_service_v2.can_infer():
        logger.info("Servicio ML global inicializado con el modelo de produccion")
    else:
        logger.error("Servicio ML global inicializado en modo MOCK")
except Exception as e:
    logger.exception("Error inicializando el servicio ML global: %s", e)
    ml_service_v2 = None


# ==================== HELPERS PARA LAS RUTAS ====================

async def predict_risk_from_weather(
    weather_data: Dict[str, Any],
    *,
    icao: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Predice riesgo a partir de datos meteorologicos y arma la respuesta
    que consumen las rutas.
    """
    if ml_service_v2 is None:
        raise RuntimeError("Servicio ML no inicializado")

    # Solo se pasan al pipeline los campos que el cliente realmente aporta;
    # el resto lo completa complete_raw_features(), que ademas deja
    # constancia de que fueron imputados.
    campos = [
        "temperatura", "humedad", "viento", "visibilidad", "presion",
        "condicion", "descripcion", "direccion_viento", "rafagas",
        "precipitacion", "techo_nubes", "punto_rocio", "tipo_nubes",
        "turbulencia", "estado_pista", "tormenta_electrica",
        "cizalladura_viento", "riesgo_hielo",
    ]
    input_data = {k: weather_data[k] for k in campos if weather_data.get(k) is not None}

    raw_df = pd.DataFrame([input_data])
    result_df = ml_service_v2.predict_batch(raw_df, icao=icao)
    row = result_df.iloc[0].to_dict()

    risk_level = str(row["riesgo"])
    model_status = row.get("model_status", MODEL_STATUS_MOCK)

    response = {
        "risk_level": risk_level,
        "confidence": float(row["confianza"]),
        "probabilities": {
            cls: float(row[f"prob_{cls}"])
            for cls in RISK_LEVELS
            if f"prob_{cls}" in row and pd.notna(row[f"prob_{cls}"])
        },
        "risk_factors": _analyze_risk_factors(weather_data),
        "recommendations": _generate_recommendations(risk_level),
        "model_status": model_status,
        "imputed_features": result_df.attrs.get("imputed_features", []),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if model_status == MODEL_STATUS_MOCK:
        response["warning"] = (
            "Prediccion generada por reglas heuristicas, NO por el modelo "
            "entrenado. No usar con fines operacionales."
        )
        response["mock_reason"] = row.get("mock_reason")

    return response


def _analyze_risk_factors(weather_data: Dict[str, Any]) -> List[str]:
    """Enumera las condiciones que elevan el riesgo."""
    factors = []

    viento = weather_data.get("viento", 0)
    visibilidad = weather_data.get("visibilidad", 10000)
    humedad = weather_data.get("humedad", 50)

    if viento > 40:
        factors.append(f"Viento muy fuerte ({viento} km/h)")
    elif viento > 25:
        factors.append(f"Viento fuerte ({viento} km/h)")
    elif viento > 15:
        factors.append(f"Viento moderado ({viento} km/h)")

    if visibilidad < 1000:
        factors.append(f"Visibilidad muy reducida ({visibilidad}m)")
    elif visibilidad < 5000:
        factors.append(f"Visibilidad reducida ({visibilidad}m)")

    if humedad > 85:
        factors.append(f"Humedad muy alta ({humedad}%)")

    if not factors:
        factors.append("Condiciones dentro de parametros normales")

    return factors


def _generate_recommendations(risk_level: str) -> List[str]:
    """Recomendaciones operacionales por nivel de riesgo."""
    if risk_level == "ALTO":
        return [
            "Implementar restricciones operacionales",
            "Solo personal experimentado",
            "Procedimientos IFR obligatorios",
            "Monitoreo continuo de condiciones",
        ]
    if risk_level == "MODERADO":
        return [
            "Precaucion en operaciones",
            "Monitoreo frecuente de condiciones",
            "Briefing completo de tripulacion",
        ]
    return [
        "Operaciones normales permitidas",
        "Monitoreo estandar de condiciones",
    ]
