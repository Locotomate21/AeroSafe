"""
Servicio de pronostico de niebla/tormenta.

Sirve el modelo CALIBRADO a partir del METAR actual de un aeropuerto.
Devuelve una probabilidad que si es una probabilidad: tras la calibracion
(ECE 0.008), un 0.30 significa ~30% de ocurrencia real, que es lo que un
despachador necesita para decidir.

Composicion del pipeline (mismas piezas que en entrenamiento, para no
introducir desajuste train/serve):

    METAR crudo
      -> METARTAFService._parse_metar        (parseo)
      -> parsed_metar_to_schema              (a vocabulario del modelo)
      -> complete_raw_features(icao)         (rumbo de pista, altitud,
                                              viento cruzado, temporales)
      -> add_forecast_features               (precip, persistencia, ciclicas)
      -> modelo calibrado                    (probabilidad)
"""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from core.config import settings
from features.adapters.metar_adapter import parsed_metar_to_schema
from features.defaults import complete_raw_features
from features.forecast_features import FORECAST_FEATURES, add_forecast_features
from services.metar_taf_service import METARTAFService

logger = logging.getLogger(__name__)

HORIZONTE_H = 3
UMBRAL_ALERTA = 0.5  # sobre probabilidad calibrada: 50% de ocurrencia

MODEL_DIR = settings.BASE_DIR / "models" / "forecast"


class MetarNoDisponible(RuntimeError):
    """No se pudo descargar el METAR del aeropuerto (fuente externa)."""


class MetarIncompleto(ValueError):
    """El METAR llegó pero le faltan variables imprescindibles."""


def _nivel(prob: float) -> str:
    """Traduce la probabilidad calibrada a una etiqueta operacional."""
    if prob >= 0.60:
        return "ALTO"
    if prob >= 0.30:
        return "MODERADO"
    if prob >= 0.10:
        return "BAJO"
    return "MINIMO"


class ForecastService:
    """Carga el modelo calibrado de un aeropuerto y pronostica desde METAR."""

    def __init__(self, icao: str = "SKBO", horizonte: int = HORIZONTE_H):
        self.icao = icao.upper()
        self.horizonte = horizonte
        self.modelo = None
        self._metar = METARTAFService()

        # Se prefiere el modelo calibrado; si no existe, se cae al sin
        # calibrar avisando, porque sus "probabilidades" no son fiables.
        calibrado = MODEL_DIR / f"forecast_{self.icao.lower()}_h{horizonte}_calibrado.pkl"
        crudo = MODEL_DIR / f"forecast_{self.icao.lower()}_h{horizonte}.pkl"

        if calibrado.exists():
            self.modelo = joblib.load(calibrado)
            self.calibrado = True
            logger.info("Modelo de pronostico calibrado cargado: %s", calibrado.name)
        elif crudo.exists():
            self.modelo = joblib.load(crudo)
            self.calibrado = False
            logger.warning(
                "Modelo de pronostico SIN calibrar (%s): las probabilidades "
                "no son fiables como tal.", crudo.name
            )
        else:
            self.calibrado = False
            logger.error("No hay modelo de pronostico para %s", self.icao)

    def disponible(self) -> bool:
        return self.modelo is not None

    async def pronosticar(self, icao: Optional[str] = None) -> Dict[str, Any]:
        """
        Pronostica niebla/tormenta a +Nh a partir del METAR actual.

        Args:
            icao: aeropuerto a consultar. Debe coincidir con el del modelo
                  cargado; se acepta por comodidad de la ruta.
        """
        if not self.disponible():
            raise RuntimeError(f"Modelo de pronostico no disponible para {self.icao}")

        objetivo = (icao or self.icao).upper()

        # 1. METAR actual. Un fallo aqui es de la fuente externa (NOAA),
        # no del cliente: se distingue de un METAR incompleto.
        try:
            metar = await self._metar.get_metar_data(objetivo)
        except ValueError as e:
            raise MetarNoDisponible(str(e)) from e

        raw = metar.get("raw") or metar.get("raw_metar", "")
        parsed = metar if "temperature_c" in metar else self._metar._parse_metar(raw)

        base = parsed_metar_to_schema(parsed)
        if base is None:
            raise MetarIncompleto(
                f"El METAR de {objetivo} no trae temperatura o visibilidad; "
                f"no se puede pronosticar."
            )

        # 2. Momento de la observacion, para las features temporales.
        momento = _parse_momento(parsed.get("observation_time"))

        # 3. Completar features aeronauticas + de pronostico.
        df = pd.DataFrame([base])
        completo, imputadas = complete_raw_features(df, icao=objetivo, momento=momento)
        completo = add_forecast_features(completo)

        # 4. Prediccion calibrada. Se pasa como array (.values) porque el
        # modelo se entreno con arrays sin nombres de columna; pasar un
        # DataFrame con nombres dispara un UserWarning de sklearn.
        X = completo[FORECAST_FEATURES].values
        prob = float(self.modelo.predict_proba(X)[:, 1][0])

        return {
            "icao": objetivo,
            "horizonte_horas": self.horizonte,
            "objetivo": "niebla o tormenta",
            "probabilidad": round(prob, 4),
            "nivel": _nivel(prob),
            "alerta": prob >= UMBRAL_ALERTA,
            "condicion_actual": base["descripcion"],
            "es_adverso_ahora": bool(completo["adverso_actual"].iloc[0]),
            "modelo_calibrado": self.calibrado,
            "metar": raw,
            "observacion": momento.isoformat() if momento else None,
            "features_imputadas": imputadas,
            "generado": datetime.now(timezone.utc).isoformat(),
        }


def _parse_momento(texto: Optional[str]) -> Optional[datetime]:
    """
    Interpreta la hora de observacion del METAR.

    El METAR la reporta como DDHHMMZ (dia del mes + HHMM en UTC), p. ej.
    '231100Z'. Esta hora ALIMENTA las features temporales (hora, mes,
    es_noche, ciclicas), que el modelo aprendio en UTC. Si en vez de la
    hora del METAR se usara datetime.now(), las features temporales
    quedarian con la hora del reloj del servidor: un desajuste train/serve
    que hace que un METAR de las 11 UTC se evalue como si fueran las 3 UTC.
    Ese bug existio en la primera version y lo detecto la comparacion con
    los datos historicos.
    """
    ahora = datetime.now(timezone.utc)
    if not texto:
        return ahora

    # Formato ISO (por si la fuente ya lo entrega parseado).
    try:
        return datetime.fromisoformat(str(texto).replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        pass

    # Formato METAR crudo DDHHMMZ.
    t = str(texto).strip().rstrip("Z")
    if len(t) == 6 and t.isdigit():
        dia, hora, minuto = int(t[0:2]), int(t[2:4]), int(t[4:6])
        # El METAR solo trae dia/hora/minuto; el mes y el ano se toman del
        # momento actual. Si el dia es mayor que hoy, la observacion es del
        # mes anterior (cambio de mes).
        anio, mes = ahora.year, ahora.month
        if dia > ahora.day:
            mes -= 1
            if mes == 0:
                mes, anio = 12, anio - 1
        try:
            return datetime(anio, mes, dia, hora, minuto, tzinfo=timezone.utc)
        except ValueError:
            return ahora

    return ahora


# Cache de servicios por aeropuerto: cargar el .pkl en cada peticion seria
# lento. Se cargan bajo demanda y se reutilizan.
_servicios: Dict[str, ForecastService] = {}


def get_forecast_service(icao: str = "SKBO", horizonte: int = HORIZONTE_H) -> ForecastService:
    clave = f"{icao.upper()}_h{horizonte}"
    if clave not in _servicios:
        _servicios[clave] = ForecastService(icao, horizonte)
    return _servicios[clave]
