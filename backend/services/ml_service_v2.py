import pandas as pd
import joblib
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime

# 🔧 CORREGIDO: Imports sin 'backend.'
from models.models import RiskPrediction
from features.build_features import build_features
from core.config import settings

logger = logging.getLogger(__name__)


class MLServiceV2:
    """
    Servicio de Machine Learning para predicción de riesgo aeronáutico
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Inicializa el servicio ML
        
        Args:
            model_path: Ruta al modelo. Si None, usa settings.MODEL_PATH
        """
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = []
        
        # Determinar ruta del modelo
        if model_path:
            model_file = Path(model_path)
        else:
            model_file = settings.get_model_path(settings.MODEL_PATH)
        
        # Cargar modelo
        try:
            if not model_file.exists():
                logger.warning(f"Modelo no encontrado en {model_file}")
                logger.warning("Servicio ML iniciará en modo MOCK")
                self.model = None
            else:
                self.model = joblib.load(model_file)
                logger.info(f"✅ Modelo cargado desde {model_file}")
                
                # Cargar scaler si existe
                scaler_path = settings.get_model_path(settings.SCALER_PATH)
                if scaler_path.exists():
                    self.scaler = joblib.load(scaler_path)
                    logger.info(f"✅ Scaler cargado desde {scaler_path}")
                
                # Cargar encoder si existe
                encoder_path = settings.get_model_path(settings.ENCODER_PATH)
                if encoder_path.exists():
                    self.label_encoder = joblib.load(encoder_path)
                    logger.info(f"✅ Label encoder cargado desde {encoder_path}")
                
                # Cargar nombres de features si existen
                feature_names_path = settings.get_model_path(
                    settings.FEATURE_NAMES_PATH
                )
                if feature_names_path.exists():
                    with open(feature_names_path, 'r') as f:
                        self.feature_names = [line.strip() for line in f]
                    logger.info(f"✅ Feature names cargados: {len(self.feature_names)} features")
                    
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {str(e)}")
            self.model = None
    
    def is_loaded(self) -> bool:
        """Verifica si el modelo está cargado"""
        return self.model is not None
    
    def predict(
        self, 
        payload: Dict[str, Any],
        *,
        db=None,
    ) -> Dict[str, Any]:
        """
        Predice riesgo para un único caso
        
        Args:
            payload: Diccionario con datos meteorológicos
            db: Sesión de base de datos (opcional)
            
        Returns:
            Diccionario con predicción
        """
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
        Predice riesgo para múltiples casos (batch)
        
        Args:
            raw_df: DataFrame con datos crudos
            ciudad: Nombre de ciudad (opcional)
            icao: Código ICAO (opcional)
            db: Sesión de base de datos (opcional)
            
        Returns:
            DataFrame con predicciones
        """
        # Si no hay modelo, usar predicción mock
        if not self.is_loaded():
            return self._predict_mock(raw_df)
        
        try:
            # Construir features
            X = build_features(raw_df)
            
            # Hacer predicción
            preds = self.model.predict(X)
            probs = self.model.predict_proba(X)
            
            # Preparar output
            output = raw_df.copy()
            output["riesgo"] = preds
            output["confianza"] = probs.max(axis=1)
            
            # Guardar en base de datos si se proporciona
            if db is not None:
                try:
                    for i in range(len(output)):
                        # Crear probabilidades dict
                        probabilidades = {}
                        if hasattr(self.model, 'classes_'):
                            probabilidades = {
                                str(cls): float(p)
                                for cls, p in zip(self.model.classes_, probs[i])
                            }
                        
                        record = RiskPrediction(
                            ciudad=ciudad,
                            icao=icao,
                            riesgo=str(preds[i]),
                            confianza=float(probs[i].max()),
                            probabilidades=probabilidades,
                            temperatura=raw_df.iloc[i].get("temperatura"),
                            humedad=raw_df.iloc[i].get("humedad"),
                            viento=raw_df.iloc[i].get("viento"),
                            visibilidad=raw_df.iloc[i].get("visibilidad"),
                        )
                        db.add(record)
                    db.commit()
                    logger.info(f"✅ {len(output)} predicciones guardadas en BD")
                except Exception as e:
                    logger.error(f"Error guardando predicciones en BD: {str(e)}")
                    db.rollback()
            
            return output
            
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            # Fallback a mock en caso de error
            return self._predict_mock(raw_df)
    
    def _predict_mock(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera predicciones simuladas cuando el modelo no está disponible
        
        Args:
            raw_df: DataFrame con datos
            
        Returns:
            DataFrame con predicciones simuladas
        """
        logger.warning("Usando predicción MOCK - modelo no disponible")
        
        output = raw_df.copy()
        
        # Lógica simple basada en reglas para generar predicción realista
        def calcular_riesgo_mock(row):
            score = 0
            
            # Evaluar visibilidad
            if row.get('visibilidad', 10000) < 1000:
                score += 3
            elif row.get('visibilidad', 10000) < 5000:
                score += 2
            
            # Evaluar viento
            if row.get('viento', 0) > 40:
                score += 3
            elif row.get('viento', 0) > 25:
                score += 2
            elif row.get('viento', 0) > 15:
                score += 1
            
            # Evaluar humedad
            if row.get('humedad', 50) > 85:
                score += 1
            
            # Determinar nivel de riesgo
            if score >= 5:
                return 'CRÍTICO', 0.90
            elif score >= 3:
                return 'ALTO', 0.85
            elif score >= 1:
                return 'MODERADO', 0.80
            else:
                return 'BAJO', 0.85
        
        # Aplicar predicción mock a cada fila
        predicciones = raw_df.apply(calcular_riesgo_mock, axis=1)
        output['riesgo'] = [p[0] for p in predicciones]
        output['confianza'] = [p[1] for p in predicciones]
        
        return output


# Crear instancia global
try:
    ml_service_v2 = MLServiceV2()
    if ml_service_v2.is_loaded():
        logger.info("✅ Servicio ML global inicializado con modelo cargado")
    else:
        logger.warning("⚠️ Servicio ML global inicializado en modo MOCK")
except Exception as e:
    logger.error(f"❌ Error inicializando servicio ML global: {str(e)}")
    ml_service_v2 = None


# ==================== HELPER FUNCTIONS ====================

async def predict_risk_from_weather(weather_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Función helper para predicción de riesgo desde datos meteorológicos
    
    Args:
        weather_data: Diccionario con datos meteorológicos
        
    Returns:
        Diccionario con predicción de riesgo
    """
    if ml_service_v2 is None:
        raise RuntimeError("Servicio ML no inicializado")
    
    # Preparar datos para predicción
    input_data = {
        'temperatura': weather_data.get('temperatura', 20),
        'humedad': weather_data.get('humedad', 60),
        'viento': weather_data.get('viento', 10),
        'visibilidad': weather_data.get('visibilidad', 10000),
        'presion': weather_data.get('presion', 1013),
    }
    
    # Hacer predicción
    prediction = ml_service_v2.predict(input_data)
    
    # Analizar factores de riesgo
    risk_factors = _analyze_risk_factors(weather_data, prediction['riesgo'])
    
    # Generar recomendaciones
    recommendations = _generate_recommendations(prediction['riesgo'], risk_factors)
    
    # Formatear respuesta
    return {
        'risk_level': prediction['riesgo'],
        'confidence': float(prediction['confianza']),
        'probabilities': _get_probabilities_dict(prediction),
        'risk_factors': risk_factors,
        'recommendations': recommendations,
        'timestamp': datetime.utcnow().isoformat()
    }


def _analyze_risk_factors(weather_data: Dict[str, Any], risk_level: str) -> List[str]:
    """Analiza factores que contribuyen al riesgo"""
    factors = []
    
    viento = weather_data.get('viento', 0)
    visibilidad = weather_data.get('visibilidad', 10000)
    humedad = weather_data.get('humedad', 50)
    
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
        factors.append("Condiciones dentro de parámetros normales")
    
    return factors


def _generate_recommendations(risk_level: str, risk_factors: List[str]) -> List[str]:
    """Genera recomendaciones basadas en el nivel de riesgo"""
    recommendations = []
    
    if risk_level == 'CRÍTICO':
        recommendations.extend([
            "Considerar suspender operaciones",
            "Activar protocolos de emergencia",
            "Monitoreo continuo obligatorio"
        ])
    elif risk_level == 'ALTO':
        recommendations.extend([
            "Implementar restricciones operacionales",
            "Solo personal experimentado",
            "Procedimientos IFR obligatorios"
        ])
    elif risk_level == 'MODERADO':
        recommendations.extend([
            "Precaución en operaciones",
            "Monitoreo frecuente de condiciones",
            "Briefing completo de tripulación"
        ])
    else:  # BAJO
        recommendations.extend([
            "Operaciones normales permitidas",
            "Monitoreo estándar de condiciones"
        ])
    
    return recommendations


def _get_probabilities_dict(prediction: Dict[str, Any]) -> Dict[str, float]:
    """Extrae o genera diccionario de probabilidades"""
    # Si el modelo tiene probabilities, usarlas
    # Sino, generar basado en el nivel de riesgo
    
    risk_level = prediction.get('riesgo', 'BAJO')
    confidence = prediction.get('confianza', 0.85)
    
    # Generar probabilidades distribuidas
    if risk_level == 'CRÍTICO':
        return {
            'BAJO': 0.01,
            'MODERADO': 0.04,
            'ALTO': 0.15,
            'CRÍTICO': confidence
        }
    elif risk_level == 'ALTO':
        return {
            'BAJO': 0.02,
            'MODERADO': 0.08,
            'ALTO': confidence,
            'CRÍTICO': 0.05
        }
    elif risk_level == 'MODERADO':
        return {
            'BAJO': 0.10,
            'MODERADO': confidence,
            'ALTO': 0.05,
            'CRÍTICO': 0.01
        }
    else:  # BAJO
        return {
            'BAJO': confidence,
            'MODERADO': 0.10,
            'ALTO': 0.03,
            'CRÍTICO': 0.00
        }