from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, func
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import logging

from models.database import WeatherRecord, RiskPrediction, Airport

logger = logging.getLogger(__name__)


class WeatherRepository:

    
    def __init__(self, db: Session):
        self.db = db
    
    # ==================== WEATHER RECORDS ====================
    
    def create_weather_record(self, data: Dict) -> WeatherRecord:
        try:
            record = WeatherRecord(**data)
            self.db.add(record)
            self.db.commit()
            self.db.refresh(record)
            
            logger.info(f"✅ Registro guardado: {record.ciudad} - {record.timestamp}")
            return record
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Error guardando registro: {str(e)}")
            raise
    
    def get_weather_by_id(self, record_id: int) -> Optional[WeatherRecord]:
        return self.db.query(WeatherRecord).filter(
            WeatherRecord.id == record_id
        ).first()
    
    def get_latest_weather(self, ciudad: str, limit: int = 10) -> List[WeatherRecord]:
        return self.db.query(WeatherRecord).filter(
            WeatherRecord.ciudad == ciudad
        ).order_by(
            desc(WeatherRecord.timestamp)
        ).limit(limit).all()
    
    def get_weather_by_airport(
        self, 
        icao: str, 
        hours: int = 24
    ) -> List[WeatherRecord]:
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return self.db.query(WeatherRecord).filter(
            and_(
                WeatherRecord.icao == icao,
                WeatherRecord.timestamp >= cutoff_time
            )
        ).order_by(
            desc(WeatherRecord.timestamp)
        ).all()
    
    def get_weather_range(
        self,
        ciudad: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[WeatherRecord]:
        return self.db.query(WeatherRecord).filter(
            and_(
                WeatherRecord.ciudad == ciudad,
                WeatherRecord.timestamp >= start_date,
                WeatherRecord.timestamp <= end_date
            )
        ).order_by(
            WeatherRecord.timestamp
        ).all()
    
    def get_dangerous_conditions(
        self,
        hours: int = 24,
        min_wind: float = 40,
        max_visibility: float = 2000
    ) -> List[WeatherRecord]:
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return self.db.query(WeatherRecord).filter(
            and_(
                WeatherRecord.timestamp >= cutoff_time,
                (
                    (WeatherRecord.viento >= min_wind) |
                    (WeatherRecord.visibilidad <= max_visibility)
                )
            )
        ).order_by(
            desc(WeatherRecord.timestamp)
        ).all()
    
    def get_weather_statistics(
        self,
        icao: str,
        days: int = 7
    ) -> Dict:
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        stats = self.db.query(
            func.avg(WeatherRecord.temperatura).label('temp_avg'),
            func.max(WeatherRecord.temperatura).label('temp_max'),
            func.min(WeatherRecord.temperatura).label('temp_min'),
            func.avg(WeatherRecord.humedad).label('humedad_avg'),
            func.avg(WeatherRecord.viento).label('viento_avg'),
            func.max(WeatherRecord.viento).label('viento_max'),
            func.avg(WeatherRecord.visibilidad).label('visibilidad_avg'),
            func.count(WeatherRecord.id).label('total_records')
        ).filter(
            and_(
                WeatherRecord.icao == icao,
                WeatherRecord.timestamp >= cutoff_time
            )
        ).first()
        
        if not stats:
            return {}
        
        return {
            "icao": icao,
            "period_days": days,
            "temperatura": {
                "promedio": round(stats.temp_avg, 2) if stats.temp_avg else None,
                "maxima": round(stats.temp_max, 2) if stats.temp_max else None,
                "minima": round(stats.temp_min, 2) if stats.temp_min else None,
            },
            "humedad_promedio": round(stats.humedad_avg, 2) if stats.humedad_avg else None,
            "viento": {
                "promedio": round(stats.viento_avg, 2) if stats.viento_avg else None,
                "maximo": round(stats.viento_max, 2) if stats.viento_max else None,
            },
            "visibilidad_promedio": round(stats.visibilidad_avg, 2) if stats.visibilidad_avg else None,
            "total_registros": stats.total_records
        }
    
    def delete_old_records(self, days: int = 30) -> int:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        deleted_count = self.db.query(WeatherRecord).filter(
            WeatherRecord.timestamp < cutoff_date
        ).delete()
        
        self.db.commit()
        logger.info(f"🗑️ Eliminados {deleted_count} registros antiguos")
        
        return deleted_count
    
    # ==================== RISK PREDICTIONS ====================
    
    def create_risk_prediction(self, data: Dict) -> RiskPrediction:
        try:
            prediction = RiskPrediction(**data)
            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)
            
            logger.info(f"✅ Predicción guardada: {prediction.riesgo} - {prediction.confianza:.2f}")
            return prediction
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Error guardando predicción: {str(e)}")
            raise
    
    def get_predictions_by_airport(
        self,
        icao: str,
        hours: int = 24
    ) -> List[RiskPrediction]:
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return self.db.query(RiskPrediction).filter(
            and_(
                RiskPrediction.icao == icao,
                RiskPrediction.timestamp >= cutoff_time
            )
        ).order_by(
            desc(RiskPrediction.timestamp)
        ).all()
    
    def get_risk_statistics(
        self,
        icao: str,
        days: int = 7
    ) -> Dict:
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        # Contar por nivel de riesgo
        risk_counts = self.db.query(
            RiskPrediction.riesgo,
            func.count(RiskPrediction.id).label('count')
        ).filter(
            and_(
                RiskPrediction.icao == icao,
                RiskPrediction.timestamp >= cutoff_time
            )
        ).group_by(
            RiskPrediction.riesgo
        ).all()
        
        total = sum(count for _, count in risk_counts)
        
        return {
            "icao": icao,
            "period_days": days,
            "total_predicciones": total,
            "distribucion": {
                riesgo: {
                    "cantidad": count,
                    "porcentaje": round((count / total * 100), 2) if total > 0 else 0
                }
                for riesgo, count in risk_counts
            }
        }
    
    # ==================== AIRPORTS ====================
    
    def get_airport(self, icao: str) -> Optional[Airport]:
        """Obtiene información de aeropuerto"""
        return self.db.query(Airport).filter(
            Airport.icao == icao
        ).first()
    
    def get_all_airports(self) -> List[Airport]:
        """Obtiene todos los aeropuertos"""
        return self.db.query(Airport).all()
    
    def create_airport(self, data: Dict) -> Airport:
        """Crea un nuevo aeropuerto"""
        try:
            airport = Airport(**data)
            self.db.add(airport)
            self.db.commit()
            self.db.refresh(airport)
            
            logger.info(f"✅ Aeropuerto creado: {airport.icao} - {airport.nombre}")
            return airport
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Error creando aeropuerto: {str(e)}")
            raise


def get_weather_repository(db: Session) -> WeatherRepository:
    return WeatherRepository(db)