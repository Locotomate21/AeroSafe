"""
Recolectar datos históricos de SKBO con eventos reales documentados.
"""

class HistoricalDataCollector:
    """
    Recolecta datos históricos y los correlaciona con eventos operacionales.
    """
    
    def collect_skbo_historical(self, start_date: str, end_date: str):
        """
        Recolecta METAR/TAF históricos + eventos operacionales.
        
        Fuentes:
        - NOAA Aviation Weather (históricos METAR)
        - Aerocivil (reportes de incidentes)
        - NOTAM históricos
        """
        pass
    
    def label_operational_events(self, weather_df: pd.DataFrame):
        """
        Labela eventos como:
        - Desvíos por clima
        - Cancelaciones
        - Aproximaciones frustradas
        - Demoras por clima
        """
        pass