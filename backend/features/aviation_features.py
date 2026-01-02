"""
Specific to aviation according to Colombian regulations RAC (Aeronautical Regulations of Colombia). 
"""

class AviationFeatureEngineer:
    """
    Generates features based on Colombian aeronautical regulations (RAC).
    """
    
    # Mínimos meteorológicos según RAC 121 (Aviación comercial)
    MINIMOS_RAC = {
        'CAT_I': {
            'visibility_min': 550,  # metros
            'ceiling_min': 200,     # pies
        },
        'CAT_II': {
            'visibility_min': 300,
            'ceiling_min': 100,
        },
        'VFR': {
            'visibility_min': 5000,
            'ceiling_min': 1500,
        }
    }
    
    def calcular_riesgo_aproximacion(self, weather_data: dict) -> dict:
        """
        Calcula riesgo según mínimos para aproximación ILS CAT I/II.
        """
        visibility = weather_data['visibility']
        ceiling = self._calculate_ceiling(weather_data['clouds'])
        
        features = {
            'cumple_cat_i': self._check_cat_i(visibility, ceiling),
            'cumple_cat_ii': self._check_cat_ii(visibility, ceiling),
            'margen_visibilidad': visibility - self.MINIMOS_RAC['CAT_I']['visibility_min'],
            'riesgo_windshear': self._detect_windshear(weather_data),
            'riesgo_cizalladura': self._calculate_wind_shear_risk(weather_data),
        }
        
        return features
    
    def calcular_riesgo_hielo(self, temp: float, dewpoint: float, altitude: int) -> float:
        """
        Calcula probabilidad de formación de hielo según temperatura y punto de rocío.
        Basado en normas FAA y OACI.
        """
        # Temperatura entre 0°C y -20°C + alta humedad = RIESGO ALTO
        if -20 <= temp <= 0:
            humidity_factor = 1 - abs(temp - dewpoint) / 10
            return min(1.0, humidity_factor * 1.5)
        return 0.0
    
    def validar_condiciones_despegue(self, weather: dict, runway: str) -> dict:
        """
        Valida si las condiciones cumplen para despegue según RAC.
        """
        return {
            'apto_despegue': True/False,
            'restricciones': [],
            'alternativas': [],
        }