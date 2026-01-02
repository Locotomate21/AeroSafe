"""
Validación con casos reales documentados de SKBO
"""

def test_bogota_storm_2024_03_15():
    """
    Caso real: Tormenta eléctrica severa marzo 15, 2024
    Resultado esperado: RIESGO ALTO
    Eventos reales: 12 vuelos desviados, 8 cancelados
    """
    weather_data = {
        'metar': 'METAR SKBO 151800Z 27020G35KT 1200 +TSRA...',
        # ... datos reales del evento
    }
    
    prediction = model.predict(weather_data)
    
    assert prediction['riesgo'] == 'ALTO'
    assert prediction['confidence'] > 0.8
    # Validar que las razones coinciden con el evento real