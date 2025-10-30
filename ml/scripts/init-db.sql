-- Crear extensiones útiles
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Crear índices adicionales si es necesario
-- (Las tablas se crean automáticamente con SQLAlchemy)

-- Insertar aeropuertos colombianos
INSERT INTO airports (icao, iata, nombre, ciudad, pais, latitud, longitud, elevacion, activo)
VALUES 
    ('SKBO', 'BOG', 'El Dorado', 'Bogotá', 'Colombia', 4.7016, -74.1469, 2548, true),
    ('SKCL', 'CLO', 'Alfonso Bonilla Aragón', 'Cali', 'Colombia', 3.5432, -76.3816, 965, true),
    ('SKMR', 'EOH', 'Olaya Herrera', 'Medellín', 'Colombia', 6.2205, -75.5909, 1499, true),
    ('SKRG', 'MDE', 'José María Córdova', 'Rionegro', 'Colombia', 6.1645, -75.4233, 2142, true),
    ('SKCG', 'CTG', 'Rafael Núñez', 'Cartagena', 'Colombia', 10.4424, -75.5130, 1, true),
    ('SKBQ', 'BGA', 'Palonegro', 'Bucaramanga', 'Colombia', 7.1265, -73.1848, 1191, true),
    ('SKPE', 'PEI', 'Matecaña', 'Pereira', 'Colombia', 4.8127, -75.7395, 1343, true)
ON CONFLICT (icao) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO aerosafe;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO aerosafe;