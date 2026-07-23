"""
Configuración de seguridad.

La API estuvo abierta con API keys de ejemplo hardcodeadas en el código
(`dev-key-123,prod-key-456`), DEBUG=true por defecto filtrando trazas, y
un rate limiter implementado que nunca se conectó. Estos tests fijan ese
comportamiento para que no vuelva sin que nadie se entere.
"""
import importlib

import pytest
from fastapi.testclient import TestClient

from api.dependencies import RateLimiter, verify_api_key
from core.config import Settings


# =========================================================================
# Configuración por defecto
# =========================================================================

def test_no_hay_api_keys_por_defecto():
    """
    Unas credenciales escritas en el repositorio son credenciales
    públicas. Si REQUIRE_API_KEY se activa, las claves tienen que venir
    del entorno.
    """
    config = Settings(_env_file=None)
    assert config.get_valid_api_keys() == set()


def test_debug_desactivado_por_defecto():
    """Un despliegue que olvide definir DEBUG debe quedar en modo seguro."""
    config = Settings(_env_file=None)
    assert config.DEBUG is False


def test_auditoria_detecta_api_key_sin_claves():
    config = Settings(_env_file=None, REQUIRE_API_KEY=True, VALID_API_KEYS="")
    problemas = " ".join(config.validate_security())
    assert "VALID_API_KEYS" in problemas


def test_auditoria_detecta_api_abierta_en_produccion():
    config = Settings(_env_file=None, DEBUG=False, REQUIRE_API_KEY=False)
    problemas = " ".join(config.validate_security())
    assert "sin autenticación" in problemas


def test_auditoria_detecta_debug_activo():
    config = Settings(_env_file=None, DEBUG=True)
    problemas = " ".join(config.validate_security())
    assert "DEBUG=true" in problemas


def test_auditoria_detecta_cors_abierto():
    config = Settings(_env_file=None, ALLOWED_ORIGINS="*")
    problemas = " ".join(config.validate_security())
    assert "ALLOWED_ORIGINS" in problemas


def test_configuracion_segura_no_reporta_problemas():
    config = Settings(
        _env_file=None,
        DEBUG=False,
        REQUIRE_API_KEY=True,
        VALID_API_KEYS="clave-larga-y-aleatoria",
        ALLOWED_ORIGINS="https://aerosafe.example.com",
    )
    assert config.validate_security() == []


def test_claves_vacias_se_descartan():
    """'a,,b,' no debe producir una clave vacía que autentique a nadie."""
    config = Settings(_env_file=None, VALID_API_KEYS="a,,b, ")
    assert config.get_valid_api_keys() == {"a", "b"}


# =========================================================================
# Autenticación
# =========================================================================

@pytest.mark.asyncio
async def test_sin_api_key_requerida_pasa_todo(monkeypatch):
    from core import config as config_mod

    monkeypatch.setattr(config_mod.settings, "REQUIRE_API_KEY", False)
    assert await verify_api_key(None) == "public"


@pytest.mark.asyncio
async def test_api_key_faltante_da_401(monkeypatch):
    from core import config as config_mod
    from fastapi import HTTPException

    monkeypatch.setattr(config_mod.settings, "REQUIRE_API_KEY", True)
    monkeypatch.setattr(config_mod.settings, "VALID_API_KEYS", "clave-valida")

    with pytest.raises(HTTPException) as exc:
        await verify_api_key(None)
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_api_key_invalida_da_401(monkeypatch):
    from core import config as config_mod
    from fastapi import HTTPException

    monkeypatch.setattr(config_mod.settings, "REQUIRE_API_KEY", True)
    monkeypatch.setattr(config_mod.settings, "VALID_API_KEYS", "clave-valida")

    with pytest.raises(HTTPException) as exc:
        await verify_api_key("clave-incorrecta")
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_api_key_valida_pasa(monkeypatch):
    from core import config as config_mod

    monkeypatch.setattr(config_mod.settings, "REQUIRE_API_KEY", True)
    monkeypatch.setattr(config_mod.settings, "VALID_API_KEYS", "clave-valida")

    assert await verify_api_key("clave-valida") == "clave-valida"


# =========================================================================
# Rate limiting
# =========================================================================

def test_rate_limiter_deja_pasar_bajo_el_limite():
    limiter = RateLimiter()
    for _ in range(5):
        assert limiter.check_rate_limit("ip:1.2.3.4", max_requests=5) is True


def test_rate_limiter_bloquea_al_superar_el_limite():
    limiter = RateLimiter()
    for _ in range(3):
        limiter.check_rate_limit("ip:1.2.3.4", max_requests=3)

    assert limiter.check_rate_limit("ip:1.2.3.4", max_requests=3) is False


def test_rate_limiter_aisla_por_cliente():
    """El exceso de una IP no debe penalizar a otra."""
    limiter = RateLimiter()
    for _ in range(3):
        limiter.check_rate_limit("ip:1.1.1.1", max_requests=3)

    assert limiter.check_rate_limit("ip:2.2.2.2", max_requests=3) is True


def test_rate_limiter_expira_la_ventana():
    """Pasada la ventana, la cuota se restablece."""
    limiter = RateLimiter()
    limiter.check_rate_limit("ip:1.2.3.4", max_requests=1, window_seconds=0)
    assert limiter.check_rate_limit("ip:1.2.3.4", max_requests=1, window_seconds=0) is True


def test_middleware_de_rate_limit_responde_429(monkeypatch):
    """
    Regresión: RATE_LIMIT_ENABLED y MAX_REQUESTS_PER_MINUTE existían en la
    configuración pero ningún middleware los aplicaba.
    """
    import main

    monkeypatch.setattr(main.settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(main.settings, "MAX_REQUESTS_PER_MINUTE", 3)
    main.rate_limiter.requests.clear()

    client = TestClient(main.app)

    codigos = [client.get("/").status_code for _ in range(5)]

    assert 429 in codigos, f"El rate limit no se aplicó: {codigos}"
    assert codigos[:3] == [200, 200, 200]


def test_health_exento_de_rate_limit(monkeypatch):
    """El orquestador consulta /health constantemente; no debe agotar cuota."""
    import main

    monkeypatch.setattr(main.settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(main.settings, "MAX_REQUESTS_PER_MINUTE", 2)
    main.rate_limiter.requests.clear()

    client = TestClient(main.app)

    for _ in range(5):
        assert client.get("/health").status_code != 429


# =========================================================================
# Fuga de información
# =========================================================================

def test_error_500_no_filtra_traza_sin_debug(monkeypatch):
    """
    Con DEBUG=false el cliente recibe un mensaje genérico, no str(exc).
    """
    import main

    monkeypatch.setattr(main.settings, "DEBUG", False)
    monkeypatch.setattr(main.settings, "RATE_LIMIT_ENABLED", False)

    client = TestClient(main.app, raise_server_exceptions=False)

    @main.app.get("/_boom_test")
    async def _boom():
        raise RuntimeError("detalle interno secreto: password=hunter2")

    r = client.get("/_boom_test")

    assert r.status_code == 500
    assert "hunter2" not in r.text
    assert "secreto" not in r.text
