"""Excepciones específicas del agente. Cada una lleva un mensaje accionable."""


class AgenteError(Exception):
    """Base."""


class MexcError(AgenteError):
    """MEXC sin respuesta, con límite de solicitudes o con error persistente."""


class DatosInvalidosError(AgenteError):
    """Velas vacías, desordenadas, duplicadas o con valores no válidos."""


class DatoAtrasadoError(AgenteError):
    """La última vela cerrada supera la antigüedad máxima permitida."""


class OllamaNoDisponibleError(AgenteError):
    """Ollama detenido o inaccesible."""


class PropuestaInvalidaError(AgenteError):
    """La respuesta del LLM no es JSON válido o contiene valores no permitidos."""


class OrdenRechazadaError(AgenteError):
    """El simulador Paper rechazó la orden (saldo, tamaño mínimo, etc.)."""


class ConfiguracionError(AgenteError):
    """Configuración faltante o fuera de límites."""
