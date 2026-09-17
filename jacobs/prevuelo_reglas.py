"""Jacobs — reglas PURAS del pre-vuelo (spec 2026-09-17 §4).

Sin DB ni red: reciben lo que el catálogo dijo (prevuelo_catalogo) y devuelven
violaciones y costo. El orquestador (jacobs/prevuelo.py) junta todo.

Los tipos viven acá y no en prevuelo.py para que sonda.py pueda importarlos
sin un import circular (prevuelo -> sonda -> prevuelo).

Cuatro chequeos (D4 de Fernando): tope suficiente, credencial viva, faceta
sana (la sonda, en el orquestador) y costo máximo. Un paso que no se puede
acotar NO bloquea: se declara (`hay_no_acotados`) y la Mesa pide confirmación.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal

from contrato_dispatch import ModelDispatchConfigError, _max_output_tokens_value, errores_del_contrato

from jacobs.models import Step

REGLAS = frozenset({
    "tope_insuficiente", "sin_contrato_de_salida", "credencial_ausente",
    "faceta_caida", "faceta_inexistente",
})

# Transportes que cobran por token y necesitan credencial de proveedor
# (§4.3). ollama es local y subprocess (hyde) es suscripción.
TRANSPORTES_QUE_COBRAN = frozenset({"http_openai_compat", "http_gemini"})

_UN_MILLON = Decimal(1_000_000)
# axioma_usage.cost_usd es DECIMAL(10,6): mismo grano, redondeado HACIA ARRIBA
# (un costo máximo que redondea para abajo deja de ser máximo).
_GRANO_USD = Decimal("0.000001")


@dataclass(frozen=True)
class Violacion:
    paso: int
    faceta: str
    regla: str
    detalle: str

    def __post_init__(self) -> None:
        if self.regla not in REGLAS:
            raise ValueError(f"regla de pre-vuelo desconocida: {self.regla!r}")

    def to_dict(self) -> dict:
        return {"paso": self.paso, "faceta": self.faceta, "regla": self.regla, "detalle": self.detalle}


@dataclass(frozen=True)
class CostoPaso:
    """Motivos: "acotado" (precio y tope conocidos), "sin_precio" (el catálogo
    no declara precio_in/precio_out), "local" (ollama, sin costo), "suscripcion"
    (subprocess -- hyde y cualquier otra faceta subprocess, R11), "mecanico"
    (assemble), "sin_contrato_de_salida" (bloqueado, incluye transporte
    desconocido -- R10) y "faceta_inexistente". "herramientas_sin_tope"
    (Ruling R9a, 2026-09-17): un motor con herramientas EN UN TRANSPORTE QUE
    COBRA (TRANSPORTES_QUE_COBRAN) no recorta el historial creciente ni los
    resultados de tools (worker.py:970) -- no hay tope de tokens de salida
    que acotar, así que NO se acota el costo (cuenta como no acotado,
    `hay_no_acotados`, aunque llamadas_max siga siendo el cálculo normal). Un
    motor con herramientas en ollama sigue siendo "local" ($0 siempre) y en
    subprocess "suscripcion" (Ruling R9a ronda 2, 2026-09-17): "sin tope" solo
    tiene sentido cuando hay un precio real que podría dispararse."""
    paso: int
    faceta: str
    modelo: str | None
    llamadas_max: int
    tokens_in_max: int
    tokens_out_max: int
    usd_max: Decimal | None
    motivo: str

    def to_dict(self) -> dict:
        return {
            "paso": self.paso, "faceta": self.faceta, "modelo": self.modelo,
            "llamadas_max": self.llamadas_max, "tokens_in_max": self.tokens_in_max,
            "tokens_out_max": self.tokens_out_max,
            "usd_max": None if self.usd_max is None else str(self.usd_max),
            "motivo": self.motivo,
        }


@dataclass(frozen=True)
class Veredicto:
    ok: bool
    violaciones: tuple[Violacion, ...]
    costo_max_usd: Decimal
    pasos_costo: tuple[CostoPaso, ...]
    sondeadas: tuple[str, ...]

    @property
    def hay_no_acotados(self) -> bool:
        return any(c.usd_max is None for c in self.pasos_costo)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "violaciones": [v.to_dict() for v in self.violaciones],
            "costo_max_usd": str(self.costo_max_usd),
            "pasos_costo": [c.to_dict() for c in self.pasos_costo],
            "sondeadas": list(self.sondeadas),
            "hay_no_acotados": self.hay_no_acotados,
        }


@dataclass(frozen=True)
class Despacho:
    """Lo que el despacho REAL de un paso va a usar, leído del catálogo."""
    clave_salud: str           # faceta HTTP, o el motor que resuelve MotorPolicy
    via_motor: bool
    transporte: str            # http_openai_compat | http_gemini | ollama | subprocess
    provider_id: str
    base_url: str | None
    modelo: str
    max_tokens_param: str | None
    max_output_tokens: int | None
    motor_max_tokens: int      # 0 = sin presupuesto propio (solo via_motor)
    precio_in: Decimal | None  # USD por millón
    precio_out: Decimal | None
    tiene_herramientas: bool
    schema_con_reintento: bool
    persona: str | None


_TRANSPORTES_CONOCIDOS = frozenset({"http_openai_compat", "http_gemini", "ollama", "subprocess"})


def errores_de_contrato(d: Despacho) -> list[str]:
    """Los MISMOS validadores que el despacho: http_openai_compat exige nombre
    y tope (contrato_dispatch.faltantes_del_contrato); ollama y http_gemini
    exigen el tope de la fila porque sin él no hay costo máximo (§4.4,
    desvío 5). subprocess no lleva (R11: lo maneja evaluar_paso ANTES de
    llegar acá, así que este caso no se ejercita desde ahí, pero se deja
    fail-open a propósito por si algo más lo llama directo).

    Un transporte fuera de este conjunto (Ruling R10, ronda de arreglo 1) NO
    es "sin contrato que exigir": es uno que el pre-vuelo todavía no conoce.
    Antes de R10 cualquier transporte desconocido caía en la rama de
    http_gemini/ollama, y si max_output_tokens venía seteado (aunque ese
    transporte ni lo usara) pasaba sin error -- un transporte nuevo real
    quedaba sin cubrir en vez de rechazado. Fail-closed: se rechaza hasta que
    alguien le enseñe la regla al pre-vuelo."""
    if d.transporte == "subprocess":
        return []
    if d.transporte == "http_openai_compat":
        return [str(e) for _, e in errores_del_contrato(d.modelo, d.max_tokens_param, d.max_output_tokens)]
    if d.transporte in ("http_gemini", "ollama"):
        try:
            _max_output_tokens_value(d.modelo, d.max_output_tokens)
        except ModelDispatchConfigError as e:
            return [str(e)]
        return []
    return [
        f"transporte '{d.transporte}' desconocido: el pre-vuelo solo sabe evaluar "
        f"{sorted(_TRANSPORTES_CONOCIDOS)}; enseñale la regla antes de despachar por acá",
    ]


def tope_efectivo(d: Despacho) -> tuple[int, str]:
    """El mismo cálculo que worker._limite_del_motor: el menor entre
    motor.max_tokens (0 = sin presupuesto) y el tope del catálogo. Presupone
    contrato válido."""
    if d.via_motor and d.motor_max_tokens and d.motor_max_tokens < d.max_output_tokens:
        return d.motor_max_tokens, "motor.max_tokens"
    return d.max_output_tokens, "model.max_output_tokens"


def llamadas_max(d: Despacho, max_iteraciones: int) -> int:
    """Desvío 2 del plan. Motor: el bucle de worker.py está acotado por
    MOTOR_TOOL_LOOP_MAX_ITERATIONS y el reintento de schema es una iteración
    de ese bucle. HTTP gemini: _invoke_http_gemini reintenta sin grounding."""
    if d.via_motor:
        if d.tiene_herramientas:
            return max_iteraciones
        return min(2 if d.schema_con_reintento else 1, max_iteraciones)
    if d.transporte == "http_gemini":
        return 2
    return 1


def tokens_de_entrada(chars: int, chars_por_token: int) -> int:
    return -(-chars // chars_por_token)


def costo_usd(llamadas: int, tokens_in: int, tokens_out: int,
              precio_in: Decimal, precio_out: Decimal) -> Decimal:
    bruto = Decimal(llamadas) * (Decimal(tokens_in) * precio_in + Decimal(tokens_out) * precio_out) / _UN_MILLON
    return bruto.quantize(_GRANO_USD, rounding=ROUND_CEILING)


def costo_usd_acumulado_motor(llamadas: int, tokens_in: int, tokens_out: int,
                               precio_in: Decimal, precio_out: Decimal) -> Decimal:
    """Ruling R9b (ronda de arreglo 1): en un motor SIN herramientas con
    reintento de schema (worker.py:832-838, rama `validation_retried`), la
    llamada k reenvía la respuesta de las k-1 llamadas previas como parte del
    prompt -- a diferencia del reintento de grounding de Gemini (mismo
    payload, R9c), acá la entrada CRECE. La llamada k lleva
    tokens_in + (k-1)*tokens_out de entrada; el costo es la SUMA sobre las
    `llamadas` llamadas, no `llamadas` veces la misma -- costo_usd
    subestimaría (spec §4.6: "sobreestima, nunca subestima")."""
    bruto = sum(
        (Decimal(tokens_in + (k - 1) * tokens_out) * precio_in + Decimal(tokens_out) * precio_out)
        for k in range(1, llamadas + 1)
    ) / _UN_MILLON
    return bruto.quantize(_GRANO_USD, rounding=ROUND_CEILING)


def costo_sin_evaluar(step: Step) -> CostoPaso | None:
    """Pasos que no consultan catálogo: `assemble` es mecánico
    (executor._assemble_mechanical) y `hyde` es suscripción (sin tope,
    credencial ni sonda, igual que el canario de la Mesa)."""
    if step.capability == "assemble":
        return CostoPaso(step.step_index, step.facet, None, 0, 0, 0, Decimal(0), "mecanico")
    if step.facet == "hyde":
        return CostoPaso(step.step_index, step.facet, None, 0, 0, 0, Decimal(0), "suscripcion")
    return None


def evaluar_paso(
    paso: int,
    faceta: str,
    d: Despacho | None,
    *,
    min_output_tokens: int,
    credencial_activa: bool,
    chars_entrada: int,
    chars_por_token: int,
    max_iteraciones: int,
    detalle_inexistente: str = "",
) -> tuple[list[Violacion], CostoPaso]:
    if d is None:
        detalle = detalle_inexistente or f"la faceta '{faceta}' no tiene fila activa con binding primario"
        return ([Violacion(paso, faceta, "faceta_inexistente", detalle)],
                CostoPaso(paso, faceta, None, 0, 0, 0, None, "faceta_inexistente"))

    if d.transporte == "subprocess":
        # R11 (ronda de arreglo 1): subprocess (hyde, y cualquier otra faceta
        # que despache por acá) no declara max_output_tokens -- tope_efectivo
        # comparando None < min_output_tokens revienta con TypeError. Es
        # suscripción, igual que costo_sin_evaluar() para hyde: sin tope,
        # credencial ni sonda que chequear.
        return [], CostoPaso(paso, faceta, d.modelo, 0, 0, 0, Decimal(0), "suscripcion")

    violaciones: list[Violacion] = []
    tokens_in = tokens_de_entrada(chars_entrada, chars_por_token)
    llamadas = llamadas_max(d, max_iteraciones)

    errores = errores_de_contrato(d)
    tokens_out = 0
    if errores:
        violaciones.append(Violacion(paso, faceta, "sin_contrato_de_salida", " | ".join(errores)))
    else:
        tokens_out, que_subir = tope_efectivo(d)
        if tokens_out < min_output_tokens:
            de_quien = d.clave_salud if que_subir == "motor.max_tokens" else d.modelo
            violaciones.append(Violacion(
                paso, faceta, "tope_insuficiente",
                f"tope efectivo {tokens_out} < capability.min_output_tokens {min_output_tokens}: "
                f"subí {que_subir} de '{de_quien}' a {min_output_tokens} o más",
            ))

    if d.transporte in TRANSPORTES_QUE_COBRAN and not credencial_activa:
        violaciones.append(Violacion(
            paso, faceta, "credencial_ausente",
            f"el proveedor '{d.provider_id}' no tiene credencial con state='active' en `credential`",
        ))

    if errores:
        costo = CostoPaso(paso, faceta, d.modelo, llamadas, tokens_in, 0, None, "sin_contrato_de_salida")
    elif d.via_motor and d.tiene_herramientas and d.transporte in TRANSPORTES_QUE_COBRAN:
        # R9a (ronda de arreglo 2): sin tope de costo conocido, pero SOLO en
        # un transporte que cobra por token (ver docstring de CostoPaso).
        # ollama es gratis pase lo que pase en el bucle de herramientas --
        # "sin tope" no significa nada cuando el costo real es siempre $0
        # (motivo "local", más abajo); subprocess ya salió antes de acá
        # (motivo "suscripcion", ver el corte temprano al inicio de esta
        # función) y nunca llega a esta rama.
        costo = CostoPaso(paso, faceta, d.modelo, llamadas, tokens_in, tokens_out, None, "herramientas_sin_tope")
    elif d.transporte == "ollama":
        costo = CostoPaso(paso, faceta, d.modelo, llamadas, tokens_in, tokens_out, Decimal(0), "local")
    elif d.precio_in is None or d.precio_out is None:
        costo = CostoPaso(paso, faceta, d.modelo, llamadas, tokens_in, tokens_out, None, "sin_precio")
    elif d.via_motor and llamadas > 1:
        # R9b: motor sin herramientas con reintento de schema -- acumula.
        # (Gemini nunca entra acá: via_motor es False en ese transporte.)
        costo = CostoPaso(paso, faceta, d.modelo, llamadas, tokens_in, tokens_out,
                          costo_usd_acumulado_motor(llamadas, tokens_in, tokens_out, d.precio_in, d.precio_out),
                          "acotado")
    else:
        costo = CostoPaso(paso, faceta, d.modelo, llamadas, tokens_in, tokens_out,
                          costo_usd(llamadas, tokens_in, tokens_out, d.precio_in, d.precio_out),
                          "acotado")
    return violaciones, costo


def armar_veredicto(violaciones: list[Violacion], costos: list[CostoPaso],
                    sondeadas: list[str]) -> Veredicto:
    total = sum((c.usd_max for c in costos if c.usd_max is not None), Decimal(0))
    return Veredicto(
        ok=not violaciones,
        violaciones=tuple(violaciones),
        costo_max_usd=total,
        pasos_costo=tuple(sorted(costos, key=lambda c: c.paso)),
        sondeadas=tuple(sorted(sondeadas)),
    )
