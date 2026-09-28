from jax.ejecutor.codigo.diff import Cambio


def C(ruta, mas=(), estado="M"):
    return Cambio(ruta, estado, None, tuple(mas), ())
