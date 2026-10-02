"""El rele del Puerto: stdio de un lado, el socket Unix `/run/faro/<run_id>.sock` del otro.

Es lo unico del Faro que viaja DENTRO de la jaula del motor (spec §3 «Transporte»), asi que es
minimo a proposito: SOLO biblioteca estandar (una prueba lo vigila: no importa `mcp`, ni
pydantic, ni nada pesado), sin logica de protocolo (copia bytes en los dos sentidos) y SIN
identidad propia: quien pide lo dice el socket (las credenciales del par) y el TOKEN de la
ejecucion, que el rele lee de un ARCHIVO montado en la jaula y manda como primera linea
(`FARO-TOKEN <token>\n`). El token nunca viaja por argv ni por entorno.

    python -m jax.faro.relay --socket /faro/puerto.sock --token-file /faro/token

La jaula recibe por bind de archivo SOLO esos dos (ver el plan, argv de bwrap).

Cierre: si el cliente cierra su stdin, el rele cierra solo el lado de escritura del socket
(`write_eof`) y sigue copiando hasta que el Puerto responda lo pendiente y cierre. Termina con 0
cuando cualquiera de los dos extremos cierra del todo; con 2 si no puede leer el token o conectar.
Por stdout solo sale protocolo; los mensajes propios van a stderr.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

_TROZO = 65536


async def _copiar(origen: asyncio.StreamReader, destino: asyncio.StreamWriter) -> None:
    while True:
        datos = await origen.read(_TROZO)
        if not datos:
            return
        destino.write(datos)
        await destino.drain()


async def _correr(ruta: str, ruta_token: str) -> int:
    try:
        with open(ruta_token, encoding="utf-8") as f:
            token = f.read().strip()
    except OSError as exc:
        print(f"faro-relay: no se pudo leer el archivo del token {ruta_token}: {exc.strerror}", file=sys.stderr)
        return 2
    if not token:
        print(f"faro-relay: el archivo del token {ruta_token} esta vacio", file=sys.stderr)
        return 2
    try:
        lector_sock, escritor_sock = await asyncio.open_unix_connection(ruta)
    except OSError as exc:
        print(f"faro-relay: no se pudo conectar a {ruta}: {exc}", file=sys.stderr)
        return 2
    escritor_sock.write(b"FARO-TOKEN " + token.encode() + b"\n")
    del token
    bucle = asyncio.get_running_loop()
    lector_in = asyncio.StreamReader()
    await bucle.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(lector_in), sys.stdin.buffer)
    transporte, protocolo = await bucle.connect_write_pipe(asyncio.streams.FlowControlMixin, sys.stdout.buffer)
    escritor_out = asyncio.StreamWriter(transporte, protocolo, None, bucle)
    ida = asyncio.create_task(_copiar(lector_in, escritor_sock))
    vuelta = asyncio.create_task(_copiar(lector_sock, escritor_out))
    try:
        hechas, _ = await asyncio.wait({ida, vuelta}, return_when=asyncio.FIRST_COMPLETED)
        if vuelta not in hechas:
            # El cliente cerro su stdin: half-close, y se espera lo pendiente del Puerto.
            if escritor_sock.can_write_eof():
                escritor_sock.write_eof()
            await vuelta
    finally:
        for t in (ida, vuelta):
            t.cancel()
        await asyncio.gather(ida, vuelta, return_exceptions=True)
        escritor_sock.close()
        escritor_out.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="jax.faro.relay", allow_abbrev=False, description="Rele stdio <-> socket Unix del Puerto del Faro")
    ap.add_argument("--socket", required=True, help="ruta del socket Unix del run")
    ap.add_argument("--token-file", required=True, help="archivo con el token de la ejecucion (nunca argv ni entorno)")
    args = ap.parse_args(argv)
    try:
        return asyncio.run(_correr(args.socket, args.token_file))
    except (BrokenPipeError, ConnectionResetError):
        return 0


if __name__ == "__main__":
    sys.exit(main())
