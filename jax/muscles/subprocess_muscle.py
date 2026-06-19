"""
JAX 2.0 — SubprocessMuscle (Hyde).

Hyde es la faceta tecnica de JAX: Claude via Claude Code (binario `claude`)
invocado como subproceso async en modo headless (-p).

Decisiones firmes (Fernando + DeepSeek + Claude), medidas en hall9000:
  - Salida: --output-format text (texto limpio a stdout, parseo cero).
  - Modelo: --model con alias corto (sonnet/opus/haiku). Verificado.
  - Identidad: --append-system-prompt (suma a Claude Code, no lo reemplaza).
  - --fallback-model APAGADO: el fallo duro de modelo lo maneja JAX, no la CLI.
  - Invocacion: create_subprocess_exec (lista de args, sin shell). Sin inyeccion.
  - Prompt: por argumento, truncado a 32k chars.
  - Timeout: proc.kill() (en asyncio kill() = SIGKILL, muerte inmediata — es lo
    que queremos para un proceso colgado) + await proc.wait() para cosechar
    el zombie. Sin esto quedan huerfanos y con concurrencia=1 traban todo.

MEMORIA DE CONVERSACION (caso especial de Hyde):
  Las otras facetas (HTTP/Ollama) reciben el historial como array de mensajes
  con roles. Pero `claude -p` recibe UN solo prompt de texto, no un array.
  Por eso, para Hyde el historial se SERIALIZA a texto y se antepone al
  mensaje actual, marcado claramente como contexto. Es la unica via limpia
  sin meternos en --input-format stream-json (eso queda para Fase 2 si hace
  falta streaming). El truncado a 32k se aplica al conjunto (contexto+mensaje).

En memoria de Jairo Urbina.
"""

from __future__ import annotations

import asyncio
import os

from jax.muscles.base import Muscle, MuscleInvocationError, MuscleTimeoutError

MAX_PROMPT_CHARS = 32000


class SubprocessMuscle(Muscle):

    def __init__(
        self,
        name: str,
        model_default: str,
        models_allowed: list[str],
        system_prompt: str,
        timeout: float,
        cli_path: str = "claude",
        workspace_dir: str = "/home/fruiz/jax/workspace",
        authority_origin: str = "",
    ) -> None:
        super().__init__(
            name, model_default, models_allowed, system_prompt, timeout,
            authority_origin=authority_origin,
        )
        self.cli_path = cli_path
        self.workspace_dir = workspace_dir
        os.makedirs(self.workspace_dir, exist_ok=True)


    def _serialize_history(self, history: list[dict] | None) -> str:
        """Convierte el historial neutro a texto, para anteponerlo al prompt.
        Hyde no tiene array de mensajes, asi que el contexto va como texto
        marcado. Vacio si no hay historial."""
        if not history:
            return ""
        lineas = ["[Contexto de la conversacion hasta ahora:]"]
        for m in history:
            quien = "Fernando" if m["role"] == "user" else "Vos (JAX)"
            lineas.append(f"{quien}: {m['content']}")
        lineas.append(
            "[Fin del contexto. Responde unicamente al ultimo mensaje de "
            "Fernando, tomando en cuenta lo anterior.]"
        )
        return "\n".join(lineas) + "\n\n"

    def _sanitize(self, prompt: str) -> str:
        if len(prompt) > MAX_PROMPT_CHARS:
            return prompt[:MAX_PROMPT_CHARS] + "\n[...truncado por JAX...]"
        return prompt

    def _check_error(self, returncode: int | None, stderr: str) -> None:
        # Error real: codigo de salida distinto de 0.
        if returncode != 0:
            raise MuscleInvocationError(
                f"[{self.name}] claude exit {returncode}: {stderr[:200]}"
            )
        # Codigo 0 pero stderr con senal de error -> tambien error.
        low = stderr.lower()
        if any(t in low for t in ("error", "fatal", "exception", "failed")):
            raise MuscleInvocationError(
                f"[{self.name}] error en stderr: {stderr[:200]}"
            )
        # Warnings benignos: se ignoran (se podrian loguear en Fase 2).

    async def _call(
        self, prompt: str, model: str, history: list[dict] | None = None
    ) -> str:
        # Anteponer el contexto serializado, luego truncar el conjunto.
        contexto = self._serialize_history(history)
        safe_prompt = self._sanitize(contexto + prompt)

        cmd = [
            self.cli_path,
            "--model", model,
            "--append-system-prompt", self.system_prompt,
            "--print",
            "--output-format", "text",
            "--permission-mode", "acceptEdits",
            "--allowedTools", "Write,Edit,Read,Bash",
            "--add-dir", self.workspace_dir,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.workspace_dir,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=safe_prompt.encode("utf-8")),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            # kill() = SIGKILL en asyncio. Matamos el proceso colgado y
            # cosechamos el zombie con wait() antes de propagar el error.
            proc.kill()
            await proc.wait()
            raise MuscleTimeoutError(
                f"[{self.name}] sin respuesta en {self.timeout}s"
            )

        stdout_str = stdout.decode("utf-8", errors="replace")
        stderr_str = stderr.decode("utf-8", errors="replace")

        self._check_error(proc.returncode, stderr_str)

        return stdout_str.strip()
