"""Bounded, cancellable local tools, never a shell or a network fallback."""
import os
import subprocess
import time

from .domain import StopToken


def run_local(args: list[str], stop: StopToken, timeout: float = 180.0):
    if stop.cancelled():
        raise InterruptedError("Cancelado antes de iniciar herramienta")
    env = dict(os.environ, LC_ALL="C")
    child = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, env=env)
    start = time.monotonic()
    try:
        while True:
            if stop.cancelled():
                raise InterruptedError("Herramienta cancelada")
            if time.monotonic() - start > timeout:
                raise TimeoutError("Tiempo máximo de herramienta agotado")
            try:
                out, err = child.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                continue
        if child.returncode:
            raise RuntimeError(f"{args[0]} terminó con código {child.returncode}: {err[-2000:]}")
        return out, err
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                child.kill()
                child.communicate()
