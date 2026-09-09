"""Ejecución supervisada, límites de memoria y cierre de procesos hijos."""
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

_processes = set()
_lock = threading.Lock()
_closing = False


def memory_limit():
    explicit = os.environ.get('WEB_MEMORY_LIMIT_MB')
    if explicit:
        value = int(explicit)
        if value <= 0:
            raise ValueError('WEB_MEMORY_LIMIT_MB debe ser positivo.')
        return value * 1024**2
    for path in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            value = int(Path(path).read_text().strip())
            if 0 < value < 2**60:
                return value
        except (OSError, ValueError):
            pass
    return None


def memory_usage():
    # cgroup incluye todos los procesos y la caché que cuenta para Render.
    if sys.platform != 'win32':
        for path in ('/sys/fs/cgroup/memory.current', '/sys/fs/cgroup/memory/memory.usage_in_bytes'):
            try:
                return int(Path(path).read_text().strip())
            except (OSError, ValueError):
                pass
    process = psutil.Process()
    total = 0
    for child in [process, *process.children(recursive=True)]:
        try:
            total += child.memory_info().rss
        except psutil.Error:
            pass
    return total


def stop_tree(process):
    if sys.platform != 'win32':
        # El grupo sobrevive aunque el padre haya terminado antes que un nieto.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        try:
            parent = psutil.Process(process.pid)
            children = parent.children(recursive=True)
            for child in reversed(children):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            parent.kill()
            psutil.wait_procs([parent, *children], timeout=5)
        except psutil.NoSuchProcess:
            pass
    process.wait(timeout=10)


def reopen():
    global _closing
    with _lock:
        _closing = False


def shutdown():
    global _closing
    with _lock:
        _closing = True
        for process in tuple(_processes):
            stop_tree(process)


def run(command, *, cwd, log, timeout):
    limit = memory_limit()
    # Reservar margen para que la API pueda responder y guardar el fallo.
    ceiling = int(limit * .85) if limit else None
    if ceiling and memory_usage() >= ceiling:
        raise RuntimeError('No hay memoria disponible para iniciar el trabajo. Reduce la carga o aumenta la RAM.')
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'ML_NUM_THREADS'):
        env.setdefault(name, '1')
    with _lock:
        if _closing:
            raise RuntimeError('El servicio se está cerrando. Vuelve a intentarlo después del reinicio.')
        process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=sys.platform != 'win32',
                                   creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
        _processes.add(process)
    started = time.monotonic()
    peak = 0
    try:
        while process.poll() is None:
            used = memory_usage()
            peak = max(peak, used)
            if ceiling and used >= ceiling:
                raise RuntimeError('Se detuvo el trabajo al alcanzar el margen de memoria del servidor. Usa modo Base o una instancia con más RAM.')
            if time.monotonic()-started > timeout:
                raise RuntimeError(f'Se superó el límite de {timeout} segundos.')
            time.sleep(.2)
        return process.returncode, round(peak/1024**2, 2)
    finally:
        with _lock:
            stop_tree(process)
            _processes.discard(process)

