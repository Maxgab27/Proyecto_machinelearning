"""Inicia API y frontend locales. Ctrl+C detiene únicamente sus procesos hijos."""
import argparse
import importlib.util
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Iniciar Mercado Lab localmente')
    parser.add_argument('--sin-abrir', action='store_true', help='No abrir el navegador automáticamente')
    parser.add_argument('--check', action='store_true', help='Comprobar requisitos sin iniciar servidores')
    args = parser.parse_args()
    node = shutil.which('node')
    vite = ROOT/'frontend'/'node_modules'/'vite'/'bin'/'vite.js'
    missing = [name for name in ['fastapi','uvicorn','multipart','psutil'] if importlib.util.find_spec(name) is None]
    if missing:
        parser.exit(1, 'Faltan dependencias web. Ejecuta: .\\.venv\\Scripts\\python.exe -m pip install -r requirements-web.txt\n')
    if not node or not vite.is_file():
        parser.exit(1, 'Instala Node.js y ejecuta: npm.cmd ci --prefix frontend\n')
    if args.check:
        print('Python web, Node.js y Vite disponibles.')
        return
    children = []
    try:
        children.append(subprocess.Popen([sys.executable,'-m','uvicorn','webapp.api:app','--host','127.0.0.1','--port','8000'], cwd=ROOT))
        children.append(subprocess.Popen([node,str(vite),'--host','127.0.0.1','--port','8501','--strictPort'], cwd=ROOT/'frontend'))
        print('\nMercado Lab: http://127.0.0.1:8501/\nMantén esta terminal abierta. Ctrl+C para detener.\n', flush=True)
        opened = False
        started = time.monotonic()
        while True:
            if any(child.poll() is not None for child in children):
                raise RuntimeError('Un servidor terminó. Revisa los mensajes anteriores; los puertos 8000 y 8501 deben estar libres.')
            if not opened:
                try:
                    for url in ['http://127.0.0.1:8000/api/health','http://127.0.0.1:8501/']:
                        with urllib.request.urlopen(url, timeout=1) as response:
                            if response.status != 200:
                                raise OSError('Servidor no listo')
                    opened = True
                    if not args.sin_abrir:
                        webbrowser.open('http://127.0.0.1:8501/')
                except OSError:
                    if time.monotonic()-started > 60:
                        raise RuntimeError('El arranque superó un minuto. Revisa los mensajes de la terminal.')
            time.sleep(.5)
    except KeyboardInterrupt:
        print('\nCerrando Mercado Lab...')
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        for child in reversed(children):
            if child.poll() is None:
                if sys.platform == 'win32':
                    subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    child.terminate()
                child.wait(timeout=10)
    return 0


if __name__ == '__main__':
    sys.exit(main())
