"""Servicio local; un entrenamiento a la vez y salidas aisladas por ejecución."""
import json
import os
import re
import secrets
import sys
import threading
import uuid
import tempfile
import heapq
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException, Request
from starlette.datastructures import UploadFile
from starlette.concurrency import run_in_threadpool
from starlette.formparsers import MultiPartException
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from .presentacion import dashboard
from . import runtime

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get('WEB_DATA_DIR', str(ROOT))).resolve()
RUNS = DATA_ROOT/'web_runs'
UPLOADS = DATA_ROOT/'web_uploads'
RUN_TOKEN = os.environ.get('WEB_RUN_TOKEN', '')
lock = threading.Lock()
active = None
active_thread = None
capacity = threading.BoundedSemaphore(1)
MAX_UPLOAD = 10*1024*1024
MAX_ROWS = int(os.environ.get('WEB_MAX_ROWS', '50000'))


def allow_full():
    return os.environ.get('WEB_ALLOW_FULL', 'false' if os.environ.get('RENDER') else 'true').lower() == 'true'


@asynccontextmanager
async def lifespan(app):
    runtime.reopen()
    runtime.memory_limit()  # Fallar en el arranque si hay un límite inválido.
    if MAX_ROWS < 180 or int(os.environ.get('WEB_STORAGE_LIMIT_MB', '512')) <= 0:
        raise ValueError('Configura WEB_MAX_ROWS >= 180 y WEB_STORAGE_LIMIT_MB > 0.')
    try:
        yield
    finally:
        await run_in_threadpool(runtime.shutdown)
        if active_thread is not None:
            await run_in_threadpool(active_thread.join, 15)


app = FastAPI(title='Mercado Lab API', version='1.0.0', lifespan=lifespan)
origins = [o.strip() for o in os.environ.get('WEB_ORIGINS', 'http://localhost:8501,http://127.0.0.1:8501').split(',') if o.strip()]


class RequestPolicy:
    """Autorización antes de recibir cuerpos; ASGI directo sin tareas auxiliares."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http' and scope.get('method') == 'POST':
            from starlette.datastructures import Headers
            from fastapi.responses import JSONResponse
            headers = Headers(scope=scope)
            error, code = None, None
            if headers.get('origin') not in [None, *origins]:
                error, code = 'Origen no permitido.', 403
            elif os.environ.get('RENDER') and not RUN_TOKEN:
                error, code = 'Configura WEB_RUN_TOKEN en el servidor antes de ejecutar análisis.', 503
            elif RUN_TOKEN and not secrets.compare_digest(headers.get('authorization', '').encode(), ('Bearer '+RUN_TOKEN).encode()):
                error, code = 'Introduce la clave de ejecución correcta en Nuevo análisis.', 401
            if error:
                return await JSONResponse({'detail':error}, status_code=code)(scope, receive, send)
        await self.app(scope, receive, send)


def run_dir(run_id):
    if run_id == 'actual':
        return ROOT/'resultados'
    if not re.fullmatch(r'[a-f0-9]{32}', run_id):
        raise HTTPException(404, 'Análisis inexistente.')
    folder = RUNS/run_id
    if not folder.is_dir():
        raise HTTPException(404, 'Análisis inexistente.')
    return folder


def save_state(folder, state):
    temp = folder/'estado.tmp'
    temp.write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')
    temp.replace(folder/'estado.json')


@app.get('/api/health')
def health():
    return {'service': 'mercado-lab', 'status': 'ok', 'active': active,
            'key_required': bool(RUN_TOKEN) or bool(os.environ.get('RENDER')),
            'allowed_modes': ['base', 'completo'] if allow_full() else ['base'],
            'max_upload_mb': MAX_UPLOAD//1024**2, 'max_rows': MAX_ROWS,
            'default_csv_available': (ROOT/'datos'/'arch_financiero.csv').is_file()}


@app.get('/api/dashboard/{run_id}')
def get_dashboard(run_id: str):
    try:
        return dashboard(run_dir(run_id))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get('/api/runs')
def list_runs():
    def states():
        for path in RUNS.glob('*/estado.json'):
            try:
                state = json.loads(path.read_text(encoding='utf-8'))
                if not isinstance(state, dict) or not {'id', 'status', 'created'} <= state.keys():
                    continue
            except (OSError, ValueError):
                continue
            if state['status'] == 'running' and state['id'] != active:
                state.update(status='failed', error='El servicio se reinició antes de completar este análisis.')
            yield state
    return heapq.nlargest(30, states(), key=lambda r:r['created'])


@app.get('/api/runs/{run_id}')
def status(run_id: str):
    folder = run_dir(run_id)
    if not (folder/'estado.json').is_file():
        raise HTTPException(404, 'Estado no disponible.')
    state = json.loads((folder/'estado.json').read_text(encoding='utf-8'))
    if state['status'] == 'running' and active != run_id:
        state.update(status='failed', error='El servicio se reinició durante el análisis.')
    log = folder/'ejecucion.log'
    state['log'] = ''
    if log.exists():
        with log.open('rb') as handle:
            handle.seek(max(0, log.stat().st_size-20000))
            state['log'] = handle.read(20000).decode('utf-8', errors='replace')[-5000:]
    return state


def check_storage(reserve):
    limit = int(os.environ.get('WEB_STORAGE_LIMIT_MB', '512'))*1024**2
    used = sum(p.stat().st_size for root in (UPLOADS, RUNS) if root.exists()
               for p in root.rglob('*') if p.is_file())
    if used+reserve > limit:
        raise HTTPException(507, 'El almacenamiento de análisis está lleno. Descarga y limpia ejecuciones antiguas antes de continuar.')


def validate_csv(path):
    with tempfile.TemporaryDirectory(prefix='.validacion-', dir=UPLOADS) as tmp:
        result = Path(tmp)/'resultado.json'
        with (Path(tmp)/'validacion.log').open('w', encoding='utf-8') as log:
            code, _ = runtime.run([sys.executable, '-X', 'utf8', '-m', 'webapp.validate_csv',
                                   str(path), str(result), str(MAX_ROWS)],
                                  cwd=ROOT, log=log, timeout=60)
        if code or not result.exists():
            raise ValueError('La validación no terminó correctamente.')
        payload = json.loads(result.read_text(encoding='utf-8'))
        if 'error' in payload:
            raise ValueError(payload['error'])
        return payload['audit']


@app.post('/api/upload')
async def upload(request: Request):
    if not capacity.acquire(blocking=False):
        raise HTTPException(409, 'Hay una validación o análisis en ejecución. Espera a que termine.')
    path = None
    accepted = False
    try:
        check_storage(MAX_UPLOAD)
        try:
            async with request.form(max_files=1, max_fields=0) as form:
                file = form.get('file')
                if not isinstance(file, UploadFile) or not (file.filename or '').lower().endswith('.csv'):
                    raise HTTPException(422, 'Selecciona un archivo CSV.')
                UPLOADS.mkdir(parents=True, exist_ok=True)
                token = uuid.uuid4().hex
                path = UPLOADS/f'{token}.csv'
                size = 0
                with path.open('wb') as handle:
                    while chunk := await file.read(64*1024):
                        size += len(chunk)
                        if size > MAX_UPLOAD:
                            raise HTTPException(413, 'El límite es 10 MB por archivo.')
                        handle.write(chunk)
                name = Path(file.filename).name
        except StarletteHTTPException as exc:
            if exc.detail == 'Carga demasiado grande.':
                raise HTTPException(413, 'El límite es 10 MB por archivo.') from exc
            raise
        try:
            audit = await run_in_threadpool(validate_csv, path)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(422, f'No se pudo validar el CSV: {exc}') from exc
        accepted = True
        return {'id': token, 'name': name, 'audit': audit}
    finally:
        try:
            if path is not None and not accepted:
                path.unlink(missing_ok=True)
        finally:
            capacity.release()


class RunRequest(BaseModel):
    mode: Literal['base','completo'] = 'base'
    epochs: int = Field(default=30, ge=1, le=50)
    upload_id: str | None = None


def worker(folder, state, csvpath, request):
    global active
    try:
        command = [sys.executable, '-X', 'utf8', str(ROOT/'main.py'), '--csv', str(csvpath),
                   '--salida', str(folder), '--modo', request.mode, '--epochs', str(request.epochs)]
        command.extend(['--max-filas', str(MAX_ROWS)])
        with (folder/'ejecucion.log').open('w', encoding='utf-8') as log:
            code, peak = runtime.run(command, cwd=ROOT, log=log, timeout=1800)
        state['peak_memory_mb'] = peak
        if code != 0 or not (folder/'COMPLETADO.json').exists():
            raise RuntimeError('El análisis no terminó correctamente. Consulta el registro de ejecución.')
        state['status'] = 'completed'
    except Exception as exc:
        state.update(status='failed', error=str(exc))
    finally:
        state['finished'] = datetime.now(timezone.utc).isoformat()
        try:
            save_state(folder, state)
        finally:
            with lock:
                active = None
            capacity.release()


@app.post('/api/runs', status_code=202)
def start(request: RunRequest):
    global active, active_thread
    if request.mode == 'completo' and not allow_full():
        raise HTTPException(422, 'Este servidor admite modo Base. Para habilitar las redes, configura WEB_ALLOW_FULL=true en una instancia con RAM suficiente.')
    csvpath = ROOT/'datos'/'arch_financiero.csv'
    if request.upload_id:
        if not re.fullmatch(r'[a-f0-9]{32}', request.upload_id):
            raise HTTPException(422, 'Identificador de CSV inválido.')
        csvpath = UPLOADS/f'{request.upload_id}.csv'
        if not csvpath.is_file():
            raise HTTPException(404, 'Vuelve a cargar el CSV.')
    if not csvpath.is_file():
        raise HTTPException(422, 'El CSV original no está disponible. Selecciona y carga un archivo CSV.')
    if not capacity.acquire(blocking=False):
        raise HTTPException(409, 'Hay una validación o análisis en ejecución. Espera a que termine.')
    try:
        check_storage(64*1024**2)
        with lock:
            run_id = uuid.uuid4().hex
            folder = RUNS/run_id
            folder.mkdir(parents=True)
            state = {'id':run_id, 'status':'running', 'mode':request.mode,
                     'dataset':'CSV cargado' if request.upload_id else 'GGAL original',
                     'created':datetime.now(timezone.utc).isoformat()}
            save_state(folder, state)
            active = run_id
            active_thread = threading.Thread(target=worker, args=(folder,state,csvpath,request), daemon=True)
            active_thread.start()
    except BaseException:
        with lock:
            active = None
        capacity.release()
        raise
    return state


@app.get('/api/artifacts/{run_id}/{filename:path}')
def artifact(run_id: str, filename: str):
    folder = run_dir(run_id).resolve()
    target = (folder/filename).resolve()
    if (not target.is_relative_to(folder) or 'modelos' in target.relative_to(folder).parts
            or target.suffix not in {'.csv','.html','.md','.png','.json','.npz'} or not target.is_file()):
        raise HTTPException(404, 'Archivo no disponible.')
    return FileResponse(target, headers={'X-Content-Type-Options':'nosniff'})


class BodyLimit:
    """Limitar el multipart durante la recepción, también sin Content-Length."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('method') != 'POST':
            return await self.app(scope, receive, send)
        maximum = MAX_UPLOAD+64*1024 if scope['path'] == '/api/upload' else 16*1024
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            received += len(message.get('body', b''))
            if received > maximum:
                raise MultiPartException('Carga demasiado grande.')
            return message

        from fastapi.responses import JSONResponse
        headers = dict(scope.get('headers', []))
        try:
            length = int(headers.get(b'content-length', b'0'))
        except ValueError:
            return await JSONResponse({'detail': 'Content-Length inválido.'}, status_code=400)(scope, receive, send)
        if length > maximum:
            return await JSONResponse({'detail': 'La solicitud supera el tamaño permitido.'}, status_code=413)(scope, receive, send)
        await self.app(scope, limited_receive, send)


app.add_middleware(BodyLimit)
app.add_middleware(RequestPolicy)
# CORS debe envolver también las respuestas 401/503 del control de ejecución.
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=['GET','POST'], allow_headers=['Content-Type','Authorization'])
