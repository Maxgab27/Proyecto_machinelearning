"""Servicio local; un entrenamiento a la vez y salidas aisladas por ejecución."""
import json
import os
import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from .presentacion import dashboard

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/'web_runs'
UPLOADS = ROOT/'web_uploads'
lock = threading.Lock()
active = None
app = FastAPI(title='Mercado Lab API', version='1.0.0')
origins = [o.strip() for o in os.environ.get('WEB_ORIGINS', 'http://localhost:8501,http://127.0.0.1:8501').split(',') if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=['GET','POST'], allow_headers=['Content-Type'])


@app.middleware('http')
async def local_origin(request: Request, call_next):
    # Rechazar POST desde páginas ajenas, también si usan un formulario simple.
    if request.method == 'POST' and request.headers.get('origin') not in [None, *origins]:
        from fastapi.responses import JSONResponse
        return JSONResponse({'detail': 'Origen no permitido.'}, status_code=403)
    return await call_next(request)


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
    return {'service': 'mercado-lab', 'status': 'ok', 'active': active}


@app.get('/api/dashboard/{run_id}')
def get_dashboard(run_id: str):
    try:
        return dashboard(run_dir(run_id))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get('/api/runs')
def list_runs():
    states = []
    if RUNS.exists():
        for path in RUNS.glob('*/estado.json'):
            state = json.loads(path.read_text(encoding='utf-8'))
            if state['status'] == 'running' and state['id'] != active:
                state.update(status='failed', error='El servicio se reinició antes de completar este análisis.')
            states.append(state)
    return sorted(states, key=lambda r:r['created'], reverse=True)[:30]


@app.get('/api/runs/{run_id}')
def status(run_id: str):
    folder = run_dir(run_id)
    if not (folder/'estado.json').is_file():
        raise HTTPException(404, 'Estado no disponible.')
    state = json.loads((folder/'estado.json').read_text(encoding='utf-8'))
    if state['status'] == 'running' and active != run_id:
        state.update(status='failed', error='El servicio se reinició durante el análisis.')
    log = folder/'ejecucion.log'
    state['log'] = log.read_text(encoding='utf-8', errors='replace')[-5000:] if log.exists() else ''
    return state


@app.post('/api/upload')
async def upload(file: UploadFile = File(...)):
    if not (file.filename or '').lower().endswith('.csv'):
        raise HTTPException(422, 'Selecciona un archivo CSV.')
    content = await file.read(10*1024*1024+1)
    await file.close()
    if len(content) > 10*1024*1024:
        raise HTTPException(413, 'El límite es 10 MB por archivo.')
    UPLOADS.mkdir(exist_ok=True)
    token = uuid.uuid4().hex
    path = UPLOADS/f'{token}.csv'
    path.write_bytes(content)
    try:
        from mercado.datos import preparar
        df, audit = preparar(path)
        if len(df)>50000:
            raise ValueError('El límite local es 50.000 observaciones útiles.')
        if not re.fullmatch(r'[A-Za-z0-9.^=_-]{1,24}', audit['ticker']):
            raise ValueError('El ticker debe ser un identificador bursátil de hasta 24 caracteres.')
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(422, f'No se pudo validar el CSV: {exc}') from exc
    return {'id': token, 'name': Path(file.filename).name, 'audit': audit}


class RunRequest(BaseModel):
    mode: Literal['base','completo'] = 'completo'
    epochs: int = Field(default=30, ge=1, le=50)
    upload_id: str | None = None


def worker(folder, state, csvpath, request):
    global active
    try:
        command = [sys.executable, '-X', 'utf8', str(ROOT/'main.py'), '--csv', str(csvpath),
                   '--salida', str(folder), '--modo', request.mode, '--epochs', str(request.epochs)]
        with (folder/'ejecucion.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
            try:
                code = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise RuntimeError('Se superó el límite de 30 minutos.')
        if code != 0 or not (folder/'COMPLETADO.json').exists():
            raise RuntimeError('El análisis no terminó correctamente. Consulta el registro de ejecución.')
        state['status'] = 'completed'
    except Exception as exc:
        state.update(status='failed', error=str(exc))
    finally:
        state['finished'] = datetime.now(timezone.utc).isoformat()
        save_state(folder, state)
        with lock:
            active = None


@app.post('/api/runs', status_code=202)
def start(request: RunRequest):
    global active
    csvpath = ROOT/'datos'/'arch_financiero.csv'
    if request.upload_id:
        if not re.fullmatch(r'[a-f0-9]{32}', request.upload_id):
            raise HTTPException(422, 'Identificador de CSV inválido.')
        csvpath = UPLOADS/f'{request.upload_id}.csv'
        if not csvpath.is_file():
            raise HTTPException(404, 'Vuelve a cargar el CSV.')
    with lock:
        if active:
            raise HTTPException(409, 'Ya hay un análisis en ejecución. Espera a que termine.')
        run_id = uuid.uuid4().hex
        folder = RUNS/run_id
        folder.mkdir(parents=True)
        state = {'id':run_id, 'status':'running', 'mode':request.mode,
                 'dataset':'CSV cargado' if request.upload_id else 'GGAL original',
                 'created':datetime.now(timezone.utc).isoformat()}
        save_state(folder, state)
        active = run_id
        threading.Thread(target=worker, args=(folder,state,csvpath,request), daemon=True).start()
    return state


@app.get('/api/artifacts/{run_id}/{filename:path}')
def artifact(run_id: str, filename: str):
    folder = run_dir(run_id).resolve()
    target = (folder/filename).resolve()
    if (not target.is_relative_to(folder) or 'modelos' in target.relative_to(folder).parts
            or target.suffix not in {'.csv','.html','.md','.png','.json','.npz'} or not target.is_file()):
        raise HTTPException(404, 'Archivo no disponible.')
    return FileResponse(target, headers={'X-Content-Type-Options':'nosniff'})
