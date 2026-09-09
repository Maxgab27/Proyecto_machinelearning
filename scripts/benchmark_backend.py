"""Prueba HTTP local con datos sintéticos y medición de toda la familia de procesos.

Ejemplo: python scripts/benchmark_backend.py --mode base --memory-limit-mb 512
Los archivos se guardan en tmp/; nunca se sustituyen los resultados del proyecto.
"""
import argparse
import csv
import json
import math
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import httpx
import psutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['base', 'completo'], default='base')
    parser.add_argument('--rows', type=int, default=6014)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--memory-limit-mb', type=int)
    args = parser.parse_args()
    folder = ROOT/'tmp'/'backend-audit'/f'http-{args.mode}-{time.time_ns()}'
    folder.mkdir(parents=True)
    dataset = folder/'synthetic.csv'
    with dataset.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Date', 'ticker', 'open', 'high', 'low', 'close', 'volume'])
        day = date(2000, 1, 3)
        for i in range(args.rows):
            while day.weekday() >= 5:
                day += timedelta(days=1)
            price = 100+i*.01+math.sin(i*.7)
            writer.writerow([day, 'TEST', price, price+1, price-1, price, 1000])
            day += timedelta(days=1)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env.update(WEB_DATA_DIR=str(folder/'data'), WEB_ALLOW_FULL=str(args.mode == 'completo').lower(),
               WEB_RUN_TOKEN=secrets.token_hex(24), WEB_ORIGINS='http://localhost:8501', RENDER='true')
    if args.memory_limit_mb:
        env['WEB_MEMORY_LIMIT_MB'] = str(args.memory_limit_mb)
    else:
        env.pop('WEB_MEMORY_LIMIT_MB', None)
    report = {'mode':args.mode, 'rows':args.rows, 'epochs':args.epochs, 'memory_limit_mb':args.memory_limit_mb}
    stop = threading.Event()
    samples = []
    started = time.monotonic()
    with (folder/'server.log').open('w', encoding='utf-8') as log:
        child = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'webapp.api:app', '--host',
                                  '127.0.0.1', '--port', str(port), '--workers', '1', '--limit-concurrency', '32'],
                                 cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)

        def tree_rss():
            # En Windows el ejecutable del venv puede ser solo un lanzador.
            rss = 0
            for p in [process, *process.children(recursive=True)]:
                try:
                    rss += p.memory_info().rss
                except psutil.Error:
                    pass
            return rss

        def sample():
            while not stop.wait(.1):
                try:
                    samples.append(tree_rss())
                except psutil.Error:
                    pass

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        try:
            with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=90, trust_env=False) as client:
                deadline = time.monotonic()+30
                while True:
                    if child.poll() is not None:
                        raise RuntimeError(f'Falló el arranque. Consulta {folder}/server.log')
                    try:
                        response = client.get('/api/health')
                        response.raise_for_status()
                        break
                    except httpx.TransportError:
                        if time.monotonic() > deadline:
                            raise
                        time.sleep(.2)
                report['startup_seconds'] = round(time.monotonic()-started, 3)
                report['idle_rss_mib'] = round(tree_rss()/1024**2, 2)
                headers = {'Authorization':'Bearer '+env['WEB_RUN_TOKEN'], 'Origin':env['WEB_ORIGINS']}
                assert client.post('/api/runs', json={}).status_code == 401
                cors = client.options('/api/upload', headers={'Origin':env['WEB_ORIGINS'],
                    'Access-Control-Request-Method':'POST', 'Access-Control-Request-Headers':'authorization'})
                assert cors.headers['access-control-allow-origin'] == env['WEB_ORIGINS']
                latencies = []

                def health():
                    t = time.monotonic()
                    response = client.get('/api/health', timeout=5)
                    response.raise_for_status()
                    latencies.append((time.monotonic()-t)*1000)

                upload_rss = []
                for _ in range(3):
                    with dataset.open('rb') as handle, ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(client.post, '/api/upload', files={'file':('synthetic.csv', handle)}, headers=headers)
                        while not future.done():
                            health()
                            time.sleep(.1)
                        uploaded = future.result()
                    uploaded.raise_for_status()
                    upload_rss.append(round(tree_rss()/1024**2, 2))
                report['api_rss_after_uploads_mib'] = upload_rss
                response = client.post('/api/runs', json={'mode':args.mode, 'epochs':args.epochs,
                                      'upload_id':uploaded.json()['id']}, headers=headers)
                response.raise_for_status()
                run_id = response.json()['id']
                assert client.post('/api/runs', json={'upload_id':uploaded.json()['id']}, headers=headers).status_code == 409
                deadline = time.monotonic()+180
                while True:
                    health()
                    response = client.get('/api/runs/'+run_id)
                    response.raise_for_status()
                    state = response.json()
                    if state['status'] != 'running':
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError('El análisis no terminó en 180 segundos.')
                    time.sleep(.25)
                report['state'] = {k:v for k,v in state.items() if k != 'log'}
                assert state['status'] == 'completed', state
                dashboard = client.get('/api/dashboard/'+run_id)
                dashboard.raise_for_status()
                expected = 4 if args.mode == 'completo' else 2
                assert len(dashboard.json()['summary']['metricas_test']) == expected
                assert client.get(f'/api/artifacts/{run_id}/INFORME_TECNICO.md').status_code == 200
                report['health_requests'] = len(latencies)
                report['health_max_ms'] = round(max(latencies), 2)
                report['health_p95_ms'] = round(sorted(latencies)[int((len(latencies)-1)*.95)], 2)
                report['final_api_rss_mib'] = round(tree_rss()/1024**2, 2)
                report['status'] = 'passed'
        finally:
            stop.set()
            sampler.join(2)
            for p in reversed(process.children(recursive=True)) if process.is_running() else []:
                try:
                    p.kill()
                except psutil.Error:
                    pass
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=10)
            report['peak_tree_rss_mib'] = round(max(samples, default=0)/1024**2, 2)
            report['total_seconds'] = round(time.monotonic()-started, 2)
            (folder/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(f'Informe: {folder}/report.json')


if __name__ == '__main__':
    main()
