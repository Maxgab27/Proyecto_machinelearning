import json
import tempfile
import unittest
import threading
import pandas as pd
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from webapp import api


class WebTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # Fixture sintético independiente del CSV del usuario.
        self.raw = pd.DataFrame({'Date':pd.bdate_range('2020-01-01',periods=200),
                                 'open':10.,'high':12.,'low':9.,
                                 'close':[10.+i%2 for i in range(200)],
                                 'volume':1000,'ticker':'TEST'}).to_csv(index=False).encode()
        (self.root/'datos').mkdir()
        (self.root/'datos'/'arch_financiero.csv').write_bytes(self.raw)
        self.patches = [patch.object(api, 'RUNS', self.root/'runs'),
                        patch.object(api, 'UPLOADS', self.root/'uploads'),
                        patch.object(api, 'active', None), patch.object(api,'RUN_TOKEN',''),
                        patch.object(api, 'active_thread', None),
                        patch.object(api, 'capacity', threading.BoundedSemaphore(1))]
        for p in self.patches:
            p.start(); self.addCleanup(p.stop)
        self.client = TestClient(api.app)

    def test_health_and_results(self):
        self.assertEqual(self.client.get('/api/health').json()['service'], 'mercado-lab')
        res = self.client.get('/api/dashboard/actual')
        self.assertEqual(res.status_code, 200)
        self.assertIn('metricas_test', res.json()['summary'])
        self.assertGreater(len(res.json()['series']), 10)

    def test_csv_validation(self):
        bad = self.client.post('/api/upload', files={'file': ('bad.csv', b'close\n1\n')})
        self.assertEqual(bad.status_code, 422)
        raw = self.raw
        res = self.client.post('/api/upload', files={'file': ('sample.csv', raw)})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()['audit']['ticker'], 'TEST')
        self.assertTrue((self.root/'uploads'/f"{res.json()['id']}.csv").is_file())

    def test_reject_foreign_origin_and_missing_upload(self):
        self.assertEqual(self.client.post('/api/runs', json={}, headers={'Origin':'https://unrelated.example'}).status_code, 403)
        self.assertEqual(self.client.post('/api/runs', json={'upload_id':'../escape'}).status_code, 422)
        self.assertEqual(self.client.post('/api/runs', json={'upload_id':'a'*32}).status_code, 404)
        self.assertEqual(self.client.post('/api/runs', json={'epochs':0}).status_code, 422)

    def test_artifacts_do_not_expose_models_or_parent_paths(self):
        self.assertEqual(self.client.get('/api/artifacts/actual/modelos/logistica.joblib').status_code, 404)
        self.assertEqual(self.client.get('/api/artifacts/actual/%2E%2E%2Fmain.py').status_code, 404)
        self.assertEqual(self.client.get('/api/artifacts/actual/metricas.csv').status_code, 200)

    def test_only_one_job_and_no_overwrite_of_original_results(self):
        before = (api.ROOT/'resultados'/'resumen.json').read_bytes()
        with patch.object(api.threading, 'Thread'), patch.object(api,'ROOT',self.root):
            first = self.client.post('/api/runs', json={'mode':'base', 'epochs':1})
            self.assertEqual(first.status_code, 202)
            self.assertEqual(self.client.post('/api/runs', json={}).status_code, 409)
        state = first.json()
        self.assertEqual((self.root/'runs'/state['id']/'estado.json').is_file(), True)
        self.assertEqual(before, (api.ROOT/'resultados'/'resumen.json').read_bytes())

    def test_restart_marks_incomplete_job_failed(self):
        folder = self.root/'runs'/('b'*32)
        folder.mkdir(parents=True)
        (folder/'estado.json').write_text(json.dumps({'id':'b'*32,'status':'running','created':'2026-01-01'}))
        self.assertEqual(self.client.get('/api/runs/'+('b'*32)).json()['status'], 'failed')

    def test_remote_key_and_cors(self):
        origin = 'http://localhost:8501'
        with patch.object(api,'RUN_TOKEN','test-key'):
            self.assertTrue(self.client.get('/api/health').json()['key_required'])
            denied=self.client.post('/api/upload',files={'file':('test.csv',self.raw)},headers={'Origin':origin})
            self.assertEqual(denied.status_code,401)
            self.assertEqual(denied.headers['access-control-allow-origin'],origin)
            allowed=self.client.post('/api/upload',files={'file':('test.csv',self.raw)},headers={'Origin':origin,'Authorization':'Bearer test-key'})
            self.assertEqual(allowed.status_code,200)
        preflight=self.client.options('/api/upload',headers={'Origin':origin,'Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'authorization'})
        self.assertEqual(preflight.status_code,200)

    def test_missing_default_csv_is_actionable(self):
        (self.root/'datos'/'arch_financiero.csv').unlink()
        with patch.object(api,'ROOT',self.root):
            self.assertFalse(self.client.get('/api/health').json()['default_csv_available'])
            response=self.client.post('/api/runs',json={})
            self.assertEqual(response.status_code,422)
            self.assertIn('carga un archivo',response.json()['detail'])

    def test_render_requires_key_configuration(self):
        with patch.dict(api.os.environ,{'RENDER':'true'}):
            self.assertEqual(self.client.post('/api/runs',json={}).status_code,503)

    def test_base_profile_advertises_and_enforces_modes(self):
        with patch.dict(api.os.environ, {'WEB_ALLOW_FULL':'false'}):
            self.assertEqual(self.client.get('/api/health').json()['allowed_modes'], ['base'])
            res = self.client.post('/api/runs', json={'mode':'completo'})
            self.assertEqual(res.status_code, 422)
        self.assertEqual(api.RunRequest().mode, 'base')

    def test_upload_limit_and_failed_validation_release_slot(self):
        with patch.object(api, 'MAX_UPLOAD', 128):
            res = self.client.post('/api/upload', files={'file':('big.csv', b'x'*129)})
            self.assertEqual(res.status_code, 413)
        self.assertFalse(list((self.root/'uploads').glob('*.csv')))
        with patch.object(api, 'validate_csv', side_effect=RuntimeError('Memoria insuficiente')):
            res = self.client.post('/api/upload', files={'file':('test.csv', self.raw)})
            self.assertEqual(res.status_code, 422)
        self.assertFalse(list((self.root/'uploads').glob('*.csv')))
        self.assertTrue(api.capacity.acquire(blocking=False))
        api.capacity.release()

    def test_chunked_upload_is_limited_before_validation(self):
        boundary = 'test-boundary'
        def chunks():
            yield (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="big.csv"\r\nContent-Type: text/csv\r\n\r\n').encode()
            for _ in range(20):
                yield b'x'*8192
            yield f'\r\n--{boundary}--\r\n'.encode()
        with patch.object(api, 'MAX_UPLOAD', 128), patch.object(api, 'validate_csv') as validate:
            response = self.client.post('/api/upload', content=chunks(), headers={'Content-Type':f'multipart/form-data; boundary={boundary}'})
            self.assertEqual(response.status_code, 413)
            validate.assert_not_called()
        self.assertTrue(api.capacity.acquire(blocking=False))
        api.capacity.release()

    def test_validation_keeps_health_responsive_and_excludes_other_work(self):
        entered, release = threading.Event(), threading.Event()
        def validate(path):
            entered.set()
            release.wait(5)
            return {'ticker':'TEST'}
        from concurrent.futures import ThreadPoolExecutor
        with patch.object(api, 'validate_csv', side_effect=validate), ThreadPoolExecutor() as pool:
            result = pool.submit(self.client.post, '/api/upload', files={'file':('test.csv', self.raw)})
            try:
                self.assertTrue(entered.wait(3))
                self.assertEqual(self.client.get('/api/health').status_code, 200)
                self.assertEqual(self.client.post('/api/upload', files={'file':('test.csv', self.raw)}).status_code, 409)
                with patch.object(api, 'ROOT', self.root):
                    self.assertEqual(self.client.post('/api/runs', json={}).status_code, 409)
            finally:
                release.set()
            self.assertEqual(result.result().status_code, 200)

    def test_storage_limit_releases_capacity(self):
        with patch.dict(api.os.environ, {'WEB_STORAGE_LIMIT_MB':'1'}):
            self.assertEqual(self.client.post('/api/upload', files={'file':('test.csv', self.raw)}).status_code, 507)
        self.assertTrue(api.capacity.acquire(blocking=False))
        api.capacity.release()

    def test_worker_failure_releases_capacity_and_persists_error(self):
        folder = self.root/'runs'/('c'*32)
        folder.mkdir(parents=True)
        state = {'id':'c'*32, 'status':'running'}
        api.active = state['id']
        api.capacity.acquire()
        with patch.object(api.runtime, 'run', side_effect=RuntimeError('Sin memoria')):
            api.worker(folder, state, self.root/'datos/arch_financiero.csv', api.RunRequest())
        saved = json.loads((folder/'estado.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['status'], 'failed')
        self.assertIn('Sin memoria', saved['error'])
        self.assertIsNone(api.active)
        self.assertTrue(api.capacity.acquire(blocking=False))
        api.capacity.release()


if __name__ == '__main__':
    unittest.main()
