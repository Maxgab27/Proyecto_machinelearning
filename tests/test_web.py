import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from webapp import api


class WebTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.patches = [patch.object(api, 'RUNS', self.root/'runs'),
                        patch.object(api, 'UPLOADS', self.root/'uploads'),
                        patch.object(api, 'active', None)]
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
        raw = (api.ROOT/'datos'/'arch_financiero.csv').read_bytes()
        res = self.client.post('/api/upload', files={'file': ('sample.csv', raw)})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()['audit']['ticker'], 'GGAL')
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
        with patch.object(api.threading, 'Thread'):
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


if __name__ == '__main__':
    unittest.main()
