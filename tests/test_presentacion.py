import csv
import tempfile
import tracemalloc
import unittest
from pathlib import Path

from webapp.presentacion import dashboard


class PresentationTest(unittest.TestCase):
    def test_large_dashboard_uses_bounded_memory_and_preserves_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name in ('COMPLETADO.json', 'resumen.json'):
                (folder/name).write_text('{}', encoding='utf-8')
            with (folder/'datos_preparados.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow(['Date', 'close', 'Media_Movil_7', 'Media_Movil_30'])
                for i in range(50000):
                    writer.writerow(['2020-01-01', i, i, i])
            for name in ('nlp_terminos.csv', 'predicciones.csv'):
                with (folder/name).open('w', newline='', encoding='utf-8') as handle:
                    writer = csv.writer(handle)
                    writer.writerow(['index'])
                    writer.writerows((i,) for i in range(50000))
            tracemalloc.start()
            try:
                result = dashboard(folder)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            self.assertLess(peak, 4*1024**2)
            self.assertEqual(result['series'][0]['close'], 0)
            self.assertEqual(result['series'][-1]['close'], 49999)
            self.assertLessEqual(len(result['series']), 300)
            self.assertEqual(len(result['preview']), 8)
            self.assertEqual(len(result['terms']), 15)
            self.assertEqual([int(row['index']) for row in result['predictions']], list(range(49980, 50000)))


if __name__ == '__main__':
    unittest.main()
