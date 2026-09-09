import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from scipy import sparse
from mercado.nlp import analizar, tokenizar
from mercado.modelos import evaluar


class NLPMetricasTest(unittest.TestCase):
    def test_tokenizacion_sin_corpus(self):
        self.assertEqual(tokenizar('The risk and CREDIT rose.'), ['risk', 'credit', 'rose'])

    def test_extraccion_y_matriz_recuperable(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            # Texto sintético exclusivo de prueba; no es evidencia del análisis real.
            (folder/'ejemplo.txt').write_text('Credit risk increased while deposits remained stable. Capital ratio reached 12.5%. Net profit was Ps.1,200 million after operating expenses.', encoding='utf-8')
            (folder/'fuentes.json').write_text(json.dumps([{'archivo':'ejemplo.txt'}]), encoding='utf-8')
            meta, terms = analizar(folder, folder)
            self.assertEqual(meta['paginas_analizadas'], 1)
            rows = pd.read_csv(folder/'nlp_paginas.csv')
            self.assertIn('12.5%', rows.porcentajes_detectados.iloc[0])
            self.assertIn('1,200', rows.montos_detectados.iloc[0])
            self.assertEqual(sparse.load_npz(folder/'nlp_tfidf.npz').shape[1], len(terms))

    def test_metricas_referencia_conocida(self):
        result = evaluar([0, 0, 1, 1], [.1, .9, .8, .7])
        self.assertEqual(result['exactitud'], .75)
        self.assertEqual(result['matriz_confusion'], [[1, 1], [0, 2]])
        with self.assertRaisesRegex(ValueError, 'inválidas'):
            evaluar([0, 1], [float('nan'), .3])


if __name__ == '__main__':
    unittest.main()
