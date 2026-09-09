import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from mercado.datos import preparar, particiones, secuencias, FEATURES


class DatosTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'serie.csv'
        prices = 100 + np.arange(250)*.1 + np.sin(np.arange(250))
        self.raw = pd.DataFrame({'Date': pd.bdate_range('2020-01-01', periods=250),
                                 'ticker': 'TEST', 'open': prices, 'high': prices+1,
                                 'low': prices-1, 'close': prices, 'volume': 1000})
        self.raw.to_csv(self.path, index=False)

    def test_objetivo_siguiente_sesion_y_ultima_fila(self):
        df, _ = preparar(self.path)
        self.assertEqual(len(df), 220)
        self.assertNotIn(self.raw.Date.iloc[-1], df.Date.to_list())
        first = self.raw.index[self.raw.Date == df.Date.iloc[0]][0]
        self.assertEqual(df.Objetivo.iloc[0], int(self.raw.close.iloc[first+1] > self.raw.close.iloc[first]))

    def test_cambiar_futuro_no_modifica_indicadores_pasados(self):
        before, _ = preparar(self.path)
        cutoff = self.raw.Date.iloc[180]
        self.raw.loc[180:, ['open', 'high', 'low', 'close']] *= 2
        self.raw.to_csv(self.path, index=False)
        after, _ = preparar(self.path)
        np.testing.assert_array_equal(before.loc[before.Date < cutoff, FEATURES],
                                      after.loc[after.Date < cutoff, FEATURES])

    def test_fronteras_purgadas_y_secuencias_sin_futuro(self):
        df, _ = preparar(self.path)
        parts = particiones(df)
        for a, b in [('train', 'val'), ('val', 'test')]:
            self.assertLess(df.Fecha_Objetivo.iloc[parts[a]].max(), df.Date.iloc[parts[b]].min())
            self.assertEqual(len(set(parts[a]) & set(parts[b])), 0)
        x = np.arange(40).reshape(20, 2)
        windows = secuencias(x, [5, 10], 4)
        np.testing.assert_array_equal(windows[0], x[2:6])
        np.testing.assert_array_equal(windows[1, -1], x[10])

    def test_rechaza_activos_mezclados(self):
        self.raw.loc[0, 'ticker'] = 'OTRO'
        self.raw.to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, 'ticker'):
            preparar(self.path)

    def test_rechaza_infinito_y_conflictos(self):
        self.raw.loc[0, 'close'] = np.inf
        self.raw.to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, 'infinitos'):
            preparar(self.path)
        self.raw.loc[0, 'close'] = 100
        self.raw.loc[1, 'Date'] = self.raw.Date.iloc[0]
        self.raw.to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, 'conflictivos'):
            preparar(self.path)

    def test_limite_de_filas_y_cabecera_antes_de_preparar(self):
        with self.assertRaisesRegex(ValueError, 'filas de entrada'):
            preparar(self.path, max_rows=200)
        df, _ = preparar(self.path, max_rows=250)
        self.assertEqual(len(df), 220)
        self.path.write_text(','.join(f'col{i}' for i in range(65))+'\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'demasiadas columnas'):
            preparar(self.path)


if __name__ == '__main__':
    unittest.main()
