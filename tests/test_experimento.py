"""Verificaciones de los artefactos generados; nunca se abre test.csv."""

import csv
import json
import unittest
from pathlib import Path


RAIZ = Path(__file__).resolve().parents[1]
SPLITS = RAIZ / 'data' / 'splits'
METRICAS = RAIZ / 'outputs' / 'baselines_validacion' / 'metricas.csv'


def leer_csv(ruta):
    with ruta.open(encoding='utf-8', newline='') as archivo:
        return list(csv.DictReader(archivo))


@unittest.skipUnless((SPLITS / 'metadata.json').exists() and METRICAS.exists(),
                     'Ejecutar primero los notebooks 01 y 02')
class ExperimentoTest(unittest.TestCase):
    def test_asignacion_sin_solapamiento_ni_perdidas(self):
        asignacion = leer_csv(SPLITS / 'asignacion.csv')
        fuente = leer_csv(RAIZ / 'archive' / 'WA_Fn-UseC_-Telco-Customer-Churn_clean.csv')
        ids = [fila['customerID'] for fila in asignacion]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {fila['customerID'] for fila in fuente})
        metadata = json.loads((SPLITS / 'metadata.json').read_text(encoding='utf-8'))
        for conjunto, proporcion in [('train', 0.70), ('validacion', 0.15), ('test', 0.15)]:
            cantidad = sum(fila['conjunto'] == conjunto for fila in asignacion)
            self.assertEqual(cantidad, metadata['filas'][conjunto])
            self.assertLessEqual(abs(cantidad - len(ids) * proporcion), 1)
        self.assertEqual(metadata['random_state'], 42)

    def test_validacion_estratificada_y_coherente_con_asignacion(self):
        validacion = leer_csv(SPLITS / 'validacion.csv')
        asignacion = leer_csv(SPLITS / 'asignacion.csv')
        ids_validacion = {fila['customerID'] for fila in asignacion
                          if fila['conjunto'] == 'validacion'}
        self.assertEqual({fila['customerID'] for fila in validacion}, ids_validacion)
        fuente = leer_csv(RAIZ / 'archive' / 'WA_Fn-UseC_-Telco-Customer-Churn_clean.csv')
        proporcion = sum(fila['Churn'] == 'Yes' for fila in fuente) / len(fuente)
        for nombre in ['train', 'validacion']:
            filas = leer_csv(SPLITS / f'{nombre}.csv')
            positivos = sum(fila['Churn'] == 'Yes' for fila in filas)
            self.assertLessEqual(abs(positivos - len(filas) * proporcion), 1)

    def test_metricas_y_costos_conteo_independiente(self):
        validacion = leer_csv(SPLITS / 'validacion.csv')
        positivos = sum(fila['Churn'] == 'Yes' for fila in validacion)
        negativos = len(validacion) - positivos
        resultados = {fila['estrategia']: fila for fila in leer_csv(METRICAS)}
        esperados = {
            'No contactar a nadie': (negativos, 0, positivos, 0),
            'Contactar a todos': (0, negativos, 0, positivos),
        }
        for nombre, (tn, fp, fn, tp) in esperados.items():
            fila = resultados[nombre]
            self.assertEqual(tuple(int(fila[c]) for c in ['TN', 'FP', 'FN', 'TP']),
                             (tn, fp, fn, tp))
            self.assertEqual(int(fila['costo_total_usd']), fn * 120 + fp * 15)
            self.assertAlmostEqual(float(fila['accuracy']), (tn + tp) / len(validacion))
            self.assertAlmostEqual(float(fila['precision']), tp / (tp + fp) if tp + fp else 0)
            self.assertAlmostEqual(float(fila['recall']), tp / (tp + fn))


if __name__ == '__main__':
    unittest.main()
