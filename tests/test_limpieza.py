import csv
import tempfile
import unittest
from pathlib import Path

from src.limpieza import limpiar


class LimpiezaTest(unittest.TestCase):
    def test_regla_condicional_y_conservacion(self):
        with tempfile.TemporaryDirectory() as carpeta:
            raw = Path(carpeta) / 'raw.csv'
            clean = Path(carpeta) / 'clean.csv'
            original = (
                'customerID,tenure,TotalCharges,SeniorCitizen,OnlineSecurity,MultipleLines\n'
                'A,0, ,0,No internet service,No phone service\n'
                'B,4, ,1,Yes,Yes\n'
                'C,0,25.50,0,No,No\n'
            )
            raw.write_text(original, encoding='utf-8')
            original_bytes = raw.read_bytes()
            resumen = limpiar(raw, clean)
            with clean.open(newline='', encoding='utf-8') as archivo:
                filas = list(csv.DictReader(archivo))
            self.assertEqual([f['customerID'] for f in filas], ['A', 'B', 'C'])
            self.assertEqual([f['TotalCharges'] for f in filas], ['0.00', '', '25.50'])
            self.assertEqual([f['SeniorCitizen'] for f in filas], ['No', 'Yes', 'No'])
            self.assertEqual(filas[0]['OnlineSecurity'], 'No internet service')
            self.assertEqual(filas[0]['MultipleLines'], 'No phone service')
            self.assertEqual(resumen['totalcharges_imputados'], 1)
            self.assertEqual(resumen['totalcharges_pendientes'], 1)
            self.assertEqual(raw.read_bytes(), original_bytes)
            primera_salida = clean.read_bytes()
            limpiar(raw, clean)
            self.assertEqual(clean.read_bytes(), primera_salida)

    def test_rechaza_sobrescribir_raw(self):
        with self.assertRaises(ValueError):
            limpiar('raw.csv', 'raw.csv')

    def test_dato_invalido_no_reemplaza_salida(self):
        with tempfile.TemporaryDirectory() as carpeta:
            raw = Path(carpeta) / 'raw.csv'
            clean = Path(carpeta) / 'clean.csv'
            raw.write_text('customerID,tenure,TotalCharges,SeniorCitizen\nA,2,error,0\n', encoding='utf-8')
            clean.write_text('salida anterior', encoding='utf-8')
            with self.assertRaises(ValueError):
                limpiar(raw, clean)
            self.assertEqual(clean.read_text(encoding='utf-8'), 'salida anterior')


if __name__ == '__main__':
    unittest.main()
