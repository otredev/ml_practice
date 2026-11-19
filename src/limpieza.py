"""Genera el CSV limpio sin modificar el original ni entrenar modelos."""

import argparse
import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path


RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / 'archive' / 'WA_Fn-UseC_-Telco-Customer-Churn_raw.csv'
CLEAN = RAIZ / 'archive' / 'WA_Fn-UseC_-Telco-Customer-Churn_clean.csv'


def numero(texto, columna, linea):
    """Lee un decimal y rechaza datos inválidos en lugar de imputarlos."""
    try:
        valor = Decimal(texto.strip())
    except InvalidOperation as exc:
        raise ValueError(f'Línea {linea}: {columna} no es numérico: {texto!r}') from exc
    if not valor.is_finite() or valor < 0:
        raise ValueError(f'Línea {linea}: {columna} debe ser finito y no negativo')
    return valor


def limpiar(entrada=RAW, salida=CLEAN):
    entrada, salida = Path(entrada), Path(salida)
    if entrada.resolve() == salida.resolve():
        raise ValueError('La salida debe ser distinta del archivo raw')

    filas = []
    imputados = pendientes = 0
    with entrada.open(encoding='utf-8-sig', newline='') as archivo:
        lector = csv.DictReader(archivo)
        columnas = lector.fieldnames
        requeridas = {'customerID', 'tenure', 'TotalCharges', 'SeniorCitizen'}
        if not columnas or not requeridas.issubset(columnas):
            raise ValueError(f'Faltan columnas requeridas: {sorted(requeridas - set(columnas or []))}')
        for linea, fila in enumerate(lector, start=2):
            if None in fila or any(valor is None for valor in fila.values()):
                raise ValueError(f'Línea {linea}: cantidad de campos incorrecta')
            tenure = numero(fila['tenure'], 'tenure', linea)
            total = fila['TotalCharges'].strip()
            if not total:
                if tenure == 0:
                    fila['TotalCharges'] = '0.00'
                    imputados += 1
                else:
                    fila['TotalCharges'] = ''
                    pendientes += 1
            else:
                # El formato fijo conserva la precisión decimal del importe original.
                importe = numero(total, 'TotalCharges', linea)
                fila['TotalCharges'] = format(importe, '.2f') if importe == importe.quantize(Decimal('0.01')) else format(importe, 'f')
            senior = fila['SeniorCitizen']
            if senior not in {'0', '1'}:
                raise ValueError(f'Línea {linea}: SeniorCitizen debe ser 0 o 1')
            fila['SeniorCitizen'] = {'0': 'No', '1': 'Yes'}[senior]
            filas.append(fila)

    # Validar todas las filas antes de crear o reemplazar la salida.
    with salida.open('w', encoding='utf-8', newline='') as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)
    return {'filas': len(filas), 'columnas': len(columnas),
            'totalcharges_imputados': imputados, 'totalcharges_pendientes': pendientes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entrada', type=Path, default=RAW)
    parser.add_argument('--salida', type=Path, default=CLEAN)
    args = parser.parse_args()
    resumen = limpiar(args.entrada, args.salida)
    print(f'Archivo limpio: {args.salida.resolve()}')
    for nombre, valor in resumen.items():
        print(f'{nombre}: {valor}')
    if resumen['totalcharges_pendientes']:
        print('AVISO: quedan TotalCharges faltantes con tenure > 0; se conservaron vacíos.')


if __name__ == '__main__':
    main()
