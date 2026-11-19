"""Genera los PDFs de exploración y baselines, sin cargar el conjunto de test."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from decimal import Decimal
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas


RAIZ = Path(__file__).resolve().parents[1]
AZUL = colors.HexColor('#245A78')
ROJO = colors.HexColor('#C46041')
TEXTO = colors.HexColor('#1D3040')
GRIS = colors.HexColor('#607080')
FONDO = colors.HexColor('#F2F5F7')


def leer(ruta):
    with ruta.open(encoding='utf-8-sig', newline='') as archivo:
        return list(csv.DictReader(archivo))


def porcentaje(valor):
    return f'{valor * 100:.1f}'.replace('.', ',') + ' %'


def entero(valor):
    return f'{valor:,}'.replace(',', '.')


def agrupar(filas, columna, categorias):
    resultado = []
    for valor, etiqueta in categorias:
        grupo = [fila for fila in filas if fila[columna] == valor]
        if not grupo:
            raise ValueError(f'Grupo vacío: {columna}={valor}')
        abandonos = sum(fila['Churn'] == 'Yes' for fila in grupo)
        resultado.append((etiqueta, len(grupo), abandonos / len(grupo)))
    return resultado


def texto(pdf, x, y, contenido, tamano=9, color=TEXTO, negrita=False):
    pdf.setFillColor(color)
    pdf.setFont('Helvetica-Bold' if negrita else 'Helvetica', tamano)
    pdf.drawString(x, y, contenido)


def panel(pdf, x, arriba, ancho, alto, titulo, grupos, clave,
          escala=0.60, paso=33, general=False):
    pdf.setFillColor(FONDO)
    pdf.roundRect(x, arriba - alto, ancho, alto, 9, stroke=0, fill=1)
    texto(pdf, x + 13, arriba - 23, titulo, 12, negrita=True)
    subtitulo = 'Distribución de clientes' if general else 'Tasa de abandono dentro de cada grupo'
    texto(pdf, x + 13, arriba - 39, subtitulo, 8, GRIS)
    inicio_barra = x + 100
    ancho_barra = ancho - 155
    for indice, (etiqueta, cantidad, tasa) in enumerate(grupos):
        y = arriba - 63 - indice * paso
        texto(pdf, x + 13, y + 2, etiqueta, 9)
        texto(pdf, x + 13, y - 9, f'n = {entero(cantidad)}', 7, GRIS)
        pdf.setFillColor(colors.HexColor('#DFE6EB'))
        pdf.roundRect(inicio_barra, y - 3, ancho_barra, 10, 3, stroke=0, fill=1)
        pdf.setFillColor(AZUL if general and indice == 0 else ROJO)
        pdf.roundRect(inicio_barra, y - 3, ancho_barra * tasa / escala, 10, 3,
                      stroke=0, fill=1)
        texto(pdf, inicio_barra + ancho_barra + 7, y - 1, porcentaje(tasa), 9,
              negrita=True)
    texto(pdf, x + 13, arriba - alto + 14, clave, 8.5, AZUL, negrita=True)


def generar(salida):
    splits = RAIZ / 'data' / 'splits'
    ruta_train = splits / 'train.csv'
    metadata = json.loads((splits / 'metadata.json').read_text(encoding='utf-8'))
    if hashlib.sha256(ruta_train.read_bytes()).hexdigest() != metadata['archivos_sha256']['train.csv']:
        raise ValueError('Train no coincide con la separación persistida.')
    train = leer(ruta_train)
    if not train or any(f['Churn'] not in {'No', 'Yes'} for f in train):
        raise ValueError('Train vacío o con etiquetas Churn inválidas.')
    # Las etiquetas para los gráficos proceden solo de train.
    n = len(train)
    abandonos = sum(f['Churn'] == 'Yes' for f in train)
    general = [('Permanece', n - abandonos, (n - abandonos) / n),
               ('Abandona', abandonos, abandonos / n)]
    contrato = agrupar(train, 'Contract', [
        ('Month-to-month', 'Mes a mes'), ('One year', 'Un año'), ('Two year', 'Dos años')])
    internet = agrupar(train, 'InternetService', [
        ('Fiber optic', 'Fibra óptica'), ('DSL', 'DSL'), ('No', 'Sin internet')])
    for fila in train:
        meses = Decimal(fila['tenure'])
        if meses != int(meses) or not 0 <= meses <= 72:
            raise ValueError('tenure debe ser entero entre 0 y 72 para estos tramos.')
        fila['tramo'] = str(min(max((int(meses) - 1) // 12, 0), 5))
    antiguedad = agrupar(train, 'tramo', [
        (str(i), etiqueta) for i, etiqueta in enumerate(
            ['0-12 meses', '13-24 meses', '25-36 meses', '37-48 meses', '49-60 meses', '61-72 meses'])])

    # Auditoría estructural del original: se descartan las etiquetas Churn.
    raw = RAIZ / 'archive' / 'WA_Fn-UseC_-Telco-Customer-Churn_raw.csv'
    with raw.open(encoding='utf-8-sig', newline='') as archivo:
        calidad = [{k: fila[k] for k in ['customerID', 'tenure', 'TotalCharges', 'SeniorCitizen']}
                   for fila in csv.DictReader(archivo)]
    vacios = [f for f in calidad if not f['TotalCharges'].strip()]
    cero = sum(Decimal(f['tenure']) == 0 for f in vacios)
    pendientes = len(vacios) - cero
    duplicados_id = sum(cantidad - 1 for cantidad in Counter(f['customerID'] for f in calidad).values())

    salida = Path(salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(salida), pagesize=A4, invariant=1)
    pdf.setTitle('Telco | Exploración de datos')
    pdf.setAuthor('Proyecto ML aplicado a negocios')
    ancho, alto = A4
    margen = 36
    texto(pdf, margen, alto - 40, 'TELCO / EXPLORACIÓN DE DATOS', 9, AZUL, True)
    texto(pdf, margen, alto - 69, '¿Dónde se concentra el abandono?', 23, negrita=True)
    texto(pdf, margen, alto - 89,
          f'Gráficos: train, {entero(n)} clientes. Validación y test quedan reservados.', 9, GRIS)
    texto(pdf, margen, alto - 104,
          'n = clientes del grupo. Barras por grupo: 0-60 %; distribución general: 0-100 %.', 8, GRIS)
    espacio = 14
    w = (ancho - 2 * margen - espacio) / 2
    derecha = margen + w + espacio
    panel(pdf, margen, 720, w, 184, '01  Abandono general', general,
          f'{porcentaje(abandonos / n)} abandona ({entero(abandonos)} clientes).',
          escala=1, general=True, paso=42)
    panel(pdf, derecha, 720, w, 184, '02  Por contrato', contrato,
          f'Mes a mes: {porcentaje(contrato[0][2])} de abandono.')
    panel(pdf, margen, 522, w, 224, '03  Por antigüedad', antiguedad,
          f'0-12 meses: {porcentaje(antiguedad[0][2])} de abandono.', paso=24)
    panel(pdf, derecha, 522, w, 224, '04  Por tipo de internet', internet,
          f'Fibra óptica: {porcentaje(internet[0][2])} de abandono.', paso=43)

    texto(pdf, margen, 274, 'Calidad de datos y decisiones', 13, negrita=True)
    texto(pdf, margen, 258, f'Auditoría del raw: {entero(len(calidad))} filas; {duplicados_id} IDs duplicados.', 8, GRIS)
    filas_calidad = [
        (f'TotalCharges: {len(vacios)} vacíos',
         f'Decimal; 0 solo si falta y tenure = 0 ({cero} casos).\nSi tenure > 0: dejar vacío y reportar ({pendientes} actuales).'),
        ('SeniorCitizen: 0/1', 'Convertir a No/Yes; tratar como categoría.'),
        ('No internet / phone service', 'Conservar ambas categorías de "no aplica".'),
        ('Importes y antigüedad extremos', 'Conservar: los extremos no prueban errores.'),
        ('customerID: identificador', 'Conservar en el limpio; excluir al entrenar.'),
    ]
    y = 242
    for problema, decision in filas_calidad:
        lineas = []
        for parrafo in decision.split('\n'):
            lineas.extend(simpleSplit(parrafo, 'Helvetica', 8.5, ancho - 2 * margen - 182))
        altura = max(28, 12 + 11 * len(lineas))
        pdf.setStrokeColor(colors.HexColor('#DCE3E8'))
        pdf.line(margen, y, ancho - margen, y)
        texto(pdf, margen, y - 15, problema, 8.5, negrita=True)
        for i, linea in enumerate(lineas):
            texto(pdf, margen + 182, y - 15 - 11 * i, linea, 8.5)
        y -= altura
    texto(pdf, margen, 59, 'Fuente: Telco Customer Churn; train de la separación 70/15/15, semilla 42.', 7.5, GRIS)
    texto(pdf, margen, 46, 'Asociaciones descriptivas; no implican causalidad. Sin entrenamiento ni evaluación de test.', 7.5, GRIS)
    texto(pdf, ancho - margen - 5, 25, '1', 8, GRIS)
    pdf.save()
    print(json.dumps({'train_clientes': n, 'train_abandonos': abandonos,
                      'contrato': contrato, 'antiguedad': antiguedad, 'internet': internet,
                      'raw_vacios_totalcharges': len(vacios), 'raw_imputables_cero': cero},
                     ensure_ascii=False, indent=2))
    print(f'PDF generado: {salida.resolve()}')


def matriz_confusion(pdf, x, arriba, ancho, fila):
    pdf.setFillColor(FONDO)
    pdf.roundRect(x, arriba - 201, ancho, 201, 9, stroke=0, fill=1)
    texto(pdf, x + 14, arriba - 24, fila['estrategia'], 11, negrita=True)
    izquierda = x + 87
    for i, etiqueta in enumerate(['No contactar', 'Contactar']):
        texto(pdf, izquierda + 75 * i + 4, arriba - 52, etiqueta, 8, GRIS)
    for i, etiqueta in enumerate(['No', 'Yes']):
        texto(pdf, x + 15, arriba - 85 - i * 57, etiqueta, 10, negrita=True)
    for i, clave in enumerate(['TN', 'FP', 'FN', 'TP']):
        columna, fila_indice = i % 2, i // 2
        base_x = izquierda + 75 * columna
        base_y = arriba - 112 - 57 * fila_indice
        error = clave in {'FP', 'FN'} and fila[clave] > 0
        pdf.setFillColor(colors.HexColor('#F4D9CF') if error else colors.HexColor('#E0EAF0'))
        pdf.roundRect(base_x, base_y, 70, 51, 5, stroke=0, fill=1)
        texto(pdf, base_x + 9, base_y + 33, clave, 8, GRIS)
        texto(pdf, base_x + 9, base_y + 12, entero(fila[clave]), 18,
              ROJO if error else AZUL, True)


def generar_baseline(salida):
    carpeta = RAIZ / 'outputs' / 'baselines_validacion'
    metadata = json.loads((carpeta / 'metadata.json').read_text(encoding='utf-8'))
    splits = RAIZ / 'data' / 'splits'
    particion = json.loads((splits / 'metadata.json').read_text(encoding='utf-8'))
    ruta_validacion = splits / 'validacion.csv'
    huella = hashlib.sha256(ruta_validacion.read_bytes()).hexdigest()
    if (metadata['conjunto'] != 'validacion'
            or huella != metadata['validacion_sha256']
            or huella != particion['archivos_sha256']['validacion.csv']):
        raise ValueError('Los resultados no corresponden a la validación persistida.')
    if (metadata['costo_fn_usd'], metadata['costo_fp_usd']) != (120, 15):
        raise ValueError('Los costos guardados no coinciden con FN=120 y FP=15.')
    validacion = leer(ruta_validacion)
    if not validacion or any(f['Churn'] not in {'No', 'Yes'} for f in validacion):
        raise ValueError('Etiquetas inválidas en validación.')
    n = len(validacion)
    positivos = sum(f['Churn'] == 'Yes' for f in validacion)
    negativos = n - positivos
    por_nombre = {f['estrategia']: f for f in leer(carpeta / 'metricas.csv')}
    filas = []
    for nombre, esperado in [
        ('No contactar a nadie', (negativos, 0, positivos, 0)),
        ('Contactar a todos', (0, negativos, 0, positivos)),
    ]:
        fila = dict(por_nombre[nombre])
        for clave in ['TN', 'FP', 'FN', 'TP', 'costo_total_usd']:
            fila[clave] = int(fila[clave])
        if tuple(fila[k] for k in ['TN', 'FP', 'FN', 'TP']) != esperado:
            raise ValueError(f'Matriz incoherente para {nombre}')
        tn, fp, fn, tp = esperado
        comprobaciones = {
            'accuracy': (tn + tp) / n,
            'precision': tp / (tp + fp) if tp + fp else 0,
            'recall': tp / (tp + fn) if tp + fn else 0,
            'costo_total_usd': 120 * fn + 15 * fp,
        }
        for clave, esperado_valor in comprobaciones.items():
            if not math.isclose(float(fila[clave]), esperado_valor, abs_tol=1e-12):
                raise ValueError(f'{clave} incoherente para {nombre}')
        filas.append(fila)

    salida = Path(salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(salida), pagesize=A4, invariant=1)
    pdf.setTitle('Telco | Baselines en validación')
    pdf.setAuthor('Proyecto ML aplicado a negocios')
    ancho, alto = A4
    m = 36
    texto(pdf, m, alto - 40, 'TELCO / BASELINES EN VALIDACIÓN', 9, AZUL, True)
    texto(pdf, m, alto - 69, 'Dos decisiones, costos muy distintos', 23, negrita=True)
    texto(pdf, m, alto - 90,
          f'{entero(n)} clientes de validación: {entero(positivos)} abandonan y {entero(negativos)} permanecen.', 9, GRIS)
    texto(pdf, m, alto - 106, 'Positivo = Churn Yes; decisión positiva = contactar. Sin entrenamiento ni uso de test.', 8, GRIS)

    texto(pdf, m, 711, '01  Métricas', 13, negrita=True)
    columnas = [m + 10, 260, 334, 407, 478]
    pdf.setFillColor(AZUL)
    pdf.roundRect(m, 666, ancho - 2 * m, 28, 4, stroke=0, fill=1)
    for x, etiqueta in zip(columnas, ['Estrategia', 'Accuracy', 'Precision', 'Recall', 'Costo USD']):
        texto(pdf, x, 676, etiqueta, 9, colors.white, True)
    for i, fila in enumerate(filas):
        y = 645 - i * 36
        texto(pdf, columnas[0], y, fila['estrategia'], 10, negrita=True)
        for x, clave in zip(columnas[1:4], ['accuracy', 'precision', 'recall']):
            etiqueta = f'{float(fila[clave]) * 100:.2f}'.replace('.', ',') + ' %'
            if clave == 'precision' and i == 0:
                etiqueta += '*'
            texto(pdf, x, y, etiqueta, 9)
        texto(pdf, columnas[4], y, entero(fila['costo_total_usd']), 10, AZUL, True)
    texto(pdf, m + 10, 585, '* Precision = 0 por convención: no hay predicciones positivas.', 8, GRIS)

    texto(pdf, m, 554, '02  Costo total de los errores', 13, negrita=True)
    texto(pdf, m, 537, 'USD = 120 × FN + 15 × FP; TP y TN sin costo en el ejercicio.', 8, GRIS)
    inicio, longitud = 205, 273
    maximo = max(f['costo_total_usd'] for f in filas)
    escala = max(10000, math.ceil(maximo / 10000) * 10000)
    for i, fila in enumerate(filas):
        y = 502 - i * 48
        texto(pdf, m, y + 7, fila['estrategia'], 9)
        pdf.setFillColor(FONDO)
        pdf.roundRect(inicio, y, longitud, 23, 4, stroke=0, fill=1)
        pdf.setFillColor(ROJO if i == 0 else AZUL)
        longitud_barra = longitud * fila['costo_total_usd'] / escala
        pdf.roundRect(inicio, y, longitud_barra, 23, 4, stroke=0, fill=1)
        texto(pdf, inicio + longitud_barra + 8, y + 7,
              entero(fila['costo_total_usd']), 10, negrita=True)
    for i in range(5):
        valor = escala * i / 4
        texto(pdf, inicio + longitud * i / 4 - 5, 434, entero(int(valor)), 7, GRIS)

    texto(pdf, m, 394, '03  Matrices de confusión', 13, negrita=True)
    texto(pdf, m, 377, 'Filas: Churn real [No, Yes]. Columnas: decisión [No contactar, Contactar].', 8, GRIS)
    w = (ancho - 2 * m - 14) / 2
    matriz_confusion(pdf, m, 361, w, filas[0])
    matriz_confusion(pdf, m + w + 14, 361, w, filas[1])
    ahorro = filas[0]['costo_total_usd'] - filas[1]['costo_total_usd']
    conclusion = f'Contactar a todos reduce el costo en USD {entero(ahorro)}, pese a su menor accuracy.'
    if ahorro < 0:
        conclusion = f'No contactar a nadie reduce el costo en USD {entero(-ahorro)} frente a contactar a todos.'
    texto(pdf, m, 113, conclusion, 10.3, AZUL, True)
    texto(pdf, m, 58, 'Fuente: resultados persistidos del notebook 02; separación 70/15/15, semilla 42.', 7.5, GRIS)
    texto(pdf, m, 45, 'Costos supuestos del ejercicio; no se estima el efecto de una campaña de retención.', 7.5, GRIS)
    texto(pdf, ancho - m - 5, 25, '1', 8, GRIS)
    pdf.save()
    print(f'PDF generado: {salida.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--informe', choices=['exploracion', 'baseline', 'todos'], default='todos')
    parser.add_argument('--salida', type=Path, help='Ruta personalizada para un solo informe')
    args = parser.parse_args()
    if args.salida and args.informe == 'todos':
        parser.error('--salida requiere elegir --informe exploracion o baseline')
    if args.informe in {'exploracion', 'todos'}:
        generar(args.salida or RAIZ / 'reports' / '01_exploracion.pdf')
    if args.informe in {'baseline', 'todos'}:
        generar_baseline(args.salida or RAIZ / 'reports' / '02_baseline_validacion.pdf')
