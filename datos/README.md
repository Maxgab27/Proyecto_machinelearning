# Fuentes de datos

## CSV original

`arch_financiero.csv` fue proporcionado con el proyecto: 6.014 registros de GGAL,
25/07/2000 a 20/06/2024. No se modificó el archivo original.
Columnas: fecha (encabezado vacío), open, high, low, close, adjclose, volume, ticker.
La fuente, licencia, moneda y ajustes originales no constan: deben confirmarse con
quien obtuvo el archivo. No se atribuye automáticamente a Kaggle ni a Yahoo Finance.
El usuario indicó que no dispone del enlace de origen; esta limitación queda registrada.
El enunciado permite fuentes abiertas como Kaggle, pero no obliga a usar esa plataforma.

El análisis conserva close como base del objetivo; adjclose no se utiliza.
Usar una serie ajustada requeriría documentar su disponibilidad histórica y tratamiento
de dividendos/splits. Los modelos no representan un retorno total ni una estrategia.

## Reporte para NLP

Archivo: `reportes/galicia_2q2024.pdf`. Fuente institucional:
[Información financiera de Galicia](https://www.gfgsa.com/en/financial-information).
Título: Financial Report, 2nd quarter 2024. Publicación: 22/08/2024. Idioma: inglés.
El [manifiesto](reportes/fuentes.json) contiene el enlace directo del emisor.
Derechos del emisor; uso local académico. No se atribuye una licencia abierta no declarada.

Si falta el PDF, descargarlo con PowerShell desde la raíz del proyecto:

```powershell
Invoke-WebRequest -Uri 'https://gfgsa-stag.s3.amazonaws.com/app/uploads/2024/08/22204706/Press_Release_GFG_2Q2024_EN.pdf' -OutFile datos/reportes/galicia_2q2024.pdf
```

Se extraen términos y candidatos a cifras por página. El reporte es posterior al período
del CSV y no se integra como variable en el modelo predictivo. Para ampliar NLP, agregar
PDF o TXT y entradas al manifiesto; verificar idioma y adaptar las palabras filtradas.
No se necesita descargar corpus de NLTK. Un PDF escaneado sin texto necesitaría OCR.
