# Trabajo final PIAD-425: tendencias financieras

## Interfaz web local

```powershell
.\.venv\Scripts\python.exe iniciar_web.py
```

Abre [Mercado Lab](http://127.0.0.1:8501/): consulta resultados, carga CSV y ejecuta
análisis desde el navegador. Instalación web, pruebas y futura publicación en Vercel:
[FRONTEND.md](FRONTEND.md). Para ejecutar toda la suite, instalar las dependencias web.

Para habilitar cargas y entrenamientos desde Vercel: [configurar backend en Render](BACKEND_RENDER.md).
Evaluación local, consumo medido y pruebas: [auditoría del backend](BACKEND_AUDIT.md).

Solución académica modular con Pandas/NumPy, regresión logística (Scikit-learn),
MLP (PyTorch), LSTM (TensorFlow/Keras), NLP (NLTK/SciPy), Matplotlib y Seaborn.
Predice subida frente a bajada/igualdad del cierre de la próxima sesión de GGAL.

## Ejecutar en Windows (Python 3.12 de 64 bits)

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements-web-lock.txt
./.venv/Scripts/python.exe -m unittest discover -s tests -v
./.venv/Scripts/python.exe main.py --epochs 30
./.venv/Scripts/python.exe verificar_entrega.py
```

`requirements-lock.txt` registra versiones del entorno verificado. Para resolver
versiones compatibles en otro entorno, usar `requirements.txt`.
La instalación inicial requiere Internet; con el reporte descargado, el análisis
y los entrenamientos no requieren conexiones ni descargas NLTK.

El punto de entrada resuelve los datos respecto de su propia ubicación, por lo que
también se puede ejecutar desde otro directorio. Las rutas indicadas explícitamente
por argumentos se interpretan respecto del directorio de trabajo.

```powershell
./.venv/Scripts/python.exe main.py --help
./.venv/Scripts/python.exe main.py --modo base --salida resultados_base
./.venv/Scripts/python.exe main.py --epochs 5 --salida resultados_prueba
```

El modo base omite las redes y no cubre por sí solo el enunciado completo.
Usar una carpeta distinta para experimentos evita confundir resultados anteriores.

## Resultados

- [Informe técnico legible](resultados/INFORME_TECNICO.html), con preguntas guía, esquema y gráficos.
- [Informe en Markdown](resultados/INFORME_TECNICO.md).
- `resultados/metricas.csv`: comparación de los cuatro métodos sobre las mismas fechas.
- `resultados/predicciones.csv`: fecha de observación, fecha objetivo, probabilidad y clase.
- `resultados/validacion_cruzada.csv`: selección temporal de la regresión logística.
- `resultados/historial_*.csv`: pérdidas de entrenamiento y validación de las redes.
- `resultados/nlp_*.csv`, `nlp_tfidf.npz`, `nlp_vocabulario.json`: resultados de texto.
- `resultados/resumen.json`: parámetros, auditoría de datos, versiones y métricas.
- `resultados/modelos/`: logística/base joblib, escalador, pesos PyTorch y modelo Keras.
- `resultados/COMPLETADO.json`: acredita que esa corrida terminó; comprobar también su modo.

El inicio elimina el marcador anterior. Si hay un fallo, pueden quedar archivos parciales;
no usar una carpeta sin marcador como resultado completo.
Los modelos guardados solo deben cargarse desde fuentes confiables.

## Estructura

```text
main.py                 CLI y coordinación
mercado/datos.py         validación, indicadores, ventanas y cortes temporales
mercado/modelos.py       validación, entrenamiento, métricas y guardado
mercado/nlp.py           tokenización y extracción del reporte real
mercado/graficos.py      visualizaciones exportables
mercado/informe.py       informe automático con resultados reales
datos/                  CSV original, reporte público y fuentes
tests/                  pruebas de temporalidad, entradas, NLP y métricas
verificar_entrega.py     recarga modelos y contrasta predicciones sin reentrenar
resultados/             entrega generada
visualizaciones/        gráficos históricos de la versión inicial
```

## Metodología

Se separa aproximadamente 60/20/20 y se purga una etiqueta en cada frontera.
Solo entrenamiento ajusta escaladores. TimeSeriesSplit con gap selecciona C;
validación selecciona pesos de redes mediante parada temprana. Prueba permanece
fuera de esas selecciones. Las ventanas LSTM usan pasado hasta el cierre actual.
La referencia usa la clase mayoritaria de entrenamiento. Umbral fijo 0.5.

NLP analiza un reporte de Galicia de agosto de 2024, posterior al CSV. Por eso no
se mezcla con los datos predictivos. Se cumple tokenización (alternativa permitida
frente a sentimiento); no se afirma tener un modelo de sentimiento validado.

## Pendientes de identificación

Confirmar fuente/licencia del CSV recibido y completar autor/instructor del informe.
El programa no inventa esa procedencia. Consultar [datos/README.md](datos/README.md).
No se requiere web, API o base de datos para este entregable.
