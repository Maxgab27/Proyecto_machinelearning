# Interfaz local y publicación futura

## Abrir el programa

Desde PowerShell, en la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe iniciar_web.py
```

Abre [Mercado Lab](http://127.0.0.1:8501/) automáticamente. Mantener la terminal abierta.
`Ctrl+C` detiene los servidores que inició ese comando. Si los puertos están ocupados,
cerrar primero la terminal de la ejecución anterior. El programa no mata servicios ajenos.
También se puede usar `--sin-abrir` para abrir el navegador manualmente.

## Primer uso en otra computadora

Python 3.12 de 64 bits y Node.js 22.12+ o 24 LTS.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-web-lock.txt
npm.cmd ci --prefix frontend
.\.venv\Scripts\python.exe iniciar_web.py
```

Si se necesita resolver versiones en un sistema diferente, usar `requirements-web.txt`.
Las fuentes tipográficas externas son opcionales: el panel usa fuentes del sistema si
no hay Internet. El CSV, reporte, resultados y análisis funcionan localmente.

## Qué se puede hacer

- Resumen: cierre histórico, períodos del gráfico y métricas del modelo seleccionado.
- Modelos: tabla comparativa, matrices de confusión, curvas y descarga de métricas.
- Texto: vocabulario, cifras por página y enlace a la fuente real.
- Datos: filas, particiones temporales y vista previa.
- Nuevo análisis: CSV original o carga de otro CSV, modo base/completo y épocas.
- Historial: selector de análisis finalizados. Cada ejecución queda en `web_runs/<id>`;
  el resultado original de `resultados/` no se sobrescribe.
- Seguimiento: estado y registro de ejecución; los resultados se muestran al terminar.
- Informe: acceso al informe HTML correspondiente al análisis seleccionado.

Una carga requiere un ticker, las columnas esperadas y al menos 150 filas útiles;
máximo 10 MB y 50.000 filas útiles. Solo se permite un entrenamiento simultáneo.
El modo base omite redes; modo completo usa logística, PyTorch y Keras.
El reporte NLP es el documento de Galicia incluido, incluso si se carga otro ticker.

## Pruebas

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe iniciar_web.py --check
npm.cmd run build --prefix frontend
```

Para probar el flujo: abrir el panel, seleccionar Nuevo análisis, mantener el CSV de
GGAL y elegir Base. Ejecutar y esperar el estado de finalización. Revisar el selector
de análisis, las métricas y el informe. Luego se puede ejecutar Todos para incluir redes.

La API está en `http://127.0.0.1:8000/docs`. Vite reenvía `/api` a ese servicio.
El navegador no ejecuta TensorFlow: el entrenamiento ocurre en un subproceso de Python.
La suite comprueba carga/errores, orígenes, archivos permitidos y exclusión mutua de trabajos.
El build verifica JSX/CSS y genera la versión de distribución; no sustituye una prueba de interacción en navegador.

## Vercel: frontend de consulta

1. Generar o actualizar los resultados con `main.py`.
2. Ejecutar `.\.venv\Scripts\python.exe exportar_web.py`.
3. Guardar en el repositorio `frontend/`, incluido `public/demo/` y `package-lock.json`.
4. Importar el repositorio en Vercel, con **Root Directory: frontend**, framework **Vite**,
   build **npm run build** y salida **dist**. `frontend/vercel.json` ya lo configura.
5. Dejar `VITE_API_URL` vacío para consulta sin servidor.

En ese modo se ven métricas, gráficos e informes exportados. Cargar y ejecutar quedan
deshabilitados con un aviso. Se publica solo el contenido exportado: revisar los datos
antes de publicar. No se incluyen modelos, entorno Python, archivos subidos o registros.
Este trabajo no realiza una publicación en Vercel.

## Vercel: frontend conectado a Python

Para cargar y entrenar desde Internet, desplegar la API en un servicio Python persistente
con almacenamiento y recursos para las redes. Configurar **VITE_API_URL=https://tu-api**
en Vercel (sin `/api` final) y reconstruir el frontend. Configurar **WEB_ORIGINS** en el
backend con el origen HTTPS exacto del frontend, separando varios con comas.

El servicio actual es para uso local. Antes de exponerlo, agregar autenticación,
límites por usuario, cola de trabajos durable y almacenamiento compartido. CORS por sí
solo no autentica. Usar un solo worker mientras el bloqueo de trabajos sea en memoria.
No apuntar Vercel a `localhost`: sería la computadora de cada visitante.

La separación es una decisión de arquitectura: las funciones tienen límites de tiempo,
memoria y tamaño; un servidor Python persistente conserva las ejecuciones de ML.
Referencias oficiales: [Vite en Vercel](https://vercel.com/docs/frameworks/frontend/vite),
[límites de funciones](https://vercel.com/docs/functions/limitations) y
[CORS en FastAPI](https://fastapi.tiangolo.com/tutorial/cors/).

## Estructura añadida

```text
iniciar_web.py       un solo comando para los dos servidores
exportar_web.py      resultados estáticos para Vercel
webapp/api.py        API, cargas y trabajos aislados
webapp/presentacion.py datos compactos del panel
frontend/src/        React y estilos responsivos
frontend/public/demo/ resultados exportados para consulta
frontend/vercel.json configuración de Vercel
```
