# Evaluación y optimización local del backend

Fecha: 9 de septiembre de 2026. Estado: **validado localmente para preparar el despliegue**.
No se ha subido el código ni modificado Render. No se afirma una optimización absoluta
ni se garantiza que todas las cargas quepan en una instancia concreta.

## Hallazgos y cambios

1. **PyTorch y TensorFlow se acumulaban en el mismo proceso.** Ahora cada motor corre
   en un hijo independiente y secuencial; al salir se libera su runtime. Se mantienen
   los cuatro modelos, sus archivos y los criterios de validación temporal.
2. **La LSTM materializaba todas las ventanas.** Ahora construye lotes de 64 ventanas
   mediante `tf.data`, con un lote de anticipación y un pool privado de un hilo.
3. **La API retenía Pandas/NumPy al validar una carga.** La validación se ejecuta en
   un proceso breve, fuera del bucle HTTP. Las consultas siguen respondiendo.
4. **CSV, términos, predicciones y logs se leían completos para vistas pequeñas.**
   El panel conserva solo sus muestras; términos y predicciones se leen con memoria
   acotada y el estado lee únicamente la cola del log. El historial conserva los
   últimos 30 estados durante la selección y tolera archivos corruptos.
5. **Los trabajos carecían de supervisión del árbol completo.** Se añadieron límites
   de tiempo, vigilancia de memoria, cierre de hijos y limpieza del cupo de ejecución
   incluso si falla el guardado del estado.
6. **Las cargas podían competir con entrenamientos.** Ambos comparten un único cupo.
   Se limita el multipart durante la recepción, incluso sin Content-Length, además
   del tamaño del archivo, sus filas y su cabecera. La cuota de disco evita aceptar
   trabajo indefinidamente; los archivos existentes no se borran automáticamente.
7. **Render instalaba siempre las bibliotecas de redes.** Hay un perfil Base ligero
   y otro completo opcional. La interfaz consulta las capacidades del servidor,
   selecciona Base por defecto y deshabilita Todos cuando no está habilitado.

## Resultados medidos

Equipo local Windows, Python 3.12, CPU; dataset sintético de 6.014 filas (5.984 útiles),
dos épocas y el PDF financiero incluido en el proyecto. El dataset sintético se
utilizó solo para comprobar funcionamiento y rendimiento; sus métricas no representan
el desempeño financiero de GGAL.

Se muestreó cada 100 ms la suma de RSS del proceso y todos sus descendientes. En
Windows puede existir un ejecutable lanzador adicional; está incluido en el total.
Las cifras son picos observados, no máximos garantizados ni una medición de Render.
RSS agregado puede contar páginas compartidas más de una vez.

| Escenario | Pico observado | Duración |
|---|---:|---:|
| CLI completo anterior | 708,36 MiB | 26,48 s |
| CLI completo optimizado | 583,85 MiB | 24,36 s |
| API + tres cargas + análisis Base | 305,44 MiB | 12,84 s |
| API + tres cargas + análisis completo | 655,17 MiB | 28,75 s |

La comparación equivalente del CLI redujo el pico **17,6 %**. Las duraciones son
una ejecución por escenario y pueden variar con cachés y carga del equipo.

La API arrancó en aproximadamente 1,2 s. Su RSS agregado pasó de unos 55 MiB a unos
57 MiB tras completar el flujo. Después de las tres validaciones fue 55,51 / 55,55 /
55,57 MiB en Base y 55,61 / 55,70 / 55,68 MiB en completo. Esto es una comprobación
corta de recuperación de memoria, no una prueba prolongada de ausencia de fugas.

Durante carga y entrenamiento se realizaron 62 consultas de salud en Base y 125
en completo: todas respondieron y el máximo observado fue 16 ms. Se verificaron
autorización, preflight CORS, exclusión de un segundo trabajo, panel e informe.

## Corrección y pruebas

- **30 pruebas automatizadas aprobadas:** temporalidad y datos, NLP y métricas,
  CORS y clave, archivos y rutas, concurrencia, límite de uploads con y sin longitud,
  cuota de almacenamiento, fallo del worker, límite de memoria, timeout y apagado.
- Panel con 50.000 filas: conserva primera/última observación, ocho filas de vista
  previa, 15 términos y 20 predicciones; menos de 4 MiB de asignaciones Python
  rastreadas por `tracemalloc` durante la lectura.
- En la comparación anterior/posterior, probabilidades de logística y referencia
  idénticas; diferencia máxima de PyTorch 6,60e-8 y Keras 1,21e-7. F1 sin cambios.
- Los cuatro modelos guardados se recargaron y verificaron contra **1.197
  predicciones por modelo**, con tolerancia 1e-6, clases coincidentes, hash de entrada
  correcto y escalador ajustado únicamente con entrenamiento.
- Compilación de producción del frontend aprobada; `pip check` sin conflictos;
  sintaxis Bash de ambos scripts de Render validada.

## Despliegue resultante

El perfil Base es la opción predeterminada en Render. Terminó la prueba local con
`WEB_MEMORY_LIMIT_MB=512`; el supervisor usa un margen de 85 %. El modo completo
observado no cabe en 512 MiB y requiere más memoria o ejecución local. Se habilita
con `WEB_ALLOW_FULL=true` **durante el build y el arranque**.

El supervisor intenta detectar límites cgroup en Linux y revisa el consumo cada
200 ms. Picos entre muestras todavía pueden causar un OOM. La rama Linux/cgroup
y un despliegue limpio en Render no se ejecutaron en este equipo Windows; comprueba
la RAM y los logs reales al publicar. Las dependencias locales incluyen ambos
motores; no se hizo una instalación separada del perfil Base en un entorno vacío.

La configuración usa una instancia y un worker. El CSV original ya estaba ausente
al comenzar esta revisión; no se restauró ni se sustituyó con datos sintéticos.
Los resultados originales existentes se conservaron.

Guía de valores y pasos: [BACKEND_RENDER.md](BACKEND_RENDER.md).

## Reproducir las comprobaciones

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe scripts/benchmark_backend.py --mode base --memory-limit-mb 512
.\.venv\Scripts\python.exe scripts/benchmark_backend.py --mode completo
npm.cmd run build --prefix frontend
.\.venv\Scripts\python.exe -m pip check
```

El benchmark genera un CSV de prueba, arranca un servidor en un puerto local libre,
mide su árbol de procesos y guarda `report.json` y logs bajo `tmp/backend-audit/`.
Para verificar los modelos de una ejecución concreta:

```powershell
.\.venv\Scripts\python.exe verificar_entrega.py --csv RUTA_AL_CSV --salida CARPETA_DE_EJECUCION
```

Evidencia conservada en [output/backend-audit-2026-09-09](output/backend-audit-2026-09-09):
`before.json`, `after.json`, `http-base.json`, `http-completo.json` y
`model-verification.json`.
