# Backend preparado para Render

La API admite un trabajo pesado a la vez (validación o entrenamiento). La configuración
predeterminada de Render utiliza **Base**, que entrena logística y referencia mayoritaria.
Los resultados completos guardados siguen disponibles para consulta. Las redes se pueden
habilitar explícitamente; no se eliminó el modo completo local.

## Archivos y comandos de despliegue

Sube `webapp/`, `mercado/`, `main.py`, `requirements*.txt`, `.python-version`, `deploy/`,
`datos/reportes/`, `resultados/` y los cambios de `frontend/`. No incluyas `.venv/`,
`tmp/`, claves, `node_modules/`, `web_uploads/` ni `web_runs/`.
El CSV original está ausente en la copia local: carga uno desde la interfaz.

| Campo de Render | Valor |
|---|---|
| Runtime | Python 3 |
| Root Directory | Vacío (raíz del repositorio) |
| Build Command | `bash deploy/render-build.sh` |
| Start Command | `bash deploy/render-start.sh` |
| Health Check Path | `/api/health` |

Se mantiene **un worker y una instancia**: el bloqueo de trabajos vive en un proceso.
No aumentes workers/réplicas sin migrar el bloqueo y la cola a un servicio compartido.
El arranque limita la concurrencia HTTP a 32; un exceso recibe 503 para proteger memoria.

## Variables de entorno

| Variable | Valor / función |
|---|---|
| `WEB_ORIGINS` | Origen HTTPS exacto de Vercel, sin barra final; varios separados por comas |
| `WEB_RUN_TOKEN` | Clave privada de ejecución; obligatoria en Render |
| `WEB_ALLOW_FULL` | `false` para comenzar; `true` instala y habilita PyTorch y TensorFlow |
| `WEB_MAX_ROWS` | `50000` por defecto; filas originales, antes de descartar indicadores iniciales |
| `WEB_STORAGE_LIMIT_MB` | `512` por defecto, para cargas y ejecuciones; rechaza trabajo nuevo al llenarse |
| `WEB_MEMORY_LIMIT_MB` | Opcional: RAM de la instancia en MiB; normalmente se detecta desde cgroup |
| `WEB_DATA_DIR` | Opcional: directorio de cargas y resultados; por ejemplo `/var/data/mercado-lab` con disco montado |

El supervisor comprueba memoria cada 0,2 s y detiene el trabajo al llegar al 85 %
del límite, dejando margen para responder y guardar el error. Es una protección
preventiva, no una garantía contra picos más rápidos que el muestreo. No configures
un límite manual superior a la RAM real de tu instancia.

El perfil Base instala `requirements-api.txt`, sin necesitar los runtimes de redes.
Al cambiar `WEB_ALLOW_FULL` a `true`, ejecuta nuevamente el build: cambiar solo el
valor durante el arranque no instala las dependencias. En este perfil PyTorch se
instala desde el índice CPU. Ambos perfiles ejecutan `pip check`.

## Memoria comprobada localmente

Pruebas en Windows, 6.014 filas sintéticas, 2 épocas; incluyen el reporte PDF del
proyecto. El consumo en Linux/Render debe comprobarse después del despliegue.

| Prueba | Pico de RSS agregado |
|---|---:|
| Análisis completo antes de optimizar (sin API) | 708,36 MiB |
| Análisis completo optimizado (sin API) | 583,85 MiB |
| API + validaciones + entrenamiento Base | 305,44 MiB |
| API + validaciones + entrenamiento completo | 655,17 MiB |

Base terminó con el supervisor configurado a 512 MiB. El modo completo excede
512 MiB en esta prueba y debe usar una instancia con más memoria o ejecutarse localmente.
No se ha contratado ni modificado ningún plan. Consulta `BACKEND_AUDIT.md` para evidencia.

## Conexión con Vercel

1. Comprueba que `https://TU-BACKEND.onrender.com/api/health` responde con
   `service: mercado-lab`, `status: ok`, `key_required: true` y `allowed_modes: ["base"]`.
2. En Vercel configura `VITE_API_URL` con la URL HTTPS real de Render, sin `/api`
   y sin barra final; nunca con la dirección del dashboard de Render.
3. Redespliega el frontend: Vite incorpora la URL durante la compilación.
4. Introduce la clave en Nuevo análisis, carga un CSV y ejecuta Base.
5. Comprueba historial, panel, informe descargado y Memory en las métricas de Render.

La interfaz selecciona Base por defecto y deshabilita Todos cuando el servidor no
lo admite. La clave nunca debe incluirse en variables `VITE_` ni subirse a Git.

## Límites y operación

- CSV: 10 MiB por archivo, un archivo por solicitud y 50.000 filas originales como máximo.
- Validación: 60 segundos; entrenamiento: 30 minutos. Se terminan también sus hijos.
- Las consultas y `/api/health` continúan disponibles mientras se trabaja.
- Al detener el servidor se cancelan los procesos de trabajo. Un reinicio inesperado
  se muestra como fallo en el historial.
- Al alcanzar la cuota de almacenamiento se rechazan nuevos trabajos (507). Descarga
  los resultados y limpia solo los archivos que ya no necesites; no se borra tu historial automáticamente.
- Sin disco persistente, cargas y ejecuciones pueden perderse en reinicios/despliegues.
  Los resultados incluidos en el repositorio siguen disponibles.

La clave protege POST. Las consultas, informes e historial son compartidos y públicos:
utiliza datos aptos para compartir. No es un sistema con privacidad por usuario.

## Diagnóstico

`Failed to fetch` puede indicar servidor caído, URL incorrecta o CORS. Comprueba
primero `/api/health` directamente. Si responde, compara `WEB_ORIGINS` con el origen
exacto de la página de Vercel. Un 401 indica clave incorrecta; un 503 en POST puede
indicar que falta `WEB_RUN_TOKEN`. El frontend conserva resultados exportados cuando
no hay conexión, por lo que ver GGAL no demuestra que el backend esté funcionando.

## Referencias

- https://render.com/docs/deploy-fastapi
- https://render.com/docs/service-metrics
- https://render.com/docs/disks
