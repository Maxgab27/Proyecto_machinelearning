#!/usr/bin/env bash
set -euo pipefail
# El perfil base evita instalar los dos runtimes de redes en instancias pequeñas.
full="${WEB_ALLOW_FULL:-false}"
if [[ "${full,,}" == "true" ]]; then
  python -m pip install 'torch>=2.5,<3' --index-url https://download.pytorch.org/whl/cpu
  python -m pip install -r requirements-web.txt
  python -c "import torch; print('PyTorch CPU disponible')"
  python -c "import tensorflow; print('TensorFlow disponible')"
else
  python -m pip install -r requirements-api.txt
fi
python -m pip check
python -c "import webapp.api; print('API disponible')"
