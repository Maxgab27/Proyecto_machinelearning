#!/usr/bin/env bash
set -euo pipefail
# Evitar pools de cálculo grandes; los hijos de entrenamiento heredan estos límites.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export ML_NUM_THREADS="${ML_NUM_THREADS:-1}"
# El bloqueo de trabajos es local a un proceso: mantener un solo worker.
exec python -m uvicorn webapp.api:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1 --limit-concurrency 32 --timeout-keep-alive 5 --timeout-graceful-shutdown 20
