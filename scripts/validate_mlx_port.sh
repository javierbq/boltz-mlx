#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

# Prefer the project's durable venv (boltz + boltz_mlx_export + pytest + ruff).
if [[ -z "${PYTHON:-}" && -x "$HOME/.venvs/boltz-mlx/bin/python" ]]; then
  PYTHON="$HOME/.venvs/boltz-mlx/bin/python"
fi
python="${PYTHON:-python3.12}"

echo "== Python: exporter tests =="
"$python" -m pytest tests/mlx_export -q

echo "== Python: ruff =="
"$python" -m ruff check src/boltz_mlx_export tests/mlx_export

echo "== Swift: strict format lint =="
xcrun swift-format lint --strict --recursive \
  Package.swift Sources/BoltzMLX Sources/BoltzMLXCLI tests/BoltzMLXTests

echo "== Swift: macOS tests =="
# The env-gated BoundaryDumpTests need the real int8 artifact + a feature bundle;
# they are a diagnostic harness, not part of the unit gate, so skip them here.
xcodebuild test \
  -scheme BoltzMLX-Package \
  -destination 'platform=macOS,arch=arm64' \
  -skipPackagePluginValidation \
  -skip-testing:BoltzMLXTests/BoundaryDumpTests

echo "== Swift: CLI build =="
xcodebuild build \
  -scheme BoltzMLXCLI \
  -destination 'platform=macOS,arch=arm64' \
  -skipPackagePluginValidation

echo "== iOS demo: simulator build =="
scripts/build_ios_demo.sh simulator

echo "== Quality report present and generated (not 'not_run') =="
report="validation/quality/report.json"
status="$("$python" -c "import json,sys; print(json.load(open('$report'))['status'])")"
if [[ "$status" == "not_run" ]]; then
  echo "ERROR: $report is still 'not_run' -- generate structural quality metrics first." >&2
  exit 1
fi
echo "quality report status: $status"

# Optional: re-export the production int8 model when a checkpoint is provided.
if [[ -n "${BOLTZ_CHECKPOINT:-}" && -n "${BOLTZ_MLX_MODEL:-}" ]]; then
  echo "== Re-export int8 model =="
  "$python" -m boltz_mlx_export export-model \
    --checkpoint "$BOLTZ_CHECKPOINT" \
    --output "$BOLTZ_MLX_MODEL"
fi

echo "ALL VALIDATION CHECKS PASSED"
