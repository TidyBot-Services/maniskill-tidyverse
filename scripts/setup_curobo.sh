#!/usr/bin/env bash
# Install cuRobo (NVIDIA GPU motion planner) + Tidybot custom robot config.
#
# cuRobo source is NOT vendored — every developer clones + builds it on
# their own machine. Pinning a commit + copying our custom robot config
# (franka_tidyverse.yml + urdf + mesh sphere yml) keeps all machines on
# the same version + same robot model.
#
# Usage:
#   ./scripts/setup_curobo.sh
#
# Override defaults with env vars:
#   CUROBO_DIR=$HOME/other/path CUROBO_COMMIT=abc123 ./scripts/setup_curobo.sh
set -euo pipefail

# v0.8 / cuRoboV2 research release. The Tidybot maniskill_sim + curobo_service
# code (planner_core.py) is written against the v2 API. The legacy v0.7.6 commit
# (2fbffc35) is preserved here as a reference but is INCOMPATIBLE with the
# current codebase.
CUROBO_DIR="${CUROBO_DIR:-$HOME/workspace/curobo}"
CUROBO_COMMIT="${CUROBO_COMMIT:-4ea77366ca48ee453e7df139e39fa6532af49f3b}"   # v0.8 / cuRoboV2 research release
CUROBO_REPO="${CUROBO_REPO:-https://github.com/NVlabs/curobo.git}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="$REPO_ROOT/curobo_assets"

echo "=== cuRobo setup ==="
echo "  dir:        $CUROBO_DIR"
echo "  commit:     $CUROBO_COMMIT"
echo "  repo:       $CUROBO_REPO"
echo "  assets src: $ASSETS"
echo

# Sanity check
if [ ! -f "$ASSETS/franka_tidyverse.yml" ]; then
  echo "ERROR: $ASSETS/franka_tidyverse.yml not found — wrong repo root?"
  exit 1
fi

# CUDA toolkit: must match PyTorch's CUDA version. Prefer the conda env's
# nvcc (if installed) over system nvcc to avoid "CUDA version mismatch".
# Note: v0.8 needs torch 2.10+ with CUDA 12.8 (driver 570+) for Blackwell GPUs
# (5070 Ti / 5090). The pip wheel of cuRobo lacks sm_120 kernels — source
# compile is mandatory on Blackwell.
if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/nvcc" ]; then
  export CUDA_HOME="$CONDA_PREFIX"
  export PATH="$CUDA_HOME/bin:$PATH"
  echo "  using CUDA_HOME=$CUDA_HOME (conda env nvcc)"
fi

# 1. Clean any stale editable install (e.g. an abandoned /tmp/curobo .pth leftover)
echo "[1/5] Removing any stale nvidia_curobo install..."
pip uninstall -y nvidia_curobo 2>/dev/null || true

# 2. Clone or update the source tree
echo "[2/5] Fetching cuRobo source..."
if [ ! -d "$CUROBO_DIR/.git" ]; then
  mkdir -p "$(dirname "$CUROBO_DIR")"
  git clone "$CUROBO_REPO" "$CUROBO_DIR"
fi
cd "$CUROBO_DIR"
git fetch origin
git checkout "$CUROBO_COMMIT"

# 3. Copy Tidybot custom robot config into cuRobo's content tree.
#    cuRobo's MotionGen loads robot YAMLs from get_robot_configs_path() which
#    resolves to <CUROBO_DIR>/curobo/content/configs/robot/. We must copy our
#    franka_tidyverse.yml + sphere yml there so cuRobo can find them.
echo "[3/5] Installing Tidybot robot config into cuRobo content tree..."
ROBOT_CONF_DIR="$CUROBO_DIR/curobo/content/configs/robot"
SPHERES_DIR="$ROBOT_CONF_DIR/spheres"
ASSET_ROBOT_DIR="$CUROBO_DIR/curobo/content/assets/robot/franka_description"

mkdir -p "$ROBOT_CONF_DIR" "$SPHERES_DIR" "$ASSET_ROBOT_DIR"

cp -v "$ASSETS/franka_tidyverse.yml"                                  "$ROBOT_CONF_DIR/franka_tidyverse.yml"
cp -v "$ASSETS/spheres/franka_tidyverse_mesh.yml"                     "$SPHERES_DIR/franka_tidyverse_mesh.yml"
cp -v "$ASSETS/robot/franka_description/franka_panda_tidyverse.urdf"  "$ASSET_ROBOT_DIR/franka_panda_tidyverse.urdf"

# 4. Build + editable install (CUDA kernels compile on first import, ~15-30 min)
echo "[4/5] Building + installing (editable) — CUDA compile takes 15-30 min on first run..."
cd "$CUROBO_DIR"
pip install -e . --no-build-isolation

# 5. Smoke test (load the robot config + tiny IK to verify everything fits)
echo "[5/5] Verifying import + Tidybot robot config..."
python -c "
import curobo
print('cuRobo OK at:', curobo.__file__)
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
cfg = load_yaml(join_path(get_robot_configs_path(), 'franka_tidyverse.yml'))
print('franka_tidyverse.yml loaded, format:', cfg.get('robot_cfg', {}).get('kinematics', {}).get('format_version', 'v1'))
"

echo
echo "✅ cuRobo + Tidybot robot config installed."
echo "Currently pinned to:"
echo "  CUROBO_COMMIT=$(git rev-parse HEAD)"
echo
echo "If you edit franka_tidyverse.yml in this repo, re-run this script to"
echo "propagate the change into cuRobo's content tree."
