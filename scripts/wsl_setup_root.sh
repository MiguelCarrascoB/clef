#!/usr/bin/env bash
# Root bootstrap for ROCm-on-WSL2 (AMD Radeon RX 7900 XTX, Ubuntu 22.04).
# Installs base tools, AMD package repo, ROCm core runtime and librocdxg (the WSL GPU enabler).
# Run as root: sudo bash scripts/wsl_setup_root.sh
set -euo pipefail

ROCM_VERSION="${ROCM_VERSION:-7.2.4}"
AMDGPU_INSTALL_DEB="${AMDGPU_INSTALL_DEB:-amdgpu-install_7.2.4.70204-1_all.deb}"
UBUNTU_CODENAME="${UBUNTU_CODENAME:-jammy}"
ROCDXG_VERSION="${ROCDXG_VERSION:-1.2.2}"

if [ "$(id -u)" -ne 0 ]; then
  echo "must run as root (sudo bash $0)" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

echo "== [1/5] Base packages =="
apt-get update -qq
apt-get install -y -qq git-lfs python3-venv python3-pip libnuma1 libdrm2 libelf1 \
    ca-certificates curl wget shellcheck

echo "== [2/5] Register AMD package repository (amdgpu-install $ROCM_VERSION, $UBUNTU_CODENAME) =="
wget -q "https://repo.radeon.com/amdgpu-install/$ROCM_VERSION/ubuntu/$UBUNTU_CODENAME/$AMDGPU_INSTALL_DEB"
apt-get install -y -qq "./$AMDGPU_INSTALL_DEB"
apt-get update -qq

echo "== [3/5] Install ROCm core runtime (user-space; kernel driver comes from the Windows host) =="
# No amdgpu-dkms: the Windows Adrenalin driver is the host driver.
apt-get install -y -qq rocm-core python3-setuptools python3-wheel

echo "== [4/5] Install librocdxg $ROCDXG_VERSION (ROCDXG WSL runtime) =="
BASE="https://github.com/ROCm/librocdxg/releases/download/v$ROCDXG_VERSION"
for pkg in rocdxg-roct rocdxg-amd-smi-lib; do
  wget -q "$BASE/${pkg}_${ROCDXG_VERSION}_amd64.deb"
done
# apt resolves dependencies of local debs; a failure here must stop the script.
apt-get install -y -qq "./rocdxg-roct_${ROCDXG_VERSION}_amd64.deb" "./rocdxg-amd-smi-lib_${ROCDXG_VERSION}_amd64.deb"

echo "== [5/5] Environment profile (required for ROCm <= 7.13 on WSL) =="
cat > /etc/profile.d/rocm-wsl.sh << 'PROFILE'
# Required for ROCm under WSL2 until ROCm 7.13+: detect GPU via /dev/dxg
export HSA_ENABLE_DXG_DETECTION=1
export HSA_OVERRIDE_GFX_VERSION=11.0.0
PROFILE
chmod 644 /etc/profile.d/rocm-wsl.sh

echo "DONE: WSL root bootstrap complete"
HSA_ENABLE_DXG_DETECTION=1 /opt/rocm/bin/rocminfo 2>/dev/null | grep -E "Marketing Name|Name:" | head -6 \
  || echo "(rocminfo probe will be verified later as user: clef doctor)"
