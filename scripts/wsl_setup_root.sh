#!/bin/bash
# Root bootstrap for ROCm-on-WSL2 (AMD Radeon RX 7900 XTX, Ubuntu 22.04)
# Installs: base tools, AMD package repos (7.2.4), ROCm core runtime, librocdxg (WSL enabler)
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

echo "== [1/5] Base packages =="
apt-get update -qq
apt-get install -y -qq git-lfs python3-venv python3-pip libnuma1 libdrm2 libelf1 \
    ca-certificates curl wget

echo "== [2/5] Register AMD package repository (amdgpu-install 7.2.4, jammy) =="
cd /tmp
wget -q https://repo.radeon.com/amdgpu-install/7.2.4/ubuntu/jammy/amdgpu-install_7.2.4.70204-1_all.deb
apt-get install -y -qq ./amdgpu-install_7.2.4.70204-1_all.deb
apt-get update -qq

echo "== [3/5] Install ROCm core runtime (user-space; kernel driver comes from Windows host) =="
# NOTE: no amdgpu-dkms / no kernel driver inside WSL - the Windows Adrenalin driver is the host driver.
apt-get install -y -qq rocm-core python3-setuptools python3-wheel

echo "== [4/5] Install librocdxg (ROCDXG WSL runtime v1.2.2, production for ROCm 7.2.x + RX 7900 XTX) =="
wget -q https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-roct_1.2.2_amd64.deb
wget -q https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-amd-smi-lib_1.2.2_amd64.deb
dpkg -i rocdxg-roct_1.2.2_amd64.deb || apt-get -f install -y -qq
dpkg -i rocdxg-amd-smi-lib_1.2.2_amd64.deb || apt-get -f install -y -qq
rm -f rocdxg-roct_1.2.2_amd64.deb rocdxg-amd-smi-lib_1.2.2_amd64.deb amdgpu-install_*.deb

echo "== [5/5] Environment profile (required for ROCm <= 7.13 on WSL) =="
cat > /etc/profile.d/rocm-wsl.sh << 'EOF'
# Required for ROCm under WSL2 until ROCm 7.13+: detect GPU via /dev/dxg
export HSA_ENABLE_DXG_DETECTION=1
export HSA_OVERRIDE_GFX_VERSION=11.0.0
EOF
chmod 644 /etc/profile.d/rocm-wsl.sh

echo "DONE: WSL root bootstrap complete"
/opt/rocm/bin/rocminfo 2>/dev/null | grep -E "Marketing Name|Name:" | head -6 || echo "(rocminfo probe will be verified later as user)"
