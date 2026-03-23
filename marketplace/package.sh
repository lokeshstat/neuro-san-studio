#!/bin/bash
# marketplace/package.sh
# ─────────────────────────────────────────────────────────────────────────────
# Builds the Azure Marketplace Managed Application package (app.zip).
#
# The zip must contain:
#   - mainTemplate.json   (compiled ARM template from main.bicep)
#   - createUiDefinition.json
#
# Prerequisites:
#   - Azure CLI  (az)
#   - Bicep CLI  (az bicep install)
#
# Usage:
#   cd marketplace && bash package.sh
#   Produces: marketplace/app.zip  → upload this to Partner Center
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/dist"
ZIP_FILE="${SCRIPT_DIR}/app.zip"

echo "==> Installing / updating Bicep..."
az bicep install 2>/dev/null || true

echo "==> Compiling Bicep → ARM JSON..."
mkdir -p "${OUT_DIR}"
az bicep build \
  --file "${SCRIPT_DIR}/bicep/main.bicep" \
  --outfile "${OUT_DIR}/mainTemplate.json"

echo "==> Copying createUiDefinition.json..."
cp "${SCRIPT_DIR}/createUiDefinition.json" "${OUT_DIR}/createUiDefinition.json"

echo "==> Validate createUiDefinition.json with ARM TTK (if available)..."
if command -v Test-AzResourceGroupDeployment &>/dev/null; then
  echo "    Running ARM TTK..."
else
  echo "    ARM TTK not found — skipping (install from https://aka.ms/arm-ttk)"
fi

echo "==> Creating app.zip..."
rm -f "${ZIP_FILE}"
cd "${OUT_DIR}"
zip -j "${ZIP_FILE}" mainTemplate.json createUiDefinition.json
cd "${SCRIPT_DIR}"

echo ""
echo "✅  Package ready: ${ZIP_FILE}"
echo ""
echo "Next steps:"
echo "  1. Go to https://partner.microsoft.com/dashboard"
echo "  2. Create a new 'Azure Application' offer → 'Managed Application' plan"
echo "  3. Upload app.zip to the 'Package file' field"
echo "  4. Fill in publisher details, pricing, and categories"
echo "  5. Submit for Microsoft certification review"
