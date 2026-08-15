#!/usr/bin/env bash
# Apply RoboGPT-R1's EasyR1 reward-input patch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${SCRIPT_DIR}/0001-pass-reward-model-content.patch"
DEFAULT_EASYR1_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)/EasyR1"
EASYR1_DIR="${1:-${DEFAULT_EASYR1_DIR}}"

if [[ ! -d "${EASYR1_DIR}/.git" ]]; then
    echo "EasyR1 Git repository not found: ${EASYR1_DIR}" >&2
    echo "Usage: $0 [/absolute/path/to/EasyR1]" >&2
    exit 1
fi

if git -C "${EASYR1_DIR}" apply --reverse --check "${PATCH_FILE}"; then
    echo "Patch is already applied: ${EASYR1_DIR}"
    exit 0
fi

if ! git -C "${EASYR1_DIR}" apply --check "${PATCH_FILE}"; then
    echo "Patch cannot be applied cleanly." >&2
    echo "Check that this is the supported EasyR1 revision, or adapt the patch manually." >&2
    exit 1
fi

git -C "${EASYR1_DIR}" apply "${PATCH_FILE}"
echo "Patch applied successfully: ${EASYR1_DIR}"
