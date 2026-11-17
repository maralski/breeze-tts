#!/bin/zsh
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
# Builds dist/breeze-tts-VERSION-macos-arm64.zip: the command-line tool with LICENSE, NOTICE
# and README.
#
#   scripts/build-release.sh VERSION
#
# The binary is ad-hoc signed unless BREEZE_SIGN_IDENTITY names a Developer ID Application
# identity. With BREEZE_NOTARY_PROFILE (a notarytool keychain profile) as well, the zip is
# submitted to Apple's notary service. A bare executable cannot carry a stapled ticket, so
# Gatekeeper looks the notarization up online the first time the tool runs.
set -euo pipefail

VERSION="${1:?usage: build-release.sh VERSION}"
PROJECT_DIRECTORY="${0:A:h:h}"
NAME="breeze-tts-${VERSION}-macos-arm64"
STAGE="${PROJECT_DIRECTORY}/dist/${NAME}"
ZIP_ARCHIVE="${PROJECT_DIRECTORY}/dist/${NAME}.zip"
IDENTITY="${BREEZE_SIGN_IDENTITY:--}"

cd "${PROJECT_DIRECTORY}"
swift build -c release --arch arm64 --product breeze-tts
BIN_DIRECTORY="$(swift build -c release --arch arm64 --show-bin-path)"

rm -rf "${STAGE}" "${ZIP_ARCHIVE}"
mkdir -p "${STAGE}"
cp "${BIN_DIRECTORY}/breeze-tts" LICENSE NOTICE README.md "${STAGE}/"
chmod 755 "${STAGE}/breeze-tts"

if [[ "${IDENTITY}" == "-" ]]; then
  codesign --force --sign - "${STAGE}/breeze-tts"
else
  codesign --force --sign "${IDENTITY}" --options runtime --timestamp \
    --identifier io.github.maralski.breeze-tts "${STAGE}/breeze-tts"
fi
codesign --verify --strict "${STAGE}/breeze-tts"
ditto -c -k --keepParent "${STAGE}" "${ZIP_ARCHIVE}"

if [[ -n "${BREEZE_NOTARY_PROFILE:-}" ]]; then
  [[ "${IDENTITY}" != "-" ]] || { echo "notarization needs BREEZE_SIGN_IDENTITY" >&2; exit 1; }
  xcrun notarytool submit "${ZIP_ARCHIVE}" --keychain-profile "${BREEZE_NOTARY_PROFILE}" --wait
fi

echo "Built ${ZIP_ARCHIVE}"
shasum -a 256 "${ZIP_ARCHIVE}"
