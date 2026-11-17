# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
# Sourced by doctor.sh and drive.sh. Resolves the repo, the binary, the checkpoint and the goldens.
# shellcheck shell=bash

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BIN="$REPO/.build/release/breeze-tts"
EVIDENCE_ROOT="${BREEZE_VERIFY_DIR:-$REPO/.audit/verify}"
ASR="$REPO/tools/asr/transcribe.py"

# The files breeze-tts reads; BreezeCheckpoint.synthesisFiles holds the same list.
checkpoint_complete() {
    local f
    for f in model.safetensors.index.json model-00001-of-00002.safetensors model-00002-of-00002.safetensors \
             tokenizer.json audio_tokenizer/model.safetensors; do
        [[ -f "$1/$f" ]] || return 1
    done
}

# Prints the checkpoint directory, or returns 1.
# Order: BREEZE_WEIGHTS, then the newest complete snapshot in the Hugging Face cache, as breeze-tts searches.
resolve_weights() {
    if [[ -n "${BREEZE_WEIGHTS:-}" ]]; then
        echo "$BREEZE_WEIGHTS"
        return 0
    fi
    local hub="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}" candidate
    for candidate in $(ls -td "$hub"/models--BreezeBlue--Breeze-TTS-2/snapshots/*/ 2>/dev/null); do
        if checkpoint_complete "$candidate"; then
            (cd "$candidate" && pwd)
            return 0
        fi
    done
    return 1
}

# Prints the directory of golden parity cases, or returns 1. They are not published.
resolve_goldens() {
    local candidate
    for candidate in "${BREEZE_GOLDENS:-}" "$REPO/.audit/goldens"; do
        if [[ -n "$candidate" && -d "$candidate" ]]; then
            (cd "$candidate" && pwd)
            return 0
        fi
    done
    return 1
}

# Whisper weights the ASR script reads from the Hugging Face cache.
asr_ready() {
    local hub="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
    [[ -x "$ASR" ]] && [[ -x "$(head -1 "$ASR" | cut -c3-)" ]] && command -v ffmpeg >/dev/null \
        && ls -d "$hub/models--mlx-community--whisper-small.en-mlx/snapshots/"*/ >/dev/null 2>&1
}
