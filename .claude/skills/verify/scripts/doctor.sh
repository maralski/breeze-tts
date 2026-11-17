#!/usr/bin/env bash
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
# Read-only check: is this checkout worth driving? Exits 0 when every required line is ok.
#   .claude/skills/verify/scripts/doctor.sh
set -u
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

fail=0
ok()   { echo "ok    $*"; }
warn() { echo "warn  $*"; }
bad()  { echo "FAIL  $*"; fail=1; }

[[ "$(uname -m)" == "arm64" ]] && ok "Apple silicon" || bad "needs Apple silicon, this is $(uname -m)"

if [[ -x "$BIN" ]]; then
    stale="$(cd "$REPO" && find Sources Package.swift -type f -newer "$BIN" | head -5)"
    if [[ -z "$stale" ]]; then
        ok "binary $BIN is newer than every source file"
    else
        bad "binary is older than these sources; run: swift build -c release"
        echo "$stale" | sed 's/^/        /'
    fi
else
    bad "no binary at $BIN; run: swift build -c release"
fi

if weights="$(resolve_weights)"; then
    if checkpoint_complete "$weights"; then
        ok "checkpoint $weights"
    else
        bad "checkpoint $weights lacks files; breeze-tts names them"
    fi
else
    bad "no complete checkpoint in BREEZE_WEIGHTS or the Hugging Face cache; see the README's Get the weights"
fi

if goldens="$(resolve_goldens)"; then
    ok "goldens $goldens ($(find "$goldens" -mindepth 2 -maxdepth 2 -name manifest.json | wc -l | tr -d ' ') cases)"
else
    warn "no goldens (BREEZE_GOLDENS or .audit/goldens); --parity cannot run"
fi

if asr_ready; then
    ok "whisper (tools/asr/transcribe.py, cached MLX weights, ffmpeg): drive.sh will transcribe --say output"
else
    warn "no speech recognition: tools/asr/transcribe.py needs openai-whisper and torch in \$BREEZE_ASR_PYTHON (default python3), ffmpeg, and mlx-community/whisper-small.en-mlx in the Hugging Face cache; drive.sh will skip transcripts"
fi

running="$(pgrep -fl "$BIN" || true)"
[[ -z "$running" ]] && ok "no other breeze-tts from this checkout is running" \
    || warn "breeze-tts already running (timings will be skewed): $running"

echo "evidence root: $EVIDENCE_ROOT"
exit $fail
