#!/usr/bin/env bash
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
# Runs the real breeze-tts binary once and keeps the evidence.
#
#   .claude/skills/verify/scripts/drive.sh <name> -- <breeze-tts arguments>
#
#   drive.sh plain    -- --say "The garden stays open until sunset."
#   drive.sh design   -- --say "Welcome aboard." --instruction "A warm young woman."
#   drive.sh bad-flag -- --say hi --guidance 4          # exit 2 is the evidence
#   drive.sh parity   -- --parity "$G/plain_en"
#   drive.sh bench    -- --bench --repeat 5
#
# It adds --weights (see common.sh for the search order) and, for --say, --out
# <evidence dir>/out.wav, unless you pass them yourself. Every run holds the GPU lock
# through scripts/gpulock.py. Paths you pass are resolved from your own working directory.
#
# Evidence goes to .audit/verify/<UTC time>-<name>/ (override the root with BREEZE_VERIFY_DIR):
#   command.txt     the exact command line
#   stdout.txt      what the tool printed
#   stderr.txt      its error output
#   exit.txt        its exit code
#   out.wav         the audio, with out.fragN.wav when the text split into fragments
#   wavstat.json    format, length, peak, RMS and edge silence of each wav
#   transcript.txt  Whisper's reading of out.wav against the --say text, with the word error
#                   rate (character error rate for Chinese), when tools/asr can run
#
# The script exits with breeze-tts's exit code and prints the evidence directory last.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [[ $# -lt 3 || "$2" != "--" ]]; then
    sed -n '4,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 64
fi
name="$1"
shift 2
args=("$@")

[[ -x "$BIN" ]] || { echo "no binary at $BIN; run: swift build -c release" >&2; exit 69; }

has() { local flag; for flag in "${args[@]}"; do [[ "$flag" == "$1" ]] && return 0; done; return 1; }
value_of() { local i; for i in "${!args[@]}"; do [[ "${args[$i]}" == "$1" ]] && { echo "${args[$((i + 1))]:-}"; return 0; }; done; return 1; }

dir="$EVIDENCE_ROOT/$(date -u +%Y%m%dT%H%M%SZ)-$name"
mkdir -p "$dir"

# Added flags go first, so a trailing flag that lacks its value still fails as the user would see it.
if has --say && ! has --out; then args=(--out "$dir/out.wav" "${args[@]}"); fi
if ! has --weights && weights="$(resolve_weights)"; then args=(--weights "$weights" "${args[@]}"); fi

# Every run holds the GPU lock, so it neither skews nor is skewed by a benchmark in another worktree.
cmd=("$REPO/scripts/gpulock.py" -- "$BIN" "${args[@]}")

printf '%q ' "${cmd[@]}" > "$dir/command.txt"
echo >> "$dir/command.txt"

"${cmd[@]}" > "$dir/stdout.txt" 2> "$dir/stderr.txt"
code=$?
echo "$code" > "$dir/exit.txt"

cat "$dir/stdout.txt"
cat "$dir/stderr.txt" >&2

shopt -s nullglob
wavs=("$dir"/*.wav)
if [[ ${#wavs[@]} -gt 0 ]]; then
    {
        echo "{"
        for i in "${!wavs[@]}"; do
            [[ $i -gt 0 ]] && echo ","
            printf '"%s": ' "$(basename "${wavs[$i]}")"
            "$(dirname "${BASH_SOURCE[0]}")/wavstat.py" "${wavs[$i]}"
        done
        echo "}"
    } > "$dir/wavstat.json"
    echo "--- wavstat"
    cat "$dir/wavstat.json"
fi

# Transcribe the speech so the proof covers the words, not only the waveform.
if has --say && [[ $code -eq 0 && -f "$dir/out.wav" && -z "${BREEZE_VERIFY_NO_ASR:-}" ]]; then
    if asr_ready; then
        text="$(value_of --say)"
        lang=en
        if printf '%s' "$text" | grep -q '[一-龥]'; then lang=zh; fi
        "$ASR" --lang "$lang" --max-error 1 "$dir/out.wav" "$text" > "$dir/transcript.txt" 2> "$dir/asr-stderr.txt"
        echo "--- transcript"
        cat "$dir/transcript.txt"
    else
        echo "--- transcript skipped: speech recognition is not set up (run doctor.sh)"
    fi
fi

echo "--- exit $code, evidence in $dir"
exit $code
