// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0

// Code embeddings and token sampling, both reading and writing codes in GPU buffers so a frame
// never returns to the CPU. The sampling rule is specified in Sampling.swift, next to the CPU
// implementation of the same rule.
extension ShaderSource {
    static let sampling = #"""
    struct EmbedParams { uint width; uint count; uint vocab; uint ldy; uint q0; uint rowStride; };

    // out[r] = sum over c < count of E[codes[r * rowStride + q0 + c] + (q0 + c) * vocab], added in
    // ascending c in fp32. count 16, q0 0 is a frame's backbone input; count 1 is one code's row.
    // rowStride 0 writes the same frame to every output row, 16 walks consecutive frames.
    // A thread owns four dims.
    kernel void embed_codes(device const ushort* E [[buffer(0)]],
                            device const uint* codes [[buffer(1)]],
                            device float* out [[buffer(2)]],
                            constant EmbedParams& p [[buffer(3)]],
                            uint2 gid [[thread_position_in_grid]]) {
        const uint d = gid.x * 4, r = gid.y;
        if (d >= p.width) return;
        float4 acc = float4(0.0f);
        for (uint c = 0; c < p.count; c++) {
            const uint q = p.q0 + c;
            const uint idx = codes[r * p.rowStride + q] + q * p.vocab;
            const uint2 w = *(device const uint2*)(E + idx * p.width + d);
            acc += float4(bf16_to_float(ushort(w.x & 0xFFFF)), bf16_to_float(ushort(w.x >> 16)),
                          bf16_to_float(ushort(w.y & 0xFFFF)), bf16_to_float(ushort(w.y >> 16)));
        }
        *(device float4*)(out + r * p.ldy + d) = acc;
    }

    struct SampleParams {
        uint N; uint rows; uint ld; uint topK; uint greedy; uint slot;
        uint suppressLo; uint suppressHi; uint clampMax;
        float cfgScale; float temperature; float penalty;
    };
    constant int SAMPLE_TG = 256;
    constant int SAMPLE_SG = SAMPLE_TG / 32;
    constant int SAMPLE_OWN = 9;
    constant int SAMPLE_SPAN = 32 * SAMPLE_OWN;
    constant uint SAMPLE_NONE = 0xFFFFFFFFu;
    constant uint STATUS_BAD_LOGITS = 1;

    // Floats as unsigned keys in the same order (-inf lowest, +inf highest).
    inline uint order_key(float v) {
        // v + 0 turns -0 into +0, which compare equal as floats and must share a key.
        const uint b = as_type<uint>(v + 0.0f);
        return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    }

    // One draw as one threadgroup of 256 threads: larger single threadgroups wait for a whole
    // GPU core whenever another process draws. Thread t owns the nine token ids from 9 t, so a
    // simdgroup owns a contiguous id range in lane order. The top-k cut is found by bisection on
    // the score's bit pattern, one count over all threads per bit, instead of sorting or ranking;
    // the survivors are compacted per simdgroup with a prefix sum, and one thread then walks them
    // in ascending id for the running sums the rule prescribes.
    // BACKBONE applies the repetition penalty over the `seen` set and keeps top-k ties, then
    // adds the drawn token to the set; the depth variant keeps exactly top-k and clamps.
    template <int BACKBONE>
    kernel void sample_kernel(device const float* logits [[buffer(0)]],
                              device const float* uniforms [[buffer(1)]],
                              device uint* codes [[buffer(2)]],
                              device uint* seen [[buffer(3)]],
                              device uint* status [[buffer(4)]],
                              constant SampleParams& p [[buffer(5)]],
                              uint sgid [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
        threadgroup float redv[SAMPLE_SG];
        threadgroup uint redi[2][SAMPLE_SG];
        threadgroup uint redbad[SAMPLE_SG];
        threadgroup uint keptID[SAMPLE_SG][SAMPLE_SPAN];
        threadgroup float keptE[SAMPLE_SG][SAMPLE_SPAN];
        const uint id0 = (sgid * 32 + lane) * SAMPLE_OWN;
        const bool writer = sgid == 0 && lane == 0;

        // Scores: guidance, penalty, suppression, temperature, in the reference's order.
        float own[SAMPLE_OWN];
        uint key[SAMPLE_OWN];
        uint bad = 0;
        float best = -INFINITY;
        uint bestAt = SAMPLE_NONE;
        #pragma clang loop unroll(full)
        for (int a = 0; a < SAMPLE_OWN; a++) {
            const uint i = id0 + a;
            float v = -INFINITY;
            if (i < p.N) {
                v = logits[i];
                if (p.rows == 2) {
                    // Three statements: each result must round to float before the next uses it.
                    const float un = logits[p.ld + i];
                    const float diff = v - un;
                    const float scaled = p.cfgScale * diff;
                    v = un + scaled;
                }
                bad |= uint(isnan(v));
                if (BACKBONE && p.penalty != 1.0f && seen[i] != 0) v = v > 0.0f ? v / p.penalty : v * p.penalty;
                if (i >= p.suppressLo && i < p.suppressHi) v = -INFINITY;
                if (!p.greedy) v = v / p.temperature;
                if (v > best) { best = v; bestAt = i; }
            }
            own[a] = v;
            key[a] = order_key(v);
        }
        const float sgBest = simd_max(best);
        const uint sgAt = simd_min(best == sgBest ? bestAt : SAMPLE_NONE);
        const uint sgBad = simd_or(bad);
        if (lane == 0) { redv[sgid] = sgBest; redi[1][sgid] = sgAt; redbad[sgid] = sgBad; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float top = redv[0];
        #pragma clang loop unroll(full)
        for (int t = 1; t < SAMPLE_SG; t++) top = max(top, redv[t]);
        uint token = SAMPLE_NONE;
        uint anyBad = 0;
        #pragma clang loop unroll(full)
        for (int t = 0; t < SAMPLE_SG; t++) {
            if (redv[t] == top) token = min(token, redi[1][t]);
            anyBad |= redbad[t];
        }

        if (!p.greedy) {
            // Counts over all 256 threads alternate between two slots, so one barrier a round
            // separates this round's reads from the next round's writes.
            uint slot = 0;
            #define SAMPLE_COUNT(result, condition) { \
                uint n_ = 0; \
                for (int a = 0; a < SAMPLE_OWN; a++) n_ += uint(condition); \
                n_ = simd_sum(n_); \
                if (lane == 0) redi[slot][sgid] = n_; \
                threadgroup_barrier(mem_flags::mem_threadgroup); \
                n_ = 0; \
                for (int t = 0; t < SAMPLE_SG; t++) n_ += redi[slot][t]; \
                slot ^= 1; \
                result = n_; \
            }
            // cut = the k-th largest key: the largest value that at least k keys reach.
            uint cut = 0;
            for (int bit = 31; bit >= 0; bit--) {
                const uint candidate = cut | (1u << bit);
                uint reach;
                SAMPLE_COUNT(reach, key[a] >= candidate)
                if (reach >= p.topK) cut = candidate;
            }
            // The depth sampler keeps exactly k: of the tokens at the cut, the lowest ids, up to
            // lastTie, the id of the last one there is room for.
            uint lastTie = SAMPLE_NONE;
            if (!BACKBONE) {
                uint above;
                SAMPLE_COUNT(above, key[a] > cut)
                const uint room = p.topK - above;
                lastTie = 0;
                for (int bit = 11; bit >= 0; bit--) {
                    const uint candidate = lastTie | (1u << bit);
                    uint before;
                    SAMPLE_COUNT(before, key[a] == cut && id0 + a < candidate)
                    if (before < room) lastTie = candidate;
                }
            }
            #undef SAMPLE_COUNT
            // Weights of the survivors, packed per simdgroup in ascending id.
            float e[SAMPLE_OWN];
            bool kept[SAMPLE_OWN];
            uint mine = 0;
            #pragma clang loop unroll(full)
            for (int a = 0; a < SAMPLE_OWN; a++) {
                kept[a] = own[a] > -INFINITY && (key[a] > cut || (key[a] == cut && id0 + a <= lastTie));
                e[a] = kept[a] ? exp(own[a] - top) : 0.0f;
                mine += uint(kept[a]);
            }
            uint at = simd_prefix_exclusive_sum(mine);
            const uint inGroup = simd_sum(mine);
            #pragma clang loop unroll(full)
            for (int a = 0; a < SAMPLE_OWN; a++) {
                if (kept[a]) {
                    keptID[sgid][at] = id0 + a;
                    keptE[sgid][at] = e[a];
                    at++;
                }
            }
            if (lane == 0) redi[slot][sgid] = inGroup;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (!writer) return;
            // Running sums in ascending token id, one fp32 add each; tokens outside the kept
            // set would add zero.
            float total = 0.0f;
            #pragma clang loop unroll(disable)
            for (int t = 0; t < SAMPLE_SG; t++) {
                for (uint j = 0; j < redi[slot][t]; j++) total += keptE[t][j];
            }
            const float target = uniforms[p.slot] * total;
            float c = 0.0f;
            uint last = SAMPLE_NONE;
            token = SAMPLE_NONE;
            #pragma clang loop unroll(disable)
            for (int t = 0; t < SAMPLE_SG; t++) {
                for (uint j = 0; j < redi[slot][t]; j++) {
                    const float w = keptE[t][j];
                    c += w;
                    if (w > 0.0f) last = keptID[t][j];
                    if (c > target && token == SAMPLE_NONE) token = keptID[t][j];
                }
            }
            if (token == SAMPLE_NONE) token = last;
            if (!(total > 0.0f) || isinf(total)) anyBad = 1;
        }
        if (!writer) return;
        if (token == SAMPLE_NONE || isinf(top)) anyBad = 1;
        if (anyBad) status[0] = status[0] | STATUS_BAD_LOGITS;
        token = min(token, BACKBONE ? p.N - 1 : p.clampMax);
        codes[p.slot] = token;
        if (BACKBONE) seen[token] = 1;
    }

    template [[host_name("sample_backbone")]] kernel void sample_kernel<1>(device const float*, device const float*, device uint*, device uint*, device uint*, constant SampleParams&, uint, uint);
    template [[host_name("sample_depth")]] kernel void sample_kernel<0>(device const float*, device const float*, device uint*, device uint*, device uint*, constant SampleParams&, uint, uint);

    """#
}
