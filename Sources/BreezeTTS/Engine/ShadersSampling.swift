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
    constant int SAMPLE_MAXN = 2056;
    constant int SAMPLE_OWN = 9;
    constant uint SAMPLE_NONE = 0xFFFFFFFFu;
    constant uint STATUS_BAD_LOGITS = 1;

    // One draw as one threadgroup of 256 threads: larger single threadgroups wait for a whole
    // GPU core whenever another process draws. Thread t owns tokens t, t + 256, ...
    // BACKBONE applies the repetition penalty over the `seen` set and keeps top-k ties, then
    // adds the drawn token to the set; the depth variant keeps exactly top-k and clamps.
    template <int BACKBONE>
    kernel void sample_kernel(device const float* logits [[buffer(0)]],
                              device const float* uniforms [[buffer(1)]],
                              device uint* codes [[buffer(2)]],
                              device uint* seen [[buffer(3)]],
                              device uint* status [[buffer(4)]],
                              constant SampleParams& p [[buffer(5)]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint sgid [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
        threadgroup float4 x4[SAMPLE_MAXN / 4];
        threadgroup float redv[SAMPLE_SG];
        threadgroup uint redi[SAMPLE_SG];
        threadgroup uint redbad[SAMPLE_SG];
        threadgroup float total;
        threadgroup uint last;
        threadgroup float* x = (threadgroup float*)x4;

        // Scores: guidance, penalty, suppression, temperature, in the reference's order.
        float own[SAMPLE_OWN];
        uint bad = 0;
        float best = -INFINITY;
        uint bestAt = SAMPLE_NONE;
        #pragma clang loop unroll(full)
        for (int a = 0; a < SAMPLE_OWN; a++) {
            const uint i = tid + a * SAMPLE_TG;
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
            if (i < uint(SAMPLE_MAXN)) x[i] = v;
        }
        const float sgBest = simd_max(best);
        const uint sgAt = simd_min(best == sgBest ? bestAt : SAMPLE_NONE);
        const uint sgBad = simd_or(bad);
        if (lane == 0) { redv[sgid] = sgBest; redi[sgid] = sgAt; redbad[sgid] = sgBad; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float top = redv[0];
        #pragma clang loop unroll(full)
        for (int t = 1; t < SAMPLE_SG; t++) top = max(top, redv[t]);
        uint token = SAMPLE_NONE;
        uint anyBad = 0;
        #pragma clang loop unroll(full)
        for (int t = 0; t < SAMPLE_SG; t++) {
            if (redv[t] == top) token = min(token, redi[t]);
            anyBad |= redbad[t];
        }

        if (!p.greedy) {
            // A token survives top-k when fewer than k tokens dominate it: strictly greater
            // scores, and in the depth variant equal scores at a lower id as well.
            uint dominators[SAMPLE_OWN];
            #pragma clang loop unroll(full)
            for (int a = 0; a < SAMPLE_OWN; a++) dominators[a] = 0;
            for (uint j = 0; j < uint(SAMPLE_MAXN); j += 4) {
                const float4 xv = x4[j / 4];
                const uint4 jv = j + uint4(0, 1, 2, 3);
                #pragma clang loop unroll(full)
                for (int a = 0; a < SAMPLE_OWN; a++) {
                    bool4 over = xv > own[a];
                    if (!BACKBONE) over = over | ((xv == own[a]) & (jv < uint4(tid + a * SAMPLE_TG)));
                    const uint4 one = select(uint4(0), uint4(1), over);
                    dominators[a] += one.x + one.y + one.z + one.w;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (int a = 0; a < SAMPLE_OWN; a++) {
                const uint i = tid + a * SAMPLE_TG;
                if (i < uint(SAMPLE_MAXN)) {
                    const bool kept = i < p.N && dominators[a] < p.topK && own[a] > -INFINITY;
                    x[i] = kept ? exp(own[a] - top) : 0.0f;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Running sums in ascending token id, one fp32 add each.
            if (tid == 0) {
                float c = 0.0f;
                uint lastKept = SAMPLE_NONE;
                for (uint j = 0; j < p.N; j++) {
                    const float e = x[j];
                    if (e > 0.0f) lastKept = j;
                    c += e;
                    x[j] = c;
                }
                total = c;
                last = lastKept;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float target = uniforms[p.slot] * total;
            uint firstOver = SAMPLE_NONE;
            #pragma clang loop unroll(full)
            for (int a = SAMPLE_OWN - 1; a >= 0; a--) {
                const uint i = tid + a * SAMPLE_TG;
                if (i < p.N && x[i] > target) firstOver = i;
            }
            firstOver = simd_min(firstOver);
            if (lane == 0) redi[sgid] = firstOver;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            token = redi[0];
            #pragma clang loop unroll(full)
            for (int t = 1; t < SAMPLE_SG; t++) token = min(token, redi[t]);
            if (token == SAMPLE_NONE) token = last;
            if (!(total > 0.0f) || isinf(total)) anyBad = 1;
        }
        if (tid != 0) return;
        if (token == SAMPLE_NONE || isinf(top)) anyBad = 1;
        if (anyBad) status[0] = status[0] | STATUS_BAD_LOGITS;
        token = min(token, BACKBONE ? p.N - 1 : p.clampMax);
        codes[p.slot] = token;
        if (BACKBONE) seen[token] = 1;
    }

    template [[host_name("sample_backbone")]] kernel void sample_kernel<1>(device const float*, device const float*, device uint*, device uint*, device uint*, constant SampleParams&, uint, uint, uint);
    template [[host_name("sample_depth")]] kernel void sample_kernel<0>(device const float*, device const float*, device uint*, device uint*, device uint*, constant SampleParams&, uint, uint, uint);

    """#
}
