// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0

// Three attention kernels, one per workload. They share the row and cache layout described in
// ShadersNorm.swift and nothing else.
extension ShaderSource {
    static let attention = #"""
    struct AttnParams { uint B; uint M; uint hq; uint hkv; uint ldq; uint ldo; uint cap; uint seqStride; float scale; uint pos0[4]; };
    constant int ATT_U = 8;

    // (a) Backbone: causal GQA over a cache of up to thousands of keys, head dim 128.
    // An item is (row, KV head): its G query heads share every key and value load. A threadgroup
    // is one item split over S simdgroups, which take the keys in interleaved chunks of ATT_U.
    // A lane owns four of the 128 dims, so a key is one contiguous 512-byte load across the
    // simdgroup and a score is one simd_sum. Scores never leave registers: each chunk is folded
    // into a running softmax (max, sum and weighted V rescaled when the max rises), so the key
    // count is bounded by the cache alone. The S partial softmaxes meet in threadgroup memory.
    // Decode uses S > 1 (few items, many keys); prefill S = 1 (thousands of items).
    template <int S, int G>
    kernel void attention_causal(device const float* Q [[buffer(0)]],
                                 device const float* Kc [[buffer(1)]],
                                 device const float* Vc [[buffer(2)]],
                                 device float* O [[buffer(3)]],
                                 constant AttnParams& p [[buffer(4)]],
                                 uint2 tgid [[threadgroup_position_in_grid]],
                                 uint sgid [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
        threadgroup float red[S][G][2];
        threadgroup float4 part[S][G][32];
        const uint kvh = tgid.x, row = tgid.y;
        const uint s = row / p.M;
        const uint nKeys = p.pos0[s] + row % p.M + 1;
        device const float* q = Q + row * p.ldq + kvh * G * 128 + 4 * lane;
        device const float* kb = Kc + s * p.seqStride + kvh * p.cap * 128 + 4 * lane;
        device const float* vb = Vc + s * p.seqStride + kvh * p.cap * 128 + 4 * lane;
        float4 qv[G], acc[G];
        float mx[G], sum[G];
        #pragma clang loop unroll(full)
        for (int g = 0; g < G; g++) {
            qv[g] = *(device const float4*)(q + g * 128);
            acc[g] = float4(0.0f);
            mx[g] = -INFINITY;
            sum[g] = 0.0f;
        }
        for (uint j0 = sgid * ATT_U; j0 < nKeys; j0 += S * ATT_U) {
            float sc[G][ATT_U];
            #pragma clang loop unroll(full)
            for (int u = 0; u < ATT_U; u++) {
                const bool ok = j0 + u < nKeys;
                float4 k = float4(0.0f);
                if (ok) k = *(device const float4*)(kb + (j0 + u) * 128);
                #pragma clang loop unroll(full)
                for (int g = 0; g < G; g++) sc[g][u] = ok ? simd_sum(dot(qv[g], k)) * p.scale : -INFINITY;
            }
            #pragma clang loop unroll(full)
            for (int g = 0; g < G; g++) {
                float top = mx[g];
                #pragma clang loop unroll(full)
                for (int u = 0; u < ATT_U; u++) top = max(top, sc[g][u]);
                const float carry = exp(mx[g] - top);
                mx[g] = top;
                sum[g] *= carry;
                acc[g] *= carry;
                #pragma clang loop unroll(full)
                for (int u = 0; u < ATT_U; u++) {
                    sc[g][u] = exp(sc[g][u] - top);
                    sum[g] += sc[g][u];
                }
            }
            #pragma clang loop unroll(full)
            for (int u = 0; u < ATT_U; u++) {
                if (j0 + u < nKeys) {
                    const float4 v = *(device const float4*)(vb + (j0 + u) * 128);
                    #pragma clang loop unroll(full)
                    for (int g = 0; g < G; g++) acc[g] += sc[g][u] * v;
                }
            }
        }
        if (S > 1) {
            #pragma clang loop unroll(full)
            for (int g = 0; g < G; g++) {
                if (lane == 0) { red[sgid][g][0] = mx[g]; red[sgid][g][1] = sum[g]; }
                part[sgid][g][lane] = acc[g];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgid != 0) return;
            #pragma clang loop unroll(full)
            for (int g = 0; g < G; g++) {
                float top = red[0][g][0];
                #pragma clang loop unroll(full)
                for (int t = 1; t < S; t++) top = max(top, red[t][g][0]);
                float total = 0.0f;
                float4 a = float4(0.0f);
                #pragma clang loop unroll(full)
                for (int t = 0; t < S; t++) {
                    const float w = exp(red[t][g][0] - top);
                    total += red[t][g][1] * w;
                    a += part[t][g][lane] * w;
                }
                sum[g] = total;
                acc[g] = a;
            }
        }
        #pragma clang loop unroll(full)
        for (int g = 0; g < G; g++) *(device float4*)(O + row * p.ldo + (kvh * G + g) * 128 + 4 * lane) = acc[g] / sum[g];
    }

    #define ATTN_CAUSAL(S) template [[host_name("attention_causal_" #S)]] kernel void attention_causal<S, 2>(device const float*, device const float*, device const float*, device float*, constant AttnParams&, uint2, uint, uint);
    ATTN_CAUSAL(1) ATTN_CAUSAL(4) ATTN_CAUSAL(8)
    #undef ATTN_CAUSAL

    struct ShortAttnParams { uint B; uint M; uint hq; uint hkv; uint ldq; uint ldo; uint cap; uint seqStride; float scale; uint pos0; };
    constant int SHORT_SG = 4;
    constant int SHORT_MAXKEYS = 17;

    inline void rotate_pairs(thread float2& lo, thread float2& hi, float2 angle) {
        const float2 c = cos(angle), s = sin(angle);
        const float2 r = lo * c - hi * s;
        hi = hi * c + lo * s;
        lo = r;
    }

    // (b) Depth decoder: causal GQA over at most SHORT_MAXKEYS keys, head dim 128, every sequence
    // at the same positions pos0 ..< pos0 + M. With so few keys the dispatch itself is the cost,
    // so this one does the work of three: rotate-half RoPE on q, k and v of the new rows into
    // the cache, and the attention. One simdgroup per (row, query head); a lane owns pairs
    // (lane, lane + 64) and (lane + 32, lane + 96). Keys before pos0 come from the cache; the
    // keys and values of this dispatch's own rows are read from QKV and rotated in registers,
    // never from the cache, so the cache writes (made by the first query head of each KV group)
    // race with nothing.
    kernel void attention_short(device const float* QKV [[buffer(0)]],
                                device float* Kc [[buffer(1)]],
                                device float* Vc [[buffer(2)]],
                                device const float* invFreq [[buffer(3)]],
                                device float* O [[buffer(4)]],
                                constant ShortAttnParams& p [[buffer(5)]],
                                uint tgid [[threadgroup_position_in_grid]],
                                uint sgid [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
        const uint work = tgid * SHORT_SG + sgid;
        if (work >= p.B * p.M * p.hq) return;
        const uint row = work / p.hq, h = work % p.hq;
        const uint s = row / p.M, m = row % p.M, pos = p.pos0 + m;
        const uint G = p.hq / p.hkv, kvh = h / G;
        const bool writer = h % G == 0;
        const uint i0 = lane, i1 = lane + 32;
        const float2 freq = float2(invFreq[i0], invFreq[i1]);
        device const float* rows = QKV + s * p.M * p.ldq;
        const uint kOff = (p.hq + kvh) * 128, vOff = (p.hq + p.hkv + kvh) * 128;
        device float* kc = Kc + s * p.seqStride + kvh * p.cap * 128;
        device float* vc = Vc + s * p.seqStride + kvh * p.cap * 128;

        device const float* qp = rows + m * p.ldq + h * 128;
        float2 qlo = float2(qp[i0], qp[i1]), qhi = float2(qp[i0 + 64], qp[i1 + 64]);
        rotate_pairs(qlo, qhi, float(pos) * freq);

        float sc[SHORT_MAXKEYS];
        float top = -INFINITY;
        for (uint j = 0; j <= pos; j++) {
            float2 klo, khi;
            if (j < p.pos0) {
                device const float* kp = kc + j * 128;
                klo = float2(kp[i0], kp[i1]);
                khi = float2(kp[i0 + 64], kp[i1 + 64]);
            } else {
                device const float* kp = rows + (j - p.pos0) * p.ldq + kOff;
                klo = float2(kp[i0], kp[i1]);
                khi = float2(kp[i0 + 64], kp[i1 + 64]);
                rotate_pairs(klo, khi, float(j) * freq);
                if (writer && j == pos) {
                    device float* dst = kc + pos * 128;
                    dst[i0] = klo.x; dst[i1] = klo.y; dst[i0 + 64] = khi.x; dst[i1 + 64] = khi.y;
                }
            }
            sc[j] = simd_sum(dot(qlo, klo) + dot(qhi, khi)) * p.scale;
            top = max(top, sc[j]);
        }
        float sum = 0.0f;
        float2 alo = float2(0.0f), ahi = float2(0.0f);
        for (uint j = 0; j <= pos; j++) {
            const float e = exp(sc[j] - top);
            sum += e;
            device const float* vp = j < p.pos0 ? (device const float*)(vc + j * 128) : rows + (j - p.pos0) * p.ldq + vOff;
            const float2 vlo = float2(vp[i0], vp[i1]), vhi = float2(vp[i0 + 64], vp[i1 + 64]);
            alo += e * vlo;
            ahi += e * vhi;
            if (writer && j == pos) {
                device float* dst = vc + pos * 128;
                dst[i0] = vlo.x; dst[i1] = vlo.y; dst[i0 + 64] = vhi.x; dst[i1 + 64] = vhi.y;
            }
        }
        device float* o = O + row * p.ldo + h * 128;
        alo /= sum;
        ahi /= sum;
        o[i0] = alo.x; o[i1] = alo.y; o[i0 + 64] = ahi.x; o[i1 + 64] = ahi.y;
    }

    struct WindowAttnParams { uint M; uint row0; uint total; uint before; uint after; uint ldq; uint ldo; float scale; };
    constant int WIN_U = 4;
    constant int WIN_G = 4;

    // (c) Text encoder: bidirectional attention of one sequence over itself, four query heads on
    // one KV head, head dim 256. Query i sees keys max(0, i - before) ... min(total - 1, i + after);
    // Gemma's sliding layers pass before 255 and after 256, full layers pass the sequence length.
    // Every row is a query, so an item is a row and a simdgroup: its lanes own eight dims each and
    // the four heads share each key and value load, folded WIN_U keys at a time into a running
    // softmax like (a). All keys must be in the cache before this runs.
    kernel void attention_window(device const float* Q [[buffer(0)]],
                                 device const float* Kc [[buffer(1)]],
                                 device const float* Vc [[buffer(2)]],
                                 device float* O [[buffer(3)]],
                                 constant WindowAttnParams& p [[buffer(4)]],
                                 uint tgid [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
        const uint row = tgid, pos = p.row0 + row;
        const uint first = pos > p.before ? pos - p.before : 0;
        const uint end = min(p.total, pos + min(p.after, p.total) + 1);
        device const float* q = Q + row * p.ldq + 8 * lane;
        device const float* kb = Kc + 8 * lane;
        device const float* vb = Vc + 8 * lane;
        float4 qa[WIN_G], qb[WIN_G], aa[WIN_G], ab[WIN_G];
        float mx[WIN_G], sum[WIN_G];
        #pragma clang loop unroll(full)
        for (int g = 0; g < WIN_G; g++) {
            qa[g] = *(device const float4*)(q + g * 256);
            qb[g] = *(device const float4*)(q + g * 256 + 4);
            aa[g] = float4(0.0f);
            ab[g] = float4(0.0f);
            mx[g] = -INFINITY;
            sum[g] = 0.0f;
        }
        for (uint j0 = first; j0 < end; j0 += WIN_U) {
            float sc[WIN_G][WIN_U];
            #pragma clang loop unroll(full)
            for (int u = 0; u < WIN_U; u++) {
                const bool ok = j0 + u < end;
                float4 ka = float4(0.0f), kb4 = float4(0.0f);
                if (ok) {
                    ka = *(device const float4*)(kb + (j0 + u) * 256);
                    kb4 = *(device const float4*)(kb + (j0 + u) * 256 + 4);
                }
                #pragma clang loop unroll(full)
                for (int g = 0; g < WIN_G; g++) sc[g][u] = ok ? simd_sum(dot(qa[g], ka) + dot(qb[g], kb4)) * p.scale : -INFINITY;
            }
            #pragma clang loop unroll(full)
            for (int g = 0; g < WIN_G; g++) {
                float top = mx[g];
                #pragma clang loop unroll(full)
                for (int u = 0; u < WIN_U; u++) top = max(top, sc[g][u]);
                const float carry = exp(mx[g] - top);
                mx[g] = top;
                sum[g] *= carry;
                aa[g] *= carry;
                ab[g] *= carry;
                #pragma clang loop unroll(full)
                for (int u = 0; u < WIN_U; u++) {
                    sc[g][u] = exp(sc[g][u] - top);
                    sum[g] += sc[g][u];
                }
            }
            #pragma clang loop unroll(full)
            for (int u = 0; u < WIN_U; u++) {
                if (j0 + u < end) {
                    const float4 va = *(device const float4*)(vb + (j0 + u) * 256);
                    const float4 vb4 = *(device const float4*)(vb + (j0 + u) * 256 + 4);
                    #pragma clang loop unroll(full)
                    for (int g = 0; g < WIN_G; g++) {
                        aa[g] += sc[g][u] * va;
                        ab[g] += sc[g][u] * vb4;
                    }
                }
            }
        }
        #pragma clang loop unroll(full)
        for (int g = 0; g < WIN_G; g++) {
            device float* o = O + row * p.ldo + g * 256 + 8 * lane;
            *(device float4*)o = aa[g] / sum[g];
            *(device float4*)(o + 4) = ab[g] / sum[g];
        }
    }

    """#
}
