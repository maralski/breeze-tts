// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0

// Row layout shared by every kernel below and by the attention kernels: a dispatch covers B
// sequences of M new rows each, sequence-major (row = s * M + m), and row (s, m) sits at
// position pos0[s] + m of sequence s. A layer's KV cache is fp32 [sequence][kvHead][cap][D],
// K and V apart; `seqStride` is the float count of one sequence's block.
extension ShaderSource {
    static let norm = #"""
    constant uint NORM_ONE_PLUS = 1;
    constant uint NORM_ACCUM = 2;
    struct NormParams { uint K; uint ldx; uint ldy; float eps; uint flags; };

    // One simdgroup per row, K a multiple of 4. y = x * rsqrt(mean(x^2) + eps) * w in fp32, the
    // Qwen and Llama convention; NORM_ONE_PLUS scales by (1 + w), Gemma's, whose stored weights
    // are offsets from one. NORM_ACCUM adds the result to y (Gemma's post-norms close a residual).
    // Per-head norms are this kernel with K = head dim and one row per head.
    kernel void rmsnorm(device const float* X [[buffer(0)]],
                        device const ushort* W [[buffer(1)]],
                        device float* Y [[buffer(2)]],
                        constant NormParams& p [[buffer(3)]],
                        uint gid [[thread_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
        const uint row = gid / 32;
        device const float* x = X + row * p.ldx;
        float ss = 0.0f;
        for (uint k = lane * 4; k < p.K; k += 128) {
            const float4 v = *(device const float4*)(x + k);
            ss += dot(v, v);
        }
        ss = simd_sum(ss);
        const float scale = rsqrt(ss / float(p.K) + p.eps);
        const float one = (p.flags & NORM_ONE_PLUS) ? 1.0f : 0.0f;
        device float* y = Y + row * p.ldy;
        for (uint k = lane * 4; k < p.K; k += 128) {
            const float4 v = *(device const float4*)(x + k);
            const uint2 w01 = *(device const uint2*)(W + k);
            const float4 w = one + float4(bf16_to_float(ushort(w01.x & 0xFFFF)), bf16_to_float(ushort(w01.x >> 16)),
                                          bf16_to_float(ushort(w01.y & 0xFFFF)), bf16_to_float(ushort(w01.y >> 16)));
            float4 r = v * scale * w;
            if (p.flags & NORM_ACCUM) r += *(device float4*)(y + k);
            *(device float4*)(y + k) = r;
        }
    }

    kernel void copy_rows(device const float* X [[buffer(0)]],
                          device float* Y [[buffer(1)]],
                          uint gid [[thread_position_in_grid]]) {
        Y[gid] = X[gid];
    }

    struct RopeParams { uint B; uint M; uint hq; uint hkv; uint D; uint ldq; uint cap; uint seqStride; uint normMode; float eps; uint pos0[4]; };
    constant uint QK_NORM_NONE = 0;
    constant uint QK_NORM_WEIGHT = 1;
    constant uint QK_NORM_ONE_PLUS = 2;

    // Finishes a row of [q heads | k heads | v heads] projections in one dispatch: optional
    // per-head RMSNorm of q and k (weights W = q_norm then k_norm, D values each), rotate-half
    // RoPE with angle = position * invFreq[i] for i < D/2, q back in place, k and v into the cache
    // at the row's position. One simdgroup per (row, head slot). A lane owns pairs i = lane,
    // lane + 32, ... and their partners i + D/2, loads them once and keeps them in registers, so
    // the norm's reads cannot meet the rotation's writes. D is a multiple of 64, at most 256.
    kernel void norm_rope_kv(device float* QKV [[buffer(0)]],
                             device float* Kc [[buffer(1)]],
                             device float* Vc [[buffer(2)]],
                             device const float* invFreq [[buffer(3)]],
                             device const ushort* W [[buffer(4)]],
                             constant RopeParams& p [[buffer(5)]],
                             uint2 tgid [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
        const uint slot = tgid.x, row = tgid.y;
        const uint s = row / p.M, pos = p.pos0[s] + row % p.M;
        const uint D = p.D, halfD = D / 2, pairs = halfD / 32;
        device float* x = QKV + row * p.ldq + slot * D;
        if (slot >= p.hq + p.hkv) {
            device float* dst = Vc + s * p.seqStride + ((slot - p.hq - p.hkv) * p.cap + pos) * D;
            for (uint k = lane * 4; k < D; k += 128) *(device float4*)(dst + k) = *(device const float4*)(x + k);
            return;
        }
        const bool isK = slot >= p.hq;
        float lo[4], hi[4];
        float ss = 0.0f;
        for (uint j = 0; j < pairs; j++) {
            const uint i = lane + 32 * j;
            lo[j] = x[i];
            hi[j] = x[i + halfD];
            ss += lo[j] * lo[j] + hi[j] * hi[j];
        }
        if (p.normMode != QK_NORM_NONE) {
            const float scale = rsqrt(simd_sum(ss) / float(D) + p.eps);
            const float one = (p.normMode == QK_NORM_ONE_PLUS) ? 1.0f : 0.0f;
            device const ushort* w = W + (isK ? D : 0);
            for (uint j = 0; j < pairs; j++) {
                const uint i = lane + 32 * j;
                lo[j] = lo[j] * scale * (one + bf16_to_float(w[i]));
                hi[j] = hi[j] * scale * (one + bf16_to_float(w[i + halfD]));
            }
        }
        device float* dst = isK ? Kc + s * p.seqStride + ((slot - p.hq) * p.cap + pos) * D : x;
        for (uint j = 0; j < pairs; j++) {
            const uint i = lane + 32 * j;
            const float angle = float(pos) * invFreq[i];
            const float c = cos(angle), sn = sin(angle);
            dst[i] = lo[j] * c - hi[j] * sn;
            dst[i + halfD] = hi[j] * c + lo[j] * sn;
        }
    }

    """#
}
