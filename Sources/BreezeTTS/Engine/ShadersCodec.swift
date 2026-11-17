// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0

// The audio codec's kernels. Weights and activations are fp32. Activations are time-major:
// one row per time step, `ld` floats apart, channels within the row.
extension ShaderSource {
    static let codec = #"""
    // Coefficients fitted to erf in double precision (Chebyshev nodes, residual under 1e-9), so
    // the fp32 evaluation is within 2e-7 of erff. |x| < 1: x p(x^2). 1 <= |x| < 4.2:
    // 1 - exp(-x^2) q(1 / |x|). Past 4.2 erf rounds to 1.
    inline float erf_f32(float x) {
        const float ax = fabs(x);
        float r;
        if (ax < 1.0f) {
            const float z = ax * ax;
            float p = 1.0649200031440543e-06f;
            p = p * z - 1.3933974069297213e-05f;
            p = p * z + 0.0001196070996393302f;
            p = p * z - 0.0008543004813673027f;
            p = p * z + 0.005223801684604858f;
            p = p * z - 0.02686613824426957f;
            p = p * z + 0.11283791378771743f;
            p = p * z - 0.37612638893684613f;
            p = p * z + 1.1283791670951964f;
            r = ax * p;
        } else if (ax < 4.2f) {
            const float u = 1.0f / ax;
            float q = -0.03534794346911333f;
            q = q * u + 0.24726732523491182f;
            q = q * u - 0.7642586966893693f;
            q = q * u + 1.352664195888857f;
            q = q * u - 1.4539191673121454f;
            q = q * u + 0.8535617598755811f;
            q = q * u - 0.049023510054079025f;
            q = q * u - 0.2911079270496105f;
            q = q * u + 0.0041133130369221445f;
            q = q * u + 0.5636029271389651f;
            q = q * u + 3.1298922114897376e-05f;
            r = 1.0f - exp(-ax * ax) * q;
        } else {
            r = 1.0f;
        }
        return x < 0.0f ? -r : r;
    }

    // A convolution as a matrix product. Output row m, column n is
    //   sum over tap j < taps, channel c < cin of  x[m * rowStep + j * tapStep][c] * W[n][j * cin + c]
    // where x[r] is the time row r * xld floats past the bound X offset. A causal conv binds X at
    // its oldest left-context row; a transposed conv of stride s and kernel 2s is this with two
    // taps and N = s * channels, its weights regrouped by the host; a linear layer has one tap.
    //
    // v = sum (+ bias[n]) (+ Y[m][n] with CG_ACCUM). CG_WRITE_Y stores v at Y[m * yld + n].
    // CG_WRITE_Z stores act(v) as time rows of zc channels: column n is channel n % zc of time
    // row zRow0 + m * (N / zc) + n / zc. With zRows > 0, Z is a ring of zRows time rows stored
    // twice over (row r and row r + zRows hold the same values), so a reader's window of
    // consecutive rows is contiguous wherever it starts.
    struct CGemmParams {
        uint M; uint N; uint cin; uint taps;
        uint xld; uint rowStep; uint tapStep; uint yld;
        uint flags; uint sg; uint zRows; uint zRow0;
        uint zc; uint zld;
    };
    constant uint CG_BIAS = 1;
    constant uint CG_ACCUM = 2;
    constant uint CG_WRITE_Y = 4;
    constant uint CG_WRITE_Z = 8;
    constant uint CG_SNAKE = 16;
    constant uint CG_ELU = 32;
    constant uint CG_GELU = 64;
    constant uint CG_CLAMP = 128;

    // actA and actB are the SnakeBeta terms per channel: exp(alpha) and 1 / (exp(beta) + 1e-9).
    inline void cg_emit(float v, uint m, uint n, constant CGemmParams& p, device const float* bias, device float* Y, device float* Z,
                        device const float* actA, device const float* actB) {
        if (p.flags & CG_BIAS) v += bias[n];
        if (p.flags & CG_ACCUM) v += Y[m * p.yld + n];
        if (p.flags & CG_WRITE_Y) Y[m * p.yld + n] = v;
        if (!(p.flags & CG_WRITE_Z)) return;
        const uint c = n % p.zc;
        float z = v;
        if (p.flags & CG_SNAKE) {
            const float s = sin(actA[c] * v);
            z = v + actB[c] * s * s;
        }
        if (p.flags & CG_ELU) z = v > 0.0f ? v : exp(v) - 1.0f;
        if (p.flags & CG_GELU) z = v * 0.5f * (1.0f + erf_f32(v * 0.7071067811865476f));
        if (p.flags & CG_CLAMP) z = clamp(v, -1.0f, 1.0f);
        uint row = p.zRow0 + m * (p.N / p.zc) + n / p.zc;
        if (p.zRows != 0) {
            row %= p.zRows;
            Z[(row + p.zRows) * p.zld + c] = z;
        }
        Z[row * p.zld + c] = z;
    }

    // Each simdgroup owns an 8 TM x 8 TN block of the output and walks K in steps of eight,
    // loading 8x8 tiles of X and of W straight from the buffers; p.sg simdgroups make a
    // threadgroup. cin is a multiple of 8. Rows of X past M are read (they must exist) and their
    // results dropped; W holds zero rows up to a multiple of 8 TN past N.
    // thread_elements() gives a lane elements (r, c) and (r, c + 1) of a matrix, with
    // r = ((lane >> 1) & 3) | ((lane >> 4) << 2) and c = ((lane & 1) << 1) | (((lane >> 3) & 1) << 2).
    template <int TM, int TN>
    kernel void cgemm_kernel(device const float* X [[buffer(0)]],
                             device const float* W [[buffer(1)]],
                             device const float* bias [[buffer(2)]],
                             device float* Y [[buffer(3)]],
                             device float* Z [[buffer(4)]],
                             device const float* actA [[buffer(5)]],
                             device const float* actB [[buffer(6)]],
                             constant CGemmParams& p [[buffer(7)]],
                             uint2 tgid [[threadgroup_position_in_grid]],
                             uint sgid [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
        const uint m0 = tgid.y * 8 * TM;
        const uint n0 = (tgid.x * p.sg + sgid) * 8 * TN;
        if (n0 >= p.N) return;
        const uint K = p.cin * p.taps;
        const uint xstride = p.rowStep * p.xld;
        simdgroup_float8x8 acc[TM][TN];
        #pragma clang loop unroll(full)
        for (int i = 0; i < TM; i++) {
            #pragma clang loop unroll(full)
            for (int t = 0; t < TN; t++) acc[i][t] = simdgroup_float8x8(0.0f);
        }
        for (uint j = 0; j < p.taps; j++) {
            device const float* xr = X + m0 * xstride + j * p.tapStep * p.xld;
            device const float* wr = W + n0 * K + j * p.cin;
            for (uint c = 0; c < p.cin; c += 8) {
                simdgroup_float8x8 a[TM], b[TN];
                #pragma clang loop unroll(full)
                for (int i = 0; i < TM; i++) simdgroup_load(a[i], xr + i * 8 * xstride + c, xstride);
                #pragma clang loop unroll(full)
                for (int t = 0; t < TN; t++) simdgroup_load(b[t], wr + t * 8 * K + c, K, ulong2(0, 0), true);
                #pragma clang loop unroll(full)
                for (int i = 0; i < TM; i++) {
                    #pragma clang loop unroll(full)
                    for (int t = 0; t < TN; t++) simdgroup_multiply_accumulate(acc[i][t], a[i], b[t], acc[i][t]);
                }
            }
        }
        const uint r = ((lane >> 1) & 3) | ((lane >> 4) << 2);
        const uint cc = ((lane & 1) << 1) | (((lane >> 3) & 1) << 2);
        #pragma clang loop unroll(full)
        for (int i = 0; i < TM; i++) {
            const uint m = m0 + 8 * i + r;
            if (m >= p.M) continue;
            #pragma clang loop unroll(full)
            for (int t = 0; t < TN; t++) {
                const uint n = n0 + 8 * t + cc;
                if (n < p.N) cg_emit(acc[i][t].thread_elements()[0], m, n, p, bias, Y, Z, actA, actB);
                if (n + 1 < p.N) cg_emit(acc[i][t].thread_elements()[1], m, n + 1, p, bias, Y, Z, actA, actB);
            }
        }
    }

    #define CGEMM(TM, TN) template [[host_name("cgemm_" #TM "_" #TN)]] kernel void cgemm_kernel<TM, TN>(device const float*, device const float*, device const float*, device float*, device float*, device const float*, device const float*, constant CGemmParams&, uint2, uint, uint);
    CGEMM(1, 2) CGEMM(1, 4) CGEMM(1, 8)
    CGEMM(2, 2) CGEMM(2, 4)
    CGEMM(4, 2) CGEMM(4, 4)
    #undef CGEMM

    // The same product with one thread per output element, for shapes the tiles do not fit: a
    // single input channel, a single output column. Any cin, any N.
    kernel void cconv_direct(device const float* X [[buffer(0)]],
                             device const float* W [[buffer(1)]],
                             device const float* bias [[buffer(2)]],
                             device float* Y [[buffer(3)]],
                             device float* Z [[buffer(4)]],
                             device const float* actA [[buffer(5)]],
                             device const float* actB [[buffer(6)]],
                             constant CGemmParams& p [[buffer(7)]],
                             uint2 gid [[thread_position_in_grid]]) {
        const uint n = gid.x, m = gid.y;
        if (n >= p.N || m >= p.M) return;
        const uint K = p.cin * p.taps;
        float acc = 0.0f;
        for (uint j = 0; j < p.taps; j++) {
            device const float* x = X + (m * p.rowStep + j * p.tapStep) * p.xld;
            device const float* w = W + n * K + j * p.cin;
            float4 part = 0.0f;
            uint c = 0;
            for (; c + 4 <= p.cin; c += 4) part += float4(x[c], x[c + 1], x[c + 2], x[c + 3]) * float4(w[c], w[c + 1], w[c + 2], w[c + 3]);
            acc += (part.x + part.y) + (part.z + part.w);
            for (; c < p.cin; c++) acc += x[c] * w[c];
        }
        cg_emit(acc, m, n, p, bias, Y, Z, actA, actB);
    }

    struct CNormParams { uint K; uint ldx; uint ldy; float eps; uint mode; };
    constant uint CNORM_RMS = 0;
    constant uint CNORM_LAYER = 1;

    // One simdgroup per row, K a multiple of 4. RMS: y = x * rsqrt(mean(x^2) + eps) * w.
    // Layer: y = (x - mean) / sqrt(var + eps) * w + b, biased variance.
    kernel void codec_norm(device const float* X [[buffer(0)]],
                           device const float* W [[buffer(1)]],
                           device const float* B [[buffer(2)]],
                           device float* Y [[buffer(3)]],
                           constant CNormParams& p [[buffer(4)]],
                           uint gid [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
        const uint row = gid / 32;
        device const float* x = X + row * p.ldx;
        device float* y = Y + row * p.ldy;
        float mean = 0.0f;
        if (p.mode == CNORM_LAYER) {
            float4 s = 0.0f;
            for (uint k = lane * 4; k < p.K; k += 128) s += *(device const float4*)(x + k);
            mean = simd_sum((s.x + s.y) + (s.z + s.w)) / float(p.K);
        }
        float ss = 0.0f;
        for (uint k = lane * 4; k < p.K; k += 128) {
            const float4 v = *(device const float4*)(x + k) - mean;
            ss += dot(v, v);
        }
        ss = simd_sum(ss) / float(p.K) + p.eps;
        const float scale = p.mode == CNORM_LAYER ? 1.0f / sqrt(ss) : rsqrt(ss);
        for (uint k = lane * 4; k < p.K; k += 128) {
            float4 r = (*(device const float4*)(x + k) - mean) * scale * *(device const float4*)(W + k);
            if (p.mode == CNORM_LAYER) r += *(device const float4*)(B + k);
            *(device float4*)(y + k) = r;
        }
    }

    struct CDwParams { uint C; uint taps; uint xld; uint ldy; float eps; };

    // The ConvNeXt block's front: a causal depthwise conv (W [C][taps], bias) followed by
    // LayerNorm over channels. X is bound at the oldest context row; output row m reads time
    // rows m ... m + taps - 1. One simdgroup per row.
    kernel void codec_dwconv_ln(device const float* X [[buffer(0)]],
                                device const float* W [[buffer(1)]],
                                device const float* bias [[buffer(2)]],
                                device const float* lnW [[buffer(3)]],
                                device const float* lnB [[buffer(4)]],
                                device float* Y [[buffer(5)]],
                                constant CDwParams& p [[buffer(6)]],
                                uint gid [[thread_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
        const uint row = gid / 32;
        device const float* x = X + row * p.xld;
        device float* y = Y + row * p.ldy;
        float sum = 0.0f;
        for (uint c = lane; c < p.C; c += 32) {
            float v = 0.0f;
            for (uint j = 0; j < p.taps; j++) v += x[j * p.xld + c] * W[c * p.taps + j];
            v += bias[c];
            y[c] = v;
            sum += v;
        }
        const float mean = simd_sum(sum) / float(p.C);
        float ss = 0.0f;
        for (uint c = lane; c < p.C; c += 32) {
            const float d = y[c] - mean;
            ss += d * d;
        }
        const float scale = 1.0f / sqrt(simd_sum(ss) / float(p.C) + p.eps);
        for (uint c = lane; c < p.C; c += 32) y[c] = (y[c] - mean) * scale * lnW[c] + lnB[c];
    }

    struct CSwigluParams { uint H; uint ldg; uint ldy; };

    // y[m][n] = silu(g[m][n]) * g[m][H + n]: the gate and up projections arrive as one row.
    kernel void codec_swiglu(device const float* G [[buffer(0)]],
                             device float* Y [[buffer(1)]],
                             constant CSwigluParams& p [[buffer(2)]],
                             uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= p.H) return;
        device const float* g = G + gid.y * p.ldg;
        Y[gid.y * p.ldy + gid.x] = silu(g[gid.x]) * g[p.H + gid.x];
    }

    kernel void codec_zero(device float* Y [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
        *(device float4*)(Y + gid * 4) = float4(0.0f);
    }

    struct CRvqSumParams { uint books; uint dim; uint vocab; uint ldy; uint maxCode; };

    // Split RVQ lookup. Row m of `codes` holds `books` codes. y[m][d] is the semantic codebook's
    // vector, y[m][dim + d] the sum of the acoustic ones in ascending codebook order.
    // E is [books][vocab][dim].
    kernel void codec_rvq_sum(device const float* E [[buffer(0)]],
                              device const uint* codes [[buffer(1)]],
                              device float* Y [[buffer(2)]],
                              constant CRvqSumParams& p [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= p.dim) return;
        device const uint* c = codes + gid.y * p.books;
        device float* y = Y + gid.y * p.ldy;
        y[gid.x] = E[min(c[0], p.maxCode) * p.dim + gid.x];
        float acc = 0.0f;
        for (uint q = 1; q < p.books; q++) acc += E[(q * p.vocab + min(c[q], p.maxCode)) * p.dim + gid.x];
        y[p.dim + gid.x] = acc;
    }

    struct CVqParams { uint dim; uint vocab; uint ldx; uint books; uint slot; };

    // One residual quantizer step: for each row, the codebook entry nearest x by the distance
    // torch.cdist forms above 25 rows, sqrt(max(|x|^2 + |e|^2 - 2 x.e, 0)), first index on ties.
    // Writes the index to codes[m][slot] and subtracts the entry from x in place.
    // One 256-thread threadgroup per row; dim is 256.
    kernel void codec_vq_nearest(device float* X [[buffer(0)]],
                                 device const float* E [[buffer(1)]],
                                 device const float* norms [[buffer(2)]],
                                 device uint* codes [[buffer(3)]],
                                 constant CVqParams& p [[buffer(4)]],
                                 uint2 tgid [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
        threadgroup float bestDist[256];
        threadgroup uint bestIndex[256];
        threadgroup uint chosen;
        device float* x = X + tgid.y * p.ldx;
        float4 xn4 = 0.0f;
        for (uint d = 0; d < p.dim; d += 4) {
            const float4 v = *(device const float4*)(x + d);
            xn4 += v * v;
        }
        const float xn = (xn4.x + xn4.y) + (xn4.z + xn4.w);
        float best = INFINITY;
        uint index = 0;
        for (uint i = tid; i < p.vocab; i += 256) {
            device const float* e = E + i * p.dim;
            float4 part = 0.0f;
            for (uint d = 0; d < p.dim; d += 4) part += *(device const float4*)(x + d) * *(device const float4*)(e + d);
            const float dotp = (part.x + part.y) + (part.z + part.w);
            const float dist = sqrt(max(xn + norms[i] - 2.0f * dotp, 0.0f));
            if (dist < best) { best = dist; index = i; }
        }
        bestDist[tid] = best;
        bestIndex[tid] = index;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float b = bestDist[0];
            uint bi = bestIndex[0];
            for (uint t = 1; t < 256; t++) {
                if (bestDist[t] < b || (bestDist[t] == b && bestIndex[t] < bi)) { b = bestDist[t]; bi = bestIndex[t]; }
            }
            chosen = bi;
            codes[tgid.y * p.books + p.slot] = bi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < p.dim) x[tid] -= E[chosen * p.dim + tid];
    }

    struct CRopeParams { uint M; uint heads; uint ldq; uint pos0; uint ring; };

    // Finishes rows of [q | k | v] projections with head dim 64: rotate-half RoPE on q (in place)
    // and k, then k and v into slot (position % ring) of their caches, [head][ring][64] each.
    // Row m is at absolute position pos0 + m. `table` holds, per position, 32 cosines then 32
    // sines. One simdgroup per (row, head slot), one rotation pair per lane.
    kernel void codec_rope_kv(device float* QKV [[buffer(0)]],
                              device float* Kc [[buffer(1)]],
                              device float* Vc [[buffer(2)]],
                              device const float* table [[buffer(3)]],
                              constant CRopeParams& p [[buffer(4)]],
                              uint2 tgid [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
        const uint slot = tgid.x, row = tgid.y;
        const uint pos = p.pos0 + row;
        device float* x = QKV + row * p.ldq + slot * 64;
        const float lo = x[lane], hi = x[lane + 32];
        if (slot >= 2 * p.heads) {
            device float* dst = Vc + ((slot - 2 * p.heads) * p.ring + pos % p.ring) * 64;
            dst[lane] = lo;
            dst[lane + 32] = hi;
            return;
        }
        const float c = table[pos * 64 + lane], s = table[pos * 64 + 32 + lane];
        device float* dst = slot < p.heads ? x : Kc + ((slot - p.heads) * p.ring + pos % p.ring) * 64;
        dst[lane] = lo * c - hi * s;
        dst[lane + 32] = hi * c + lo * s;
    }

    struct CAttnParams { uint M; uint heads; uint ldq; uint ldo; uint pos0; uint window; uint ring; float scale; };

    // Causal attention with head dim 64 over the caches codec_rope_kv fills. Query row m at
    // absolute position pos0 + m sees the `window` positions ending at its own, at most CAP of
    // them. One simdgroup per (head, row): lanes share the keys out for the scores and the
    // softmax, which meet in threadgroup memory, then each lane sums two output dims.
    template <int CAP>
    kernel void codec_attention_kernel(device const float* Q [[buffer(0)]],
                                       device const float* Kc [[buffer(1)]],
                                       device const float* Vc [[buffer(2)]],
                                       device float* O [[buffer(3)]],
                                       constant CAttnParams& p [[buffer(4)]],
                                       uint2 tgid [[threadgroup_position_in_grid]],
                                       uint lane [[thread_index_in_simdgroup]]) {
        threadgroup float score[CAP];
        const uint head = tgid.x, row = tgid.y;
        const uint pos = p.pos0 + row;
        const uint first = pos + 1 > p.window ? pos + 1 - p.window : 0;
        const uint count = pos - first + 1;
        device const float* q = Q + row * p.ldq + head * 64;
        device const float* kc = Kc + head * p.ring * 64;
        device const float* vc = Vc + head * p.ring * 64;
        float top = -INFINITY;
        for (uint i = lane; i < count; i += 32) {
            device const float* k = kc + ((first + i) % p.ring) * 64;
            float4 part = 0.0f;
            #pragma clang loop unroll(full)
            for (int d = 0; d < 64; d += 4) part += *(device const float4*)(q + d) * *(device const float4*)(k + d);
            const float s = ((part.x + part.y) + (part.z + part.w)) * p.scale;
            score[i] = s;
            top = max(top, s);
        }
        top = simd_max(top);
        float sum = 0.0f;
        for (uint i = lane; i < count; i += 32) {
            const float e = exp(score[i] - top);
            score[i] = e;
            sum += e;
        }
        const float inv = 1.0f / simd_sum(sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float2 acc = 0.0f;
        for (uint i = 0; i < count; i++) {
            acc += (score[i] * inv) * *(device const float2*)(vc + ((first + i) % p.ring) * 64 + lane * 2);
        }
        *(device float2*)(O + row * p.ldo + head * 64 + lane * 2) = acc;
    }

    template [[host_name("codec_attention_96")]] kernel void codec_attention_kernel<96>(device const float*, device const float*, device const float*, device float*, constant CAttnParams&, uint2, uint);
    template [[host_name("codec_attention_1024")]] kernel void codec_attention_kernel<1024>(device const float*, device const float*, device const float*, device float*, constant CAttnParams&, uint2, uint);

    """#
}
