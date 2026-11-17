// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
// Metal shader source, compiled at load time so the kernels work inside an .app bundle
// without a resource bundle. Weights are bf16, activations and accumulation fp32.
enum ShaderSource {
    static let all = common + gemv + gemvInt8 + gemm + norm + attention + sampling + codec
}

// The projection kernels, adapted from MingTTS.
extension ShaderSource {
    static let common = #"""
    #include <metal_stdlib>
    using namespace metal;

    inline float bf16_to_float(ushort v) { return as_type<float>(uint(v) << 16); }

    inline void unpack8(uint4 v, thread float* out) {
        out[0] = bf16_to_float(ushort(v.x & 0xFFFF)); out[1] = bf16_to_float(ushort(v.x >> 16));
        out[2] = bf16_to_float(ushort(v.y & 0xFFFF)); out[3] = bf16_to_float(ushort(v.y >> 16));
        out[4] = bf16_to_float(ushort(v.z & 0xFFFF)); out[5] = bf16_to_float(ushort(v.z >> 16));
        out[6] = bf16_to_float(ushort(v.w & 0xFFFF)); out[7] = bf16_to_float(ushort(v.w >> 16));
    }

    // Metal's tanh misbehaves past |u| ~ 44 (returns 0 or NaN); tanh(15) is already 1.0f, so the
    // clamp changes nothing representable.
    inline float gelu_tanh(float x) {
        const float k = 0.7978845608028654f;
        const float u = clamp(k * (x + 0.044715f * x * x * x), -15.0f, 15.0f);
        return 0.5f * x * (1.0f + tanh(u));
    }

    inline float silu(float x) { return x / (1.0f + exp(-x)); }

    """#

    // Y[M,N] = X[M,K] * W[N,K]^T with an epilogue, for M <= 2 rows: a decode token, or the two
    // sequences of a guided step sharing one read of the weights. A simdgroup computes NR output
    // columns, p.sg simdgroups make a threadgroup, lanes stride K by 8 bf16 (16 bytes). MR rows of
    // X share every weight load and NR weight rows share every load of X; a column's sum is the
    // same whatever NR and p.sg are. Any N; K a multiple of 8. With a gate flag the epilogue is
    // act(x W^T) * (x W2^T). The matrix-unit kernel below wastes 8/M of its multiply-accumulates
    // at tiny M, which costs more than these M scalar FMAs per weight. Grid y walks M in MR-row
    // chunks; rows past M are never read.
    static let gemv = #"""
    struct GemvParams { uint M; uint N; uint K; uint ldx; uint ldy; uint flags; uint sg; };
    constant uint GEMV_BIAS = 1;
    constant uint GEMV_ACCUM = 2;
    constant uint GEMV_GELU = 4;
    constant uint GEMV_SILU_GATE = 8;
    constant uint GEMV_SILU = 16;
    constant uint GEMV_GELU_GATE = 32;
    constant uint GEMV_GATED = GEMV_SILU_GATE | GEMV_GELU_GATE;

    inline float gate_act(float v, uint flags) { return (flags & GEMV_GELU_GATE) ? gelu_tanh(v) : silu(v); }

    template <int MR, int NR>
    kernel void gemv_kernel(device const float* X [[buffer(0)]],
                            device const ushort* W [[buffer(1)]],
                            device const ushort* W2 [[buffer(2)]],
                            device const ushort* bias [[buffer(3)]],
                            device float* Y [[buffer(4)]],
                            constant GemvParams& p [[buffer(5)]],
                            uint2 tgid [[threadgroup_position_in_grid]],
                            uint sgid [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
        const uint n0 = (tgid.x * p.sg + sgid) * NR;
        if (n0 >= p.N) return;
        const uint row0 = tgid.y * MR;
        const uint rows = min(uint(MR), p.M - row0);
        device const float* x = X + row0 * p.ldx;
        // Columns past N read the last column's weights and are never written.
        device const ushort* w[NR];
        device const ushort* w2[NR];
        #pragma clang loop unroll(full)
        for (int r = 0; r < NR; r++) {
            const uint n = min(n0 + r, p.N - 1);
            w[r] = W + n * p.K;
            w2[r] = W2 + n * p.K;
        }
        const bool gated = (p.flags & GEMV_GATED) != 0;
        float acc[NR][MR], acc2[NR][MR];
        #pragma clang loop unroll(full)
        for (int r = 0; r < NR; r++) {
            #pragma clang loop unroll(full)
            for (int m = 0; m < MR; m++) { acc[r][m] = 0.0f; acc2[r][m] = 0.0f; }
        }
        for (uint k = lane * 8; k < p.K; k += 256) {
            // The rows of X are loaded ahead of the weights, with the row guard on the loads, so
            // the multiply-adds below run unguarded. With the guard around the multiply-adds
            // the two-row shapes were slow: 1024 x 8192 read 286 GB/s, and 353 this way.
            float4 a[MR], b[MR];
            #pragma clang loop unroll(full)
            for (int m = 0; m < MR; m++) {
                a[m] = 0.0f; b[m] = 0.0f;
                if (uint(m) < rows) {
                    a[m] = *(device const float4*)(x + m * p.ldx + k);
                    b[m] = *(device const float4*)(x + m * p.ldx + k + 4);
                }
            }
            #pragma clang loop unroll(full)
            for (int r = 0; r < NR; r++) {
                float wf[8];
                unpack8(*(device const uint4*)(w[r] + k), wf);
                float wg[8];
                if (gated) unpack8(*(device const uint4*)(w2[r] + k), wg);
                #pragma clang loop unroll(full)
                for (int m = 0; m < MR; m++) {
                    acc[r][m] += a[m].x * wf[0] + a[m].y * wf[1] + a[m].z * wf[2] + a[m].w * wf[3]
                               + b[m].x * wf[4] + b[m].y * wf[5] + b[m].z * wf[6] + b[m].w * wf[7];
                    if (gated) {
                        acc2[r][m] += a[m].x * wg[0] + a[m].y * wg[1] + a[m].z * wg[2] + a[m].w * wg[3]
                                    + b[m].x * wg[4] + b[m].y * wg[5] + b[m].z * wg[6] + b[m].w * wg[7];
                    }
                }
            }
        }
        #pragma clang loop unroll(full)
        for (int r = 0; r < NR; r++) {
            #pragma clang loop unroll(full)
            for (int m = 0; m < MR; m++) {
                acc[r][m] = simd_sum(acc[r][m]);
                if (gated) acc2[r][m] = simd_sum(acc2[r][m]);
            }
        }
        if (lane != 0) return;
        #pragma clang loop unroll(full)
        for (int r = 0; r < NR; r++) {
            const uint n = n0 + r;
            if (n >= p.N) continue;
            const float bval = (p.flags & GEMV_BIAS) ? bf16_to_float(bias[n]) : 0.0f;
            #pragma clang loop unroll(full)
            for (int m = 0; m < MR; m++) {
                if (uint(m) < rows) {
                    float v = acc[r][m] + bval;
                    if (gated) v = gate_act(v, p.flags) * acc2[r][m];
                    if (p.flags & GEMV_GELU) v = gelu_tanh(v);
                    if (p.flags & GEMV_SILU) v = silu(v);
                    device float* y = Y + (row0 + m) * p.ldy + n;
                    if (p.flags & GEMV_ACCUM) v += *y;
                    *y = v;
                }
            }
        }
    }

    #define GEMV_INSTANCE(MR, NR) template [[host_name("gemv_" #MR "_" #NR)]] kernel void gemv_kernel<MR, NR>(device const float*, device const ushort*, device const ushort*, device const ushort*, device float*, constant GemvParams&, uint2, uint, uint);
    GEMV_INSTANCE(1, 1) GEMV_INSTANCE(1, 2)
    GEMV_INSTANCE(2, 1) GEMV_INSTANCE(2, 2)
    #undef GEMV_INSTANCE

    // Matrix-unit GEMV for M <= 8 rows per chunk: Y^T[n, m] = sum_k W[n, k] X[m, k] as 8x8 simdgroup
    // matrix products, so a lane issues one 16-byte weight load and four multiply-accumulates per
    // 512-byte weight tile instead of M scalar FMAs per weight. A threadgroup is KS simdgroups over the
    // same 8 NC output columns, each owning K/KS of the reduction; partial tiles meet in threadgroup memory.
    // Inside a 32-wide k step the tiles use a permuted k order, tile u, tile column j <-> k = k0 + 4 j + u,
    // applied identically to A and B: a lane's two weight columns (c, c + 1) over the four tiles are
    // the 8 contiguous bf16 at k0 + 4 c, and its B row j over the four tiles is one float4 of X at k0 + 4 j.
    // thread_elements() lane layout: row = ((lane >> 1) & 3) | ((lane >> 4) << 2),
    // columns (c, c + 1) with c = ((lane & 1) << 1) | (((lane >> 3) & 1) << 2); KernelTests pins it.
    // thread_elements() is typed as the whole matrix; only elements 0 and 1 exist per lane.
    inline void g8_set(thread simdgroup_float8x8& m, float2 v) {
        m.thread_elements()[0] = v.x;
        m.thread_elements()[1] = v.y;
    }

    inline float2 g8_bf16x2(uint v) { return float2(bf16_to_float(ushort(v & 0xFFFF)), bf16_to_float(ushort(v >> 16))); }

    // A's two elements for tile u out of the lane's 8 weights (k0 + 4 c .. + 7): offsets u and u + 4.
    inline float2 g8_a(uint4 wv, int u) {
        const uint lo = (u & 2) ? wv.y : wv.x, hi = (u & 2) ? wv.w : wv.z;
        const uint shift = (u & 1) * 16;
        return float2(bf16_to_float(ushort(lo >> shift)), bf16_to_float(ushort(hi >> shift)));
    }

    struct G8Lane {
        device const float* x0;
        device const float* x1;
        bool has0, has1, gated;
        uint r, c;
    };

    // One 32-k step over NC column tiles: the lane's B values, one float4 of X per row, are loaded once
    // and multiplied into every tile's weights. wrow is the lane's weight row in tile 0; tile t's row is
    // 8 t rows further on. Tile t accumulates into C[t][u % NA] so four chains stay in flight at any NC.
    template <int NC>
    inline void g8_step32(G8Lane L, device const ushort* wrow, device const ushort* w2row, uint K, uint k0,
                          thread simdgroup_float8x8 (&C)[NC][4 / NC], thread simdgroup_float8x8 (&C2)[NC][4 / NC]) {
        constexpr int NA = 4 / NC;
        const uint kx = k0 + 4 * L.r;
        float4 b0 = 0.0f, b1 = 0.0f;
        if (L.has0) b0 = *(device const float4*)(L.x0 + kx);
        if (L.has1) b1 = *(device const float4*)(L.x1 + kx);
        simdgroup_float8x8 B[4];
        #pragma clang loop unroll(full)
        for (int u = 0; u < 4; u++) g8_set(B[u], float2(b0[u], b1[u]));
        #pragma clang loop unroll(full)
        for (int t = 0; t < NC; t++) {
            const uint off = t * 8 * K + k0 + 4 * L.c;
            const uint4 wv = *(device const uint4*)(wrow + off);
            uint4 wv2 = 0;
            if (L.gated) wv2 = *(device const uint4*)(w2row + off);
            #pragma clang loop unroll(full)
            for (int u = 0; u < 4; u++) {
                simdgroup_float8x8 A;
                g8_set(A, g8_a(wv, u));
                simdgroup_multiply_accumulate(C[t][u % NA], A, B[u], C[t][u % NA]);
                if (L.gated) {
                    g8_set(A, g8_a(wv2, u));
                    simdgroup_multiply_accumulate(C2[t][u % NA], A, B[u], C2[t][u % NA]);
                }
            }
        }
    }

    // One tile in natural k order, weight rows guarded against N (the tail of K, and column tiles past N).
    inline void g8_tile8(G8Lane L, device const ushort* W, device const ushort* W2, uint n, uint N, uint K, uint k,
                         thread simdgroup_float8x8& C, thread simdgroup_float8x8& C2) {
        const float2 b = float2(L.has0 ? L.x0[k + L.r] : 0.0f, L.has1 ? L.x1[k + L.r] : 0.0f);
        simdgroup_float8x8 A, B;
        g8_set(B, b);
        g8_set(A, n < N ? g8_bf16x2(*(device const uint*)(W + n * K + k + L.c)) : float2(0.0f));
        simdgroup_multiply_accumulate(C, A, B, C);
        if (L.gated) {
            g8_set(A, n < N ? g8_bf16x2(*(device const uint*)(W2 + n * K + k + L.c)) : float2(0.0f));
            simdgroup_multiply_accumulate(C2, A, B, C2);
        }
    }

    // KS simdgroups split K; each walks NC column tiles (8 NC output columns per threadgroup) so one X
    // load feeds NC weight loads. The host passes NC > 1 only when N % (8 NC) == 0; the guarded tail
    // path below is for NC == 1. Simdgroup t sums the partials of tile t.
    template <int KS, int NC>
    kernel void gemm8_kernel(device const float* X [[buffer(0)]],
                             device const ushort* W [[buffer(1)]],
                             device const ushort* W2 [[buffer(2)]],
                             device const ushort* bias [[buffer(3)]],
                             device float* Y [[buffer(4)]],
                             constant GemvParams& p [[buffer(5)]],
                             uint2 tgid [[threadgroup_position_in_grid]],
                             uint sgid [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
        constexpr int NA = 4 / NC;
        threadgroup float part[KS][NC][2][64];
        const uint n0 = tgid.x * 8 * NC;
        const uint row0 = tgid.y * 8;
        const uint rows = min(8u, p.M - row0);
        G8Lane L;
        L.r = ((lane >> 1) & 3) | ((lane >> 4) << 2);
        L.c = ((lane & 1) << 1) | (((lane >> 3) & 1) << 2);
        L.gated = (p.flags & GEMV_GATED) != 0;
        L.has0 = L.c < rows;
        L.has1 = L.c + 1 < rows;
        device const float* x = X + row0 * p.ldx;
        L.x0 = x + L.c * p.ldx;
        L.x1 = x + (L.c + 1) * p.ldx;
        const uint kLen = p.K / KS, kBegin = sgid * kLen, kEnd = kBegin + kLen;

        simdgroup_float8x8 C[NC][NA], C2[NC][NA];
        #pragma clang loop unroll(full)
        for (int t = 0; t < NC; t++) {
            #pragma clang loop unroll(full)
            for (int a = 0; a < NA; a++) { C[t][a] = simdgroup_float8x8(0.0f); C2[t][a] = simdgroup_float8x8(0.0f); }
        }
        const uint n = n0 + L.r;
        uint k = kBegin;
        if (n0 + 8 * NC <= p.N) {
            device const ushort* wrow = W + n * p.K;
            device const ushort* w2row = W2 + n * p.K;
            for (; k + 64 <= kEnd; k += 64) {
                g8_step32<NC>(L, wrow, w2row, p.K, k, C, C2);
                g8_step32<NC>(L, wrow, w2row, p.K, k + 32, C, C2);
            }
            for (; k + 32 <= kEnd; k += 32) g8_step32<NC>(L, wrow, w2row, p.K, k, C, C2);
        }
        for (; k < kEnd; k += 8) {
            #pragma clang loop unroll(full)
            for (int t = 0; t < NC; t++) g8_tile8(L, W, W2, n + 8 * t, p.N, p.K, k, C[t][0], C2[t][0]);
        }

        #pragma clang loop unroll(full)
        for (int t = 0; t < NC; t++) {
            float2 acc = float2(0.0f), acc2 = float2(0.0f);
            #pragma clang loop unroll(full)
            for (int a = 0; a < NA; a++) {
                acc += float2(C[t][a].thread_elements()[0], C[t][a].thread_elements()[1]);
                acc2 += float2(C2[t][a].thread_elements()[0], C2[t][a].thread_elements()[1]);
            }
            part[sgid][t][0][lane * 2] = acc.x;
            part[sgid][t][0][lane * 2 + 1] = acc.y;
            if (L.gated) {
                part[sgid][t][1][lane * 2] = acc2.x;
                part[sgid][t][1][lane * 2 + 1] = acc2.y;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid >= uint(NC)) return;
        const uint t = sgid, nt = n + 8 * t;
        if (nt >= p.N) return;

        float2 v = float2(0.0f), v2 = float2(0.0f);
        #pragma clang loop unroll(full)
        for (int s = 0; s < KS; s++) {
            v += float2(part[s][t][0][lane * 2], part[s][t][0][lane * 2 + 1]);
            if (L.gated) v2 += float2(part[s][t][1][lane * 2], part[s][t][1][lane * 2 + 1]);
        }
        v += (p.flags & GEMV_BIAS) ? bf16_to_float(bias[nt]) : 0.0f;
        if (L.gated) v = float2(gate_act(v.x, p.flags), gate_act(v.y, p.flags)) * v2;
        if (p.flags & GEMV_GELU) v = float2(gelu_tanh(v.x), gelu_tanh(v.y));
        if (p.flags & GEMV_SILU) v = float2(silu(v.x), silu(v.y));
        if (L.has0) {
            device float* y = Y + (row0 + L.c) * p.ldy + nt;
            *y = (p.flags & GEMV_ACCUM) ? *y + v.x : v.x;
        }
        if (L.has1) {
            device float* y = Y + (row0 + L.c + 1) * p.ldy + nt;
            *y = (p.flags & GEMV_ACCUM) ? *y + v.y : v.y;
        }
    }

    #define GEMM8(KS, NC) template [[host_name("gemm8_" #KS "_" #NC)]] kernel void gemm8_kernel<KS, NC>(device const float*, device const ushort*, device const ushort*, device const ushort*, device float*, constant GemvParams&, uint2, uint, uint);
    GEMM8(4, 1) GEMM8(4, 2) GEMM8(4, 4)
    GEMM8(8, 1) GEMM8(8, 2) GEMM8(8, 4)
    GEMM8(16, 1) GEMM8(16, 2)
    #undef GEMM8

    // Dumps which two matrix elements each lane of a simdgroup_float8x8 holds: out[lane*2 + i] = row*8 + col.
    kernel void simd_matrix_layout(device const float* M [[buffer(0)]],
                                   device float* out [[buffer(1)]],
                                   uint lane [[thread_index_in_simdgroup]]) {
        simdgroup_float8x8 A;
        simdgroup_load(A, M, 8);
        out[lane * 2] = A.thread_elements()[0];
        out[lane * 2 + 1] = A.thread_elements()[1];
    }

    """#

    // The scalar GEMV over `Int8Stripes` weights: W is int8 [N, K], S fp16 [N, 32], and
    // w[n, k] = W[n, k] * S[n, (k / 16) % 32]. Lanes stride K by 16 int8 (16 bytes), so lane l
    // reads exactly the weights that share S[n, l] and scales its sum once, after the loop.
    // Same grid, flags and layout parameters as gemv_kernel, without a bias; K a multiple of 512.
    // A 16-byte weight load meets four float4 of x here, against two in the bf16 kernel.
    static let gemvInt8 = #"""
    inline void unpack16(uint4 v, thread float4* out) {
        out[0] = float4(as_type<char4>(v.x)); out[1] = float4(as_type<char4>(v.y));
        out[2] = float4(as_type<char4>(v.z)); out[3] = float4(as_type<char4>(v.w));
    }

    template <int MR, int NR>
    kernel void gemv_i8_kernel(device const float* X [[buffer(0)]],
                               device const char* W [[buffer(1)]],
                               device const char* W2 [[buffer(2)]],
                               device const half* S [[buffer(3)]],
                               device float* Y [[buffer(4)]],
                               constant GemvParams& p [[buffer(5)]],
                               device const half* S2 [[buffer(6)]],
                               uint2 tgid [[threadgroup_position_in_grid]],
                               uint sgid [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
        const uint n0 = (tgid.x * p.sg + sgid) * NR;
        if (n0 >= p.N) return;
        const uint row0 = tgid.y * MR;
        const uint rows = min(uint(MR), p.M - row0);
        device const float* x = X + row0 * p.ldx;
        const bool gated = (p.flags & GEMV_GATED) != 0;
        // Columns past N read the last column's weights and are never written.
        device const char* w[NR];
        device const char* w2[NR];
        float acc[NR][MR], acc2[NR][MR];
        #pragma clang loop unroll(full)
        for (int r = 0; r < NR; r++) {
            const uint n = min(n0 + r, p.N - 1);
            w[r] = W + n * p.K;
            w2[r] = W2 + n * p.K;
            #pragma clang loop unroll(full)
            for (int m = 0; m < MR; m++) { acc[r][m] = 0.0f; acc2[r][m] = 0.0f; }
        }
        for (uint k = lane * 16; k < p.K; k += 512) {
            float4 a[MR][4];
            #pragma clang loop unroll(full)
            for (int m = 0; m < MR; m++) {
                #pragma clang loop unroll(full)
                for (int j = 0; j < 4; j++) {
                    a[m][j] = 0.0f;
                    if (uint(m) < rows) a[m][j] = *(device const float4*)(x + m * p.ldx + k + 4 * j);
                }
            }
            #pragma clang loop unroll(full)
            for (int r = 0; r < NR; r++) {
                float4 wf[4], wg[4];
                unpack16(*(device const uint4*)(w[r] + k), wf);
                if (gated) unpack16(*(device const uint4*)(w2[r] + k), wg);
                #pragma clang loop unroll(full)
                for (int m = 0; m < MR; m++) {
                    acc[r][m] += dot(a[m][0], wf[0]) + dot(a[m][1], wf[1]) + dot(a[m][2], wf[2]) + dot(a[m][3], wf[3]);
                    if (gated) acc2[r][m] += dot(a[m][0], wg[0]) + dot(a[m][1], wg[1]) + dot(a[m][2], wg[2]) + dot(a[m][3], wg[3]);
                }
            }
        }
        #pragma clang loop unroll(full)
        for (int r = 0; r < NR; r++) {
            const uint n = min(n0 + r, p.N - 1);
            const float s = float(S[n * 32 + lane]);
            const float s2 = gated ? float(S2[n * 32 + lane]) : 0.0f;
            #pragma clang loop unroll(full)
            for (int m = 0; m < MR; m++) {
                acc[r][m] = simd_sum(acc[r][m] * s);
                if (gated) acc2[r][m] = simd_sum(acc2[r][m] * s2);
            }
        }
        if (lane != 0) return;
        #pragma clang loop unroll(full)
        for (int r = 0; r < NR; r++) {
            const uint n = n0 + r;
            if (n >= p.N) continue;
            #pragma clang loop unroll(full)
            for (int m = 0; m < MR; m++) {
                if (uint(m) < rows) {
                    float v = acc[r][m];
                    if (gated) v = gate_act(v, p.flags) * acc2[r][m];
                    if (p.flags & GEMV_GELU) v = gelu_tanh(v);
                    if (p.flags & GEMV_SILU) v = silu(v);
                    device float* y = Y + (row0 + m) * p.ldy + n;
                    if (p.flags & GEMV_ACCUM) v += *y;
                    *y = v;
                }
            }
        }
    }

    #define GEMV_I8_INSTANCE(MR, NR) template [[host_name("gemv_i8_" #MR "_" #NR)]] kernel void gemv_i8_kernel<MR, NR>(device const float*, device const char*, device const char*, device const half*, device float*, constant GemvParams&, device const half*, uint2, uint, uint);
    GEMV_I8_INSTANCE(1, 1) GEMV_I8_INSTANCE(1, 2) GEMV_I8_INSTANCE(1, 4)
    GEMV_I8_INSTANCE(2, 1) GEMV_I8_INSTANCE(2, 2) GEMV_I8_INSTANCE(2, 4)
    #undef GEMV_I8_INSTANCE

    """#

    // Tiled GEMM for prefill: Y[M,N] = X[M,K] W[N,K]^T with the GEMV's flags and buffer layout.
    // A BM x BN output tile per threadgroup, SG simdgroups each owning BN/SG columns, fp32
    // simdgroup matrices accumulating over K in BK slices staged through threadgroup memory.
    // Rows past M are clamped on load and skipped on store; the caller guarantees N % BN == 0
    // and K % BK == 0. With a gate flag the W2 tile is accumulated too and the epilogue is
    // act(x W^T) * (x W2^T), like the GEMV.
    static let gemm = #"""
    template <int BM, int BN, int BK, int SG>
    kernel void gemm_kernel(device const float* X [[buffer(0)]],
                            device const ushort* W [[buffer(1)]],
                            device const ushort* W2 [[buffer(2)]],
                            device const ushort* bias [[buffer(3)]],
                            device float* Y [[buffer(4)]],
                            constant GemvParams& p [[buffer(5)]],
                            uint2 tgid [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint sgid [[simdgroup_index_in_threadgroup]]) {
        constexpr int LD = BK + 4;
        constexpr int THREADS = SG * 32;
        constexpr int SN = BN / SG;
        constexpr int TM = BM / 8, TN = SN / 8;
        constexpr int OUT_LD = BN + 4;
        threadgroup float stage[(BM + 2 * BN) * LD];
        threadgroup float* Xs = stage;
        threadgroup float* Ws = stage + BM * LD;
        threadgroup float* Ws2 = Ws + BN * LD;
        const uint m0 = tgid.y * BM, n0 = tgid.x * BN;
        const bool gated = (p.flags & GEMV_GATED) != 0;

        simdgroup_float8x8 acc[TM][TN];
        simdgroup_float8x8 acc2[TM][TN];
        #pragma clang loop unroll(full)
        for (int i = 0; i < TM; i++) {
            #pragma clang loop unroll(full)
            for (int j = 0; j < TN; j++) { acc[i][j] = simdgroup_float8x8(0.0f); acc2[i][j] = simdgroup_float8x8(0.0f); }
        }

        for (uint k0 = 0; k0 < p.K; k0 += BK) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint u = tid; u < BM * BK / 4; u += THREADS) {
                const uint row = u / (BK / 4), c4 = u % (BK / 4);
                const uint m = min(m0 + row, p.M - 1);
                const float4 v = *(device const float4*)(X + m * p.ldx + k0 + c4 * 4);
                threadgroup float* dst = Xs + row * LD + c4 * 4;
                dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;
            }
            for (uint u = tid; u < BN * BK / 8; u += THREADS) {
                const uint row = u / (BK / 8), c8 = u % (BK / 8);
                const uint off = (n0 + row) * p.K + k0 + c8 * 8;
                float vals[8];
                unpack8(*(device const uint4*)(W + off), vals);
                threadgroup float* dst = Ws + row * LD + c8 * 8;
                for (int i = 0; i < 8; i++) dst[i] = vals[i];
                if (gated) {
                    unpack8(*(device const uint4*)(W2 + off), vals);
                    threadgroup float* dst2 = Ws2 + row * LD + c8 * 8;
                    for (int i = 0; i < 8; i++) dst2[i] = vals[i];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (int kk = 0; kk < BK; kk += 8) {
                simdgroup_float8x8 a[TM];
                simdgroup_float8x8 b[TN];
                #pragma clang loop unroll(full)
                for (int i = 0; i < TM; i++) simdgroup_load(a[i], Xs + i * 8 * LD + kk, LD);
                #pragma clang loop unroll(full)
                for (int j = 0; j < TN; j++) simdgroup_load(b[j], Ws + (sgid * SN + j * 8) * LD + kk, LD, ulong2(0, 0), true);
                #pragma clang loop unroll(full)
                for (int i = 0; i < TM; i++) {
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < TN; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
                if (gated) {
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < TN; j++) simdgroup_load(b[j], Ws2 + (sgid * SN + j * 8) * LD + kk, LD, ulong2(0, 0), true);
                    #pragma clang loop unroll(full)
                    for (int i = 0; i < TM; i++) {
                        #pragma clang loop unroll(full)
                        for (int j = 0; j < TN; j++) simdgroup_multiply_accumulate(acc2[i][j], a[i], b[j], acc2[i][j]);
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* out = stage;
        threadgroup float* out2 = stage + BM * OUT_LD;
        #pragma clang loop unroll(full)
        for (int i = 0; i < TM; i++) {
            #pragma clang loop unroll(full)
            for (int j = 0; j < TN; j++) {
                simdgroup_store(acc[i][j], out + i * 8 * OUT_LD + sgid * SN + j * 8, OUT_LD);
                if (gated) simdgroup_store(acc2[i][j], out2 + i * 8 * OUT_LD + sgid * SN + j * 8, OUT_LD);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint u = tid; u < BM * BN; u += THREADS) {
            const uint row = u / BN, col = u % BN;
            const uint m = m0 + row;
            if (m >= p.M) continue;
            const uint n = n0 + col;
            float v = out[row * OUT_LD + col];
            if (p.flags & GEMV_BIAS) v += bf16_to_float(bias[n]);
            if (gated) v = gate_act(v, p.flags) * out2[row * OUT_LD + col];
            if (p.flags & GEMV_GELU) v = gelu_tanh(v);
            if (p.flags & GEMV_SILU) v = silu(v);
            device float* y = Y + m * p.ldy + n;
            if (p.flags & GEMV_ACCUM) v += *y;
            *y = v;
        }
    }

    template [[host_name("gemm_32x64x32")]] kernel void gemm_kernel<32, 64, 32, 4>(device const float*, device const ushort*, device const ushort*, device const ushort*, device float*, constant GemvParams&, uint2, uint, uint);

    """#
}
