// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// Reads the audio tokenizer's fp32 tensors and lays them out for the codec kernels.
struct CodecLoader {
    let store: any TensorStore
    let ctx: MetalContext

    static let fileName = "audio_tokenizer/model.safetensors"

    /// `directory` is the checkpoint directory that holds `audio_tokenizer/`.
    init(directory: URL, ctx: MetalContext) throws {
        store = try SafeTensors(url: directory.appendingPathComponent(Self.fileName))
        self.ctx = ctx
    }

    init(store: any TensorStore, ctx: MetalContext) {
        self.store = store
        self.ctx = ctx
    }

    func floats(_ name: String, shape: [Int]) throws -> [Float] {
        let e = try store.entry(name)
        guard e.dtype == "F32", e.shape == shape else {
            throw BreezeTTSError.badCheckpoint("\(name) is \(e.dtype) \(e.shape), expected F32 \(shape)")
        }
        return try store.floats(name)
    }

    func has(_ name: String) -> Bool { (try? store.entry(name)) != nil }

    func tensor(_ values: [Float], label: String) throws -> GPUTensor {
        GPUTensor(buffer: try ctx.makeBuffer(floats: values, label: label), offset: 0, count: values.count)
    }

    func vector(_ name: String, count: Int) throws -> GPUTensor {
        try tensor(try floats(name, shape: [count]), label: name)
    }

    /// Rows [n][k] padded with zero rows to the tile kernels' row quantum.
    func matrix(_ rows: [Float], n: Int, cin: Int, taps: Int, bias: [Float]?, label: String) throws -> CodecConv {
        let k = cin * taps
        precondition(rows.count == n * k && (bias == nil || bias!.count == n))
        let padded = (n + CodecConv.rowQuantum - 1) / CodecConv.rowQuantum * CodecConv.rowQuantum
        var w = rows
        w.append(contentsOf: [Float](repeating: 0, count: (padded - n) * k))
        return CodecConv(w: try tensor(w, label: label), bias: try bias.map { try tensor($0, label: label + ".bias") }, n: n, cin: cin, taps: taps)
    }

    /// A torch Conv1d, weight [out][in][kernel], as [out][kernel * in].
    static func packConv(_ w: [Float], out: Int, in cin: Int, kernel: Int) -> [Float] {
        var rows = [Float](repeating: 0, count: w.count)
        for o in 0..<out {
            for i in 0..<cin {
                for j in 0..<kernel { rows[(o * kernel + j) * cin + i] = w[(o * cin + i) * kernel + j] }
            }
        }
        return rows
    }

    /// A torch ConvTranspose1d, weight [in][out][kernel], whose output is trimmed to
    /// `length * stride`, as a product over the taps [x[t - 1], x[t]] (or just x[t] when the
    /// kernel equals the stride). Row `r * out + o` produces channel o of output step
    /// `t * stride + r`: kernel tap r of x[t] and tap r + stride of x[t - 1].
    static func packTransposed(_ w: [Float], in cin: Int, out: Int, kernel: Int, stride: Int) -> (rows: [Float], taps: Int) {
        precondition(kernel == stride || kernel == 2 * stride)
        let taps = kernel / stride
        var rows = [Float](repeating: 0, count: stride * out * taps * cin)
        for r in 0..<stride {
            for o in 0..<out {
                let row = (r * out + o) * taps * cin
                for i in 0..<cin {
                    rows[row + (taps - 1) * cin + i] = w[(i * out + o) * kernel + r]
                    if taps == 2 { rows[row + i] = w[(i * out + o) * kernel + r + stride] }
                }
            }
        }
        return (rows, taps)
    }

    func conv(_ prefix: String, out: Int, in cin: Int, kernel: Int) throws -> CodecConv {
        let w = try floats(prefix + ".weight", shape: [out, cin, kernel])
        let bias = has(prefix + ".bias") ? try floats(prefix + ".bias", shape: [out]) : nil
        return try matrix(Self.packConv(w, out: out, in: cin, kernel: kernel), n: out, cin: cin, taps: kernel, bias: bias, label: prefix)
    }

    func transposed(_ prefix: String, in cin: Int, out: Int, kernel: Int, stride: Int) throws -> CodecConv {
        let w = try floats(prefix + ".weight", shape: [cin, out, kernel])
        let b = try floats(prefix + ".bias", shape: [out])
        let packed = Self.packTransposed(w, in: cin, out: out, kernel: kernel, stride: stride)
        let bias = (0..<stride).flatMap { _ in b }
        return try matrix(packed.rows, n: stride * out, cin: cin, taps: packed.taps, bias: bias, label: prefix)
    }

    /// One or several [n_i][k] projections of the same input stacked along n. `rowScale`
    /// multiplies output row n of the weight and the bias (a layer scale or ConvNeXt gamma
    /// folded into the projection it follows). `squeeze` reads [n][k][1] conv weights.
    func linear(_ names: [String], n: [Int], k: Int, bias: String? = nil, rowScale: [Float]? = nil, squeeze: Bool = false) throws -> CodecConv {
        var rows: [Float] = []
        for (name, count) in zip(names, n) { rows += try floats(name, shape: squeeze ? [count, k, 1] : [count, k]) }
        let total = n.reduce(0, +)
        var b = try bias.map { try floats($0, shape: [total]) }
        if let rowScale {
            precondition(rowScale.count == total)
            for r in 0..<total {
                for c in 0..<k { rows[r * k + c] *= rowScale[r] }
                b?[r] *= rowScale[r]
            }
        }
        return try matrix(rows, n: total, cin: k, taps: 1, bias: b, label: names[0])
    }

    /// SnakeBeta's per-channel terms from its log-scale parameters.
    func snake(_ prefix: String, channels: Int) throws -> CodecActivation {
        let alpha = try floats(prefix + ".alpha", shape: [channels]), beta = try floats(prefix + ".beta", shape: [channels])
        return .snake(a: try tensor(alpha.map { exp($0) }, label: prefix + ".a"),
                      invB: try tensor(beta.map { 1 / (exp($0) + 1e-9) }, label: prefix + ".invB"))
    }

    /// Codebook vectors [2048][256]: the stored sums over their usage counts.
    func codebook(sum: String, usage: String) throws -> [Float] {
        let size = CodecShape.codebookSize, dim = CodecShape.codebookDim
        var e = try floats(sum, shape: [size, dim])
        let u = try floats(usage, shape: [size])
        for c in 0..<size {
            let d = max(u[c], 1e-5)
            for i in 0..<dim { e[c * dim + i] /= d }
        }
        return e
    }
}

struct CodecDecoderWeights {
    struct Layer {
        let norm1: GPUTensor
        /// q, k and v stacked: [3072][512].
        let qkv: CodecConv
        /// Scaled by the attention layer scale.
        let o: CodecConv
        let norm2: GPUTensor
        /// gate and up stacked: [2048][512].
        let gateUp: CodecConv
        /// Scaled by the MLP layer scale.
        let down: CodecConv
    }

    struct Upsample {
        /// Kernel 2, stride 2: [2 * 1024][1024].
        let trans: CodecConv
        let dw, dwBias, lnWeight, lnBias: GPUTensor
        let pw1: CodecConv
        /// Scaled by gamma.
        let pw2: CodecConv
    }

    struct Unit {
        let dilation: Int
        let act1: CodecActivation
        let conv1: CodecConv
        let act2: CodecActivation
        let conv2: CodecConv
    }

    struct Block {
        let channels: Int
        let stride: Int
        let inAct: CodecActivation
        let trans: CodecConv
        let units: [Unit]
    }

    static let hidden = 512
    static let heads = 16
    static let latent = 1024
    static let vocoderWidth = 1536
    static let rates = [8, 5, 4, 3]
    static let dilations = [1, 3, 9]
    static let rmsEps: Float = 1e-5
    static let layerNormEps: Float = 1e-6

    /// [16][2048][256]: the semantic codebook, then the 15 acoustic ones.
    let codebooks: GPUTensor
    /// [512][256 | 256]: the semantic and acoustic output projections side by side.
    let quantizerProj: CodecConv
    let preConv: CodecConv
    let inputProj: CodecConv
    let layers: [Layer]
    let finalNorm: GPUTensor
    let outputProj: CodecConv
    let upsample: [Upsample]
    let conv0: CodecConv
    let blocks: [Block]
    let finalAct: CodecActivation
    let finalConv: CodecConv

    init(_ l: CodecLoader) throws {
        let h = Self.hidden, lat = Self.latent, dim = CodecShape.codebookDim
        let q = "decoder.quantizer"
        var books = try l.codebook(sum: "\(q).rvq_first.vq.layers.0._codebook.embedding_sum", usage: "\(q).rvq_first.vq.layers.0._codebook.cluster_usage")
        for i in 0..<(CodecShape.codebooks - 1) {
            books += try l.codebook(sum: "\(q).rvq_rest.vq.layers.\(i)._codebook.embedding_sum", usage: "\(q).rvq_rest.vq.layers.\(i)._codebook.cluster_usage")
        }
        codebooks = try l.tensor(books, label: "decoder.codebooks")
        let first = try l.floats("\(q).rvq_first.output_proj.weight", shape: [h, dim, 1]), rest = try l.floats("\(q).rvq_rest.output_proj.weight", shape: [h, dim, 1])
        var proj = [Float](repeating: 0, count: h * 2 * dim)
        for n in 0..<h {
            for d in 0..<dim {
                proj[n * 2 * dim + d] = first[n * dim + d]
                proj[n * 2 * dim + dim + d] = rest[n * dim + d]
            }
        }
        quantizerProj = try l.matrix(proj, n: h, cin: 2 * dim, taps: 1, bias: nil, label: "decoder.quantizer.proj")
        preConv = try l.conv("decoder.pre_conv.conv", out: lat, in: h, kernel: 3)
        let t = "decoder.pre_transformer"
        inputProj = try l.linear(["\(t).input_proj.weight"], n: [h], k: lat, bias: "\(t).input_proj.bias")
        layers = try (0..<8).map { i in
            let p = "\(t).layers.\(i)", a = "\(p).self_attn"
            return Layer(norm1: try l.vector("\(p).input_layernorm.weight", count: h),
                         qkv: try l.linear(["\(a).q_proj.weight", "\(a).k_proj.weight", "\(a).v_proj.weight"], n: [lat, lat, lat], k: h),
                         o: try l.linear(["\(a).o_proj.weight"], n: [h], k: lat, rowScale: try l.floats("\(p).self_attn_layer_scale.scale", shape: [h])),
                         norm2: try l.vector("\(p).post_attention_layernorm.weight", count: h),
                         gateUp: try l.linear(["\(p).mlp.gate_proj.weight", "\(p).mlp.up_proj.weight"], n: [lat, lat], k: h),
                         down: try l.linear(["\(p).mlp.down_proj.weight"], n: [h], k: lat, rowScale: try l.floats("\(p).mlp_layer_scale.scale", shape: [h])))
        }
        finalNorm = try l.vector("\(t).norm.weight", count: h)
        outputProj = try l.linear(["\(t).output_proj.weight"], n: [lat], k: h, bias: "\(t).output_proj.bias")
        upsample = try (0..<2).map { i in
            let p = "decoder.upsample.\(i)"
            return Upsample(trans: try l.transposed("\(p).0.conv", in: lat, out: lat, kernel: 2, stride: 2),
                            dw: try l.tensor(try l.floats("\(p).1.dwconv.conv.weight", shape: [lat, 1, 7]), label: "\(p).dw"),
                            dwBias: try l.vector("\(p).1.dwconv.conv.bias", count: lat),
                            lnWeight: try l.vector("\(p).1.norm.weight", count: lat),
                            lnBias: try l.vector("\(p).1.norm.bias", count: lat),
                            pw1: try l.linear(["\(p).1.pwconv1.weight"], n: [4 * lat], k: lat, bias: "\(p).1.pwconv1.bias"),
                            pw2: try l.linear(["\(p).1.pwconv2.weight"], n: [lat], k: 4 * lat, bias: "\(p).1.pwconv2.bias", rowScale: try l.floats("\(p).1.gamma", shape: [lat])))
        }
        let d = "decoder.decoder"
        conv0 = try l.conv("\(d).0.conv", out: Self.vocoderWidth, in: lat, kernel: 7)
        var width = Self.vocoderWidth
        var blocks: [Block] = []
        for (i, rate) in Self.rates.enumerated() {
            let p = "\(d).\(i + 1).block", out = width / 2
            let units = try Self.dilations.enumerated().map { u, dilation in
                let r = "\(p).\(u + 2)"
                return Unit(dilation: dilation,
                            act1: try l.snake("\(r).act1", channels: out),
                            conv1: try l.conv("\(r).conv1.conv", out: out, in: out, kernel: 7),
                            act2: try l.snake("\(r).act2", channels: out),
                            conv2: try l.conv("\(r).conv2.conv", out: out, in: out, kernel: 1))
            }
            blocks.append(Block(channels: out, stride: rate, inAct: try l.snake("\(p).0", channels: width),
                                trans: try l.transposed("\(p).1.conv", in: width, out: out, kernel: 2 * rate, stride: rate), units: units))
            width = out
        }
        self.blocks = blocks
        finalAct = try l.snake("\(d).5", channels: width)
        finalConv = try l.conv("\(d).6.conv", out: 1, in: width, kernel: 7)
    }
}
