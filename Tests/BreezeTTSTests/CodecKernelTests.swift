// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Metal
import XCTest
@testable import BreezeTTS

/// Tensors by name from memory, in the checkpoint's layouts.
struct MemoryStore: TensorStore {
    var tensors: [String: (shape: [Int], values: [Float])] = [:]

    func entry(_ name: String) throws -> TensorEntry {
        guard let t = tensors[name] else { throw BreezeTTSError.missingTensor(name) }
        return TensorEntry(dtype: "F32", shape: t.shape, byteRange: 0..<(t.values.count * 4))
    }

    func withBytes<R>(_ name: String, _ body: (UnsafeRawBufferPointer, TensorEntry) throws -> R) throws -> R {
        let e = try entry(name)
        return try tensors[name]!.values.withUnsafeBytes { try body($0, e) }
    }
}

/// CPU references in the reference implementation's own terms (torch layouts, channel-major
/// maths written out), against which the kernels and the weight repacking are checked together.
enum CodecReference {
    /// Causal Conv1d over time-major x [T][cin]: left pad `(k - 1) * dilation + 1 - stride`
    /// zeros, right pad to a whole number of strides. w is torch's [out][cin][k].
    static func conv(_ x: [[Float]], w: [Float], bias: [Float]?, out: Int, cin: Int, kernel: Int, dilation: Int = 1, stride: Int = 1) -> [[Float]] {
        let pad = (kernel - 1) * dilation + 1 - stride
        let frames = (x.count + stride - 1) / stride
        return (0..<frames).map { t in
            (0..<out).map { o in
                var acc = Double(bias?[o] ?? 0)
                for j in 0..<kernel {
                    let r = t * stride + j * dilation - pad
                    guard r >= 0, r < x.count else { continue }
                    for i in 0..<cin { acc += Double(x[r][i]) * Double(w[(o * cin + i) * kernel + j]) }
                }
                return Float(acc)
            }
        }
    }

    /// ConvTranspose1d with no padding, trimmed on the right to T * stride rows. w is [cin][out][k].
    static func transposed(_ x: [[Float]], w: [Float], bias: [Float], cin: Int, out: Int, kernel: Int, stride: Int) -> [[Float]] {
        var y = [[Double]](repeating: bias.map(Double.init), count: (x.count - 1) * stride + kernel)
        for t in x.indices {
            for j in 0..<kernel {
                for o in 0..<out {
                    var acc = 0.0
                    for i in 0..<cin { acc += Double(x[t][i]) * Double(w[(i * out + o) * kernel + j]) }
                    y[t * stride + j][o] += acc
                }
            }
        }
        return y.prefix(x.count * stride).map { $0.map { Float($0) } }
    }

    static func snake(_ v: Float, alpha: Float, beta: Float) -> Float {
        let s = sin(Double(exp(alpha)) * Double(v))
        return Float(Double(v) + s * s / (Double(exp(beta)) + 1e-9))
    }

    static func gelu(_ v: Float) -> Float { Float(Double(v) * 0.5 * (1 + erf(Double(v) / 2.0.squareRoot()))) }
    static func elu(_ v: Float) -> Float { v > 0 ? v : Float(expm1(Double(v))) }

    static func layerNorm(_ x: [Float], w: [Float], b: [Float], eps: Double) -> [Float] {
        let mean = x.reduce(0.0) { $0 + Double($1) } / Double(x.count)
        let variance = x.reduce(0.0) { $0 + (Double($1) - mean) * (Double($1) - mean) } / Double(x.count)
        return x.indices.map { Float((Double(x[$0]) - mean) / (variance + eps).squareRoot() * Double(w[$0]) + Double(b[$0])) }
    }

    static func rmsNorm(_ x: [Float], w: [Float], eps: Double) -> [Float] {
        let ms = x.reduce(0.0) { $0 + Double($1) * Double($1) } / Double(x.count)
        return x.indices.map { Float(Double(x[$0]) / (ms + eps).squareRoot() * Double(w[$0])) }
    }
}

class CodecTestCase: GPUTestCase {
    static let codecKernels = try! CodecKernels(ctx: ctx)
    var ck: CodecKernels { Self.codecKernels }

    func loader(_ tensors: [String: (shape: [Int], values: [Float])] = [:]) -> CodecLoader {
        CodecLoader(store: MemoryStore(tensors: tensors), ctx: ctx)
    }

    /// Time-major rows in a buffer whose slack rows hold NaN.
    func rowsBuffer(_ rows: [[Float]], capacityRows: Int? = nil) throws -> MTLBuffer {
        let width = rows[0].count
        var flat = rows.flatMap { $0 }
        flat += [Float](repeating: .nan, count: ((capacityRows ?? rows.count) - rows.count + CodecRing.slackRows) * width)
        return try f32Buffer(flat)
    }

    func nanRows(_ rows: Int, _ width: Int) throws -> MTLBuffer { try nanBuffer(floats: (rows + CodecRing.slackRows) * width) }

    func random(_ rng: inout RNG, rows: Int, width: Int, scale: Float = 1) -> [[Float]] {
        (0..<rows).map { _ in rng.floats(width, scale: scale) }
    }

    func assertRows(_ got: [Float], _ want: [[Float]], atol: Float, rtol: Float = 1e-5, _ what: String, file: StaticString = #filePath, line: UInt = #line) {
        assertClose(got, want.flatMap { $0 }, atol: atol, rtol: rtol, what, file: file, line: line)
    }

    func withTile<T>(_ tile: CodecKernels.Tile?, _ body: () throws -> T) rethrows -> T {
        let saved = CodecKernels.tileOverride
        CodecKernels.tileOverride = tile
        defer { CodecKernels.tileOverride = saved }
        return try body()
    }
}

final class CodecProductTests: CodecTestCase {
    /// A conv fed through a ring in uneven steps equals the offline causal conv, for every tile
    /// and the dilations the vocoder uses. The ring is written by a 1x1 conv with a SnakeBeta
    /// epilogue, so the mirrored ring writes, the wrap and the activation are in the path.
    func testStreamedDilatedConvMatchesOfflineConv() throws {
        let cin = 16, mid = 24, out = 40, total = 61
        for (dilation, steps) in [(1, [7, 13, 30, 11]), (3, [1, 1, 29, 30]), (9, [30, 30, 1])] {
            var rng = RNG(state: UInt64(dilation))
            let x = random(&rng, rows: total, width: cin)
            let w1 = rng.floats(mid * cin), b1 = rng.floats(mid), w2 = rng.floats(out * mid * 7, scale: 0.3), b2 = rng.floats(out)
            let alpha = rng.floats(mid, scale: 0.5), beta = rng.floats(mid, scale: 0.5)
            let l = loader(["a.weight": ([mid, cin, 1], w1), "a.bias": ([mid], b1), "b.weight": ([out, mid, 7], w2), "b.bias": ([out], b2),
                            "s.alpha": ([mid], alpha), "s.beta": ([mid], beta)])
            let convA = try l.conv("a", out: mid, in: cin, kernel: 1), convB = try l.conv("b", out: out, in: mid, kernel: 7)
            let act = try l.snake("s", channels: mid)
            let y1 = CodecReference.conv(x, w: w1, bias: b1, out: mid, cin: cin, kernel: 1).map { row in row.indices.map { CodecReference.snake(row[$0], alpha: alpha[$0], beta: beta[$0]) } }
            let want = CodecReference.conv(y1, w: w2, bias: b2, out: out, cin: mid, kernel: 7, dilation: dilation)
            for tile in CodecKernels.tiles {
                try withTile(tile) {
                    let ring = try CodecRing(ctx: ctx, width: mid, pad: 6 * dilation, maxNewRows: 30, label: "ring")
                    memset(ring.buffer.contents(), 0xFF, ring.buffer.length)
                    try run { ck.zero($0, ring.buffer, bytes: ring.storedBytes) }
                    var head = 0
                    for step in steps {
                        let input = try rowsBuffer(Array(x[head..<(head + step)]))
                        let y = try nanRows(step, out)
                        try run { e in
                            ck.product(e, CodecWindow(input, ld: cin), rows: step, convA, act: ring.output(head: head, act))
                            ck.product(e, ring.window(head: head, tapStep: dilation), rows: step, convB, main: ActView(y, ld: out))
                        }
                        assertRows(y.floats(count: step * out), Array(want[head..<(head + step)]), atol: 3e-5, "dilation \(dilation) tile \(tile) rows \(head)..<\(head + step)")
                        XCTAssertTrue(y.floats(count: out, offset: step * out * 4).allSatisfy(\.isNaN), "rows past M are untouched")
                        head += step
                    }
                }
            }
        }
    }

    /// The transposed conv as a two-tap product over [x[t - 1], x[t]] equals torch's
    /// ConvTranspose1d trimmed on the right, streamed in steps, for the vocoder's four rates
    /// and the kernel-equals-stride upsample.
    func testStreamedTransposedConvMatchesTrimmedConvTranspose() throws {
        let cin = 16, out = 8, total = 23
        for (kernel, stride) in [(16, 8), (10, 5), (8, 4), (6, 3), (2, 2)] {
            var rng = RNG(state: UInt64(kernel))
            let x = random(&rng, rows: total, width: cin)
            let w = rng.floats(cin * out * kernel), b = rng.floats(out)
            let conv = try loader(["t.weight": ([cin, out, kernel], w), "t.bias": ([out], b)]).transposed("t", in: cin, out: out, kernel: kernel, stride: stride)
            let want = CodecReference.transposed(x, w: w, bias: b, cin: cin, out: out, kernel: kernel, stride: stride)
            let identity = try loader().matrix((0..<cin).flatMap { r in (0..<cin).map { $0 == r ? Float(1) : 0 } }, n: cin, cin: cin, taps: 1, bias: nil, label: "eye")
            let ring = try CodecRing(ctx: ctx, width: cin, pad: conv.taps - 1, maxNewRows: 9, label: "ring")
            try run { ck.zero($0, ring.buffer, bytes: ring.storedBytes) }
            var head = 0
            for step in [9, 1, 5, 8] {
                let input = try rowsBuffer(Array(x[head..<(head + step)]))
                let y = try nanRows(step * stride, out), z = try nanRows(step * stride, out)
                try run { e in
                    ck.product(e, CodecWindow(input, ld: cin), rows: step, identity, act: ring.output(head: head, .identity))
                    ck.product(e, ring.window(head: head), rows: step, conv, main: ActView(y, ld: stride * out),
                               act: CodecActOutput(buffer: z, ld: out, channels: out, activation: .identity))
                }
                let rows = Array(want[(head * stride)..<((head + step) * stride)])
                assertRows(y.floats(count: step * stride * out), rows, atol: 2e-5, "k \(kernel) s \(stride) main, rows from \(head)")
                assertRows(z.floats(count: step * stride * out), rows, atol: 2e-5, "k \(kernel) s \(stride) act, rows from \(head)")
                head += step
            }
        }
    }

    /// Strided causal convs (the encoder's downsampling), the ELU epilogue, the residual
    /// accumulate, and N that is not a multiple of any tile.
    func testStridedConvEluAndAccumulate() throws {
        let cin = 8, out = 20, total = 41
        for (kernel, stride) in [(8, 4), (10, 5), (12, 6), (16, 8), (3, 1), (4, 2)] {
            var rng = RNG(state: UInt64(kernel * 31 + stride))
            let x = random(&rng, rows: total, width: cin)
            let w = rng.floats(out * cin * kernel, scale: 0.5), b = rng.floats(out)
            let conv = try loader(["c.weight": ([out, cin, kernel], w), "c.bias": ([out], b)]).conv("c", out: out, in: cin, kernel: kernel)
            let want = CodecReference.conv(x, w: w, bias: b, out: out, cin: cin, kernel: kernel, stride: stride)
            let frames = want.count, pad = kernel - stride
            let padded = [[Float]](repeating: [Float](repeating: 0, count: cin), count: pad) + x
                + [[Float]](repeating: [Float](repeating: 0, count: cin), count: frames * stride - total)
            let residual = random(&rng, rows: frames, width: out)
            for tile in CodecKernels.tiles {
                try withTile(tile) {
                    let input = try rowsBuffer(padded, capacityRows: padded.count + CodecRing.slackRows * stride)
                    let y = try rowsBuffer(residual), z = try nanRows(frames, out)
                    try run { e in
                        ck.product(e, CodecWindow(input, ld: cin, rowStep: stride), rows: frames, conv, main: ActView(y, ld: out), accumulate: true,
                                   act: CodecActOutput(buffer: z, ld: out, channels: out, activation: .elu))
                    }
                    let sum = zip(want, residual).map { a, b in zip(a, b).map { $0 + $1 } }
                    assertRows(y.floats(count: frames * out), sum, atol: 2e-5, "k \(kernel) s \(stride) tile \(tile) main")
                    assertRows(z.floats(count: frames * out), sum.map { $0.map(CodecReference.elu) }, atol: 2e-5, "k \(kernel) s \(stride) tile \(tile) elu")
                }
            }
        }
    }

    /// Shapes the tiles do not fit take the one-thread-per-output kernel: one input channel
    /// (the encoder's first conv) and one output column (the vocoder's last, clamped).
    func testDirectKernelShapes() throws {
        var rng = RNG(state: 77)
        let total = 50
        let mono = random(&rng, rows: total, width: 1)
        let w0 = rng.floats(64 * 7), b0 = rng.floats(64)
        let first = try loader(["c.weight": ([64, 1, 7], w0), "c.bias": ([64], b0)]).conv("c", out: 64, in: 1, kernel: 7)
        let padded = [[Float]](repeating: [0], count: 6) + mono
        let y = try nanRows(total, 64)
        try run { ck.product($0, CodecWindow(try! self.rowsBuffer(padded), ld: 1), rows: total, first, main: ActView(y, ld: 64)) }
        assertRows(y.floats(count: total * 64), CodecReference.conv(mono, w: w0, bias: b0, out: 64, cin: 1, kernel: 7), atol: 1e-5, "cin 1")

        let wide = random(&rng, rows: total, width: 96)
        let w1 = rng.floats(96 * 7, scale: 0.2), b1: [Float] = [0.1]
        let last = try loader(["c.weight": ([1, 96, 7], w1), "c.bias": ([1], b1)]).conv("c", out: 1, in: 96, kernel: 7)
        let paddedWide = [[Float]](repeating: [Float](repeating: 0, count: 96), count: 6) + wide
        let pcm = try nanBuffer(floats: total + 8)
        try run { ck.product($0, CodecWindow(try! self.rowsBuffer(paddedWide), ld: 96), rows: total, last, act: CodecActOutput(buffer: pcm, offset: 16, ld: 1, channels: 1, activation: .clamp)) }
        let want = CodecReference.conv(wide, w: w1, bias: b1, out: 1, cin: 96, kernel: 7).map { [min(max($0[0], -1), 1)] }
        XCTAssertTrue(want.contains { abs($0[0]) == 1 } && want.contains { abs($0[0]) < 1 }, "the case must clamp some samples and not others")
        assertRows(pcm.floats(count: total, offset: 16), want, atol: 1e-5, "n 1 clamp")
        XCTAssertTrue(pcm.floats(count: 4).allSatisfy(\.isNaN) && pcm.floats(count: 4, offset: (total + 4) * 4).allSatisfy(\.isNaN))
    }

    /// The fitted erf against libm over the range GELU sees, through the GELU epilogue.
    func testGeluMatchesErf() throws {
        let n = 8, rows = 4001
        let xs = (0..<rows).map { r in (0..<n).map { c in Float(r * n + c) / Float(rows * n) * 14 - 7 } }
        let identity = try loader().matrix((0..<n).flatMap { r in (0..<n).map { $0 == r ? Float(1) : 0 } }, n: n, cin: n, taps: 1, bias: nil, label: "eye")
        let z = try nanRows(rows, n)
        try run { ck.product($0, CodecWindow(try! self.rowsBuffer(xs), ld: n), rows: rows, identity, act: CodecActOutput(buffer: z, ld: n, channels: n, activation: .gelu)) }
        assertRows(z.floats(count: rows * n), xs.map { $0.map(CodecReference.gelu) }, atol: 4e-7, rtol: 1e-6, "gelu")
    }
}

final class CodecElementwiseTests: CodecTestCase {
    func testNorms() throws {
        var rng = RNG(state: 5)
        let rows = 5, width = 512
        let x = random(&rng, rows: rows, width: width, scale: 3).map { $0.map { $0 + 0.7 } }
        let w = rng.floats(width), b = rng.floats(width)
        let l = loader()
        let wt = try l.tensor(w, label: "w"), bt = try l.tensor(b, label: "b")
        let input = try rowsBuffer(x), y = try nanRows(rows, width)
        try run { ck.norm($0, ActView(input, ld: width), rows: rows, width: width, weight: wt, ActView(y, ld: width), eps: 1e-5) }
        assertRows(y.floats(count: rows * width), x.map { CodecReference.rmsNorm($0, w: w, eps: 1e-5) }, atol: 2e-6, "rms")
        try run { ck.norm($0, ActView(input, ld: width), rows: rows, width: width, weight: wt, bias: bt, ActView(y, ld: width), eps: 1e-5) }
        assertRows(y.floats(count: rows * width), x.map { CodecReference.layerNorm($0, w: w, b: b, eps: 1e-5) }, atol: 2e-6, "layer")
        XCTAssertTrue(y.floats(count: 4, offset: rows * width * 4).allSatisfy(\.isNaN))
    }

    /// Depthwise conv (groups = channels) then LayerNorm, streamed through a ring.
    func testDepthwiseConvLayerNorm() throws {
        var rng = RNG(state: 6)
        let c = 96, total = 19
        let x = random(&rng, rows: total, width: c)
        let w = rng.floats(c * 7), b = rng.floats(c), lw = rng.floats(c), lb = rng.floats(c)
        var want: [[Float]] = []
        for t in 0..<total {
            let row = (0..<c).map { ch -> Float in
                var acc = Double(b[ch])
                for j in 0..<7 where t + j - 6 >= 0 { acc += Double(x[t + j - 6][ch]) * Double(w[ch * 7 + j]) }
                return Float(acc)
            }
            want.append(CodecReference.layerNorm(row, w: lw, b: lb, eps: 1e-6))
        }
        let l = loader()
        let tensors = try [w, b, lw, lb].map { try l.tensor($0, label: "t") }
        let identity = try l.matrix((0..<c).flatMap { r in (0..<c).map { $0 == r ? Float(1) : 0 } }, n: c, cin: c, taps: 1, bias: nil, label: "eye")
        let ring = try CodecRing(ctx: ctx, width: c, pad: 6, maxNewRows: 8, label: "ring")
        try run { ck.zero($0, ring.buffer, bytes: ring.storedBytes) }
        var head = 0
        for step in [2, 4, 8, 5] {
            let input = try rowsBuffer(Array(x[head..<(head + step)])), y = try nanRows(step, c)
            try run { e in
                ck.product(e, CodecWindow(input, ld: c), rows: step, identity, act: ring.output(head: head, .identity))
                ck.dwconvLN(e, ring.window(head: head), rows: step, channels: c, taps: 7, w: tensors[0], bias: tensors[1], lnWeight: tensors[2], lnBias: tensors[3], ActView(y, ld: c), eps: 1e-6)
            }
            assertRows(y.floats(count: step * c), Array(want[head..<(head + step)]), atol: 1e-5, "rows from \(head)")
            head += step
        }
    }

    func testSwiglu() throws {
        var rng = RNG(state: 7)
        let rows = 3, h = 1024
        let g = random(&rng, rows: rows, width: 2 * h, scale: 4)
        let y = try nanRows(rows, h)
        try run { ck.swiglu($0, ActView(try! self.rowsBuffer(g), ld: 2 * h), rows: rows, hidden: h, ActView(y, ld: h)) }
        let want = g.map { row in (0..<h).map { Float(Double(row[$0]) / (1 + exp(-Double(row[$0]))) * Double(row[h + $0])) } }
        assertRows(y.floats(count: rows * h), want, atol: 1e-6, "swiglu")
    }
}

final class CodecQuantizerTests: CodecTestCase {
    func testSplitLookupSumsCodebooksInOrder() throws {
        var rng = RNG(state: 8)
        let books = CodecShape.codebooks, size = CodecShape.codebookSize, dim = CodecShape.codebookDim, rows = 3
        let table = rng.floats(books * size * dim)
        let codes = (0..<(rows * books)).map { _ in UInt32(rng.bits() % UInt64(size)) }
        let y = try nanRows(rows, 2 * dim)
        let tensor = try loader().tensor(table, label: "books")
        let codeBuffer = try u32Buffer([7, 7] + codes)
        try run { ck.rvqSum($0, codebooks: tensor, codes: codeBuffer, codesOffset: 8, rows: rows, ActView(y, ld: 2 * dim)) }
        var want: [Float] = []
        for r in 0..<rows {
            let c = (0..<books).map { Int(codes[r * books + $0]) }
            want += (0..<dim).map { table[c[0] * dim + $0] }
            want += (0..<dim).map { d in
                var acc: Float = 0
                for q in 1..<books { acc += table[(q * size + c[q]) * dim + d] }
                return acc
            }
        }
        XCTAssertEqual(y.floats(count: rows * 2 * dim), want, "a lookup and fp32 adds in codebook order are exact")
    }

    /// Nearest-entry search with the residual update, two quantizer steps deep.
    func testNearestEntrySearchAndResidual() throws {
        var rng = RNG(state: 9)
        let size = CodecShape.codebookSize, dim = CodecShape.codebookDim, rows = 5
        let books = (0..<2).map { _ in rng.floats(size * dim) }
        var x = random(&rng, rows: rows, width: dim)
        // Row 0 sits exactly on an entry; row 1 has two identical nearest entries (first wins).
        x[0] = Array(books[0][(1234 * dim)..<(1235 * dim)])
        var tied = books[0]
        for d in 0..<dim { tied[900 * dim + d] = x[1][d] + 0.001; tied[40 * dim + d] = x[1][d] + 0.001 }
        let tables = [tied, books[1]]
        let l = loader()
        let buffer = try rowsBuffer(x.map { [Float](repeating: .nan, count: 4) + $0 })
        let codes = try u32Buffer([UInt32](repeating: 9999, count: rows * CodecShape.codebooks))
        try run { e in
            for (slot, table) in tables.enumerated() {
                let norms = (0..<size).map { c in (0..<dim).reduce(Float(0)) { $0 + table[c * dim + $1] * table[c * dim + $1] } }
                ck.vqNearest(e, ActView(buffer, offset: 16, ld: dim + 4), rows: rows, codebook: try! l.tensor(table, label: "e"), norms: try! l.tensor(norms, label: "n"), codes: codes, slot: slot + 3)
            }
        }
        let got = codes.contents().bindMemory(to: UInt32.self, capacity: rows * 16)
        var residual = x
        for (slot, table) in tables.enumerated() {
            for r in 0..<rows {
                var best = (index: -1, dist: Double.infinity)
                for c in 0..<size {
                    var d2 = 0.0
                    for d in 0..<dim { let v = Double(residual[r][d]) - Double(table[c * dim + d]); d2 += v * v }
                    if d2 < best.dist { best = (c, d2) }
                }
                XCTAssertEqual(Int(got[r * 16 + slot + 3]), best.index, "row \(r) quantizer \(slot)")
                for d in 0..<dim { residual[r][d] -= table[best.index * dim + d] }
            }
        }
        XCTAssertEqual(got[3], 1234)
        XCTAssertEqual(got[16 + 3], 40)
        XCTAssertEqual(got[0], 9999, "other code slots are untouched")
        let after = buffer.floats(count: rows * (dim + 4))
        assertRows((0..<rows).flatMap { r in Array(after[(r * (dim + 4) + 4)..<((r + 1) * (dim + 4))]) }, residual, atol: 1e-6, "residual")
    }
}

final class CodecAttentionTests: CodecTestCase {
    /// RoPE, the ring caches and windowed causal attention, streamed in steps past several ring
    /// wraps, against attention computed from the whole sequence with absolute positions.
    func testStreamedWindowAttentionMatchesFullSequence() throws {
        var rng = RNG(state: 10)
        let heads = 16, width = heads * 64, total = 200, window = 72, maxStep = 5
        let ring = window + maxStep - 1
        let q = random(&rng, rows: total, width: width), k = random(&rng, rows: total, width: width), v = random(&rng, rows: total, width: width)
        let table = try RopeTable(ctx: ctx, positions: total)
        let inv = RopeFrequencies.plain(theta: 10000, dim: 64).values
        let keys = try nanBuffer(floats: heads * ring * 64), values = try nanBuffer(floats: heads * ring * 64)
        var head = 0, step = 1
        while head < total {
            let rows = min(step, total - head)
            let qkv = try rowsBuffer((head..<(head + rows)).map { q[$0] + k[$0] + v[$0] })
            let out = try nanRows(rows, width)
            try run { e in
                ck.ropeKV(e, ActView(qkv, ld: 3 * width), rows: rows, heads: heads, pos0: head, ring: ring, k: keys, v: values, table: table)
                ck.attention(e, ActView(qkv, ld: 3 * width), rows: rows, heads: heads, pos0: head, window: window, ring: ring, k: keys, v: values, ActView(out, ld: width))
            }
            var want: [[Float]] = []
            for p in head..<(head + rows) {
                var row: [Float] = []
                for h in 0..<heads {
                    let slice = (h * 64)..<((h + 1) * 64)
                    let qr = Self.rotateHalf(Array(q[p][slice]), pos: p, invFreq: inv)
                    let first = max(0, p - window + 1)
                    let scores = (first...p).map { Self.dot(qr[...], Self.rotateHalf(Array(k[$0][slice]), pos: $0, invFreq: inv)[...]) * 0.125 }
                    row += Self.attend(scores, (first...p).map { Array(v[$0][slice]) })
                }
                want.append(row)
            }
            assertRows(out.floats(count: rows * width), want, atol: 2e-5, rtol: 1e-4, "positions \(head)..<\(head + rows)")
            head += rows
            step = step % maxStep + 1
        }
    }

    /// The long variant: every earlier position visible (the encoder's full causal attention).
    func testFullCausalAttention() throws {
        var rng = RNG(state: 11)
        let heads = 8, width = heads * 64, total = 150
        let q = random(&rng, rows: total, width: width), k = random(&rng, rows: total, width: width), v = random(&rng, rows: total, width: width)
        let table = try RopeTable(ctx: ctx, positions: total)
        let inv = RopeFrequencies.plain(theta: 10000, dim: 64).values
        let keys = try nanBuffer(floats: heads * total * 64), values = try nanBuffer(floats: heads * total * 64)
        let qkv = try rowsBuffer((0..<total).map { q[$0] + k[$0] + v[$0] })
        let out = try nanRows(total, width)
        try run { e in
            ck.ropeKV(e, ActView(qkv, ld: 3 * width), rows: total, heads: heads, pos0: 0, ring: total, k: keys, v: values, table: table)
            ck.attention(e, ActView(qkv, ld: 3 * width), rows: total, heads: heads, pos0: 0, window: CodecKernels.longWindow, ring: total, k: keys, v: values, ActView(out, ld: width))
        }
        var want: [[Float]] = []
        for p in 0..<total {
            var row: [Float] = []
            for h in 0..<heads {
                let slice = (h * 64)..<((h + 1) * 64)
                let qr = Self.rotateHalf(Array(q[p][slice]), pos: p, invFreq: inv)
                let scores = (0...p).map { Self.dot(qr[...], Self.rotateHalf(Array(k[$0][slice]), pos: $0, invFreq: inv)[...]) * 0.125 }
                row += Self.attend(scores, (0...p).map { Array(v[$0][slice]) })
            }
            want.append(row)
        }
        assertRows(out.floats(count: total * width), want, atol: 2e-5, rtol: 1e-4, "full causal")
    }
}

final class CodecResampleTests: XCTestCase {
    /// A tone keeps its frequency and level through conversions to 24 kHz, and the output has
    /// ceil(n * new / orig) samples.
    func testToneSurvivesRateConversion() {
        for rate in [48000, 16000, 44100] {
            let n = rate / 2 + 7
            let tone = (0..<n).map { Float(0.5 * sin(2 * Double.pi * 440 * Double($0) / Double(rate))) }
            let out = codecResample(tone, from: rate, to: 24000)
            XCTAssertEqual(out.count, Int((Double(n) * 24000 / Double(rate)).rounded(.up)), "rate \(rate)")
            var worst: Float = 0
            for i in 200..<(out.count - 200) { worst = max(worst, abs(out[i] - Float(0.5 * sin(2 * Double.pi * 440 * Double(i) / 24000)))) }
            XCTAssertLessThan(worst, 2e-3, "rate \(rate)")
        }
        let same = (0..<100).map { Float($0) }
        XCTAssertEqual(codecResample(same, from: 24000, to: 24000), same)
    }
}
