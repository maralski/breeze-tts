// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// How a generation samples. `guidance` nil runs one sequence; a scale runs the conditional and
/// unconditional sequences side by side and combines their logits at every draw.
struct FrameSettings: Equatable {
    var guidance: Float?
    var backbone = SamplerSettings()
    var depth = SamplerSettings()
    var sequences: Int { guidance == nil ? 1 : 2 }
}

/// The per-frame GPU program and the state it carries from frame to frame.
///
/// One audio frame is 16 codes. Everything between two frames stays on the GPU: the CPU writes
/// the frame's 16 uniforms before encoding and reads its 16 codes after the command buffer
/// completes, and a later frame can be encoded and committed before an earlier one finishes,
/// because each frame has its own uniform and code slots.
///
///     backbone step      backboneInput -> 28 layers -> hidden (row 0 of each depthInput pair)
///     codebook 0         lm_head, sample -> codes[0]
///     depth positions 0-1  [hidden, E(codes[0])] -> projector -> 12 layers -> head 0, sample -> codes[1]
///     depth position k   E(codes[k-1]) -> projector -> 12 layers -> head k-1, sample -> codes[k]   (k = 2...15)
///     embedding sum      sum over q of E(codes[q]) -> backboneInput
///
/// With guidance every stage carries two rows, [conditional, unconditional]: two sequences with
/// their own KV caches and positions, sharing each weight read. Both get the same sampled codes.
final class FrameLoop {
    let weights: FrameWeights
    let sequences: Int
    let maxFrames: Int
    let backbone: DecoderForward
    let depth: DecoderForward

    /// fp32 [sequences, 2048]: the next backbone step's input rows.
    let backboneInput: MTLBuffer
    /// fp32 [sequences, 2, 2048]: per sequence the backbone's final hidden state, then the
    /// embedding of codebook 0. The depth decoder's first two positions before the projector.
    let depthInput: MTLBuffer
    /// fp32 [sequences, 2048]: the embedding of the code a single depth step consumes.
    let stepInput: MTLBuffer
    let depthX: MTLBuffer
    let depthHidden: MTLBuffer
    /// fp32 [sequences, 2052]: the logits of the draw in progress.
    let logits: MTLBuffer
    /// uint32 [maxFrames, 16].
    let codes: MTLBuffer
    /// fp32 [maxFrames, 16]: uniform q of a frame drives the draw of codebook q.
    let uniforms: MTLBuffer
    /// uint32 [2052]: nonzero for every backbone token drawn so far (the repetition penalty's set).
    let seen: MTLBuffer
    /// uint32: bit 0 set once any draw met unusable logits.
    let status: MTLBuffer

    static let logitsStride = 2052

    /// Where `tap` is called while a frame is encoded.
    enum Tap {
        /// `depthInput` holds the backbone's final hidden state of every sequence.
        case backboneHidden
        /// `logits` holds the lm_head rows.
        case backboneLogits
        /// `logits` holds codebook `q`'s head rows.
        case depthLogits(Int)
    }

    /// Encodes extra work (copies of intermediate state) at each `Tap` point.
    var tap: ((Encoder, Tap) -> Void)?
    /// uint32 [maxFrames, 16]. When set, the embeddings read the codes here instead of the
    /// sampled ones, which still land in `codes`: teacher forcing.
    var teacher: MTLBuffer?

    init(ctx: MetalContext, weights: FrameWeights, sequences: Int, maxFrames: Int) throws {
        precondition(sequences == 1 || sequences == 2)
        self.weights = weights
        self.sequences = sequences
        self.maxFrames = maxFrames
        let bb = weights.backbone.config, dd = weights.depth.config
        backbone = try DecoderForward(ctx: ctx, weights: weights.backbone,
                                      cache: try KVCache(ctx: ctx, config: bb, sequences: sequences, label: "backbone.kv"),
                                      work: try DecoderWorkspace(ctx: ctx, config: bb, maxRows: sequences, label: "backbone"))
        depth = try DecoderForward(ctx: ctx, weights: weights.depth,
                                   cache: try KVCache(ctx: ctx, config: dd, sequences: sequences, label: "depth.kv"),
                                   work: try DecoderWorkspace(ctx: ctx, config: dd, maxRows: 2 * sequences, label: "depth"))
        let spare = 8, e = BreezeShape.embedWidth
        backboneInput = try ctx.makeBuffer(bytes: (sequences + spare) * bb.hidden * 4, label: "frame.backboneInput")
        depthInput = try ctx.makeBuffer(bytes: (2 * sequences + spare) * e * 4, label: "frame.depthInput")
        stepInput = try ctx.makeBuffer(bytes: (sequences + spare) * e * 4, label: "frame.stepInput")
        depthX = try ctx.makeBuffer(bytes: (2 * sequences + spare) * dd.hidden * 4, label: "frame.depthX")
        depthHidden = try ctx.makeBuffer(bytes: (sequences + spare) * dd.hidden * 4, label: "frame.depthHidden")
        logits = try ctx.makeBuffer(bytes: (sequences + spare) * Self.logitsStride * 4, label: "frame.logits")
        codes = try ctx.makeBuffer(bytes: maxFrames * BreezeShape.codebooks * 4, label: "frame.codes")
        uniforms = try ctx.makeBuffer(bytes: maxFrames * BreezeShape.codebooks * 4, label: "frame.uniforms")
        seen = try ctx.makeBuffer(bytes: BreezeShape.backboneVocab * 4, label: "frame.seen")
        status = try ctx.makeBuffer(bytes: 4, label: "frame.status")
    }

    /// Starts a generation whose sequences hold `positions` prompt positions in the backbone cache.
    func begin(positions: [Int]) {
        backbone.cache.setLengths(positions)
        memset(seen.contents(), 0, seen.length)
        memset(status.contents(), 0, status.length)
    }

    func setUniforms(frame: Int, _ u: [Float]) {
        precondition(u.count == BreezeShape.codebooks && frame < maxFrames)
        uniforms.write(u, offset: frame * BreezeShape.codebooks * 4)
    }

    func frameCodes(_ frame: Int) -> [Int] {
        let p = (codes.contents() + frame * BreezeShape.codebooks * 4).bindMemory(to: UInt32.self, capacity: BreezeShape.codebooks)
        return (0..<BreezeShape.codebooks).map { Int(p[$0]) }
    }

    var statusWord: UInt32 { status.contents().load(as: UInt32.self) }

    /// One backbone token: `backboneInput` through the stack, each sequence at its own position,
    /// final hidden states into `depthInput`.
    func encodeBackboneStep(_ e: Encoder) {
        let w = BreezeShape.embedWidth
        backbone.encode(e, x: ActView(backboneInput, ld: w), rows: 1, out: ActView(depthInput, ld: 2 * w))
    }

    /// The 16 codes of `frame` from the hidden states in `depthInput`, then the next backbone input.
    func encodeCodes(_ e: Encoder, frame: Int, settings: FrameSettings) {
        precondition(settings.sequences == sequences && frame < maxFrames)
        let w = BreezeShape.embedWidth, vocab = BreezeShape.codebookVocab, books = BreezeShape.codebooks
        let slots = frame * books * 4
        let dd = weights.depth.config
        let logitRows = ActView(logits, ld: Self.logitsStride)
        func draw(_ kind: SamplerKind, _ q: Int) {
            e.sample(kind, logits: logitRows, guidance: settings.guidance, settings: kind == .backbone ? settings.backbone : settings.depth,
                     uniforms: uniforms, uniformsOffset: slots, codes: codes, codesOffset: slots, slot: q, seen: seen, status: status)
        }
        func embed(_ q: Int, count: Int, _ out: ActView) {
            e.embedCodes(table: weights.embedding, width: w, vocab: vocab, codes: teacher ?? codes, codesOffset: slots, firstCodebook: q, count: count, rows: sequences, out)
        }
        let pairs = ActView(depthInput, ld: 2 * w)
        tap?(e, .backboneHidden)
        e.linear(pairs, rows: sequences, weights.lmHead, logitRows)
        tap?(e, .backboneLogits)
        draw(.backbone, 0)

        depth.cache.reset()
        let x = ActView(depthX, ld: dd.hidden), hidden = ActView(depthHidden, ld: dd.hidden)
        for position in 1..<books {
            if position == 1 {
                embed(0, count: 1, pairs.column(w))
                e.linear(ActView(depthInput, ld: w), rows: 2 * sequences, weights.depthProjector, x)
                depth.encode(e, x: x, rows: 2, out: hidden)
            } else {
                embed(position - 1, count: 1, ActView(stepInput, ld: w))
                e.linear(ActView(stepInput, ld: w), rows: sequences, weights.depthProjector, x)
                depth.encode(e, x: x, rows: 1, out: hidden)
            }
            e.linear(hidden, rows: sequences, weights.codebookHeads[position - 1], logitRows)
            tap?(e, .depthLogits(position))
            draw(.depth, position)
        }
        embed(0, count: books, ActView(backboneInput, ld: w))
    }

    func encodeFrame(_ e: Encoder, frame: Int, settings: FrameSettings) {
        encodeBackboneStep(e)
        encodeCodes(e, frame: frame, settings: settings)
    }
}
