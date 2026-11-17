// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// The knobs of one sampler. The reference runs both at temperature 0.9 and top-k 50; only the
/// backbone applies the repetition penalty.
struct SamplerSettings: Equatable {
    var temperature: Float = 0.9
    var topK = 50
    var repetitionPenalty: Float = 1.1
    /// Argmax instead of a draw.
    var greedy = false
}

/// The sampling rule both samplers follow, as scalar code. `sample_backbone` and `sample_depth`
/// implement it on the GPU, and the Python reference harness is patched to it, so every discrete
/// choice is written down here. All arithmetic is fp32, one operation at a time (no fused
/// multiply-add).
///
/// 1. Guidance, when two logit rows are given: x = uncond + scale * (cond - uncond) per token,
///    the difference, the product and the sum each rounded to fp32.
/// 2. Backbone only, with penalty != 1: for every token id in the history set (each id sampled
///    so far in this generation, the first draw included, as a set), x = x / penalty when x > 0,
///    else x * penalty.
/// 3. x = -inf for the reserved ids 2048, 2049, 2050.
/// 4. Greedy: the token is the argmax of x, the lowest id among equal maxima. Stop here.
/// 5. x = x / temperature.
/// 6. Top-k. Backbone: with t the k-th largest value, keep every token with x >= t, ties
///    included. Depth: keep exactly k tokens, ranking by x descending and by token id ascending
///    among equal x. -inf is never kept.
/// 7. Weights e = exp(x - max x) for kept tokens, 0 for the rest. No top-p step: the reference's
///    top_p is 1.0 and its cumulative-sum cut at 1.0 is rounding noise this rule leaves out.
/// 8. Inverse CDF in ascending token id: c[j] = c[j - 1] + e[j], total = c[last]. The token is
///    the first id with c[id] > u * total, for the uniform u in [0, 1) supplied by the caller.
///    If no id qualifies (u * total rounds up to total) it is the highest id with e > 0.
/// 9. Depth only: the token is clamped to 2050. Backbone only: the token joins the history set.
///
/// A draw consumes exactly one uniform, greedy or not, so draw q of a frame always reads
/// uniform q.
enum SamplingRule {
    struct Draw {
        let token: Int
        /// c[j] of step 8 and u * total, empty and 0 under greedy; a test uses them to tell a
        /// draw that sits on a rounding boundary.
        let cumulative: [Float]
        let target: Float
    }

    /// Steps 1 to 5: the scores top-k sees (step 5 skipped under greedy).
    static func scores(_ kind: SamplerKind, cond: [Float], uncond: [Float]?, guidance: Float, seen: Set<Int>, settings: SamplerSettings) -> [Float] {
        var x = cond
        if let uncond {
            for i in x.indices {
                let diff = cond[i] - uncond[i]
                let scaled = guidance * diff
                x[i] = uncond[i] + scaled
            }
        }
        if kind == .backbone, settings.repetitionPenalty != 1 {
            for i in seen where i < x.count { x[i] = x[i] > 0 ? x[i] / settings.repetitionPenalty : x[i] * settings.repetitionPenalty }
        }
        for i in BreezeShape.reservedIDs where i < x.count { x[i] = -.infinity }
        if !settings.greedy { for i in x.indices { x[i] = x[i] / settings.temperature } }
        return x
    }

    static func draw(_ kind: SamplerKind, cond: [Float], uncond: [Float]? = nil, guidance: Float = 1, seen: Set<Int> = [], settings: SamplerSettings, uniform: Float) -> Draw {
        let x = scores(kind, cond: cond, uncond: uncond, guidance: guidance, seen: seen, settings: settings)
        var top = -Float.infinity, argmax = 0
        for (i, v) in x.enumerated() where v > top { (top, argmax) = (v, i) }
        if settings.greedy { return Draw(token: argmax, cumulative: [], target: 0) }

        var cumulative = [Float](repeating: 0, count: x.count)
        var c: Float = 0
        var lastKept = argmax
        for (i, v) in x.enumerated() {
            var dominators = 0
            for (j, other) in x.enumerated() where other > v || (kind == .depth && other == v && j < i) { dominators += 1 }
            let e: Float = dominators < settings.topK && v > -.infinity ? exp(v - top) : 0
            if e > 0 { lastKept = i }
            c += e
            cumulative[i] = c
        }
        let target = uniform * c
        var token = cumulative.firstIndex { $0 > target } ?? lastKept
        if kind == .depth { token = min(token, BreezeShape.codebookVocab - 1) }
        return Draw(token: token, cumulative: cumulative, target: target)
    }
}
