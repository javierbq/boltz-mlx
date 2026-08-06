// InterfaceScores.swift — interface confidence metrics computed from a PAE matrix.
//
// These are the gate metrics a design loop filters on. They need ONLY the predicted-aligned-error
// matrix and the per-chain token ranges — not pLDDT, not pTM/ipTM, not the distogram, and not
// `compute_ptms`. That is what makes them the cheapest useful piece of the confidence work: the
// arithmetic below is exact and testable with no model, no weights and no GPU, so it can be pinned
// against the Python reference before the ConfidenceModule forward pass exists.
//
// ipSAE (Dunbrack 2025, doi:10.1101/2025.02.10.637595) fixes two problems with using ipTM to judge a
// designed interface: ipTM is normalised over ALL cross-chain pairs, so a large chain dilutes a small
// real interface, and it is dominated by whichever pairs the model happens to be confident about.
// ipSAE instead restricts to pairs whose PAE is below a cutoff and takes the BEST per-residue score,
// which is why `min_ipsae` — the worse of the two directions — is the quantity used as a gate rather
// than the symmetric mean.
//
// Reference implementation this is pinned against:
//   agent-smith/skills/boltz-predict/scripts/compute_ipsae.py
// Note that its directional score is a MAX over per-residue means, not a mean of means; getting that
// wrong produces plausible-looking numbers that are systematically too low.
//
// THE VARIANT MATTERS AND IS NOT INTERCHANGEABLE. Published ipSAE numbers come in several
// normalisations that differ in what length feeds `d0` (per-chain, combined, or interface-residue
// counts). This file computes the COMBINED-length form, matching the reference above. Any threshold
// quoted for a different form does not transfer, so none is asserted here — see `minIPSAE`.
import Foundation

public struct InterfaceScores: Sendable, Equatable {
    /// max(A->B, B->A). Reported for continuity with the reference; not the gate.
    public let ipsae: Double
    /// min(A->B, B->A) — the gate metric.
    ///
    /// NO THRESHOLD IS ASSERTED HERE, deliberately. An earlier version of this comment quoted
    /// "> 0.61", which was wrong twice over: it is not from the ipSAE paper, and it belongs to a
    /// different normalisation than the one computed below — `d0` here uses the COMBINED chain length
    /// (the d0chn variant), and the cutoff maximising F1 is not shared between variants. Carrying a
    /// number across variants silently moves the operating point, in the PERMISSIVE direction, on the
    /// single quantity a design run accepts or rejects candidates on. Calibrate against labelled
    /// interfaces before gating.
    public let minIPSAE: Double
    public let ipsaeAB: Double
    public let ipsaeBA: Double
    /// Mean PAE over all inter-chain pairs, both directions. Angstroms, lower is better;
    /// < 10 is the conventional "confident interface" mark.
    public let ipae: Double

    public init(ipsae: Double, minIPSAE: Double, ipsaeAB: Double, ipsaeBA: Double, ipae: Double) {
        self.ipsae = ipsae
        self.minIPSAE = minIPSAE
        self.ipsaeAB = ipsaeAB
        self.ipsaeBA = ipsaeBA
        self.ipae = ipae
    }
}

public enum InterfaceScoring {

    /// Conventional PAE cutoff for counting a pair as part of the interface, in Angstroms.
    public static let defaultPAECutoff: Double = 10.0

    /// TM-score d0 normalisation. Uses the COMBINED chain length, per the reference — using either
    /// chain's length alone changes every score.
    ///
    /// n <= 15 IS HANDLED EXPLICITLY, and the reference does not handle it at all: for n < 15 it
    /// evaluates `(n - 15) ** (1/3)` on a negative base, which in Python yields a COMPLEX number and
    /// makes the subsequent `max(..., 1.0)` raise TypeError — i.e. compute_ipsae.py crashes on short
    /// chains. In Swift the same expression yields NaN, which would silently poison a gate instead
    /// of crashing, which is worse. Since the reference clamps to a floor of 1.0 anyway, the intent
    /// for small n is unambiguous, so return the floor directly. For n > 15 this is identical to the
    /// reference.
    static func d0(combinedLength n: Int) -> Double {
        guard n > 15 else { return 1.0 }
        return max(1.24 * pow(Double(n) - 15.0, 1.0 / 3.0) - 1.8, 1.0)
    }

    /// One direction: for each query residue with at least one subject residue below the cutoff,
    /// the mean ptm over its valid subjects; the directional score is the MAXIMUM of those.
    static func directional(
        pae: (Int, Int) -> Double,
        queryCount: Int,
        subjectCount: Int,
        cutoff: Double
    ) -> Double {
        guard queryCount > 0, subjectCount > 0 else { return 0 }
        let d = d0(combinedLength: queryCount + subjectCount)
        var best = 0.0
        for i in 0 ..< queryCount {
            var sum = 0.0
            var valid = 0
            for j in 0 ..< subjectCount {
                let e = pae(i, j)
                guard e < cutoff else { continue }
                let r = e / d
                sum += 1.0 / (1.0 + r * r)
                valid += 1
            }
            guard valid > 0 else { continue }
            let score = sum / Double(valid)
            if score > best { best = score }
        }
        return best
    }

    /// Score the interface between two token ranges of a square PAE matrix.
    ///
    /// `pae` is indexed (row, column) over the full token axis; `a` and `b` are the two chains'
    /// half-open token ranges — take them from `BoltzFeaturizer.Layout.chainTokenRanges` so they
    /// cannot disagree with the features the PAE was computed from.
    public static func score(
        pae: (Int, Int) -> Double,
        a: Range<Int>,
        b: Range<Int>,
        cutoff: Double = defaultPAECutoff
    ) -> InterfaceScores {
        let nA = a.count, nB = b.count
        let ab = directional(pae: { i, j in pae(a.lowerBound + i, b.lowerBound + j) },
                             queryCount: nA, subjectCount: nB, cutoff: cutoff)
        let ba = directional(pae: { i, j in pae(b.lowerBound + i, a.lowerBound + j) },
                             queryCount: nB, subjectCount: nA, cutoff: cutoff)

        var total = 0.0
        for i in a { for j in b { total += pae(i, j) } }
        var totalReverse = 0.0
        for i in b { for j in a { totalReverse += pae(i, j) } }
        let meanAB = nA * nB > 0 ? total / Double(nA * nB) : 0
        let meanBA = nA * nB > 0 ? totalReverse / Double(nA * nB) : 0

        return InterfaceScores(
            ipsae: max(ab, ba),
            minIPSAE: min(ab, ba),
            ipsaeAB: ab,
            ipsaeBA: ba,
            ipae: (meanAB + meanBA) / 2.0
        )
    }

    /// Convenience for a flat row-major `[n * n]` PAE matrix.
    public static func score(
        paeMatrix: [Double],
        tokenCount n: Int,
        a: Range<Int>,
        b: Range<Int>,
        cutoff: Double = defaultPAECutoff
    ) -> InterfaceScores {
        precondition(paeMatrix.count == n * n, "PAE matrix is not \(n)x\(n)")
        return score(pae: { i, j in paeMatrix[i * n + j] }, a: a, b: b, cutoff: cutoff)
    }

    /// Expected PAE per token pair from the model's binned logits.
    ///
    /// Boltz predicts PAE as a distribution over `binCount` bins spanning [0, maxError]; the scalar
    /// error is the expectation over bin CENTRES. With the default 64 bins over 32 A that is
    /// 0.25, 0.75, ... 31.75 — the same centres the reference uses. Passing bin EDGES instead
    /// biases every value low by half a bin.
    public static func expectedError(
        logits: [Double],
        tokenCount n: Int,
        binCount: Int,
        maximumError: Double = 32.0
    ) -> [Double] {
        precondition(logits.count == n * n * binCount, "logits are not \(n)x\(n)x\(binCount)")
        let width = maximumError / Double(binCount)
        let centres = (0 ..< binCount).map { (Double($0) + 0.5) * width }
        var out = [Double](repeating: 0, count: n * n)
        for p in 0 ..< (n * n) {
            let base = p * binCount
            var maxLogit = -Double.greatestFiniteMagnitude
            for b in 0 ..< binCount { maxLogit = max(maxLogit, logits[base + b]) }
            var sum = 0.0, weighted = 0.0
            for b in 0 ..< binCount {
                let e = exp(logits[base + b] - maxLogit)
                sum += e
                weighted += e * centres[b]
            }
            out[p] = sum > 0 ? weighted / sum : 0
        }
        return out
    }
}
