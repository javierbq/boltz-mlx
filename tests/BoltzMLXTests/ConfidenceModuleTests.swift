import XCTest
import Foundation
import MLX
@testable import BoltzMLX

/// PARITY FOR THE CONFIDENCE HEAD.
///
/// The head is validated in ISOLATION: the reference script feeds it deterministic synthetic
/// (s_inputs, s, z, x_pred) rather than real trunk outputs, and records both those inputs and the
/// resulting pae_logits. That matters — driving it from a real trunk run would fold any trunk
/// difference into the comparison and a mismatch could not be attributed to the confidence head.
///
/// Logits are compared BEFORE the expectation is taken, because a bug in the head and a bug in the
/// binning are indistinguishable once reduced to a scalar PAE.
///
///   BOLTZ_CONF_MODEL=/tmp/pack_conf2   (an export including confidence_module)
///   BOLTZ_CONF_REF=/tmp/conf_ref       (from scripts/confidence_reference.py)
///   BOLTZ_REF_DIR / features           (the same bundle the reference was generated against)
final class ConfidenceModuleTests: XCTestCase {

    private func env(_ key: String) throws -> URL {
        guard let v = ProcessInfo.processInfo.environment[key] else { throw XCTSkip("set \(key)") }
        return URL(fileURLWithPath: v)
    }

    func testPAELogitsMatchPyTorch() throws {
        let packURL = try env("BOLTZ_CONF_MODEL")
        let refURL = try env("BOLTZ_CONF_REF")
        let featURL = try env("BOLTZ_CONF_FEATURES")

        let artifact = try BoltzArtifact.load(from: packURL)
        guard let configuration = artifact.configuration else {
            return XCTFail("pack has no config.json")
        }
        guard let confidenceArgs = configuration.confidence else {
            throw XCTSkip("pack predates confidence export (no confidence_model_args)")
        }
        XCTAssertTrue(confidenceArgs.useSeparateHeads, "expected the two-head PAE configuration")
        XCTAssertEqual(confidenceArgs.numPAEBins, 64)

        let store = BoltzWeightStore(artifact: artifact)
        guard let module = try store.confidenceModule(configuration: configuration) else {
            return XCTFail("confidenceModule returned nil despite config being present")
        }

        let features = try FeatureBundle.load(from: featURL).arrays
        let reference = try MLX.loadArrays(url: refURL.appending(path: "confidence_reference.safetensors"))
        for key in ["s_inputs", "s", "z", "x_pred", "pae_logits"] {
            XCTAssertNotNil(reference[key], "reference is missing \(key)")
        }

        let out = try module(
            sequenceInputs: reference["s_inputs"]!,
            sequence: reference["s"]!,
            pair: reference["z"]!,
            predictedCoordinates: reference["x_pred"]!,
            features: features
        )

        let expected = reference["pae_logits"]!
        XCTAssertEqual(expected.shape, [1, out.tokenCount, out.tokenCount, out.binCount])
        let mine = MLXArray(out.paeLogits, [1, out.tokenCount, out.tokenCount, out.binCount])
        let delta = MLX.abs(mine - expected.asType(.float32))
        let maxDelta = MLX.max(delta).item(Float.self)
        let meanDelta = MLX.mean(delta).item(Float.self)

        // Correlation is the right primary check for an int8 network: a small systematic scale
        // difference is expected from quantisation, a structural error is not.
        let a = mine.reshaped([-1]).asType(.float32)
        let b = expected.reshaped([-1]).asType(.float32)
        let am = a - MLX.mean(a), bm = b - MLX.mean(b)
        let r = (MLX.sum(am * bm) / (MLX.sqrt(MLX.sum(am * am)) * MLX.sqrt(MLX.sum(bm * bm))))
            .item(Float.self)

        print("""

        === CONFIDENCE HEAD PARITY (int8 MLX vs fp32 PyTorch) ===
        tokens \(out.tokenCount)  bins \(out.binCount)
        pae_logits  max|delta| \(String(format: "%.4f", maxDelta))   mean|delta| \(String(format: "%.5f", meanDelta))
        pearson r   \(String(format: "%.6f", r))
        """)
        XCTAssertGreaterThan(r, 0.999, "pae_logits are not structurally reproduced")
        XCTAssertLessThan(meanDelta, 0.05, "mean logit error is larger than int8 should cause")
    }

    /// The expected-error matrix, which is what min_ipSAE consumes.
    func testExpectedPAEMatchesPyTorch() throws {
        let packURL = try env("BOLTZ_CONF_MODEL")
        let refURL = try env("BOLTZ_CONF_REF")
        let featURL = try env("BOLTZ_CONF_FEATURES")
        let artifact = try BoltzArtifact.load(from: packURL)
        guard let configuration = artifact.configuration,
              let module = try BoltzWeightStore(artifact: artifact)
                .confidenceModule(configuration: configuration) else {
            throw XCTSkip("no confidence head in this pack")
        }
        let features = try FeatureBundle.load(from: featURL).arrays
        let reference = try MLX.loadArrays(url: refURL.appending(path: "confidence_reference.safetensors"))
        guard let refPAE = reference["pae"] else { throw XCTSkip("reference has no pae matrix") }

        let out = try module(
            sequenceInputs: reference["s_inputs"]!, sequence: reference["s"]!,
            pair: reference["z"]!, predictedCoordinates: reference["x_pred"]!, features: features)

        let n = out.tokenCount
        let theirs = refPAE.reshaped([n * n]).asType(.float32).asArray(Float.self)
        var maxDelta: Double = 0, sum: Double = 0
        for i in 0 ..< (n * n) {
            let d = abs(out.pae[i] - Double(theirs[i]))
            maxDelta = max(maxDelta, d); sum += d
        }
        let meanDelta = sum / Double(n * n)
        print("""
        expected PAE  max|delta| \(String(format: "%.4f", maxDelta)) A   mean|delta| \(String(format: "%.5f", meanDelta)) A
        """)
        // PAE spans 0-32 A; a mean error of a few hundredths of an Angstrom is int8 noise, and
        // min_ipSAE thresholds on a 10 A cutoff so it is far from sensitive at that scale.
        XCTAssertLessThan(meanDelta, 0.2, "expected PAE differs more than quantisation explains")
    }

    /// THE GATE, END TO END: sequences -> fold -> PAE -> min_ipSAE, entirely in Swift.
    /// This is the capability the whole design loop was blocked on.
    func testEndToEndGateFromSequencesAlone() async throws {
        let packURL = try env("BOLTZ_CONF_MODEL")
        let limits = BoltzInputLimits(maximumTokens: 512, maximumAtoms: 4_096, maximumMSADepth: 1_024)
        let predictor = try BoltzPredictor(modelDirectory: packURL,
                                          memoryPlanner: MemoryPlanner(limits: limits))
        let scorable = await predictor.canScoreInterfaces
        XCTAssertTrue(scorable, "pack should support scoring")

        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appending(path: "fixtures/twochain.yaml")
        var chains: [(String, String)] = []
        var pending: String?
        for raw in try String(contentsOf: url, encoding: .utf8).split(separator: "\n") {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.hasPrefix("id:") {
                pending = String(line.dropFirst(3)).trimmingCharacters(in: .whitespaces)
            } else if line.hasPrefix("sequence:"), let id = pending {
                chains.append((id, String(line.dropFirst(9)).trimmingCharacters(in: .whitespaces)))
                pending = nil
            }
        }
        let featurized = try BoltzFeaturizer()
            .featurize(try CanonicalStructure.fromSequences(chains))
        let scored = try await predictor.predictScored(
            featurized: featurized,
            options: BoltzPredictionOptions(recyclingSteps: 0, diffusionSteps: 20, seed: 0))

        XCTAssertEqual(scored.tokenCount, featurized.layout.tokenCount)
        XCTAssertEqual(scored.pae.count, scored.tokenCount * scored.tokenCount)
        XCTAssertTrue(scored.pae.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 32 },
                      "PAE outside [0, 32] A or non-finite")
        let s = try XCTUnwrap(scored.interfaceScores(), "two chains should yield an interface score")
        print("""

        === GATE, END TO END (no Python) ===
        tokens \(scored.tokenCount)   atoms \(scored.structure.coordinates.count)
        min_ipSAE  \(String(format: "%.4f", s.minIPSAE))
        ipSAE      \(String(format: "%.4f", s.ipsae))   (A->B \(String(format: "%.4f", s.ipsaeAB)), B->A \(String(format: "%.4f", s.ipsaeBA)))
        iPAE       \(String(format: "%.2f", s.ipae)) A
        """)
        XCTAssertTrue((0...1).contains(s.minIPSAE), "min_ipSAE must be a probability")
        XCTAssertTrue(s.ipae > 0 && s.ipae <= 32, "iPAE outside the PAE range")
        XCTAssertLessThanOrEqual(s.minIPSAE, s.ipsae, "min must not exceed max")
    }
}
