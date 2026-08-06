import XCTest
import Foundation
import MLX
@testable import BoltzMLX

/// END TO END WITHOUT PYTHON.
///
/// Proves the whole path runs in Swift/MLX alone: one-letter sequences -> CanonicalStructure ->
/// BoltzFeaturizer -> BoltzPredictor -> 3D coordinates. Nothing in this file touches Python, torch,
/// rdkit or a feature bundle on disk; the process has no interpreter loaded at all. Until the
/// featurizer landed, `FeatureBundle.load(from:)` was the ONLY way to obtain features, so every
/// prediction required a Python export step against a 2.1 GB component cache.
///
/// It also closes the validation that the featurizer commits could not: comparing a fold driven by
/// Swift features against a fold driven by Python features, at the same seed, on the same model.
/// That is the number that matters — a featurizer can be bitwise-correct on 38 tensors and still
/// produce a bad structure through the one tensor that is not bitwise-comparable (ref_pos), which is
/// exactly what happened to the first version of this port.
///
/// Requires the int8 model pack (~507 MB, not committed):
///   BOLTZ_MODEL=.artifacts/boltz2-mlx-int8
/// Optionally, for the cross-check against Python-generated features:
///   BOLTZ_REF_DIR=<dir holding ref_twochain/>
final class EndToEndNoPythonTests: XCTestCase {

    /// Deliberately small so this is runnable as a gate: recycling 0 / 20 steps is the port's default.
    /// It is NOT the production regime (upstream trains confidence at recycling 3 / 200 and this
    /// port's own quality gate fails at 20 steps), so these assertions check that the pipeline RUNS
    /// and is physically sane — not that the structure is production quality.
    private let options = BoltzPredictionOptions(recyclingSteps: 0, diffusionSteps: 20, seed: 0)

    /// Cross-backend comparison needs the PRODUCTION regime, not the port's fast defaults: at
    /// recycling 0 / 20 steps two upstream runs differing only in seed disagree by 14.5 A on this
    /// complex, so any comparison there measures sampler chaos rather than correctness.
    private var comparisonOptions: BoltzPredictionOptions {
        let env = ProcessInfo.processInfo.environment
        return BoltzPredictionOptions(
            recyclingSteps: Int(env["BOLTZ_RECYCLING"] ?? "") ?? 3,
            diffusionSteps: Int(env["BOLTZ_STEPS"] ?? "") ?? 200,
            seed: UInt64(env["BOLTZ_SEED"] ?? "") ?? 0
        )
    }

    private func model() throws -> BoltzPredictor {
        guard let dir = ProcessInfo.processInfo.environment["BOLTZ_MODEL"] else {
            throw XCTSkip("set BOLTZ_MODEL to an exported int8 model directory")
        }
        // Raise the iPhone-tuned defaults: the caps are documented as "conservative starting limits,
        // not validated device maxima", and 1779 atoms is inside the measured-safe Mac envelope.
        let limits = BoltzInputLimits(maximumTokens: 512, maximumAtoms: 4_096,
                                      maximumMSADepth: 1_024)
        return try BoltzPredictor(modelDirectory: URL(fileURLWithPath: dir),
                                  memoryPlanner: MemoryPlanner(limits: limits))
    }

    private func fixtureChains(_ name: String) throws -> [(chain: String, sequence: String)] {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appending(path: "fixtures/\(name).yaml")
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
        return chains
    }

    // MARK: The end-to-end run

    func testFoldsATwoChainComplexFromSequencesAlone() async throws {
        let predictor = try model()
        let chains = try fixtureChains("twochain")

        // The entire input: two one-letter sequences. No coordinates, no bundle, no Python.
        let canonical = try CanonicalStructure.fromSequences(chains)
        XCTAssertTrue(canonical.diagnostics.isEmpty, "\(canonical.diagnostics)")
        let featurized = try BoltzFeaturizer().featurize(canonical)

        let started = Date()
        let structure = try await predictor.predict(featurized: featurized, options: options)
        let seconds = Date().timeIntervalSince(started)

        XCTAssertEqual(structure.coordinates.count, featurized.layout.atomCount,
                       "one coordinate per real atom")
        XCTAssertTrue(structure.coordinates.allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite },
                      "coordinates contain NaN or infinity")

        // Physically sane: consecutive CA atoms of one chain should be ~3.8 A apart. CA is the second
        // atom of every canonical template, so recover CA indices from the atom layout.
        let cas = caIndices(featurized)
        XCTAssertEqual(cas.count, featurized.layout.tokenCount)
        var adjacent: [Float] = []
        for (chain, range) in featurized.layout.chainTokenRanges {
            _ = chain
            for t in range.dropLast() {
                let d = structure.coordinates[cas[t + 1]] - structure.coordinates[cas[t]]
                adjacent.append((d * d).sum().squareRoot())
            }
        }
        let mean = adjacent.reduce(0, +) / Float(adjacent.count)
        let valid = adjacent.filter { $0 > 3.6 && $0 < 4.1 }.count
        print("""

        === end-to-end, no Python ===
        tokens        \(featurized.layout.tokenCount)
        atoms         \(featurized.layout.atomCount) (padded \(featurized.layout.paddedAtomCount))
        wall clock    \(String(format: "%.2f", seconds)) s
        CA-CA mean    \(String(format: "%.3f", mean)) A
        CA-CA valid   \(valid)/\(adjacent.count)
        """)
        XCTAssertEqual(mean, 3.82, accuracy: 0.25, "CA-CA spacing is not physical")
        XCTAssertGreaterThan(Double(valid) / Double(adjacent.count), 0.90,
                             "fewer than 90% of consecutive CA-CA distances are physical")
    }

    /// Two chains must come out as two separated bodies, not one interpenetrating blob — the failure
    /// mode the unaugmented ref_pos produced.
    func testChainsAreSeparatedNotOverPacked() async throws {
        let predictor = try model()
        let featurized = try BoltzFeaturizer()
            .featurize(try CanonicalStructure.fromSequences(try fixtureChains("twochain")))
        let structure = try await predictor.predict(featurized: featurized, options: options)
        let cas = caIndices(featurized)
        let ranges = featurized.layout.chainTokenRanges
        XCTAssertEqual(ranges.count, 2)

        func centroid(_ r: Range<Int>) -> SIMD3<Float> {
            var c = SIMD3<Float>.zero
            for t in r { c += structure.coordinates[cas[t]] }
            return c / Float(r.count)
        }
        let d = centroid(ranges[0].range) - centroid(ranges[1].range)
        let separation = (d * d).sum().squareRoot()
        var contacts = 0
        for i in ranges[0].range {
            for j in ranges[1].range {
                let v = structure.coordinates[cas[i]] - structure.coordinates[cas[j]]
                if (v * v).sum().squareRoot() < 8 { contacts += 1 }
            }
        }
        print("chain centroid separation \(String(format: "%.2f", separation)) A, CA contacts <8 A: \(contacts)")
        XCTAssertGreaterThan(separation, 5, "chains collapsed onto each other")
        XCTAssertLessThan(separation, 80, "chains flew apart")
    }

    /// THE VALIDATION THE FEATURIZER COMMITS COULD NOT DO: same model, same seed, Swift features vs
    /// Python features. Compared against the Python-vs-Python noise floor (~4.4 A all-atom RMSD on
    /// this fixture), which exists because upstream re-randomises ref_pos every run.
    func testSwiftFeaturesFoldEquivalentlyToPythonFeatures() async throws {
        guard let refDir = ProcessInfo.processInfo.environment["BOLTZ_REF_DIR"] else {
            throw XCTSkip("set BOLTZ_REF_DIR to compare against Python-generated features")
        }
        let predictor = try model()
        let featurized = try BoltzFeaturizer()
            .featurize(try CanonicalStructure.fromSequences(try fixtureChains("twochain")))
        let reference = try FeatureBundle.load(
            from: URL(fileURLWithPath: refDir).appending(path: "ref_twochain"))

        let mine = try await predictor.predict(featurized: featurized, options: options)
        let theirs = try await predictor.predict(features: reference, options: options)

        XCTAssertEqual(mine.coordinates.count, theirs.coordinates.count)
        guard mine.coordinates.count == theirs.coordinates.count else { return }

        // Both are in arbitrary global frames, so superpose before comparing. Kabsch on all atoms.
        let rmsd = superposedRMSD(mine.coordinates, theirs.coordinates)
        // Radius of gyration is frame-free and was the clearest signal of the over-packing defect.
        func rg(_ p: [SIMD3<Float>]) -> Float {
            var c = SIMD3<Float>.zero
            for x in p { c += x }
            c /= Float(p.count)
            var s: Float = 0
            for x in p { let d = x - c; s += (d * d).sum() }
            return (s / Float(p.count)).squareRoot()
        }
        print("""

        === Swift features vs Python features (same model, same seed) ===
        superposed all-atom RMSD  \(String(format: "%.2f", rmsd)) A
        radius of gyration        swift \(String(format: "%.2f", rg(mine.coordinates)))  python \(String(format: "%.2f", rg(theirs.coordinates)))
        """)
        // The Python-vs-Python noise floor on this fixture is ~4.4 A; allow generous headroom but
        // catch the 13-15 A regime the unaugmented featurizer produced.
        XCTAssertLessThan(rmsd, 9.0,
                          "Swift-featurized fold diverges far beyond the Python-vs-Python noise floor")
        XCTAssertEqual(rg(mine.coordinates), rg(theirs.coordinates), accuracy: 3.0,
                       "compactness differs materially — the over-packing signature")
    }

    /// Writes the pipeline's CA coordinates so they can be compared against an upstream PyTorch run.
    /// CA-only on purpose: one CA per residue in residue order is unambiguous, whereas an all-atom
    /// comparison would depend on both sides ordering side-chain atoms identically.
    /// Set BOLTZ_DUMP_CA to a path to enable.
    func testDumpCACoordinatesForCrossBackendComparison() async throws {
        guard let out = ProcessInfo.processInfo.environment["BOLTZ_DUMP_CA"] else {
            throw XCTSkip("set BOLTZ_DUMP_CA to write CA coordinates for cross-backend comparison")
        }
        let predictor = try model()
        let featurized = try BoltzFeaturizer()
            .featurize(try CanonicalStructure.fromSequences(try fixtureChains(ProcessInfo.processInfo.environment["BOLTZ_FIXTURE"] ?? "twochain")))
        let opts = comparisonOptions
        print("dumping at recycling \(opts.recyclingSteps) / \(opts.diffusionSteps) steps, seed \(opts.seed)")
        let structure = try await predictor.predict(featurized: featurized, options: opts)
        let cas = caIndices(featurized)
        var lines = ["chain,token,x,y,z"]
        for (chain, range) in featurized.layout.chainTokenRanges {
            for t in range {
                let p = structure.coordinates[cas[t]]
                lines.append("\(chain),\(t),\(p.x),\(p.y),\(p.z)")
            }
        }
        try lines.joined(separator: "\n").write(toFile: out, atomically: true, encoding: .utf8)
        print("wrote \(cas.count) CA coordinates to \(out)")
    }


    /// MATCHED-NOISE CROSS-BACKEND COMPARISON.
    ///
    /// Consumes noise recorded from an upstream PyTorch/MPS run by
    /// scripts/matched_noise_reference.py, with ref_pos augmentation forced to identity on BOTH
    /// sides and the per-step coordinate augmentation identity on both. With all three sources of
    /// randomness pinned, any remaining difference is attributable to the implementations —
    /// features and network — rather than to the sampler, which is the only way to get an
    /// interpretable number on a target whose prediction does not converge.
    ///
    ///   BOLTZ_MATCHED=/tmp/matched   (holding noise.safetensors and reference.pdb)
    func testMatchedNoiseAgainstUpstream() async throws {
        guard let dir = ProcessInfo.processInfo.environment["BOLTZ_MATCHED"] else {
            throw XCTSkip("set BOLTZ_MATCHED to a directory from scripts/matched_noise_reference.py")
        }
        let root = URL(fileURLWithPath: dir)
        let loaded = try MLX.loadArrays(url: root.appending(path: "noise.safetensors"))
        guard let initial = loaded["initial"] else {
            return XCTFail("noise.safetensors has no `initial` tensor")
        }
        let steps = loaded.keys.filter { $0.hasPrefix("step_") }.sorted().compactMap { loaded[$0] }
        XCTAssertFalse(steps.isEmpty, "no step noise recorded")

        let predictor = try model()
        // identityAugmentation MUST match the Python side's patched featurizer.
        // The fixture is read from the matched-noise directory, not guessed: pairing recorded noise
        // with a different fixture yields a shape mismatch and a meaningless comparison.
        let fixtureName = (try? String(contentsOf: root.appending(path: "fixture.txt"), encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            ?? ProcessInfo.processInfo.environment["BOLTZ_FIXTURE"] ?? "twochain"
        let featurized = try BoltzFeaturizer(identityAugmentation: true)
            .featurize(try CanonicalStructure.fromSequences(try fixtureChains(fixtureName)))
        XCTAssertEqual(initial.shape, [1, featurized.layout.paddedAtomCount, 3],
                       "recorded noise does not match this featurization's atom axis")

        let opts = BoltzPredictionOptions(recyclingSteps: 3, diffusionSteps: steps.count, seed: 0)
        let structure = try await predictor.predict(
            featurized: featurized, options: opts,
            matchedNoise: .init(initial: initial, steps: steps))

        // Upstream reference CAs, in residue order.
        let pdb = try String(contentsOf: root.appending(path: "reference.pdb"), encoding: .utf8)
        var refCA: [SIMD3<Float>] = []
        for line in pdb.split(separator: "\n") where line.hasPrefix("ATOM") {
            let c = Array(line)
            guard c.count >= 54, String(c[12..<16]).trimmingCharacters(in: .whitespaces) == "CA" else { continue }
            func f(_ r: Range<Int>) -> Float { Float(String(c[r]).trimmingCharacters(in: .whitespaces)) ?? 0 }
            refCA.append(SIMD3(f(30..<38), f(38..<46), f(46..<54)))
        }
        let cas = caIndices(featurized)
        XCTAssertEqual(refCA.count, cas.count, "CA count mismatch with the upstream reference")
        guard refCA.count == cas.count else { return }
        let mine = cas.map { structure.coordinates[$0] }

        let rmsd = superposedRMSD(mine, refCA)
        func rg(_ p: [SIMD3<Float>]) -> Float {
            var c = SIMD3<Float>.zero
            for x in p { c += x }
            c /= Float(p.count)
            var s: Float = 0
            for x in p { let d = x - c; s += (d * d).sum() }
            return (s / Float(p.count)).squareRoot()
        }
        print("""

        === MATCHED NOISE: Swift/MLX int8 vs PyTorch fp32 (MPS) ===
        steps                \(steps.count), recycling 3
        CA RMSD (Kabsch)     \(String(format: "%.2f", rmsd)) A
        radius of gyration   mlx \(String(format: "%.2f", rg(mine)))  upstream \(String(format: "%.2f", rg(refCA)))
        """)
        // MEASURED, prot_no_msa, recycling 3: 2.06 A CA at 50 steps, 2.90 A at 200. The repo's
        // established network-only gate (Python features BOTH sides) is 2.0 A all-atom, reached at
        // 50 steps. This path additionally includes the Swift featurizer and is CA-only, so it sits
        // just outside that gate rather than inside it — close, but NOT yet demonstrating full
        // end-to-end fidelity. Asserted loosely so the test records the measurement and catches a
        // gross regression, without claiming a fidelity result that has not been earned.
        XCTAssertLessThan(rmsd, 5.0, "matched-noise CA RMSD regressed badly")
    }

    // MARK: helpers

    /// Index of each residue's CA atom in the unpadded coordinate array. CA is the second atom of
    /// every canonical template, and the atom axis runs in token order.
    private func caIndices(_ out: BoltzFeaturizer.Output) -> [Int] {
        let n = out.layout.paddedAtomCount
        let uid = out.features["ref_space_uid"]!.reshaped([n]).asArray(Int32.self)
        let mask = out.features["atom_pad_mask"]!.reshaped([n]).asArray(Float.self)
        var seen = Set<Int32>()
        var result: [Int] = []
        var unpadded = 0
        for i in 0 ..< n where mask[i] == 1 {
            if !seen.contains(uid[i]) {
                seen.insert(uid[i])
                result.append(unpadded + 1)      // +1: CA follows N
            }
            unpadded += 1
        }
        return result
    }

    private func superposedRMSD(_ a: [SIMD3<Float>], _ b: [SIMD3<Float>]) -> Float {
        func centre(_ p: [SIMD3<Float>]) -> ([SIMD3<Float>], SIMD3<Float>) {
            var c = SIMD3<Float>.zero
            for x in p { c += x }
            c /= Float(p.count)
            return (p.map { $0 - c }, c)
        }
        let (x, _) = centre(a), (y, _) = centre(b)
        // Covariance, then a small rotation search via the Kabsch SVD substitute: since we only need
        // a scalar RMSD, use the closed-form quaternion method (Horn 1987) to avoid an SVD dependency.
        var m = [Float](repeating: 0, count: 9)
        for i in 0 ..< x.count {
            m[0] += x[i].x*y[i].x; m[1] += x[i].x*y[i].y; m[2] += x[i].x*y[i].z
            m[3] += x[i].y*y[i].x; m[4] += x[i].y*y[i].y; m[5] += x[i].y*y[i].z
            m[6] += x[i].z*y[i].x; m[7] += x[i].z*y[i].y; m[8] += x[i].z*y[i].z
        }
        let k: [[Float]] = [
            [m[0]+m[4]+m[8], m[5]-m[7],       m[6]-m[2],       m[1]-m[3]],
            [m[5]-m[7],      m[0]-m[4]-m[8],  m[1]+m[3],       m[6]+m[2]],
            [m[6]-m[2],      m[1]+m[3],      -m[0]+m[4]-m[8],  m[5]+m[7]],
            [m[1]-m[3],      m[6]+m[2],       m[5]+m[7],      -m[0]-m[4]+m[8]],
        ]
        // Largest eigenvalue of K by power iteration — enough for an RMSD scalar.
        var v: [Float] = [1, 0, 0, 0]
        var lambda: Float = 0
        for _ in 0 ..< 200 {
            var w = [Float](repeating: 0, count: 4)
            for i in 0 ..< 4 { for j in 0 ..< 4 { w[i] += k[i][j] * v[j] } }
            let n = (w.reduce(0) { $0 + $1 * $1 }).squareRoot()
            if n < 1e-9 { break }
            v = w.map { $0 / n }
            lambda = n
        }
        var sq: Float = 0
        for i in 0 ..< x.count { sq += (x[i] * x[i]).sum() + (y[i] * y[i]).sum() }
        return Swift.max(0, (sq - 2 * lambda) / Float(x.count)).squareRoot()
    }
}
