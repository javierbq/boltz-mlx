import XCTest
import Foundation
import MLX
@testable import BoltzMLX

/// BENCHMARK: Swift featurizer + MLX int8 vs upstream PyTorch fp32 on MPS.
///
/// Records, per fixture: featurization time (separately, because eliminating the Python featurizer is
/// the point), prediction time, MLX peak memory, and every atom's coordinates labelled by
/// (residue ordinal, atom name) so RMSD can be paired by identity rather than by position.
///
/// MEMORY IS NOT DIRECTLY COMPARABLE ACROSS BACKENDS and the report must say so: this measures MLX's
/// own unified-memory high-water mark, while the Python side can only be measured as process peak
/// RSS. The MLX figure excludes the host process; the RSS figure includes the interpreter, torch and
/// the 2.3 GB checkpoint load. They answer different questions.
///
///   BOLTZ_CONF_MODEL=/tmp/pack_conf2   BOLTZ_BENCH_OUT=/tmp/bench
///   BOLTZ_BENCH_RECYCLING=3  BOLTZ_BENCH_STEPS=50
final class BenchmarkTests: XCTestCase {

    private static let fixtures = ["allresidues", "protnomsa", "twochain"]

    private func chains(_ name: String) throws -> [(chain: String, sequence: String)] {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appending(path: "fixtures/\(name).yaml")
        var out: [(String, String)] = []
        var pending: String?
        for raw in try String(contentsOf: url, encoding: .utf8).split(separator: "\n") {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.hasPrefix("id:") {
                pending = String(line.dropFirst(3)).trimmingCharacters(in: .whitespaces)
            } else if line.hasPrefix("sequence:"), let id = pending {
                out.append((id, String(line.dropFirst(9)).trimmingCharacters(in: .whitespaces)))
                pending = nil
            }
        }
        return out
    }

    func testBenchmarkAgainstUpstream() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["BOLTZ_CONF_MODEL"], let outPath = env["BOLTZ_BENCH_OUT"] else {
            throw XCTSkip("set BOLTZ_CONF_MODEL and BOLTZ_BENCH_OUT")
        }
        let recycling = Int(env["BOLTZ_BENCH_RECYCLING"] ?? "") ?? 3
        let steps = Int(env["BOLTZ_BENCH_STEPS"] ?? "") ?? 50
        let out = URL(fileURLWithPath: outPath)
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)

        let limits = BoltzInputLimits(maximumTokens: 512, maximumAtoms: 4_096, maximumMSADepth: 1_024)
        let predictor = try BoltzPredictor(modelDirectory: URL(fileURLWithPath: modelPath),
                                          memoryPlanner: MemoryPlanner(limits: limits))
        // One warm-up so the reported times exclude lazy weight materialisation and graph build,
        // matching the Python side which is measured after its own model load.
        let warm = try BoltzFeaturizer().featurize(
            try CanonicalStructure.fromSequences(try chains("allresidues")))
        _ = try await predictor.predict(featurized: warm,
                                        options: BoltzPredictionOptions(recyclingSteps: 0,
                                                                        diffusionSteps: 2, seed: 0))

        var rows: [String] = ["fixture,tokens,atoms,featurize_s,predict_s,mlx_peak_bytes"]
        for name in Self.fixtures {
            await predictor.resetPeakMemory()
            let c = try chains(name)

            let t0 = Date()
            let featurized = try BoltzFeaturizer()
                .featurize(try CanonicalStructure.fromSequences(c))
            let featurizeSeconds = Date().timeIntervalSince(t0)

            let t1 = Date()
            let scored = try await predictor.predictScored(
                featurized: featurized,
                options: BoltzPredictionOptions(recyclingSteps: recycling,
                                                diffusionSteps: steps, seed: 0))
            let predictSeconds = Date().timeIntervalSince(t1)
            let peak = await predictor.memorySnapshot().peakMemory

            // Coordinates labelled by identity, for RMSD pairing.
            var lines = ["res,name,x,y,z"]
            var i = 0, ordinal = 0
            for (_, sequence) in c {
                for ch in sequence {
                    guard let t = AAResidueTemplates.template(oneLetter: ch) else { continue }
                    for atom in t.atoms {
                        if i < scored.structure.coordinates.count {
                            let p = scored.structure.coordinates[i]
                            lines.append("\(ordinal),\(atom.name),\(p.x),\(p.y),\(p.z)")
                        }
                        i += 1
                    }
                    ordinal += 1
                }
            }
            try lines.joined(separator: "\n")
                .write(to: out.appending(path: "mlx_\(name).csv"), atomically: true, encoding: .utf8)

            let scores = scored.interfaceScores()
            rows.append("\(name),\(featurized.layout.tokenCount),\(featurized.layout.atomCount),"
                        + "\(featurizeSeconds),\(predictSeconds),\(peak)")
            print(String(format: "%-12s tokens %4d atoms %5d  featurize %.3f s  predict %6.2f s  "
                         + "MLX peak %6.2f GB  min_ipSAE %@",
                         (name as NSString).utf8String!, featurized.layout.tokenCount,
                         featurized.layout.atomCount, featurizeSeconds, predictSeconds,
                         Double(peak) / 1e9,
                         scores.map { String(format: "%.4f", $0.minIPSAE) } ?? "n/a"))
        }
        try rows.joined(separator: "\n")
            .write(to: out.appending(path: "mlx_timings.csv"), atomically: true, encoding: .utf8)
        print("wrote \(out.path)/mlx_timings.csv and per-fixture coordinates")
    }
}
