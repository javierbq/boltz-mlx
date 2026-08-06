import Foundation
import MLX
import XCTest

@testable import BoltzMLX

/// Diagnostic harness that runs native modules against the real int8 artifact and
/// a precomputed feature bundle, dumping intermediate tensors to safetensors for
/// PyTorch boundary comparison. It is a no-op unless a boundary config is present
/// at `~/.artifacts-boltz-boundary/config.json` (or the BOLTZ_BOUNDARY_* env
/// vars are set), so it stays inert in ordinary test runs.
final class BoundaryDumpTests: XCTestCase {
  private struct BoundaryConfig {
    let model: String
    let features: String
    let out: String
    let scoreInput: String?
    let scoreOut: String?
    let noiseInput: String?
    let sampleOut: String?
  }

  private func boundaryConfig() -> BoundaryConfig? {
    let configURL = FileManager.default.homeDirectoryForCurrentUser
      .appending(path: ".artifacts-boltz-boundary/config.json")
    guard let data = try? Data(contentsOf: configURL),
      let json = try? JSONSerialization.jsonObject(with: data) as? [String: String],
      let model = json["model"], let features = json["features"], let out = json["out"]
    else {
      return nil
    }
    return BoundaryConfig(
      model: model, features: features, out: out,
      scoreInput: json["scoreInput"], scoreOut: json["scoreOut"],
      noiseInput: json["noiseInput"], sampleOut: json["sampleOut"])
  }

  func testDumpTrunkOutput() throws {
    guard let config = boundaryConfig() else {
      throw XCTSkip("no boundary config; set BOLTZ_BOUNDARY_* or write config.json")
    }
    let artifact = try BoltzArtifact.load(from: URL(fileURLWithPath: config.model))
    guard let configuration = artifact.configuration else {
      throw XCTSkip("artifact is missing config.json")
    }
    let weights = BoltzWeightStore(artifact: artifact)
    let trunk = try weights.trunk()
    let conditioning = try weights.diffusionConditioning(configuration: configuration)
    let features = try FeatureBundle.load(from: URL(fileURLWithPath: config.features))
    let output = try trunk(features: features.arrays, recyclingSteps: 0)
    let cond = try conditioning(trunk: output, features: features.arrays)
    var arrays: [String: MLXArray] = [
      "sequence": output.sequence.asType(.float32),
      "pair": output.pair.asType(.float32),
      "cond_query": cond.query.asType(.float32),
      "cond_conditioning": cond.conditioning.asType(.float32),
      "cond_atomEncoderBias": cond.atomEncoderBias.asType(.float32),
      "cond_atomDecoderBias": cond.atomDecoderBias.asType(.float32),
      "cond_tokenTransformerBias": cond.tokenTransformerBias.asType(.float32),
    ]
    for value in arrays.values { MLX.eval(value) }
    try MLX.save(arrays: arrays, url: URL(fileURLWithPath: config.out))
    arrays.removeAll()
  }

  func testDumpScoreOutput() throws {
    guard let config = boundaryConfig() else {
      throw XCTSkip("no boundary config; write config.json")
    }
    guard let scoreInput = config.scoreInput, let scoreOut = config.scoreOut else {
      throw XCTSkip("scoreInput/scoreOut not configured")
    }
    let artifact = try BoltzArtifact.load(from: URL(fileURLWithPath: config.model))
    guard let configuration = artifact.configuration else {
      throw XCTSkip("artifact is missing config.json")
    }
    let weights = BoltzWeightStore(artifact: artifact)
    let trunk = try weights.trunk()
    let conditioning = try weights.diffusionConditioning(configuration: configuration)
    let scoreModel = try weights.diffusionScoreModel(configuration: configuration)
    let features = try FeatureBundle.load(from: URL(fileURLWithPath: config.features))
    let output = try trunk(features: features.arrays, recyclingSteps: 0)
    let cond = try conditioning(trunk: output, features: features.arrays)

    let inputs = try MLX.loadArrays(url: URL(fileURLWithPath: scoreInput))
    guard let rNoisy = inputs["r_noisy"], let times = inputs["times"] else {
      throw XCTSkip("score_io missing r_noisy/times")
    }
    let update = try scoreModel(
      noisyCoordinates: rNoisy.asType(.float32),
      noise: times.reshaped([1]).asType(.float32),
      trunk: output,
      features: features.arrays,
      conditioning: cond,
      multiplicity: 1
    )
    MLX.eval(update)
    try MLX.save(
      arrays: ["output": update.asType(.float32)],
      url: URL(fileURLWithPath: scoreOut))
  }

  /// Local (Mac) size-scaling benchmark. Reads "benchList" (comma-separated feature
  /// dirs) and "benchOut" from the boundary config, loads the model once, and runs
  /// each bundle with the MLX peak reset per protein. Writes a JSON array.
  func testBenchLocal() async throws {
    let configURL = FileManager.default.homeDirectoryForCurrentUser
      .appending(path: ".artifacts-boltz-boundary/config.json")
    guard let data = try? Data(contentsOf: configURL),
      let json = try? JSONSerialization.jsonObject(with: data) as? [String: String],
      let model = json["model"], let list = json["benchList"], let out = json["benchOut"]
    else { throw XCTSkip("no benchList/benchOut in config") }

    // Raise the provisional input cap so the size sweep can exercise large proteins.
    let planner = MemoryPlanner(
      limits: BoltzInputLimits(maximumTokens: 8192, maximumAtoms: 65536, maximumMSADepth: 8192),
      memoryLimit: 40 * 1_024 * 1_024 * 1_024,
      cacheLimit: 256 * 1_024 * 1_024)
    let predictor = try BoltzPredictor(
      modelDirectory: URL(fileURLWithPath: model), memoryPlanner: planner)
    var runs: [[String: Any]] = []
    for path in list.split(separator: ",").map(String.init) {
      await predictor.resetPeakMemory()
      let bundle = try FeatureBundle.load(from: URL(fileURLWithPath: path))
      let clock = ContinuousClock()
      let start = clock.now
      let result = try await predictor.predict(features: bundle)
      let elapsed = start.duration(to: clock.now)
      let seconds =
        Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
      let snapshot = await predictor.memorySnapshot()
      let entry: [String: Any] = [
        "name": URL(fileURLWithPath: path).lastPathComponent,
        "tokens": bundle.metadata.tokenCount,
        "atoms": result.coordinates.count,
        "elapsed_seconds": seconds,
        "mlx_peak_bytes": snapshot.peakMemory,
      ]
      runs.append(entry)
      print("BENCHLOCAL \(entry)")
    }
    let payload = try JSONSerialization.data(
      withJSONObject: runs.sorted { ($0["tokens"] as? Int ?? 0) < ($1["tokens"] as? Int ?? 0) },
      options: [.prettyPrinted, .sortedKeys])
    try payload.write(to: URL(fileURLWithPath: out))
  }

  func testSampleFixedNoise() throws {
    guard let config = boundaryConfig() else {
      throw XCTSkip("no boundary config; write config.json")
    }
    guard let noiseInput = config.noiseInput, let sampleOut = config.sampleOut else {
      throw XCTSkip("noiseInput/sampleOut not configured")
    }
    let artifact = try BoltzArtifact.load(from: URL(fileURLWithPath: config.model))
    guard let configuration = artifact.configuration else {
      throw XCTSkip("artifact is missing config.json")
    }
    let weights = BoltzWeightStore(artifact: artifact)
    let trunk = try weights.trunk()
    let conditioning = try weights.diffusionConditioning(configuration: configuration)
    let diffusion = try weights.atomDiffusion(configuration: configuration)
    let features = try FeatureBundle.load(from: URL(fileURLWithPath: config.features))
    let output = try trunk(features: features.arrays, recyclingSteps: 0)
    let cond = try conditioning(trunk: output, features: features.arrays)

    let noise = try MLX.loadArrays(url: URL(fileURLWithPath: noiseInput))
    guard let initNoise = noise["init"] else { throw XCTSkip("noise file missing init") }
    var stepNoises: [MLXArray] = []
    var index = 0
    while let step = noise["eps_\(index)"] {
      stepNoises.append(step.asType(.float32))
      index += 1
    }
    let steps = stepNoises.count
    let identity = MLX.identity(3).expandedDimensions(axis: 0)
    let zeroTranslation = MLX.zeros([1, 1, 3])
    let coordinates = try diffusion.sample(
      trunk: output,
      features: features.arrays,
      conditioning: cond,
      steps: steps,
      multiplicity: 1,
      seed: 0,
      initialNoise: initNoise.asType(.float32),
      stepNoises: stepNoises,
      rotations: Array(repeating: identity, count: steps),
      translations: Array(repeating: zeroTranslation, count: steps)
    )
    MLX.eval(coordinates)
    try MLX.save(
      arrays: ["coordinates": coordinates[0].asType(.float32)],
      url: URL(fileURLWithPath: sampleOut))
  }
}
