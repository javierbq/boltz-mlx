import ArgumentParser
import BoltzMLX
import Foundation
import MLX

@available(macOS 10.15, macCatalyst 13, iOS 13, tvOS 13, watchOS 6, *)
@main
struct BoltzMLXCommand: AsyncParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "BoltzMLXCLI",
    abstract: "Run native int8 Boltz-2 structure inference with MLX.",
    subcommands: [Predict.self]
  )
}

@available(macOS 10.15, macCatalyst 13, iOS 13, tvOS 13, watchOS 6, *)
struct Predict: AsyncParsableCommand {
  static let configuration = CommandConfiguration(
    abstract: "Predict coordinates from a precomputed Boltz feature bundle."
  )

  @Option(help: "Directory containing model.safetensors, manifest.json, and config.json.")
  var model: String

  @Option(help: "Directory containing precomputed feature artifacts.")
  var features: String

  @Option(help: "Destination SafeTensors path.")
  var output: String

  @Option(name: .customLong("recycling-steps"), help: "Pairformer recycling count.")
  var recyclingSteps = 0

  @Option(name: .customLong("diffusion-steps"), help: "Diffusion sampling steps.")
  var diffusionSteps = 20

  @Option(help: "MLX random seed.")
  var seed: UInt64 = 0

  mutating func run() async throws {
    do {
      let modelURL = URL(filePath: model, directoryHint: .isDirectory)
      let featureURL = URL(filePath: features, directoryHint: .isDirectory)
      let outputURL = URL(filePath: output)
      let predictor = try BoltzPredictor(modelDirectory: modelURL)
      let bundle = try FeatureBundle.load(from: featureURL)
      let structure = try await predictor.predict(
        features: bundle,
        options: BoltzPredictionOptions(
          recyclingSteps: recyclingSteps,
          diffusionSteps: diffusionSteps,
          seed: seed
        )
      )
      try FileManager.default.createDirectory(
        at: outputURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
      )
      let flattened = structure.coordinates.flatMap { [$0.x, $0.y, $0.z] }
      try MLX.save(
        arrays: [
          "coordinates": MLXArray(flattened, [structure.coordinates.count, 3]),
          "atom_mask": MLXArray(structure.atomMask),
        ],
        url: outputURL
      )
      let metadata = PredictionMetadata(
        atomCount: structure.coordinates.count,
        diffusionSteps: diffusionSteps,
        recyclingSteps: recyclingSteps,
        seed: seed
      )
      let metadataURL = outputURL.appendingPathExtension("json")
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      try (encoder.encode(metadata) + Data([0x0A])).write(to: metadataURL)
      print("Wrote \(structure.coordinates.count) atoms to \(outputURL.path)")
    } catch let error as BoltzError {
      writeError(error.localizedDescription)
      throw boltzExitCode(for: error)
    } catch {
      writeError(error.localizedDescription)
      throw ExitCode(5)
    }
  }
}

private struct PredictionMetadata: Codable {
  let atomCount: Int
  let diffusionSteps: Int
  let recyclingSteps: Int
  let seed: UInt64
}

private func boltzExitCode(for error: BoltzError) -> ExitCode {
  switch error {
  case .unsupportedSchema, .wrongArtifactKind, .tensorNameMismatch,
    .missingTensor, .tensorShapeMismatch, .tensorDTypeMismatch,
    .tensorLoadFailure, .missingFile, .invalidJSON:
    ExitCode(3)
  case .inputLimitExceeded:
    ExitCode(4)
  }
}

private func writeError(_ message: String) {
  FileHandle.standardError.write(Data("Error: \(message)\n".utf8))
}
