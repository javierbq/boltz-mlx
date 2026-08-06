import Foundation

/// Typed failures produced before or during native Boltz inference.
public enum BoltzError: Error, Equatable, LocalizedError, Sendable {
  case missingFile(String)
  case invalidJSON(file: String, reason: String)
  case unsupportedSchema(found: Int, supported: Int)
  case wrongArtifactKind(expected: String, found: String)
  case tensorNameMismatch(shard: String, missing: [String], unexpected: [String])
  case missingTensor(String)
  case tensorShapeMismatch(name: String, expected: [Int], actual: [Int])
  case tensorDTypeMismatch(name: String, expected: String, actual: String)
  case tensorLoadFailure(file: String, reason: String)
  case inputLimitExceeded(dimension: String, found: Int, maximum: Int)

  public var errorDescription: String? {
    switch self {
    case .missingFile(let path):
      "Required artifact file is missing: \(path)"
    case .invalidJSON(let file, let reason):
      "Invalid JSON in \(file): \(reason)"
    case .unsupportedSchema(let found, let supported):
      "Unsupported schema version \(found); this runtime supports \(supported)"
    case .wrongArtifactKind(let expected, let found):
      "Expected a \(expected) artifact, found \(found)"
    case .tensorNameMismatch(let shard, let missing, let unexpected):
      "Tensor names in \(shard) differ from the manifest; missing=\(missing), unexpected=\(unexpected)"
    case .missingTensor(let name):
      "Required model tensor is missing: \(name)"
    case .tensorShapeMismatch(let name, let expected, let actual):
      "Tensor \(name) has shape \(actual), expected \(expected)"
    case .tensorDTypeMismatch(let name, let expected, let actual):
      "Tensor \(name) has dtype \(actual), expected \(expected)"
    case .tensorLoadFailure(let file, let reason):
      "Unable to load tensors from \(file): \(reason)"
    case .inputLimitExceeded(let dimension, let found, let maximum):
      "Input \(dimension) \(found) exceeds the supported maximum \(maximum)"
    }
  }
}
