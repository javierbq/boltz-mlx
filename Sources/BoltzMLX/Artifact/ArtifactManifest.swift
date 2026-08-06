import Foundation

/// Artifact category encoded by the offline Python exporter.
public enum ArtifactKind: String, Codable, Sendable {
  case model
  case features
  case fixture
}

/// Global affine quantization settings for a model artifact.
public struct QuantizationSpec: Codable, Sendable, Equatable {
  public let bits: Int
  public let groupSize: Int
  public let mode: String
}

/// Manifest declaration for one SafeTensors array.
public struct TensorSpec: Codable, Sendable, Equatable {
  public let name: String
  public let shape: [Int]
  public let dtype: String
  public let shard: String
  public let logicalShape: [Int]?
  public let physicalShape: [Int]?
}

/// Versioned contract generated alongside every model or feature bundle.
public struct ArtifactManifest: Codable, Sendable, Equatable {
  public static let supportedSchemaVersion = 1

  public let schemaVersion: Int
  public let artifactKind: ArtifactKind
  public let sourceRevision: String
  public let sourceCommit: String
  public let sourceCheckpointSha256: String?
  public let tensors: [TensorSpec]
  public let quantization: QuantizationSpec?

  static func decode(from url: URL) throws -> ArtifactManifest {
    let data = try ArtifactIO.readData(url)
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    do {
      return try decoder.decode(ArtifactManifest.self, from: data)
    } catch {
      throw BoltzError.invalidJSON(
        file: url.lastPathComponent,
        reason: error.localizedDescription
      )
    }
  }
}

/// Dimensions and identity needed before loading a feature tensor graph.
public struct FeatureMetadata: Codable, Sendable, Equatable {
  public let schemaVersion: Int
  public let sampleID: String
  public let tokenCount: Int
  public let atomCount: Int
  public let msaDepth: Int

  private enum CodingKeys: String, CodingKey {
    case schemaVersion = "schema_version"
    case sampleID = "sample_id"
    case tokenCount = "token_count"
    case atomCount = "atom_count"
    case msaDepth = "msa_depth"
  }

  static func decode(from url: URL) throws -> FeatureMetadata {
    let data = try ArtifactIO.readData(url)
    do {
      return try JSONDecoder().decode(FeatureMetadata.self, from: data)
    } catch {
      throw BoltzError.invalidJSON(
        file: url.lastPathComponent,
        reason: error.localizedDescription
      )
    }
  }
}
