import Foundation
import MLX

/// A validated model artifact and all arrays declared by its manifest.
public struct BoltzArtifact {
  public let manifest: ArtifactManifest
  public let configuration: BoltzModelConfiguration?
  public let arrays: [String: MLXArray]

  public static func load(from directory: URL) throws -> BoltzArtifact {
    let loaded = try ArtifactIO.load(from: directory, expectedKind: .model)
    let configurationURL = directory.appending(path: "config.json")
    let configuration =
      FileManager.default.fileExists(atPath: configurationURL.path)
      ? try BoltzModelConfiguration.decode(from: configurationURL) : nil
    return BoltzArtifact(
      manifest: loaded.manifest,
      configuration: configuration,
      arrays: loaded.arrays
    )
  }
}

enum ArtifactIO {
  struct LoadedArtifact {
    let manifest: ArtifactManifest
    let arrays: [String: MLXArray]
  }

  static func readData(_ url: URL) throws -> Data {
    guard FileManager.default.fileExists(atPath: url.path) else {
      throw BoltzError.missingFile(url.path)
    }
    do {
      return try Data(contentsOf: url)
    } catch {
      throw BoltzError.invalidJSON(
        file: url.lastPathComponent,
        reason: error.localizedDescription
      )
    }
  }

  static func load(from directory: URL, expectedKind: ArtifactKind) throws -> LoadedArtifact {
    let manifest = try ArtifactManifest.decode(
      from: directory.appending(path: "manifest.json")
    )
    guard manifest.schemaVersion == ArtifactManifest.supportedSchemaVersion else {
      throw BoltzError.unsupportedSchema(
        found: manifest.schemaVersion,
        supported: ArtifactManifest.supportedSchemaVersion
      )
    }
    guard manifest.artifactKind == expectedKind else {
      throw BoltzError.wrongArtifactKind(
        expected: expectedKind.rawValue,
        found: manifest.artifactKind.rawValue
      )
    }

    var arrays: [String: MLXArray] = [:]
    let specsByShard = Dictionary(grouping: manifest.tensors, by: \.shard)
    for shard in specsByShard.keys.sorted() {
      let shardURL = directory.appending(path: shard)
      guard FileManager.default.fileExists(atPath: shardURL.path) else {
        throw BoltzError.missingFile(shardURL.path)
      }
      let loaded: [String: MLXArray]
      do {
        loaded = try MLX.loadArrays(url: shardURL)
      } catch {
        throw BoltzError.tensorLoadFailure(
          file: shard,
          reason: error.localizedDescription
        )
      }
      let expectedNames = Set(specsByShard[shard, default: []].map(\.name))
      let foundNames = Set(loaded.keys)
      guard expectedNames == foundNames else {
        throw BoltzError.tensorNameMismatch(
          shard: shard,
          missing: Array(expectedNames.subtracting(foundNames)).sorted(),
          unexpected: Array(foundNames.subtracting(expectedNames)).sorted()
        )
      }
      for spec in specsByShard[shard, default: []] {
        guard let array = loaded[spec.name] else {
          continue
        }
        guard array.shape == spec.shape else {
          throw BoltzError.tensorShapeMismatch(
            name: spec.name,
            expected: spec.shape,
            actual: array.shape
          )
        }
        let actualDType = dtypeName(array.dtype)
        guard actualDType == spec.dtype else {
          throw BoltzError.tensorDTypeMismatch(
            name: spec.name,
            expected: spec.dtype,
            actual: actualDType
          )
        }
        arrays[spec.name] = array
      }
    }
    return LoadedArtifact(manifest: manifest, arrays: arrays)
  }

  private static func dtypeName(_ dtype: DType) -> String {
    switch dtype {
    case .bool: "bool"
    case .uint8: "uint8"
    case .uint16: "uint16"
    case .uint32: "uint32"
    case .uint64: "uint64"
    case .int8: "int8"
    case .int16: "int16"
    case .int32: "int32"
    case .int64: "int64"
    case .float16: "float16"
    case .float32: "float32"
    case .bfloat16: "bfloat16"
    case .complex64: "complex64"
    case .float64: "float64"
    }
  }
}
