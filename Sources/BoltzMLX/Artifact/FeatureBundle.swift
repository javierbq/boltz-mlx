import Foundation
import MLX

/// A validated precomputed feature artifact for one complex.
public struct FeatureBundle: @unchecked Sendable {
  public let manifest: ArtifactManifest
  public let metadata: FeatureMetadata
  public let arrays: [String: MLXArray]

  public static func load(from directory: URL) throws -> FeatureBundle {
    let loaded = try ArtifactIO.load(from: directory, expectedKind: .features)
    let metadata = try FeatureMetadata.decode(
      from: directory.appending(path: "metadata.json")
    )
    guard metadata.schemaVersion == ArtifactManifest.supportedSchemaVersion else {
      throw BoltzError.unsupportedSchema(
        found: metadata.schemaVersion,
        supported: ArtifactManifest.supportedSchemaVersion
      )
    }
    return FeatureBundle(
      manifest: loaded.manifest,
      metadata: metadata,
      arrays: loaded.arrays
    )
  }
}
