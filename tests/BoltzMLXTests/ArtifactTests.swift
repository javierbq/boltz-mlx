import Foundation
import MLX
import XCTest

@testable import BoltzMLX

final class ArtifactTests: XCTestCase {
  func testRejectsUnknownSchemaBeforeLoadingTensors() throws {
    let directory = try makeDirectory()
    try write(
      """
      {
        "artifact_kind": "model",
        "schema_version": 99,
        "source_checkpoint_sha256": null,
        "source_commit": "commit",
        "source_revision": "v2.2.1",
        "tensors": []
      }
      """,
      to: directory.appending(path: "manifest.json")
    )

    XCTAssertThrowsError(try BoltzArtifact.load(from: directory)) { error in
      XCTAssertEqual(
        error as? BoltzError,
        .unsupportedSchema(found: 99, supported: 1)
      )
    }
  }

  func testLoadsAValidModelArtifact() throws {
    let directory = try makeDirectory()
    try MLX.save(
      arrays: ["trunk.s_init.weight": MLX.ones([2, 2])],
      url: directory.appending(path: "model.safetensors")
    )
    try write(
      manifest(
        kind: "model",
        tensors: [
          tensor(
            name: "trunk.s_init.weight",
            shape: [2, 2],
            dtype: "float32",
            shard: "model.safetensors"
          )
        ]
      ),
      to: directory.appending(path: "manifest.json")
    )
    try write(
      modelConfiguration(),
      to: directory.appending(path: "config.json")
    )

    let artifact = try BoltzArtifact.load(from: directory)

    XCTAssertEqual(artifact.manifest.sourceRevision, "v2.2.1")
    XCTAssertEqual(artifact.arrays["trunk.s_init.weight"]?.shape, [2, 2])
    XCTAssertEqual(artifact.configuration?.tokenS, 384)
    XCTAssertEqual(artifact.configuration?.pairformer.numBlocks, 64)
    XCTAssertEqual(artifact.configuration?.scoreModel.tokenTransformerDepth, 24)
  }

  func testRejectsTensorShapeMismatch() throws {
    let directory = try makeDirectory()
    try MLX.save(
      arrays: ["weight": MLX.ones([2, 3])],
      url: directory.appending(path: "model.safetensors")
    )
    try write(
      manifest(
        kind: "model",
        tensors: [
          tensor(
            name: "weight",
            shape: [2, 2],
            dtype: "float32",
            shard: "model.safetensors"
          )
        ]
      ),
      to: directory.appending(path: "manifest.json")
    )

    XCTAssertThrowsError(try BoltzArtifact.load(from: directory)) { error in
      XCTAssertEqual(
        error as? BoltzError,
        .tensorShapeMismatch(
          name: "weight",
          expected: [2, 2],
          actual: [2, 3]
        )
      )
    }
  }

  func testFeatureBundleLoadsMetadataAndArrays() throws {
    let directory = try makeDirectory()
    try MLX.save(
      arrays: ["token_pad_mask": MLX.ones([1, 2], type: Bool.self)],
      url: directory.appending(path: "features.safetensors")
    )
    try write(
      manifest(
        kind: "features",
        tensors: [
          tensor(
            name: "token_pad_mask",
            shape: [1, 2],
            dtype: "bool",
            shard: "features.safetensors"
          )
        ]
      ),
      to: directory.appending(path: "manifest.json")
    )
    try write(
      """
      {
        "atom_count": 4,
        "msa_depth": 3,
        "sample_id": "tiny",
        "schema_version": 1,
        "token_count": 2
      }
      """,
      to: directory.appending(path: "metadata.json")
    )

    let bundle = try FeatureBundle.load(from: directory)

    XCTAssertEqual(bundle.metadata.sampleID, "tiny")
    XCTAssertEqual(bundle.metadata.tokenCount, 2)
    XCTAssertEqual(bundle.arrays["token_pad_mask"]?.shape, [1, 2])
  }

  private func makeDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory
      .appending(path: UUID().uuidString, directoryHint: .isDirectory)
    try FileManager.default.createDirectory(
      at: directory,
      withIntermediateDirectories: true
    )
    addTeardownBlock {
      try? FileManager.default.removeItem(at: directory)
    }
    return directory
  }

  private func write(_ value: String, to url: URL) throws {
    try value.data(using: .utf8)!.write(to: url)
  }

  private func manifest(kind: String, tensors: [String]) -> String {
    """
    {
      "artifact_kind": "\(kind)",
      "quantization": null,
      "schema_version": 1,
      "source_checkpoint_sha256": null,
      "source_commit": "cb04aeccdd480fd4db707f0bbafde538397fa2ac",
      "source_revision": "v2.2.1",
      "tensors": [\(tensors.joined(separator: ","))]
    }
    """
  }

  private func tensor(
    name: String,
    shape: [Int],
    dtype: String,
    shard: String
  ) -> String {
    """
    {
      "dtype": "\(dtype)",
      "logical_shape": null,
      "name": "\(name)",
      "physical_shape": null,
      "shape": \(shape),
      "shard": "\(shard)"
    }
    """
  }

  private func modelConfiguration() -> String {
    """
    {
      "atom_feature_dim": 388,
      "atom_s": 128,
      "atom_z": 16,
      "atoms_per_window_keys": 128,
      "atoms_per_window_queries": 32,
      "bond_type_feature": true,
      "conditioning_cutoff_max": 20.0,
      "conditioning_cutoff_min": 4.0,
      "cyclic_pos_enc": true,
      "diffusion_process_args": {
        "gamma_0": 0.8,
        "gamma_min": 1.0,
        "noise_scale": 1.003,
        "num_sampling_steps": 5,
        "rho": 7.0,
        "sigma_data": 16.0,
        "sigma_max": 160.0,
        "sigma_min": 0.0001,
        "step_scale": 1.5
      },
      "embedder_args": {
        "add_cyclic_flag": true,
        "add_method_conditioning": true,
        "add_modified_flag": true,
        "add_mol_type_feat": true,
        "atom_encoder_depth": 3,
        "atom_encoder_heads": 4
      },
      "fix_sym_check": true,
      "msa_args": {
        "msa_blocks": 4,
        "msa_s": 64,
        "pairwise_head_width": 32,
        "pairwise_num_heads": 4,
        "use_paired_feature": true
      },
      "num_bins": 64,
      "pairformer_args": {"num_blocks": 64, "num_heads": 16, "v2": true},
      "schema_version": 1,
      "score_model_args": {
        "atom_decoder_depth": 3,
        "atom_decoder_heads": 4,
        "atom_encoder_depth": 3,
        "atom_encoder_heads": 4,
        "conditioning_transition_layers": 2,
        "dim_fourier": 256,
        "token_transformer_depth": 24,
        "token_transformer_heads": 16
      },
      "source_commit": "cb04aeccdd480fd4db707f0bbafde538397fa2ac",
      "source_revision": "v2.2.1",
      "template_args": {"template_blocks": 2, "template_dim": 64},
      "token_s": 384,
      "token_z": 128,
      "use_templates": true,
      "use_templates_v2": true
    }
    """
  }
}
