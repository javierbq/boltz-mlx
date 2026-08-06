import Foundation

/// Input-embedding architecture values saved with the production checkpoint.
public struct EmbedderConfiguration: Codable, Sendable, Equatable {
  public let atomEncoderDepth: Int
  public let atomEncoderHeads: Int
  public let addMethodConditioning: Bool?
  public let addModifiedFlag: Bool?
  public let addCyclicFlag: Bool?
  public let addMolTypeFeat: Bool?

  private enum CodingKeys: String, CodingKey {
    case atomEncoderDepth = "atom_encoder_depth"
    case atomEncoderHeads = "atom_encoder_heads"
    case addMethodConditioning = "add_method_conditioning"
    case addModifiedFlag = "add_modified_flag"
    case addCyclicFlag = "add_cyclic_flag"
    case addMolTypeFeat = "add_mol_type_feat"
  }
}

/// MSA stack dimensions and attention layout.
public struct MSAConfiguration: Codable, Sendable, Equatable {
  public let msaS: Int
  public let msaBlocks: Int
  public let pairwiseHeadWidth: Int
  public let pairwiseNumHeads: Int
  public let usePairedFeature: Bool

  private enum CodingKeys: String, CodingKey {
    case msaS = "msa_s"
    case msaBlocks = "msa_blocks"
    case pairwiseHeadWidth = "pairwise_head_width"
    case pairwiseNumHeads = "pairwise_num_heads"
    case usePairedFeature = "use_paired_feature"
  }
}

/// Pairformer stack dimensions.
/// Confidence-head architecture, from the checkpoint's `confidence_model_args`.
///
/// Every flag here changes which tensors exist or how they combine, so none of it can be defaulted:
/// see ConfidenceModule for what each one does. Absent when the pack has no confidence head.
public struct ConfidenceConfiguration: Codable, Sendable, Equatable {
  public let numDistBins: Int
  public let maxDist: Int
  public let addSToZProd: Bool
  public let addSInputToS: Bool
  public let addZInputToZ: Bool
  public let useSDiffusion: Bool
  public let pairformer: PairformerConfiguration
  public let confidenceArgs: ConfidenceHeadConfiguration

  public var numPAEBins: Int { confidenceArgs.numPAEBins }
  public var useSeparateHeads: Bool { confidenceArgs.useSeparateHeads }

  private enum CodingKeys: String, CodingKey {
    case numDistBins = "num_dist_bins"
    case maxDist = "max_dist"
    case addSToZProd = "add_s_to_z_prod"
    case addSInputToS = "add_s_input_to_s"
    case addZInputToZ = "add_z_input_to_z"
    case useSDiffusion = "use_s_diffusion"
    case pairformer = "pairformer_args"
    case confidenceArgs = "confidence_args"
  }
}

public struct ConfidenceHeadConfiguration: Codable, Sendable, Equatable {
  public let numPAEBins: Int
  public let numPDEBins: Int
  public let numPLDDTBins: Int
  public let useSeparateHeads: Bool

  private enum CodingKeys: String, CodingKey {
    case numPAEBins = "num_pae_bins"
    case numPDEBins = "num_pde_bins"
    case numPLDDTBins = "num_plddt_bins"
    case useSeparateHeads = "use_separate_heads"
  }
}

public struct PairformerConfiguration: Codable, Sendable, Equatable {
  public let numBlocks: Int
  public let numHeads: Int
  public let v2: Bool?
  public let postLayerNorm: Bool?

  private enum CodingKeys: String, CodingKey {
    case numBlocks = "num_blocks"
    case numHeads = "num_heads"
    case v2
    case postLayerNorm = "post_layer_norm"
  }
}

/// Structure score-network architecture.
public struct ScoreModelConfiguration: Codable, Sendable, Equatable {
  public let dimFourier: Int
  public let atomEncoderDepth: Int
  public let atomEncoderHeads: Int
  public let tokenTransformerDepth: Int
  public let tokenTransformerHeads: Int
  public let atomDecoderDepth: Int
  public let atomDecoderHeads: Int
  public let conditioningTransitionLayers: Int

  private enum CodingKeys: String, CodingKey {
    case dimFourier = "dim_fourier"
    case atomEncoderDepth = "atom_encoder_depth"
    case atomEncoderHeads = "atom_encoder_heads"
    case tokenTransformerDepth = "token_transformer_depth"
    case tokenTransformerHeads = "token_transformer_heads"
    case atomDecoderDepth = "atom_decoder_depth"
    case atomDecoderHeads = "atom_decoder_heads"
    case conditioningTransitionLayers = "conditioning_transition_layers"
  }
}

/// Inference-time diffusion schedule.
public struct DiffusionProcessConfiguration: Codable, Sendable, Equatable {
  public let numSamplingSteps: Int?
  public let sigmaMin: Float
  public let sigmaMax: Float
  public let sigmaData: Float
  public let rho: Float
  public let gamma0: Float
  public let gammaMin: Float
  public let noiseScale: Float
  public let stepScale: Float
  public let coordinateAugmentation: Bool?
  public let alignmentReverseDiff: Bool?

  private enum CodingKeys: String, CodingKey {
    case numSamplingSteps = "num_sampling_steps"
    case sigmaMin = "sigma_min"
    case sigmaMax = "sigma_max"
    case sigmaData = "sigma_data"
    case rho
    case gamma0 = "gamma_0"
    case gammaMin = "gamma_min"
    case noiseScale = "noise_scale"
    case stepScale = "step_scale"
    case coordinateAugmentation = "coordinate_augmentation"
    case alignmentReverseDiff = "alignment_reverse_diff"
  }
}

/// Optional template stack dimensions.
public struct TemplateConfiguration: Codable, Sendable, Equatable {
  public let templateDim: Int
  public let templateBlocks: Int
  public let pairwiseHeadWidth: Int?
  public let pairwiseNumHeads: Int?

  private enum CodingKeys: String, CodingKey {
    case templateDim = "template_dim"
    case templateBlocks = "template_blocks"
    case pairwiseHeadWidth = "pairwise_head_width"
    case pairwiseNumHeads = "pairwise_num_heads"
  }
}

/// Versioned architecture contract used to construct the native Swift graph.
public struct BoltzModelConfiguration: Codable, Sendable, Equatable {
  public static let supportedSchemaVersion = 1

  public let schemaVersion: Int
  public let sourceRevision: String
  public let sourceCommit: String
  public let atomS: Int
  public let atomZ: Int
  public let tokenS: Int
  public let tokenZ: Int
  public let numBins: Int
  public let atomFeatureDim: Int
  public let atomsPerWindowQueries: Int
  public let atomsPerWindowKeys: Int
  public let fixSymCheck: Bool
  public let cyclicPosEnc: Bool
  public let bondTypeFeature: Bool
  public let useTemplates: Bool
  public let useTemplatesV2: Bool
  public let conditioningCutoffMin: Float
  public let conditioningCutoffMax: Float
  public let embedder: EmbedderConfiguration
  public let msa: MSAConfiguration
  public let pairformer: PairformerConfiguration
  /// nil for a structure-only pack, which is still valid — the caller falls back to no scoring.
  public let confidence: ConfidenceConfiguration?
  public let scoreModel: ScoreModelConfiguration
  public let diffusionProcess: DiffusionProcessConfiguration
  public let template: TemplateConfiguration?

  private enum CodingKeys: String, CodingKey {
    case schemaVersion = "schema_version"
    case sourceRevision = "source_revision"
    case sourceCommit = "source_commit"
    case atomS = "atom_s"
    case atomZ = "atom_z"
    case tokenS = "token_s"
    case tokenZ = "token_z"
    case numBins = "num_bins"
    case atomFeatureDim = "atom_feature_dim"
    case atomsPerWindowQueries = "atoms_per_window_queries"
    case atomsPerWindowKeys = "atoms_per_window_keys"
    case fixSymCheck = "fix_sym_check"
    case cyclicPosEnc = "cyclic_pos_enc"
    case bondTypeFeature = "bond_type_feature"
    case useTemplates = "use_templates"
    case useTemplatesV2 = "use_templates_v2"
    case conditioningCutoffMin = "conditioning_cutoff_min"
    case conditioningCutoffMax = "conditioning_cutoff_max"
    case embedder = "embedder_args"
    case msa = "msa_args"
    case pairformer = "pairformer_args"
    case confidence = "confidence_model_args"
    case scoreModel = "score_model_args"
    case diffusionProcess = "diffusion_process_args"
    case template = "template_args"
  }

  static func decode(from url: URL) throws -> BoltzModelConfiguration {
    let data = try ArtifactIO.readData(url)
    let decoder = JSONDecoder()
    do {
      let configuration = try decoder.decode(BoltzModelConfiguration.self, from: data)
      guard configuration.schemaVersion == supportedSchemaVersion else {
        throw BoltzError.unsupportedSchema(
          found: configuration.schemaVersion,
          supported: supportedSchemaVersion
        )
      }
      return configuration
    } catch let error as BoltzError {
      throw error
    } catch {
      throw BoltzError.invalidJSON(
        file: url.lastPathComponent,
        reason: String(describing: error)
      )
    }
  }
}
