import MLX

/// Atom-aware Boltz-2 token input embedding.
public struct InputEmbedder {
  public let atomEncoder: AtomEncoder
  public let atomPairNorm: BoltzLayerNorm
  public let atomPairBias: AffineLinear
  public let atomAttention: AtomAttentionEncoder
  public let residueType: AffineLinear
  public let msaProfile: AffineLinear
  public let method: AffineEmbedding?
  public let modified: AffineEmbedding?
  public let cyclic: AffineLinear?
  public let moleculeType: AffineEmbedding?

  public func callAsFunction(_ features: [String: MLXArray]) throws -> MLXArray {
    let atom = try atomEncoder(features: features)
    let atomBias = atomPairBias(atomPairNorm(atom.pair))
    let attended = try atomAttention(
      features: features,
      query: atom.query,
      conditioning: atom.conditioning,
      bias: atomBias,
      windowing: atom.windowing
    )
    let profile = try requireFeature("profile", from: features)
    let deletionMean = try requireFeature("deletion_mean", from: features)
      .expandedDimensions(axis: -1)
    var sequence =
      attended.token
      + residueType(try requireFeature("res_type", from: features).asType(.float32))
      + msaProfile(MLX.concatenated([profile, deletionMean], axis: -1))
    if let method {
      sequence = sequence + method(try requireFeature("method_feature", from: features))
    }
    if let modified {
      sequence = sequence + modified(try requireFeature("modified", from: features))
    }
    if let cyclic {
      let period = MLX.minimum(
        try requireFeature("cyclic_period", from: features).asType(.float32),
        1
      ).expandedDimensions(axis: -1)
      sequence = sequence + cyclic(period)
    }
    if let moleculeType {
      sequence = sequence + moleculeType(try requireFeature("mol_type", from: features))
    }
    return sequence
  }
}

extension BoltzWeightStore {
  func inputEmbedder(configuration: BoltzModelConfiguration) throws -> InputEmbedder {
    let prefix = "input_embedder"
    return try InputEmbedder(
      atomEncoder: atomEncoder(
        "\(prefix).atom_encoder",
        queryWindow: configuration.atomsPerWindowQueries,
        keyWindow: configuration.atomsPerWindowKeys,
        structurePrediction: false
      ),
      atomPairNorm: layerNorm("\(prefix).atom_enc_proj_z.0"),
      atomPairBias: linear("\(prefix).atom_enc_proj_z.1"),
      atomAttention: atomAttentionEncoder(
        "\(prefix).atom_attention_encoder",
        depth: configuration.embedder.atomEncoderDepth,
        headCount: configuration.embedder.atomEncoderHeads,
        atomWidth: configuration.atomS,
        queryWindow: configuration.atomsPerWindowQueries,
        keyWindow: configuration.atomsPerWindowKeys,
        structurePrediction: false
      ),
      residueType: linear("\(prefix).res_type_encoding"),
      msaProfile: linear("\(prefix).msa_profile_encoding"),
      method: configuration.embedder.addMethodConditioning == true
        ? embedding("\(prefix).method_conditioning_init") : nil,
      modified: configuration.embedder.addModifiedFlag == true
        ? embedding("\(prefix).modified_conditioning_init") : nil,
      cyclic: configuration.embedder.addCyclicFlag == true
        ? linear("\(prefix).cyclic_conditioning_init") : nil,
      moleculeType: configuration.embedder.addMolTypeFeat == true
        ? embedding("\(prefix).mol_type_conditioning_init") : nil
    )
  }
}
