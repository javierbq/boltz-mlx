import MLX

/// One Boltz Pairformer block in inference order.
public struct PairformerLayer {
  public let sequenceNorm: BoltzLayerNorm
  public let sequenceAttention: AttentionPairBias
  public let outgoingMultiplication: TriangleMultiplication
  public let incomingMultiplication: TriangleMultiplication
  public let startingAttention: TriangleAttention
  public let endingAttention: TriangleAttention
  public let sequenceTransition: Transition
  public let pairTransition: Transition
  public let sequencePostNorm: BoltzLayerNorm?

  public init(
    sequenceNorm: BoltzLayerNorm,
    sequenceAttention: AttentionPairBias,
    outgoingMultiplication: TriangleMultiplication,
    incomingMultiplication: TriangleMultiplication,
    startingAttention: TriangleAttention,
    endingAttention: TriangleAttention,
    sequenceTransition: Transition,
    pairTransition: Transition,
    sequencePostNorm: BoltzLayerNorm?
  ) {
    self.sequenceNorm = sequenceNorm
    self.sequenceAttention = sequenceAttention
    self.outgoingMultiplication = outgoingMultiplication
    self.incomingMultiplication = incomingMultiplication
    self.startingAttention = startingAttention
    self.endingAttention = endingAttention
    self.sequenceTransition = sequenceTransition
    self.pairTransition = pairTransition
    self.sequencePostNorm = sequencePostNorm
  }

  public func callAsFunction(
    sequence: MLXArray,
    pair: MLXArray,
    mask: MLXArray,
    pairMask: MLXArray
  ) -> (sequence: MLXArray, pair: MLXArray) {
    var z = pair
    z = z + outgoingMultiplication(z, mask: pairMask)
    z = z + incomingMultiplication(z, mask: pairMask)
    z = z + startingAttention(z, mask: pairMask)
    z = z + endingAttention(z, mask: pairMask)
    z = z + pairTransition(z)

    var s = sequence.asType(.float32)
    let normalized = sequenceNorm(s)
    s =
      s
      + sequenceAttention(
        sequence: normalized,
        pair: z.asType(.float32),
        mask: mask.asType(.float32),
        keyInput: normalized
      )
    s = s + sequenceTransition(s)
    if let sequencePostNorm {
      s = sequencePostNorm(s)
    }
    return (s, z)
  }
}

/// The complete repeated Pairformer stack.
public struct Pairformer {
  public let layers: [PairformerLayer]

  public init(layers: [PairformerLayer]) {
    self.layers = layers
  }

  public func callAsFunction(
    sequence: MLXArray,
    pair: MLXArray,
    mask: MLXArray,
    pairMask: MLXArray
  ) -> (sequence: MLXArray, pair: MLXArray) {
    var s = sequence
    var z = pair
    for layer in layers {
      (s, z) = layer(sequence: s, pair: z, mask: mask, pairMask: pairMask)
    }
    return (s, z)
  }
}

extension BoltzWeightStore {
  func transition(_ prefix: String) throws -> Transition {
    try Transition(
      norm: layerNorm("\(prefix).norm"),
      first: linear("\(prefix).fc1"),
      gate: linear("\(prefix).fc2"),
      output: linear("\(prefix).fc3")
    )
  }

  func triangleMultiplication(
    _ prefix: String,
    direction: TriangleDirection
  ) throws -> TriangleMultiplication {
    try TriangleMultiplication(
      direction: direction,
      inputNorm: layerNorm("\(prefix).norm_in"),
      inputProjection: linear("\(prefix).p_in"),
      inputGate: linear("\(prefix).g_in"),
      outputNorm: layerNorm("\(prefix).norm_out"),
      outputProjection: linear("\(prefix).p_out"),
      outputGate: linear("\(prefix).g_out")
    )
  }

  func triangleAttention(
    _ prefix: String,
    startingNode: Bool,
    pairWidth: Int,
    headWidth: Int,
    headCount: Int
  ) throws -> TriangleAttention {
    try TriangleAttention(
      startingNode: startingNode,
      pairWidth: pairWidth,
      headWidth: headWidth,
      headCount: headCount,
      inputNorm: layerNorm("\(prefix).layer_norm"),
      triangleBias: linear("\(prefix).linear"),
      query: linear("\(prefix).mha.linear_q"),
      key: linear("\(prefix).mha.linear_k"),
      value: linear("\(prefix).mha.linear_v"),
      gate: linear("\(prefix).mha.linear_g"),
      output: linear("\(prefix).mha.linear_o")
    )
  }

  func attentionPairBias(
    _ prefix: String,
    sequenceWidth: Int,
    headCount: Int
  ) throws -> AttentionPairBias {
    try AttentionPairBias(
      sequenceWidth: sequenceWidth,
      headCount: headCount,
      query: linear("\(prefix).proj_q"),
      key: linear("\(prefix).proj_k"),
      value: linear("\(prefix).proj_v"),
      gate: linear("\(prefix).proj_g"),
      pairNorm: layerNorm("\(prefix).proj_z.0"),
      pairBias: linear("\(prefix).proj_z.1"),
      output: linear("\(prefix).proj_o")
    )
  }

  /// Load a pairformer stack. `prefix` selects WHICH stack: the trunk's 64-block
  /// `pairformer_module`, or the confidence head's 8-block `confidence_module.pairformer_stack`.
  /// The two are tensor-identical block for block — same 53 names and shapes — so the same loader
  /// serves both and only the prefix and block count differ.
  func pairformer(
    configuration: PairformerConfiguration,
    sequenceWidth: Int,
    pairWidth: Int,
    pairwiseHeadWidth: Int = 32,
    pairwiseHeadCount: Int = 4,
    prefix stackPrefix: String = "pairformer_module"
  ) throws -> Pairformer {
    var layers: [PairformerLayer] = []
    layers.reserveCapacity(configuration.numBlocks)
    for index in 0..<configuration.numBlocks {
      let prefix = "\(stackPrefix).layers.\(index)"
      layers.append(
        try PairformerLayer(
          sequenceNorm: layerNorm("\(prefix).pre_norm_s"),
          sequenceAttention: attentionPairBias(
            "\(prefix).attention",
            sequenceWidth: sequenceWidth,
            headCount: configuration.numHeads
          ),
          outgoingMultiplication: triangleMultiplication(
            "\(prefix).tri_mul_out",
            direction: .outgoing
          ),
          incomingMultiplication: triangleMultiplication(
            "\(prefix).tri_mul_in",
            direction: .incoming
          ),
          startingAttention: triangleAttention(
            "\(prefix).tri_att_start",
            startingNode: true,
            pairWidth: pairWidth,
            headWidth: pairwiseHeadWidth,
            headCount: pairwiseHeadCount
          ),
          endingAttention: triangleAttention(
            "\(prefix).tri_att_end",
            startingNode: false,
            pairWidth: pairWidth,
            headWidth: pairwiseHeadWidth,
            headCount: pairwiseHeadCount
          ),
          sequenceTransition: transition("\(prefix).transition_s"),
          pairTransition: transition("\(prefix).transition_z"),
          sequencePostNorm: configuration.postLayerNorm == true
            ? layerNorm("\(prefix).s_post_norm") : nil
        )
      )
    }
    return Pairformer(layers: layers)
  }
}
