import MLX
import XCTest

@testable import BoltzMLX

final class TrunkParityTests: XCTestCase {
  func testAtomWindowingMatchesBoltzIndexingMatrix() throws {
    let windowing = AtomWindowing(
      atomCount: 4,
      queryWindow: 2,
      keyWindow: 4
    )

    let keys = windowing.toKeys(MLXArray([Float(0), 1, 2, 3], [1, 4, 1]))

    MLX.eval(keys)
    XCTAssertEqual(keys.shape, [1, 2, 4, 1])
    XCTAssertEqual(keys.asArray(Float.self), [0, 0, 1, 2, 1, 2, 3, 0])
  }

  func testAdaptiveLayerNormAppliesSigmoidScale() throws {
    let layer = AdaptiveLayerNorm(
      conditioningNorm: BoltzLayerNorm(weight: MLX.ones([2]), bias: nil),
      scale: zeroLinear(2, 2),
      shift: zeroLinear(2, 2)
    )

    let output = layer(
      activation: MLXArray([Float(1), 3], [1, 2]),
      conditioning: MLX.zeros([1, 2])
    )

    MLX.eval(output)
    XCTAssertEqual(output.asArray(Float.self)[0], -0.5, accuracy: 1e-4)
    XCTAssertEqual(output.asArray(Float.self)[1], 0.5, accuracy: 1e-4)
  }

  func testTemplateDistanceBucketsMatchStrictBoundaryComparison() throws {
    let coordinates = MLXArray(
      [Float(0), 0, 0, 4, 0, 0],
      [1, 1, 2, 3]
    )

    let buckets = templateDistanceBuckets(
      coordinates,
      minimum: 3.25,
      maximum: 50.75,
      count: 3
    )

    MLX.eval(buckets)
    XCTAssertEqual(buckets.shape, [1, 1, 2, 2, 3])
    XCTAssertEqual(
      buckets.asArray(Float.self),
      [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]
    )
  }

  func testMSAInputFeaturesAppendDeletionAndPairingChannels() throws {
    let features: [String: MLXArray] = [
      "msa": MLXArray([0, 2], [1, 1, 2]),
      "has_deletion": MLXArray([Float(1), 0], [1, 1, 2]),
      "deletion_value": MLXArray([Float(0.5), 0.25], [1, 1, 2]),
      "msa_paired": MLXArray([Float(0), 1], [1, 1, 2]),
    ]

    let input = try msaInputFeatures(features, tokenClasses: 3, includePaired: true)

    MLX.eval(input)
    XCTAssertEqual(input.shape, [1, 1, 2, 6])
    XCTAssertEqual(
      input.asArray(Float.self),
      [1, 0, 0, 1, 0.5, 0, 0, 0, 1, 0, 0.25, 1]
    )
  }

  func testPairformerZeroUpdatesPreserveSequenceAndPairInputs() throws {
    let pairWidth = 2
    let sequenceWidth = 2
    let layer = PairformerLayer(
      sequenceNorm: norm(sequenceWidth),
      sequenceAttention: AttentionPairBias(
        sequenceWidth: sequenceWidth,
        headCount: 1,
        query: zeroLinear(sequenceWidth, sequenceWidth),
        key: zeroLinear(sequenceWidth, sequenceWidth),
        value: zeroLinear(sequenceWidth, sequenceWidth),
        gate: zeroLinear(sequenceWidth, sequenceWidth),
        pairNorm: norm(pairWidth),
        pairBias: zeroLinear(pairWidth, 1),
        output: zeroLinear(sequenceWidth, sequenceWidth)
      ),
      outgoingMultiplication: zeroTriangleMultiplication(pairWidth, .outgoing),
      incomingMultiplication: zeroTriangleMultiplication(pairWidth, .incoming),
      startingAttention: zeroTriangleAttention(pairWidth, starting: true),
      endingAttention: zeroTriangleAttention(pairWidth, starting: false),
      sequenceTransition: zeroTransition(sequenceWidth),
      pairTransition: zeroTransition(pairWidth),
      sequencePostNorm: nil
    )
    let sequence = MLXArray([Float(1), 2, 3, 4], [1, 2, sequenceWidth])
    let pair = MLXArray((1...8).map(Float.init), [1, 2, 2, pairWidth])

    let output = layer(
      sequence: sequence,
      pair: pair,
      mask: MLX.ones([1, 2]),
      pairMask: MLX.ones([1, 2, 2])
    )

    MLX.eval(output.sequence, output.pair)
    XCTAssertEqual(output.sequence.asArray(Float.self), sequence.asArray(Float.self))
    XCTAssertEqual(output.pair.asArray(Float.self), pair.asArray(Float.self))
  }

  func testTriangleAttentionCoreMasksWindowKeys() throws {
    let query = MLX.zeros([1, 1, 2, 1, 1])
    let key = MLX.zeros([1, 1, 2, 1, 1])
    let value = MLXArray([Float(2), 4], [1, 1, 2, 1, 1])
    let bias = MLX.zeros([1, 1, 1, 2, 2])

    let output = triangleAttentionCore(
      query: query,
      key: key,
      value: value,
      bias: bias,
      keyMask: MLXArray([Float(1), 0], [1, 1, 2]),
      headDimension: 1
    )

    MLX.eval(output)
    XCTAssertEqual(output.asArray(Float.self), [2, 2])
  }

  func testContactConditioningOverridesUnspecifiedAndUnselectedPairs() throws {
    let encoded = MLXArray([Float(1), 2, 3, 4], [1, 1, 2, 2])
    let flags = MLXArray(
      [
        Float(1), 0, 0, 0, 0,
        0, 1, 0, 0, 0,
      ],
      [1, 1, 2, 5]
    )

    let output = applyContactSelections(
      encoded: encoded,
      contactFlags: flags,
      unspecified: MLXArray([Float(10), 20]),
      unselected: MLXArray([Float(30), 40])
    )

    MLX.eval(output)
    XCTAssertEqual(output.asArray(Float.self), [10, 20, 30, 40])
  }

  func testRelativePositionFeaturesMatchBoltzBuckets() throws {
    let features: [String: MLXArray] = [
      "asym_id": MLXArray([0, 1], [1, 2]),
      "residue_index": MLXArray([5, 6], [1, 2]),
      "entity_id": MLXArray([3, 3], [1, 2]),
      "token_index": MLXArray([0, 0], [1, 2]),
      "sym_id": MLXArray([0, 1], [1, 2]),
      "cyclic_period": MLXArray([0, 0], [1, 2]),
    ]

    let encoded = try relativePositionFeatures(
      features,
      fixSymCheck: true,
      cyclicPositionEncoding: true
    )

    MLX.eval(encoded)
    XCTAssertEqual(encoded.shape, [1, 2, 2, 139])
    let crossChain = encoded[0, 0, 1]
    XCTAssertEqual(crossChain.sum().item(Float.self), 4, accuracy: 1e-6)
    XCTAssertEqual(crossChain[65].item(Float.self), 1, accuracy: 1e-6)
    XCTAssertEqual(crossChain[66 + 65].item(Float.self), 1, accuracy: 1e-6)
    XCTAssertEqual(crossChain[132].item(Float.self), 1, accuracy: 1e-6)
    XCTAssertEqual(crossChain[133 + 1].item(Float.self), 1, accuracy: 1e-6)
  }

  private func norm(_ width: Int) -> BoltzLayerNorm {
    BoltzLayerNorm(weight: MLX.ones([width]), bias: MLX.zeros([width]))
  }

  private func zeroLinear(_ inputWidth: Int, _ outputWidth: Int) -> AffineLinear {
    let physicalWidth = ((inputWidth + 63) / 64) * 64
    let (weight, scales, biases) = MLX.quantized(
      MLX.zeros([outputWidth, physicalWidth]),
      groupSize: 64,
      bits: 8,
      mode: .affine
    )
    return AffineLinear(
      weight: weight,
      scales: scales,
      quantizationBiases: biases,
      linearBias: nil,
      logicalInputWidth: inputWidth,
      physicalInputWidth: physicalWidth,
      groupSize: 64,
      bits: 8
    )
  }

  private func zeroTransition(_ width: Int) -> Transition {
    Transition(
      norm: norm(width),
      first: zeroLinear(width, width * 4),
      gate: zeroLinear(width, width * 4),
      output: zeroLinear(width * 4, width)
    )
  }

  private func zeroTriangleMultiplication(
    _ width: Int,
    _ direction: TriangleDirection
  ) -> TriangleMultiplication {
    TriangleMultiplication(
      direction: direction,
      inputNorm: norm(width),
      inputProjection: zeroLinear(width, width * 2),
      inputGate: zeroLinear(width, width * 2),
      outputNorm: norm(width),
      outputProjection: zeroLinear(width, width),
      outputGate: zeroLinear(width, width)
    )
  }

  private func zeroTriangleAttention(_ width: Int, starting: Bool) -> TriangleAttention {
    TriangleAttention(
      startingNode: starting,
      pairWidth: width,
      headWidth: 1,
      headCount: width,
      inputNorm: norm(width),
      triangleBias: zeroLinear(width, width),
      query: zeroLinear(width, width),
      key: zeroLinear(width, width),
      value: zeroLinear(width, width),
      gate: zeroLinear(width, width),
      output: zeroLinear(width, width)
    )
  }
}
