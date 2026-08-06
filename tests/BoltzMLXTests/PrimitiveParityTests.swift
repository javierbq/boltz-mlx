import MLX
import XCTest

@testable import BoltzMLX

final class PrimitiveParityTests: XCTestCase {
  func testAffineEmbeddingDequantizesOnlySelectedRows() throws {
    let physicalWeight = MLXArray(
      (0..<128).map(Float.init),
      [2, 64]
    )
    let (weight, scales, biases) = MLX.quantized(
      physicalWeight,
      groupSize: 64,
      bits: 8,
      mode: .affine
    )
    let embedding = AffineEmbedding(
      weight: weight,
      scales: scales,
      quantizationBiases: biases,
      logicalOutputWidth: 3,
      groupSize: 64,
      bits: 8
    )

    let output = embedding(MLXArray([1, 0], [2]))

    assertClose(output, [64, 65, 66, 0, 1, 2], absoluteTolerance: 0.3)
  }

  func testAffineLinearPadsAOneChannelInput() throws {
    var values = Array(repeating: Float(0), count: 2 * 64)
    values[0] = 2
    values[64] = 3
    let physicalWeight = MLXArray(values, [2, 64])
    let (weight, scales, biases) = MLX.quantized(
      physicalWeight,
      groupSize: 64,
      bits: 8,
      mode: .affine
    )
    let layer = AffineLinear(
      weight: weight,
      scales: scales,
      quantizationBiases: biases,
      linearBias: array([0.5, -0.5]),
      logicalInputWidth: 1,
      physicalInputWidth: 64,
      groupSize: 64,
      bits: 8
    )

    let output = layer(array([2.0], shape: [1, 1]))

    assertClose(output, [4.5, 5.5], absoluteTolerance: 0.03)
  }

  func testBoltzLayerNormMatchesPyTorchVarianceConvention() throws {
    let layer = BoltzLayerNorm(
      weight: array([2.0, 4.0]),
      bias: array([1.0, -1.0]),
      epsilon: 1e-5
    )

    let output = layer(array([1.0, 3.0], shape: [1, 2]))

    assertClose(output, [-0.99999, 2.99998], absoluteTolerance: 1e-4)
  }

  func testAttentionCoreAppliesMaskAndAveragesVisibleValues() throws {
    let query = MLX.zeros([1, 2, 1, 1])
    let key = MLX.zeros([1, 2, 1, 1])
    let value = array([2.0, 4.0], shape: [1, 2, 1, 1])
    let bias = MLX.zeros([1, 1, 2, 2])

    let bothVisible = attentionWithPairBias(
      query: query,
      key: key,
      value: value,
      bias: bias,
      keyMask: array([1.0, 1.0], shape: [1, 2]),
      headDimension: 1
    )
    let secondHidden = attentionWithPairBias(
      query: query,
      key: key,
      value: value,
      bias: bias,
      keyMask: array([1.0, 0.0], shape: [1, 2]),
      headDimension: 1
    )

    assertClose(bothVisible, [3, 3])
    assertClose(secondHidden, [2, 2])
  }

  func testTriangleProjectionMatchesPyTorchEinsums() throws {
    let values = array([1.0, 2.0, 3.0, 4.0], shape: [1, 2, 2, 1])

    let outgoing = triangleProjection(values, values, direction: .outgoing)
    let incoming = triangleProjection(values, values, direction: .incoming)

    assertClose(outgoing, [5, 11, 11, 25])
    assertClose(incoming, [10, 14, 14, 20])
  }

  func testOuterProductMeanUsesPairwiseSequenceCount() throws {
    let values = array([1.0, 2.0, 3.0, 4.0], shape: [1, 2, 2, 1])
    let mask = MLX.ones([1, 2, 2])

    let output = outerProductMean(values, values, mask: mask)

    assertClose(output, [5, 7, 7, 10])
  }

  func testTriangleAttentionProjectsHeadWidthBackToPairWidth() throws {
    // Regression: the template pair stack has pairWidth (token_z) = 64 while
    // headCount * headWidth = 128. The gated attention must be reshaped to
    // headCount * headWidth before the output projection maps it back to
    // pairWidth. Reshaping straight to pairWidth crashed with a size mismatch
    // (`Cannot reshape array of size ... into shape (1, N, N, 64)`).
    let pairWidth = 64
    let headCount = 4
    let headWidth = 32
    let hidden = headCount * headWidth  // 128 != pairWidth

    func affineLinear(inWidth: Int, outWidth: Int) -> AffineLinear {
      let values = (0..<(outWidth * inWidth)).map { Float($0 % 5) * 0.01 }
      let (weight, scales, biases) = MLX.quantized(
        MLXArray(values, [outWidth, inWidth]),
        groupSize: 64,
        bits: 8,
        mode: .affine
      )
      return AffineLinear(
        weight: weight,
        scales: scales,
        quantizationBiases: biases,
        linearBias: nil,
        logicalInputWidth: inWidth,
        physicalInputWidth: inWidth,
        groupSize: 64,
        bits: 8
      )
    }

    let attention = TriangleAttention(
      startingNode: true,
      pairWidth: pairWidth,
      headWidth: headWidth,
      headCount: headCount,
      inputNorm: BoltzLayerNorm(
        weight: MLX.ones([pairWidth]),
        bias: MLX.zeros([pairWidth]),
        epsilon: 1e-5
      ),
      triangleBias: affineLinear(inWidth: pairWidth, outWidth: headCount),
      query: affineLinear(inWidth: pairWidth, outWidth: hidden),
      key: affineLinear(inWidth: pairWidth, outWidth: hidden),
      value: affineLinear(inWidth: pairWidth, outWidth: hidden),
      gate: affineLinear(inWidth: pairWidth, outWidth: hidden),
      output: affineLinear(inWidth: hidden, outWidth: pairWidth)
    )

    let tokens = 3
    let result = attention(
      MLX.zeros([1, tokens, tokens, pairWidth]),
      mask: MLX.ones([1, tokens, tokens])
    )
    MLX.eval(result)

    XCTAssertEqual(result.shape, [1, tokens, tokens, pairWidth])
  }

  private func assertClose(
    _ array: MLXArray,
    _ expected: [Float],
    absoluteTolerance: Float = 1e-5,
    file: StaticString = #filePath,
    line: UInt = #line
  ) {
    MLX.eval(array)
    let actual = array.asArray(Float.self)
    XCTAssertEqual(actual.count, expected.count, file: file, line: line)
    for (actualValue, expectedValue) in zip(actual, expected) {
      XCTAssertEqual(
        actualValue,
        expectedValue,
        accuracy: absoluteTolerance,
        file: file,
        line: line
      )
    }
  }

  private func array(_ values: [Float], shape: [Int]? = nil) -> MLXArray {
    MLXArray(values, shape ?? [values.count])
  }
}
