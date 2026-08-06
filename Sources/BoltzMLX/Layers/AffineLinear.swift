import MLX

/// Row-wise lookup for an affine-int8 embedding matrix.
public struct AffineEmbedding {
  public let weight: MLXArray
  public let scales: MLXArray
  public let quantizationBiases: MLXArray?
  public let logicalOutputWidth: Int
  public let groupSize: Int
  public let bits: Int

  public init(
    weight: MLXArray,
    scales: MLXArray,
    quantizationBiases: MLXArray?,
    logicalOutputWidth: Int,
    groupSize: Int,
    bits: Int
  ) {
    self.weight = weight
    self.scales = scales
    self.quantizationBiases = quantizationBiases
    self.logicalOutputWidth = logicalOutputWidth
    self.groupSize = groupSize
    self.bits = bits
  }

  public func callAsFunction(_ indices: MLXArray) -> MLXArray {
    let flattened = indices.flattened()
    let selectedBiases = quantizationBiases.map { $0[flattened] }
    let rows = MLX.dequantized(
      weight[flattened],
      scales: scales[flattened],
      biases: selectedBiases,
      groupSize: groupSize,
      bits: bits,
      mode: .affine
    )
    let logicalRows = rows[.ellipsis, 0..<logicalOutputWidth]
    return logicalRows.reshaped(indices.shape + [logicalOutputWidth])
  }
}

/// An MLX affine-int8 matrix multiplication with optional logical-width padding.
public final class AffineLinear {
  public let weight: MLXArray
  public let scales: MLXArray
  public let quantizationBiases: MLXArray?
  public let linearBias: MLXArray?
  public let logicalInputWidth: Int
  public let physicalInputWidth: Int
  public let groupSize: Int
  public let bits: Int

  public init(
    weight: MLXArray,
    scales: MLXArray,
    quantizationBiases: MLXArray?,
    linearBias: MLXArray?,
    logicalInputWidth: Int,
    physicalInputWidth: Int,
    groupSize: Int,
    bits: Int
  ) {
    self.weight = weight
    self.scales = scales
    self.quantizationBiases = quantizationBiases
    self.linearBias = linearBias
    self.logicalInputWidth = logicalInputWidth
    self.physicalInputWidth = physicalInputWidth
    self.groupSize = groupSize
    self.bits = bits
  }

  public func callAsFunction(_ input: MLXArray) -> MLXArray {
    precondition(input.shape.last == logicalInputWidth)
    let padding = physicalInputWidth - logicalInputWidth
    let paddedInput: MLXArray
    if padding == 0 {
      paddedInput = input
    } else {
      var widths = Array(repeating: IntOrPair(0), count: input.ndim)
      widths[input.ndim - 1] = IntOrPair((0, padding))
      paddedInput = MLX.padded(input, widths: widths)
    }
    var output = MLX.quantizedMM(
      paddedInput,
      weight,
      scales: scales,
      biases: quantizationBiases,
      transpose: true,
      groupSize: groupSize,
      bits: bits,
      mode: .affine
    )
    if let linearBias {
      output = output + linearBias
    }
    return output
  }
}

/// Factory that resolves typed layers from a validated artifact.
public struct BoltzWeightStore {
  public let artifact: BoltzArtifact

  public init(artifact: BoltzArtifact) {
    self.artifact = artifact
  }

  public func linear(_ prefix: String) throws -> AffineLinear {
    let weightName = "\(prefix).weight"
    let scalesName = "\(prefix).scales"
    let quantizationBiasesName = "\(prefix).biases"
    guard let weight = artifact.arrays[weightName] else {
      throw BoltzError.missingTensor(weightName)
    }
    guard let scales = artifact.arrays[scalesName] else {
      throw BoltzError.missingTensor(scalesName)
    }
    guard let spec = artifact.manifest.tensors.first(where: { $0.name == weightName }) else {
      throw BoltzError.missingTensor(weightName)
    }
    guard let logicalShape = spec.logicalShape, let physicalShape = spec.physicalShape else {
      throw BoltzError.tensorShapeMismatch(
        name: weightName,
        expected: [2],
        actual: spec.shape
      )
    }
    guard let quantization = artifact.manifest.quantization else {
      throw BoltzError.tensorDTypeMismatch(
        name: weightName,
        expected: "affine-int8",
        actual: spec.dtype
      )
    }
    return AffineLinear(
      weight: weight,
      scales: scales,
      quantizationBiases: artifact.arrays[quantizationBiasesName],
      linearBias: artifact.arrays["\(prefix).bias"],
      logicalInputWidth: logicalShape[1],
      physicalInputWidth: physicalShape[1],
      groupSize: quantization.groupSize,
      bits: quantization.bits
    )
  }

  public func embedding(_ prefix: String) throws -> AffineEmbedding {
    let weightName = "\(prefix).weight"
    let scalesName = "\(prefix).scales"
    guard let weight = artifact.arrays[weightName] else {
      throw BoltzError.missingTensor(weightName)
    }
    guard let scales = artifact.arrays[scalesName] else {
      throw BoltzError.missingTensor(scalesName)
    }
    guard let spec = artifact.manifest.tensors.first(where: { $0.name == weightName }) else {
      throw BoltzError.missingTensor(weightName)
    }
    guard let logicalShape = spec.logicalShape else {
      throw BoltzError.tensorShapeMismatch(
        name: weightName,
        expected: [2],
        actual: spec.shape
      )
    }
    guard let quantization = artifact.manifest.quantization else {
      throw BoltzError.tensorDTypeMismatch(
        name: weightName,
        expected: "affine-int8",
        actual: spec.dtype
      )
    }
    return AffineEmbedding(
      weight: weight,
      scales: scales,
      quantizationBiases: artifact.arrays["\(prefix).biases"],
      logicalOutputWidth: logicalShape[1],
      groupSize: quantization.groupSize,
      bits: quantization.bits
    )
  }

  public func layerNorm(_ prefix: String, epsilon: Float = 1e-5) throws -> BoltzLayerNorm {
    let weightName = "\(prefix).weight"
    let biasName = "\(prefix).bias"
    guard let weight = artifact.arrays[weightName] else {
      throw BoltzError.missingTensor(weightName)
    }
    guard let bias = artifact.arrays[biasName] else {
      throw BoltzError.missingTensor(biasName)
    }
    return BoltzLayerNorm(weight: weight, bias: bias, epsilon: epsilon)
  }
}
