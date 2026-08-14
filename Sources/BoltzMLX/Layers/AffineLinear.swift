import MLX

/// How a matrix parameter is stored in the artifact it was loaded from.
///
/// An int8 pack quantizes each matrix affinely and pads its input width up to the
/// group size; a dense pack stores the matrix at its own float width, unpadded. The
/// two cannot be mixed within one artifact -- see the exporter's `Precision`.
enum MatrixStorage {
  case affineInt8(scales: MLXArray, quantizationBiases: MLXArray?, groupSize: Int, bits: Int)
  case dense
}

/// Row-wise lookup for an embedding matrix, affine-int8 or dense.
public struct AffineEmbedding {
  public let weight: MLXArray
  public let logicalOutputWidth: Int
  let storage: MatrixStorage

  public init(
    weight: MLXArray,
    scales: MLXArray,
    quantizationBiases: MLXArray?,
    logicalOutputWidth: Int,
    groupSize: Int,
    bits: Int
  ) {
    self.weight = weight
    self.logicalOutputWidth = logicalOutputWidth
    self.storage = .affineInt8(
      scales: scales,
      quantizationBiases: quantizationBiases,
      groupSize: groupSize,
      bits: bits
    )
  }

  /// An embedding whose rows are stored at full float width.
  public init(denseWeight: MLXArray) {
    self.weight = denseWeight
    self.logicalOutputWidth = denseWeight.shape[1]
    self.storage = .dense
  }

  public func callAsFunction(_ indices: MLXArray) -> MLXArray {
    let flattened = indices.flattened()
    let rows: MLXArray
    switch storage {
    case .affineInt8(let scales, let quantizationBiases, let groupSize, let bits):
      rows = MLX.dequantized(
        weight[flattened],
        scales: scales[flattened],
        biases: quantizationBiases.map { $0[flattened] },
        groupSize: groupSize,
        bits: bits,
        mode: .affine
      )
    case .dense:
      rows = weight[flattened]
    }
    let logicalRows = rows[.ellipsis, 0..<logicalOutputWidth]
    return logicalRows.reshaped(indices.shape + [logicalOutputWidth])
  }
}

/// An MLX matrix multiplication -- affine-int8 with logical-width padding, or dense.
public final class AffineLinear {
  public let weight: MLXArray
  public let linearBias: MLXArray?
  public let logicalInputWidth: Int
  public let physicalInputWidth: Int
  let storage: MatrixStorage

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
    self.linearBias = linearBias
    self.logicalInputWidth = logicalInputWidth
    self.physicalInputWidth = physicalInputWidth
    self.storage = .affineInt8(
      scales: scales,
      quantizationBiases: quantizationBiases,
      groupSize: groupSize,
      bits: bits
    )
  }

  /// A matrix stored at full float width. Dense weights need no group padding, so the
  /// logical and physical input widths coincide.
  public init(denseWeight: MLXArray, linearBias: MLXArray?) {
    self.weight = denseWeight
    self.linearBias = linearBias
    self.logicalInputWidth = denseWeight.shape[1]
    self.physicalInputWidth = denseWeight.shape[1]
    self.storage = .dense
  }

  public func callAsFunction(_ input: MLXArray) -> MLXArray {
    precondition(input.shape.last == logicalInputWidth)
    var output: MLXArray
    switch storage {
    case .affineInt8(let scales, let quantizationBiases, let groupSize, let bits):
      let padding = physicalInputWidth - logicalInputWidth
      let paddedInput: MLXArray
      if padding == 0 {
        paddedInput = input
      } else {
        var widths = Array(repeating: IntOrPair(0), count: input.ndim)
        widths[input.ndim - 1] = IntOrPair((0, padding))
        paddedInput = MLX.padded(input, widths: widths)
      }
      output = MLX.quantizedMM(
        paddedInput,
        weight,
        scales: scales,
        biases: quantizationBiases,
        transpose: true,
        groupSize: groupSize,
        bits: bits,
        mode: .affine
      )
    case .dense:
      output = MLX.matmul(input, weight.transposed())
    }
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
    if artifact.manifest.quantization == nil {
      return AffineLinear(
        denseWeight: weight,
        linearBias: artifact.arrays["\(prefix).bias"]
      )
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
    if artifact.manifest.quantization == nil {
      return AffineEmbedding(denseWeight: weight)
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
