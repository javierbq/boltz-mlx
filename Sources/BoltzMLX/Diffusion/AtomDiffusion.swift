import Foundation
import MLX

/// Boltz-2 diffusion sampler with deterministic noise injection for parity tests.
public struct AtomDiffusion {
  public let scoreModel: DiffusionScoreModel
  public let process: DiffusionProcessConfiguration

  public func preconditionedScore(
    coordinates: MLXArray,
    sigma: Float,
    trunk: TrunkOutput,
    features: [String: MLXArray],
    conditioning: DiffusionConditioningOutput,
    multiplicity: Int
  ) throws -> MLXArray {
    let sigmaArray = MLX.full([coordinates.shape[0]], values: sigma, type: Float.self)
    let paddedSigma = sigmaArray.reshaped(-1, 1, 1)
    let sigmaDataSquared = process.sigmaData * process.sigmaData
    let sigmaSquared = paddedSigma * paddedSigma
    let inputScale = 1 / MLX.sqrt(sigmaSquared + sigmaDataSquared)
    let noise = MLX.log(MLX.maximum(sigmaArray / process.sigmaData, 1e-20)) * 0.25
    let update = try scoreModel(
      noisyCoordinates: inputScale * coordinates,
      noise: noise,
      trunk: trunk,
      features: features,
      conditioning: conditioning,
      multiplicity: multiplicity
    )
    let skip = sigmaDataSquared / (sigmaSquared + sigmaDataSquared)
    let outputScale =
      paddedSigma * process.sigmaData
      / MLX.sqrt(sigmaDataSquared + sigmaSquared)
    return skip * coordinates + outputScale * update
  }

  public func sample(
    trunk: TrunkOutput,
    features: [String: MLXArray],
    conditioning: DiffusionConditioningOutput,
    steps: Int? = nil,
    multiplicity: Int = 1,
    seed: UInt64 = 0,
    initialNoise: MLXArray? = nil,
    stepNoises: [MLXArray]? = nil,
    rotations: [MLXArray]? = nil,
    translations: [MLXArray]? = nil,
    clearCacheBetweenSteps: Bool = true
  ) throws -> MLXArray {
    let stepCount = steps ?? process.numSamplingSteps ?? 5
    let schedule = diffusionSchedule(
      steps: stepCount,
      sigmaMinimum: process.sigmaMin,
      sigmaMaximum: process.sigmaMax,
      sigmaData: process.sigmaData,
      rho: process.rho
    )
    MLX.eval(schedule)
    let sigmaValues = schedule.asArray(Float.self)
    let baseMask = try requireFeature("atom_pad_mask", from: features)
    let atomMask = MLX.repeated(baseMask, count: multiplicity, axis: 0).asType(.float32)
    let shape = atomMask.shape + [3]
    let keys = MLXRandom.split(key: MLXRandom.key(seed), into: 1 + stepCount * 3)
    var coordinates =
      sigmaValues[0]
      * (initialNoise ?? MLXRandom.normal(shape, key: keys[0]))
    var previousDenoised: MLXArray?
    for index in 0..<stepCount {
      try Task.checkCancellation()
      let rotation =
        rotations?[index]
        ?? randomRotation(batch: multiplicity, key: keys[1 + index * 3])
      let translation =
        translations?[index]
        ?? MLXRandom.normal([multiplicity, 1, 3], key: keys[2 + index * 3])
      coordinates = coordinates - coordinates.mean(axis: -2, keepDims: true)
      coordinates = MLX.einsum("bmd,bds->bms", coordinates, rotation) + translation
      if let denoised = previousDenoised {
        let centered = denoised - denoised.mean(axis: -2, keepDims: true)
        previousDenoised = MLX.einsum("bmd,bds->bms", centered, rotation) + translation
      }

      let sigmaPrevious = sigmaValues[index]
      let sigmaNext = sigmaValues[index + 1]
      let gamma: Float = sigmaNext > process.gammaMin ? process.gamma0 : 0
      let tHat = sigmaPrevious * (1 + gamma)
      let noiseVariance =
        process.noiseScale * process.noiseScale
        * (tHat * tHat - sigmaPrevious * sigmaPrevious)
      let noise =
        stepNoises?[index]
        ?? MLXRandom.normal(shape, key: keys[3 + index * 3])
      let noisy = coordinates + max(0, noiseVariance).squareRoot() * noise
      let denoised = try preconditionedScore(
        coordinates: noisy,
        sigma: tHat,
        trunk: trunk,
        features: features,
        conditioning: conditioning,
        multiplicity: multiplicity
      )
      // Boltz aligns the *noisy* coordinates onto the denoised target, then steps
      // the aligned noisy coordinates toward the denoised estimate. The alignment
      // result replaces `noisy` (not `denoised`); assigning it to `denoised` made
      // the score direction a difference of two noise-scale tensors and blew up.
      var alignedNoisy = noisy
      if process.alignmentReverseDiff ?? true {
        alignedNoisy = weightedRigidAlign(
          coordinates: noisy,
          target: denoised,
          weights: atomMask,
          mask: atomMask
        )
      }
      coordinates =
        alignedNoisy
        + process.stepScale * (sigmaNext - tHat) * (alignedNoisy - denoised) / tHat
      previousDenoised = denoised
      MLX.eval(coordinates)
      if clearCacheBetweenSteps {
        Memory.clearCache()
      }
    }
    return coordinates
  }
}

func diffusionSchedule(
  steps: Int,
  sigmaMinimum: Float,
  sigmaMaximum: Float,
  sigmaData: Float,
  rho: Float
) -> MLXArray {
  precondition(steps >= 2)
  let values = MLX.arange(steps, dtype: .float32)
  let inverseRho = 1 / rho
  let start = powf(sigmaMaximum, inverseRho)
  let end = powf(sigmaMinimum, inverseRho)
  let sigmas =
    (start + values / Float(steps - 1) * (end - start)).pow(rho)
    * sigmaData
  return MLX.concatenated([sigmas, MLX.zeros([1])])
}

func randomRotation(batch: Int, key: MLXArray) -> MLXArray {
  let values = MLXRandom.normal([batch, 4], key: key)
  let magnitude = MLX.sqrt((values * values).sum(axis: 1))
  let signedMagnitude = MLX.which(values[0..., 0] .< 0, -magnitude, magnitude)
  let quaternion = values / signedMagnitude.expandedDimensions(axis: 1)
  let r = quaternion[0..., 0]
  let i = quaternion[0..., 1]
  let j = quaternion[0..., 2]
  let k = quaternion[0..., 3]
  let twoScale = 2 / (quaternion * quaternion).sum(axis: -1)
  return MLX.stacked(
    [
      1 - twoScale * (j * j + k * k),
      twoScale * (i * j - k * r),
      twoScale * (i * k + j * r),
      twoScale * (i * j + k * r),
      1 - twoScale * (i * i + k * k),
      twoScale * (j * k - i * r),
      twoScale * (i * k - j * r),
      twoScale * (j * k + i * r),
      1 - twoScale * (i * i + j * j),
    ],
    axis: -1
  ).reshaped(batch, 3, 3)
}

func weightedRigidAlign(
  coordinates: MLXArray,
  target: MLXArray,
  weights: MLXArray,
  mask: MLXArray
) -> MLXArray {
  let effectiveWeights = (weights * mask).expandedDimensions(axis: -1)
  let denominator = MLX.maximum(effectiveWeights.sum(axis: -2, keepDims: true), 1e-8)
  let coordinateCenter =
    (coordinates * effectiveWeights).sum(axis: -2, keepDims: true)
    / denominator
  let targetCenter =
    (target * effectiveWeights).sum(axis: -2, keepDims: true)
    / denominator
  let centeredCoordinates = coordinates - coordinateCenter
  let centeredTarget = target - targetCenter
  let covariance = MLX.einsum(
    "bni,bnj->bij",
    effectiveWeights * centeredTarget,
    centeredCoordinates
  ).asType(.float32)
  // MLX currently exposes the small 3x3 SVD on CPU; the surrounding graph remains on Metal.
  let (u, _, vt) = MLXLinalg.svd(covariance, stream: .cpu)
  let rotation = MLX.matmul(u, vt)
  let determinant = determinant3x3(rotation)
  let diagonal = MLX.stacked(
    [MLX.ones(determinant.shape), MLX.ones(determinant.shape), determinant],
    axis: -1
  )
  let correction =
    MLX.identity(3).expandedDimensions(axis: 0)
    * diagonal.expandedDimensions(axis: -2)
  let properRotation = MLX.matmul(MLX.matmul(u, correction), vt)
  return MLX.matmul(centeredCoordinates, properRotation.transposed(0, 2, 1))
    + targetCenter
}

private func determinant3x3(_ matrix: MLXArray) -> MLXArray {
  let a = matrix[0..., 0, 0]
  let b = matrix[0..., 0, 1]
  let c = matrix[0..., 0, 2]
  let d = matrix[0..., 1, 0]
  let e = matrix[0..., 1, 1]
  let f = matrix[0..., 1, 2]
  let g = matrix[0..., 2, 0]
  let h = matrix[0..., 2, 1]
  let i = matrix[0..., 2, 2]
  return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
}

extension BoltzWeightStore {
  func atomDiffusion(configuration: BoltzModelConfiguration) throws -> AtomDiffusion {
    try AtomDiffusion(
      scoreModel: diffusionScoreModel(configuration: configuration),
      process: configuration.diffusionProcess
    )
  }
}
