// ConfidenceModule.swift — Boltz-2 confidence head (AF3 Algorithm 31), PAE path.
//
// PURPOSE. Produces the predicted-aligned-error matrix that min_ipSAE is computed from, which is the
// gate a design loop accepts or rejects candidates on. Without this the whole pipeline can fold but
// cannot score.
//
// SCOPE: PAE + pLDDT. PDE, pTM/ipTM and resolved-ness are deliberately not computed. That is not
// laziness — `min_ipSAE` needs nothing but PAE and the per-chain token ranges, and PAE is the ONLY
// head whose inputs are already available: `pred_distogram_logits` (needed by PDE and the aggregated
// metrics) is consumed by the other heads, not by PAE. So this is the smallest change that yields a
// usable gate, and the distogram forward can stay unported.
//
// THE ARCHITECTURE IS NOT INFERABLE FROM DEFAULTS. This checkpoint sets, and every one of these
// changes which tensors exist or how they combine:
//   add_s_input_to_s   = true   -> s += s_input_to_s(s_inputs)
//   add_z_input_to_z   = true   -> z += rel_pos + token_bonds + token_bonds_type + contact_cond,
//                                  each of which is confidence's OWN copy, NOT the trunk's
//   add_s_to_z_prod    = true   -> extra outer-product term into z
//   bond_type_feature  = true   -> the token_bonds_type embedding is present
//   use_separate_heads = true   -> there is NO `to_pae_logits`; PAE is intra + inter combined by
//                                  asym_id, and a port written from the obvious reading of
//                                  confidencev2.py fails at weight lookup
//   use_s_diffusion    = false  -> no diffusion-state input, which keeps this tractable
//   num_plddt_bins     = 50     -> pLDDT is a head over the pairformer's SINGLE output `s`, not
//                                  over `z`. `s` was already computed and discarded here, so the
//                                  head costs one linear layer on top of work already done.
//
// A NOTE ON THE RESIDUAL CONNECTIONS. Upstream comments "AF3 has residual connections, we remove
// them" and assigns the pairformer outputs directly (`s = s_t; z = z_t`). Adding the residual — the
// natural thing to write — silently changes every number.
import Foundation
import MLX

public struct ConfidenceModule {
    // s / z preparation
    let sequenceInputsNorm: BoltzLayerNorm
    let sequenceNorm: BoltzLayerNorm
    let pairNorm: BoltzLayerNorm
    let sequenceInputToSequence: AffineLinear
    let relativePosition: RelativePositionEncoder
    let tokenBonds: AffineLinear
    let tokenBondType: AffineEmbedding?
    let contactConditioning: ContactConditioning
    let sequenceToPair: AffineLinear
    let sequenceToPairTranspose: AffineLinear
    let sequenceToPairProductIn1: AffineLinear
    let sequenceToPairProductIn2: AffineLinear
    let sequenceToPairProductOut: AffineLinear
    /// Distance-bin embedding over the predicted structure. `boundaries` is a buffer in the
    /// checkpoint, but it is exactly linspace(2, maxDist, bins - 1), so it is recomputed rather than
    /// loaded — one fewer tensor to thread through, and asserted against the checkpoint in tests.
    let distanceBinEmbedding: AffineEmbedding
    let boundaries: MLXArray
    let pairformer: Pairformer
    // heads
    let paeIntraLogits: AffineLinear
    /// Head over the per-token representation. pLDDT, unlike PAE, is per token not per pair.
    let plddtLogits: AffineLinear
    let paeInterLogits: AffineLinear
    let paeBinCount: Int
    let maximumPAE: Double

    public struct Output {
        /// [tokens, tokens] expected aligned error in Angstroms.
        public let pae: [Double]
        /// Raw [tokens, tokens, bins] logits, exposed so parity can be checked BEFORE the
        /// expectation is taken — a bug in the head and a bug in the binning look identical once
        /// they have been reduced to a scalar.
        public let paeLogits: [Float]
        public let tokenCount: Int
        public let binCount: Int
        /// [tokens] pLDDT in 0...100, the per-residue confidence a viewer colours by.
        public let plddt: [Double]
        /// Raw [tokens, bins] logits, exposed for the same reason as `paeLogits`: after the
        /// expectation, a wrong head and wrong bin centres are indistinguishable.
        public let plddtLogits: [Float]
        public let plddtBinCount: Int
    }

    /// - Parameters:
    ///   - sequenceInputs: `s_inputs` from the input embedder, [1, n, token_s]
    ///   - sequence: trunk `s`, [1, n, token_s]
    ///   - pair: trunk `z`, [1, n, n, token_z]
    ///   - predictedCoordinates: the PADDED atom coordinates [1, aPadded, 3]. Padded, not unpadded:
    ///     `token_to_rep_atom` is a one-hot over the padded atom axis, so handing it the unpadded
    ///     array indexes the wrong atoms and silently produces a plausible, wrong PAE.
    public func callAsFunction(
        sequenceInputs: MLXArray,
        sequence: MLXArray,
        pair: MLXArray,
        predictedCoordinates: MLXArray,
        features: [String: MLXArray]
    ) throws -> Output {
        var sInputs = sequenceInputsNorm(sequenceInputs)
        var s = sequenceNorm(sequence)
        s = s + sequenceInputToSequence(sInputs)

        var z = pairNorm(pair)
        // Confidence's own input-feature copies. These are DIFFERENT weights from the trunk's, even
        // though the layer shapes match.
        z = z + (try relativePosition(features))
        z = z + tokenBonds(try requireFeature("token_bonds", from: features).asType(.float32))
        if let bondType = tokenBondType {
            z = z + bondType(try requireFeature("type_bonds", from: features).asType(.int32))
        }
        z = z + (try contactConditioning(features))

        z = z + sequenceToPair(sInputs).expandedDimensions(axis: 2)
              + sequenceToPairTranspose(sInputs).expandedDimensions(axis: 1)
        z = z + sequenceToPairProductOut(
            sequenceToPairProductIn1(sInputs).expandedDimensions(axis: 2)
            * sequenceToPairProductIn2(sInputs).expandedDimensions(axis: 1)
        )

        // Representative atom per token, then a pairwise distance matrix, bucketed into bins.
        let tokenToRepAtom = try requireFeature("token_to_rep_atom", from: features).asType(.float32)
        let representative = MLX.matmul(tokenToRepAtom, predictedCoordinates)
        let diff = representative.expandedDimensions(axis: 2) - representative.expandedDimensions(axis: 1)
        let distance = MLX.sqrt(MLX.sum(diff * diff, axis: -1))
        // `(d > boundaries).sum(-1)` — the bin index, matching upstream's bucketize-by-summation.
        let bin = MLX.sum(
            (distance.expandedDimensions(axis: -1) .> boundaries).asType(.int32), axis: -1)
        z = z + distanceBinEmbedding(bin)

        let mask = try requireFeature("token_pad_mask", from: features).asType(.float32)
        let pairMask = mask.expandedDimensions(axis: 2) * mask.expandedDimensions(axis: 1)

        // No residual: upstream deliberately drops AF3's residual connections here.
        let (sOut, zOut) = try pairformer(sequence: s, pair: z, mask: mask, pairMask: pairMask)
        z = zOut

        // Two heads, selected per pair by whether the tokens share a chain.
        let asym = try requireFeature("asym_id", from: features)
        let sameChain = (asym.expandedDimensions(axis: 2) .== asym.expandedDimensions(axis: 1))
            .asType(.float32).expandedDimensions(axis: -1)
        let differentChain = 1 - sameChain
        let logits = paeIntraLogits(z) * sameChain + paeInterLogits(z) * differentChain

        MLX.eval(logits)
        let n = logits.dim(1)
        let flat = logits.reshaped([n * n * paeBinCount]).asType(.float32).asArray(Float.self)
        let pae = InterfaceScoring.expectedError(
            logits: flat.map(Double.init), tokenCount: n,
            binCount: paeBinCount, maximumError: maximumPAE)
        // pLDDT: one head over the single representation the pairformer already produced.
        let plddtRaw = plddtLogits(sOut)
        MLX.eval(plddtRaw)
        let plddtBins = plddtRaw.dim(-1)
        let plddtFlat = plddtRaw.reshaped([n * plddtBins]).asType(.float32).asArray(Float.self)
        let plddt = InterfaceScoring.expectedPLDDT(
            logits: plddtFlat.map(Double.init), tokenCount: n, binCount: plddtBins)

        return Output(pae: pae, paeLogits: flat, tokenCount: n, binCount: paeBinCount,
                      plddt: plddt, plddtLogits: plddtFlat, plddtBinCount: plddtBins)
    }
}

extension BoltzWeightStore {

    /// Build the confidence head. Returns nil when the pack predates confidence export, so a caller
    /// can fall back to structure-only prediction rather than crashing — a pack without these
    /// tensors is a valid structure pack.
    func confidenceModule(configuration: BoltzModelConfiguration) throws -> ConfidenceModule? {
        guard let args = configuration.confidence else { return nil }
        let p = "confidence_module"
        func rawArray(_ name: String) throws -> MLXArray {
            guard let a = artifact.arrays[name] else { throw BoltzError.missingTensor(name) }
            return a
        }
        let bins = args.numDistBins
        // linspace(2, maxDist, bins - 1) — the checkpoint stores this as a buffer, recomputed here.
        let count = bins - 1
        let step = (Double(args.maxDist) - 2.0) / Double(count - 1)
        let boundaries = MLXArray((0 ..< count).map { Float(2.0 + Double($0) * step) })

        return ConfidenceModule(
            sequenceInputsNorm: try layerNorm("\(p).s_inputs_norm"),
            sequenceNorm: try layerNorm("\(p).s_norm"),
            pairNorm: try layerNorm("\(p).z_norm"),
            sequenceInputToSequence: try linear("\(p).s_input_to_s"),
            relativePosition: RelativePositionEncoder(
                projection: try linear("\(p).rel_pos.linear_layer"),
                fixSymCheck: configuration.fixSymCheck,
                cyclicPositionEncoding: configuration.cyclicPosEnc
            ),
            tokenBonds: try linear("\(p).token_bonds"),
            tokenBondType: configuration.bondTypeFeature
                ? try embedding("\(p).token_bonds_type") : nil,
            contactConditioning: ContactConditioning(
                fourierProjection: try linear("\(p).contact_conditioning.fourier_embedding.proj"),
                encoder: try linear("\(p).contact_conditioning.encoder"),
                unspecifiedEncoding: try rawArray("\(p).contact_conditioning.encoding_unspecified"),
                unselectedEncoding: try rawArray("\(p).contact_conditioning.encoding_unselected"),
                cutoffMinimum: configuration.conditioningCutoffMin,
                cutoffMaximum: configuration.conditioningCutoffMax
            ),
            sequenceToPair: try linear("\(p).s_to_z"),
            sequenceToPairTranspose: try linear("\(p).s_to_z_transpose"),
            sequenceToPairProductIn1: try linear("\(p).s_to_z_prod_in1"),
            sequenceToPairProductIn2: try linear("\(p).s_to_z_prod_in2"),
            sequenceToPairProductOut: try linear("\(p).s_to_z_prod_out"),
            distanceBinEmbedding: try embedding("\(p).dist_bin_pairwise_embed"),
            boundaries: boundaries,
            pairformer: try pairformer(
                configuration: args.pairformer,
                sequenceWidth: configuration.tokenS,
                pairWidth: configuration.tokenZ,
                prefix: "\(p).pairformer_stack"
            ),
            paeIntraLogits: try linear("\(p).confidence_heads.to_pae_intra_logits"),
            plddtLogits: try linear("\(p).confidence_heads.to_plddt_logits"),
            paeInterLogits: try linear("\(p).confidence_heads.to_pae_inter_logits"),
            paeBinCount: args.numPAEBins,
            maximumPAE: 32.0
        )
    }
}
