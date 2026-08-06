import MLX
import XCTest

@testable import BoltzMLX

/// Diagnoses the large-N OOM in the triangle multiplicative update. Gated on
/// `~/.artifacts-boltz-boundary/trimem.json` ({"n": 384, "dim": 128}) so it is a
/// no-op in normal runs. `einsum` reproduces the ~N^3*dim single-buffer allocation;
/// `matmul` is the memory-safe reformulation (batched matmul over the channel dim).
final class TriangleMemoryTests: XCTestCase {
  private func params() throws -> (n: Int, dim: Int) {
    let url = FileManager.default.homeDirectoryForCurrentUser
      .appending(path: ".artifacts-boltz-boundary/trimem.json")
    guard let data = try? Data(contentsOf: url),
      let json = try? JSONSerialization.jsonObject(with: data) as? [String: Int],
      let n = json["n"]
    else { throw XCTSkip("no trimem.json") }
    return (n, json["dim"] ?? 128)
  }

  private func inputs(_ n: Int, _ dim: Int) -> (MLXArray, MLXArray) {
    let a = MLXRandom.normal([1, n, n, dim], key: MLXRandom.key(1))
    let b = MLXRandom.normal([1, n, n, dim], key: MLXRandom.key(2))
    MLX.eval(a, b)
    return (a, b)
  }

  private func reportPeak(_ label: String, _ n: Int, _ dim: Int, _ out: MLXArray) {
    MLX.eval(out)
    let peak = Memory.snapshot().peakMemory
    print(
      "TRIMEM \(label) n=\(n) dim=\(dim) out=\(out.shape) "
        + "peak=\(peak) bytes (~\(peak / 1_000_000) MB)")
  }

  /// The current implementation: broadcasts to [b,i,j,k,d] (~n^3*dim) before summing k.
  func testEinsumPeak() throws {
    let (n, dim) = try params()
    let (a, b) = inputs(n, dim)
    GPU.resetPeakMemory()
    let out = MLX.einsum("bikd,bjkd->bijd", a, b)
    reportPeak("einsum", n, dim, out)
  }

  private func atomParams() throws -> (tokens: Int, atoms: Int) {
    let url = FileManager.default.homeDirectoryForCurrentUser
      .appending(path: ".artifacts-boltz-boundary/trimem.json")
    guard let data = try? Data(contentsOf: url),
      let json = try? JSONSerialization.jsonObject(with: data) as? [String: Int],
      let tokens = json["tokens"], let atoms = json["atoms"]
    else { throw XCTSkip("no atom params in trimem.json") }
    return (tokens, atoms)
  }

  private func atomInputs(_ tokens: Int, _ atoms: Int) -> (MLXArray, MLXArray, MLXArray) {
    let qw = 32
    let kw = 128
    let windows = atoms / qw
    let pair = MLXRandom.normal([1, tokens, tokens, 16], key: MLXRandom.key(1))  // pairToAtom(z)
    let tq = MLXRandom.normal([1, windows, qw, tokens], key: MLXRandom.key(2))  // tokenQueries
    let tk = MLXRandom.normal([1, windows, kw, tokens], key: MLXRandom.key(3))  // tokenKeys
    MLX.eval(pair, tq, tk)
    return (pair, tq, tk)
  }

  /// The current implementation: one 3-way einsum. MLX broadcasts the token pair rep
  /// across atom windows -> [b,w,qw,i,j,d] ~ atoms*tokens^2*atom_z intermediate.
  func testAtomEinsum3Way() throws {
    let (tokens, atoms) = try atomParams()
    let (pair, tq, tk) = atomInputs(tokens, atoms)
    GPU.resetPeakMemory()
    let out = MLX.einsum("bijd,bwki,bwlj->bwkld", pair, tq, tk)
    reportPeak("atom-einsum-3way tokens=\(tokens) atoms=\(atoms)", tokens, 16, out)
  }

  /// Fix: contract i, then j — never materializes the atoms*tokens^2 intermediate.
  func testAtomEinsum2Step() throws {
    let (tokens, atoms) = try atomParams()
    let (pair, tq, tk) = atomInputs(tokens, atoms)
    GPU.resetPeakMemory()
    let step1 = MLX.einsum("bijd,bwki->bwkjd", pair, tq)  // [b,w,qw,j,d]
    let out = MLX.einsum("bwkjd,bwlj->bwkld", step1, tk)  // [b,w,qw,kw,d]
    reportPeak("atom-einsum-2step tokens=\(tokens) atoms=\(atoms)", tokens, 16, out)
    if tokens <= 96 {
      let reference = MLX.einsum("bijd,bwki,bwlj->bwkld", pair, tq, tk)
      let maxDiff = MLX.abs(out - reference).max().item(Float.self)
      XCTAssertLessThan(maxDiff, 1e-1, "2-step must match the 3-way einsum")
    }
  }

  /// Fix: batched matmul over the channel dim — O(n^2*dim) memory, no n^3 intermediate.
  func testMatmulPeak() throws {
    let (n, dim) = try params()
    let (a, b) = inputs(n, dim)
    GPU.resetPeakMemory()
    // out[b,i,j,d] = sum_k a[b,i,k,d] * b[b,j,k,d]
    let aP = a.transposed(0, 3, 1, 2)  // [b,d,i,k]
    let bP = b.transposed(0, 3, 2, 1)  // [b,d,k,j]
    let out = MLX.matmul(aP, bP).transposed(0, 2, 3, 1)  // [b,i,j,d]
    reportPeak("matmul", n, dim, out)
    // sanity: agrees with the einsum at small n (only checked when it fits)
    if n <= 64 {
      let reference = MLX.einsum("bikd,bjkd->bijd", a, b)
      let maxDiff = MLX.abs(out - reference).max().item(Float.self)
      XCTAssertLessThan(maxDiff, 1e-2, "matmul reformulation must match einsum")
    }
  }
}
