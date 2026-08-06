import BoltzMLX
import Foundation

@MainActor
final class PredictionViewModel: ObservableObject {
  @Published var modelURL: URL?
  @Published var featureURL: URL?
  @Published var phase = "Select the exported model and feature folders."
  @Published var elapsedSeconds: Double = 0
  @Published var atomCount = 0
  @Published var tokenCount = 0
  @Published var peakMemoryBytes = 0
  @Published var isRunning = false

  private var predictionTask: Task<Void, Never>?

  func predict() {
    guard let modelURL, let featureURL else {
      phase = "Both folders are required."
      return
    }
    predictionTask?.cancel()
    predictionTask = Task {
      isRunning = true
      phase = "Loading artifacts"
      let started = ContinuousClock.now
      let modelAccess = modelURL.startAccessingSecurityScopedResource()
      let featureAccess = featureURL.startAccessingSecurityScopedResource()
      defer {
        if modelAccess { modelURL.stopAccessingSecurityScopedResource() }
        if featureAccess { featureURL.stopAccessingSecurityScopedResource() }
        elapsedSeconds = started.duration(to: .now).seconds
        isRunning = false
      }
      do {
        let bundle = try FeatureBundle.load(from: featureURL)
        tokenCount = bundle.metadata.tokenCount
        phase = "Building native MLX graph"
        let predictor = try BoltzPredictor(modelDirectory: modelURL)
        phase = "Running trunk and diffusion"
        let result = try await predictor.predict(features: bundle)
        let memory = await predictor.memorySnapshot()
        atomCount = result.coordinates.count
        peakMemoryBytes = memory.peakMemory
        phase = "Complete"
      } catch is CancellationError {
        phase = "Cancelled"
      } catch {
        phase = "Failed: \(error.localizedDescription)"
      }
    }
  }

  func cancel() {
    predictionTask?.cancel()
  }

  /// True when a model + feature bundle have been pushed into the app's Documents
  /// (e.g. via `devicectl`), so an on-device test can run without the file picker.
  var hasBundledArtifacts: Bool { Self.discoverBundledArtifacts() != nil }

  /// Run a prediction directly on artifacts found in the app's own Documents
  /// (no security-scoped file picker needed).
  func runBundled() {
    guard let (bundledModel, bundledFeatures) = Self.discoverBundledArtifacts() else {
      phase = "No bundled model/features found in Documents."
      return
    }
    modelURL = bundledModel
    featureURL = bundledFeatures
    predict()
  }

  /// Locate a model dir (contains model.safetensors) and feature dir (contains
  /// features.safetensors) under Documents or its immediate subdirectories.
  static func discoverBundledArtifacts() -> (model: URL, features: URL)? {
    let fm = FileManager.default
    let docs = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let entries = (try? fm.contentsOfDirectory(at: docs, includingPropertiesForKeys: nil)) ?? []
    var model: URL?
    var features: URL?
    for candidate in entries + [docs] {
      if model == nil, fm.fileExists(atPath: candidate.appending(path: "model.safetensors").path) {
        model = candidate
      }
      if features == nil,
        fm.fileExists(atPath: candidate.appending(path: "features.safetensors").path)
      {
        features = candidate
      }
    }
    guard let model, let features else { return nil }
    return (model, features)
  }
}

private extension Duration {
  var seconds: Double {
    let parts = components
    return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
  }
}
