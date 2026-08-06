import SwiftUI
import UniformTypeIdentifiers

struct PredictionView: View {
  @StateObject private var model = PredictionViewModel()
  @State private var importingModel = false
  @State private var importingFeatures = false

  var body: some View {
    NavigationStack {
      Form {
        Section("Artifacts") {
          folderRow(
            title: "Int8 model",
            url: model.modelURL,
            action: { importingModel = true }
          )
          folderRow(
            title: "Precomputed features",
            url: model.featureURL,
            action: { importingFeatures = true }
          )
        }
        Section("Prediction") {
          LabeledContent("Phase", value: model.phase)
          LabeledContent("Tokens", value: "\(model.tokenCount)")
          LabeledContent("Output atoms", value: "\(model.atomCount)")
          LabeledContent("Elapsed", value: model.elapsedSeconds.formatted(.number.precision(.fractionLength(1))) + " s")
          LabeledContent("MLX peak", value: ByteCountFormatter.string(fromByteCount: Int64(model.peakMemoryBytes), countStyle: .memory))
          if model.isRunning {
            ProgressView()
            Button("Cancel", role: .destructive) { model.cancel() }
          } else {
            Button("Predict structure") { model.predict() }
              .disabled(model.modelURL == nil || model.featureURL == nil)
            if model.hasBundledArtifacts {
              Button("Run on-device test (bundled weights)") { model.runBundled() }
            }
          }
        }
      }
      .navigationTitle("Boltz MLX")
    }
    .fileImporter(
      isPresented: $importingModel,
      allowedContentTypes: [.folder]
    ) { result in
      model.modelURL = try? result.get()
    }
    .fileImporter(
      isPresented: $importingFeatures,
      allowedContentTypes: [.folder]
    ) { result in
      model.featureURL = try? result.get()
    }
  }

  private func folderRow(
    title: String,
    url: URL?,
    action: @escaping () -> Void
  ) -> some View {
    Button(action: action) {
      HStack {
        VStack(alignment: .leading) {
          Text(title)
          Text(url?.lastPathComponent ?? "Choose folder")
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        Spacer()
        Image(systemName: "folder")
      }
    }
  }
}
