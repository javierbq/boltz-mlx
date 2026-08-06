import BoltzMLX
import Darwin
import Foundation
import os

/// Headless on-device benchmark. When `Documents/bench/model` and
/// `Documents/bench/features` are present (pushed via `devicectl`), it runs one
/// prediction on launch and writes `Documents/bench/result.json` plus an os_log
/// line, so a physical-device run needs no document-picker interaction.
enum BenchmarkRunner {
  private static let log = Logger(subsystem: "io.github.javierbq.boltzmlx", category: "bench")

  static func runIfRequested() async {
    let fm = FileManager.default
    let docs = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let library = fm.urls(for: .libraryDirectory, in: .userDomainMask)[0]
    // devicectl pushes land under Documents/ with the source folder name intact,
    // so discover the artifacts by content rather than a fixed path: scan each
    // base and its immediate subdirectories for model.safetensors / features.safetensors.
    let bases = [docs, docs.deletingLastPathComponent(), library]
    var modelDir: URL?
    var featureDir: URL?
    for base in bases {
      let entries = (try? fm.contentsOfDirectory(at: base, includingPropertiesForKeys: nil)) ?? []
      // Subdirectories first: a clean model/ or features/ folder (with its own
      // manifest.json) must win over loose files flattened into the base.
      for entry in entries + [base] {
        if modelDir == nil, fm.fileExists(atPath: entry.appending(path: "model.safetensors").path) {
          modelDir = entry
        }
        if featureDir == nil,
          fm.fileExists(atPath: entry.appending(path: "features.safetensors").path)
        {
          featureDir = entry
        }
      }
    }
    guard let modelDir, let featureDir else {
      log.info("bench: no artifacts found under \(docs.path, privacy: .public)")
      return
    }
    let benchDir = docs
    do {
      let clock = ContinuousClock()
      let bundle = try FeatureBundle.load(from: featureDir)
      let start = clock.now
      let predictor = try BoltzPredictor(modelDirectory: modelDir)
      let footprintAfterLoad = physFootprintBytes()
      let result = try await predictor.predict(features: bundle)
      let elapsed = start.duration(to: clock.now)
      let elapsedSeconds =
        Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
      let snapshot = await predictor.memorySnapshot()
      let footprintAfterPredict = physFootprintBytes()
      let payload: [String: Any] = [
        "status": "ok",
        "device": deviceModel(),
        "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
        "tokens": bundle.metadata.tokenCount,
        "atoms": result.coordinates.count,
        "elapsed_seconds": elapsedSeconds,
        "mlx_peak_bytes": snapshot.peakMemory,
        "phys_footprint_after_load_bytes": Int(footprintAfterLoad),
        "phys_footprint_after_predict_bytes": Int(footprintAfterPredict),
      ]
      try write(payload, to: benchDir.appending(path: "result.json"))
      log.info("BENCH_RESULT \(json(payload), privacy: .public)")
    } catch {
      let payload: [String: Any] = ["status": "error", "message": error.localizedDescription]
      try? write(payload, to: benchDir.appending(path: "result.json"))
      log.error("bench failed: \(error.localizedDescription, privacy: .public)")
    }
  }

  /// Multi-size benchmark: run the model against every feature bundle found under
  /// Documents/ and write Documents/results_all.json (one entry per protein, sorted
  /// by token count). Peak memory is reset per protein for an independent reading.
  static func runAll() async {
    let fm = FileManager.default
    let docs = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let entries = ((try? fm.contentsOfDirectory(at: docs, includingPropertiesForKeys: nil)) ?? [])
    var modelDir: URL?
    var featureDirs: [URL] = []
    for entry in entries + [docs] {
      if modelDir == nil, fm.fileExists(atPath: entry.appending(path: "model.safetensors").path) {
        modelDir = entry
      }
      if fm.fileExists(atPath: entry.appending(path: "features.safetensors").path) {
        featureDirs.append(entry)
      }
    }
    guard let modelDir else {
      log.error("bench-all: no model found under \(docs.path, privacy: .public)")
      try? write(
        ["status": "error", "message": "no model"],
        to: docs.appending(path: "results_all.json"))
      return
    }
    featureDirs = featureDirs.filter { $0 != modelDir }
    do {
      // Raise the provisional input cap so large proteins run; keep the device-tuned
      // memory/cache limits (defaults) so MLX still evicts to avoid jetsam.
      let planner = MemoryPlanner(
        limits: BoltzInputLimits(maximumTokens: 8192, maximumAtoms: 65536, maximumMSADepth: 8192))
      let predictor = try BoltzPredictor(modelDirectory: modelDir, memoryPlanner: planner)
      var runs: [[String: Any]] = []
      let resultsURL = docs.appending(path: "results_all.json")
      func flush() {
        let payload: [String: Any] = [
          "device": deviceModel(),
          "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
          "runs": runs.sorted { ($0["tokens"] as? Int ?? 0) < ($1["tokens"] as? Int ?? 0) },
        ]
        try? write(payload, to: resultsURL)
      }
      for featureDir in featureDirs {
        await predictor.resetPeakMemory()
        do {
          let bundle = try FeatureBundle.load(from: featureDir)
          let clock = ContinuousClock()
          let start = clock.now
          let result = try await predictor.predict(features: bundle)
          let elapsed = start.duration(to: clock.now)
          let seconds =
            Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
          let snapshot = await predictor.memorySnapshot()
          let entry: [String: Any] = [
            "name": featureDir.lastPathComponent,
            "status": "ok",
            "tokens": bundle.metadata.tokenCount,
            "atoms": result.coordinates.count,
            "elapsed_seconds": seconds,
            "mlx_peak_bytes": snapshot.peakMemory,
            "phys_footprint_bytes": Int(physFootprintBytes()),
          ]
          runs.append(entry)
          log.info("BENCH_ONE \(json(entry), privacy: .public)")
        } catch {
          runs.append([
            "name": featureDir.lastPathComponent, "status": "error",
            "message": error.localizedDescription,
          ])
          let name = featureDir.lastPathComponent
          let message = error.localizedDescription
          log.error("bench-all \(name, privacy: .public): \(message, privacy: .public)")
        }
        flush()  // incremental: partial results survive a jetsam kill on a larger protein
      }
      log.info("BENCH_ALL_DONE")
    } catch {
      try? write(
        ["status": "error", "message": error.localizedDescription],
        to: docs.appending(path: "results_all.json"))
      log.error("bench-all setup: \(error.localizedDescription, privacy: .public)")
    }
  }

  private static func write(_ payload: [String: Any], to url: URL) throws {
    let data = try JSONSerialization.data(
      withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: url)
  }

  private static func json(_ payload: [String: Any]) -> String {
    guard let data = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys]),
      let string = String(data: data, encoding: .utf8)
    else { return "{}" }
    return string
  }

  private static func physFootprintBytes() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
      MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
      $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
      }
    }
    return result == KERN_SUCCESS ? UInt64(info.phys_footprint) : 0
  }

  private static func deviceModel() -> String {
    var systemInfo = utsname()
    uname(&systemInfo)
    let mirror = Mirror(reflecting: systemInfo.machine)
    let characters = mirror.children.compactMap { child -> Character? in
      guard let byte = child.value as? Int8, byte != 0 else { return nil }
      return Character(UnicodeScalar(UInt8(byte)))
    }
    return String(characters)
  }
}
