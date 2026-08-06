import SwiftUI
import UIKit

@main
struct BoltzMLXDemoApp: App {
  var body: some Scene {
    WindowGroup {
      PredictionView()
        .task {
          // Headless multi-size benchmark, only when explicitly automated
          // (BENCH_ALL env var passed via `devicectl process launch`); never
          // auto-runs in normal interactive use.
          if ProcessInfo.processInfo.environment["BENCH_ALL"] != nil {
            // Keep the screen awake so auto-lock doesn't suspend the app mid-run.
            await MainActor.run { UIApplication.shared.isIdleTimerDisabled = true }
            await BenchmarkRunner.runAll()
            await MainActor.run { UIApplication.shared.isIdleTimerDisabled = false }
          }
        }
    }
  }
}
