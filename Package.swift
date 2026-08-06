// swift-tools-version: 6.0

import PackageDescription

let package = Package(
  name: "BoltzMLX",
  platforms: [
    .macOS(.v14),
    .iOS(.v17),
  ],
  products: [
    .library(name: "BoltzMLX", targets: ["BoltzMLX"]),
    .executable(name: "BoltzMLXCLI", targets: ["BoltzMLXCLI"]),
  ],
  dependencies: [
    .package(
      url: "https://github.com/ml-explore/mlx-swift.git",
      exact: "0.31.6"
    ),
    .package(
      url: "https://github.com/apple/swift-argument-parser.git",
      exact: "1.8.2"
    ),
  ],
  targets: [
    .target(
      name: "BoltzMLX",
      dependencies: [
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
      ]
    ),
    .executableTarget(
      name: "BoltzMLXCLI",
      dependencies: [
        "BoltzMLX",
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
      ]
    ),
    .testTarget(
      name: "BoltzMLXTests",
      dependencies: [
        "BoltzMLX",
        .product(name: "MLX", package: "mlx-swift"),
      ],
      path: "tests/BoltzMLXTests"
    ),
  ]
)
