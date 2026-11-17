// swift-tools-version: 6.0
// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import PackageDescription

let package = Package(
    name: "BreezeTTS",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "BreezeTTS", targets: ["BreezeTTS"]),
        .executable(name: "breeze-tts", targets: ["breeze-tts"]),
    ],
    targets: [
        .target(name: "BreezeTTS", path: "Sources/BreezeTTS"),
        .executableTarget(name: "breeze-tts", dependencies: ["BreezeTTS"], path: "Sources/breeze-tts"),
        .testTarget(name: "BreezeTTSTests", dependencies: ["BreezeTTS"], path: "Tests/BreezeTTSTests", exclude: ["Fixtures"]),
    ]
)
