// swift-tools-version:5.5

import PackageDescription

let package = Package(
    name: "llama",
    platforms: [
        .macOS(.v12),
        .iOS(.v14),
        .watchOS(.v4),
        .tvOS(.v14)
    ],
    products: [
        .library(name: "llama", targets: ["llama"]),
    ],
    targets: [
        .binaryTarget(
            name: "llama",
            path: "build-ios/llama.xcframework"
        ),
        //.systemLibrary(name: "llama", pkgConfig: "llama"),
    ]
)
