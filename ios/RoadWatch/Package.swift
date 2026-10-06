// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "RoadWatch",
    platforms: [
        .iOS(.v16),
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "RoadWatch",
            targets: ["RoadWatch"]
        ),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "RoadWatch",
            dependencies: [],
            path: "Sources",
            resources: [
                .process("Resources")
            ]
        ),
        .testTarget(
            name: "RoadWatchTests",
            dependencies: ["RoadWatch"],
            path: "Tests/RoadWatchTests"
        ),
    ]
)
