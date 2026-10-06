import SwiftUI

public struct RoadWatchAppView: View {
    public init() {}
    
    public var body: some View {
        CockpitDashboardView()
            .preferredColorScheme(.dark)
    }
}
