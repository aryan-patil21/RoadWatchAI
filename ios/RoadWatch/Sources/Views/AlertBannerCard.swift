import SwiftUI

public struct AlertBannerCard: View {
    public let alert: RiderAlertDTO?
    
    public init(alert: RiderAlertDTO?) {
        self.alert = alert
    }
    
    public var body: some View {
        VStack(spacing: 16) {
            if let alert = alert {
                // ACTIVE HAZARD VIEW
                let isCritical = alert.severity == .critical
                let cardColor = isCritical ? Color.red : Color.orange
                
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Image(systemName: isCritical ? "exclamationmark.octagon.fill" : "exclamationmark.triangle.fill")
                            .font(.system(size: 32, weight: .bold))
                            .foregroundColor(.white)
                        
                        VStack(alignment: .leading, spacing: 2) {
                            Text(alert.severity.titleText)
                                .font(.system(size: 14, weight: .black))
                                .foregroundColor(.white.opacity(0.85))
                            
                            HStack(spacing: 6) {
                                Image(systemName: directionIcon(for: alert.direction))
                                    .font(.headline)
                                Text(alert.direction.uppercased())
                                    .font(.system(size: 18, weight: .black))
                            }
                            .foregroundColor(.white)
                        }
                        
                        Spacer()
                        
                        // Risk Score Badge
                        VStack(alignment: .trailing, spacing: 2) {
                            Text("\(Int(alert.riskScore))")
                                .font(.system(size: 28, weight: .heavy, design: .rounded))
                                .foregroundColor(.white)
                            Text("RISK SCORE")
                                .font(.system(size: 9, weight: .bold))
                                .foregroundColor(.white.opacity(0.8))
                        }
                    }
                    
                    Divider().background(Color.white.opacity(0.3))
                    
                    // Main Message
                    Text(alert.message)
                        .font(.system(size: 20, weight: .bold))
                        .foregroundColor(.white)
                        .lineLimit(2)
                    
                    // Suggested Action Box
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "shield.lefthalf.filled")
                            .font(.subheadline)
                            .foregroundColor(.white)
                        Text(alert.suggestedAction)
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundColor(.white)
                    }
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.black.opacity(0.25))
                    .cornerRadius(8)
                }
                .padding(20)
                .background(
                    RoundedRectangle(cornerRadius: 16)
                        .fill(cardColor)
                        .shadow(color: cardColor.opacity(0.5), radius: 14, x: 0, y: 6)
                )
            } else {
                // NORMAL SAFE STATE VIEW
                VStack(spacing: 12) {
                    Image(systemName: "checkmark.shield.fill")
                        .font(.system(size: 46))
                        .foregroundColor(.green)
                    
                    Text("ROAD CONDITIONS NORMAL")
                        .font(.system(size: 20, weight: .black, design: .rounded))
                        .foregroundColor(.primary)
                    
                    Text("No immediate trajectory conflicts detected.")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 32)
                .padding(.horizontal, 20)
                .background(
                    RoundedRectangle(cornerRadius: 16)
                        #if canImport(UIKit)
                        .fill(Color(uiColor: .secondarySystemBackground))
                        #else
                        .fill(Color(nsColor: .windowBackgroundColor))
                        #endif
                        .shadow(color: Color.black.opacity(0.08), radius: 10, x: 0, y: 4)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(Color.green.opacity(0.3), lineWidth: 2)
                )
            }
        }
    }
    
    private func directionIcon(for direction: String) -> String {
        switch direction.lowercased() {
        case "left": return "arrow.left.circle.fill"
        case "right": return "arrow.right.circle.fill"
        default: return "arrow.up.circle.fill"
        }
    }
}
