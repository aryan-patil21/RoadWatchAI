import SwiftUI

public struct CockpitDashboardView: View {
    @StateObject public var viewModel = CockpitViewModel()
    
    public init() {}
    
    public var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Header Bar
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("ROADWATCH AI")
                            .font(.system(size: 22, weight: .black, design: .rounded))
                        Text(viewModel.backendStatusText)
                            .font(.caption)
                            .foregroundColor(viewModel.isConnected ? .green : .secondary)
                    }
                    
                    Spacer()
                    
                    // Mode Toggle Badge
                    Button(action: {
                        if viewModel.isSimulationMode {
                            viewModel.enableLiveBackendMode()
                        } else {
                            viewModel.startSimulation()
                        }
                    }) {
                        HStack(spacing: 4) {
                            Circle()
                                .fill(viewModel.isSimulationMode ? Color.purple : Color.green)
                                .frame(width: 8, height: 8)
                            Text(viewModel.isSimulationMode ? "DEMO MODE" : "LIVE API")
                                .font(.system(size: 11, weight: .bold))
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(Color.secondary.opacity(0.15))
                        .cornerRadius(20)
                    }
                }
                .padding(.horizontal)
                .padding(.top, 8)
                
                // Lane Radar Strip
                HStack(spacing: 8) {
                    LaneZoneIndicator(
                        title: "LEFT LANE",
                        isActive: viewModel.activeAlert?.direction.lowercased() == "left",
                        severity: viewModel.activeAlert?.severity ?? .normal
                    )
                    LaneZoneIndicator(
                        title: "EGO PATH",
                        isActive: viewModel.activeAlert?.direction.lowercased() == "ahead" || viewModel.activeAlert?.direction.lowercased() == "center",
                        severity: viewModel.activeAlert?.severity ?? .normal
                    )
                    LaneZoneIndicator(
                        title: "RIGHT LANE",
                        isActive: viewModel.activeAlert?.direction.lowercased() == "right",
                        severity: viewModel.activeAlert?.severity ?? .normal
                    )
                }
                .padding(.horizontal)
                
                // Main Alert Card
                AlertBannerCard(alert: viewModel.activeAlert)
                    .padding(.horizontal)
                
                // Interactive Demo Controls (When in Simulation Mode)
                if viewModel.isSimulationMode {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("SIMULATION CONTROLS")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundColor(.secondary)
                        
                        HStack(spacing: 12) {
                            Button(action: { viewModel.stepSimulation() }) {
                                Label("Next Scenario", systemImage: "forward.fill")
                                    .font(.system(size: 13, weight: .bold))
                                    .padding(.vertical, 8)
                                    .padding(.horizontal, 14)
                                    .background(Color.blue.opacity(0.15))
                                    .foregroundColor(.blue)
                                    .cornerRadius(8)
                            }
                            
                            Button(action: { AudioAlertService.shared.playAlert(severity: .caution) }) {
                                Label("Chime", systemImage: "speaker.wave.2.fill")
                                    .font(.system(size: 13, weight: .bold))
                                    .padding(.vertical, 8)
                                    .padding(.horizontal, 14)
                                    .background(Color.orange.opacity(0.15))
                                    .foregroundColor(.orange)
                                    .cornerRadius(8)
                            }
                            
                            Button(action: { AudioAlertService.shared.playAlert(severity: .critical) }) {
                                Label("Alarm", systemImage: "bell.badge.fill")
                                    .font(.system(size: 13, weight: .bold))
                                    .padding(.vertical, 8)
                                    .padding(.horizontal, 14)
                                    .background(Color.red.opacity(0.15))
                                    .foregroundColor(.red)
                                    .cornerRadius(8)
                            }
                        }
                    }
                    .padding()
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.secondary.opacity(0.08))
                    .cornerRadius(12)
                    .padding(.horizontal)
                }
                
                // Recent Alerts Log
                VStack(alignment: .leading, spacing: 12) {
                    Text("SESSION ALERT HISTORY")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundColor(.secondary)
                    
                    if viewModel.recentAlerts.isEmpty {
                        Text("No warnings recorded this session.")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .padding(.vertical, 8)
                    } else {
                        ForEach(viewModel.recentAlerts) { alert in
                            HStack {
                                Circle()
                                    .fill(alert.severity == .critical ? Color.red : Color.orange)
                                    .frame(width: 8, height: 8)
                                
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(alert.title)
                                        .font(.system(size: 13, weight: .bold))
                                    Text(alert.message)
                                        .font(.system(size: 11))
                                        .foregroundColor(.secondary)
                                }
                                
                                Spacer()
                                
                                Text("\(Int(alert.riskScore)) pts")
                                    .font(.system(size: 12, weight: .bold))
                                    .foregroundColor(.secondary)
                            }
                            .padding(10)
                            .background(Color.secondary.opacity(0.08))
                            .cornerRadius(8)
                        }
                    }
                }
                .padding(.horizontal)
            }
            .padding(.bottom, 24)
        }
    }
}

// Subview for lane radar direction
private struct LaneZoneIndicator: View {
    let title: String
    let isActive: Bool
    let severity: AlertSeverity
    
    var body: some View {
        VStack(spacing: 4) {
            Text(title)
                .font(.system(size: 10, weight: .bold))
                .foregroundColor(isActive ? .white : .secondary)
            
            Rectangle()
                .fill(isActive ? (severity == .critical ? Color.red : Color.orange) : Color.secondary.opacity(0.2))
                .frame(height: 6)
                .cornerRadius(3)
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 6)
        .frame(maxWidth: .infinity)
        .background(isActive ? (severity == .critical ? Color.red.opacity(0.2) : Color.orange.opacity(0.2)) : Color.secondary.opacity(0.05))
        .cornerRadius(8)
    }
}
