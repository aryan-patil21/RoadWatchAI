import Foundation
import Combine

@MainActor
public final class CockpitViewModel: ObservableObject {
    @Published public var activeAlert: RiderAlertDTO? = nil
    @Published public var isConnected: Bool = false
    @Published public var isSimulationMode: Bool = true
    @Published public var backendStatusText: String = "Simulation Mode Active"
    @Published public var recentAlerts: [RiderAlertDTO] = []
    
    private let apiService: RoadWatchAPIService
    private let audioService: AudioAlertService
    
    private var pollingTimer: Timer?
    private var simulationTimer: Timer?
    private var simulationIndex = 0
    private let demoScenarios = RoadWatchAPIService.makeDemoScenarios()
    
    private var lastObservedAlertId: String? = nil
    
    public init(
        apiService: RoadWatchAPIService = .shared,
        audioService: AudioAlertService? = nil
    ) {
        self.apiService = apiService
        self.audioService = audioService ?? AudioAlertService.shared
        startSimulation()
    }
    
    // MARK: - Live Backend Mode (WebSocket with Polling Health Check)
    private var webSocketTask: URLSessionWebSocketTask?
    
    public func enableLiveBackendMode(baseURLString: String = "http://127.0.0.1:8000") {
        if let url = URL(string: baseURLString) {
            apiService.baseURL = url
        }
        isSimulationMode = false
        stopSimulation()
        startWebSocketStreaming()
        startPolling()
    }
    
    public func startWebSocketStreaming() {
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        webSocketTask = apiService.createWebSocketTask(
            onAlert: { [weak self] alert in
                Task { @MainActor [weak self] in
                    self?.handleReceivedAlert(alert)
                }
            },
            onError: { _ in
                // Falls back gracefully to the polling loop
            }
        )
    }
    
    public func startPolling() {
        stopPolling()
        pollingTimer = Timer.scheduledTimer(withTimeInterval: 0.35, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                await self?.pollBackend()
            }
        }
    }
    
    public func stopPolling() {
        pollingTimer?.invalidate()
        pollingTimer = nil
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        webSocketTask = nil
    }
    
    private func pollBackend() async {
        do {
            let health = try await apiService.fetchHealth()
            isConnected = true
            backendStatusText = "Connected (\(health.device)) | \(health.framesProcessed) Frames"
            
            let alert = try await apiService.fetchActiveAlert()
            handleReceivedAlert(alert)
        } catch {
            isConnected = false
            backendStatusText = "Backend Offline: Retrying..."
        }
    }
    
    // MARK: - Simulation Mode (Demo on Device)
    public func startSimulation() {
        isSimulationMode = true
        stopPolling()
        stopSimulation()
        
        simulationTimer = Timer.scheduledTimer(withTimeInterval: 3.5, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.stepSimulation()
            }
        }
    }
    
    public func stopSimulation() {
        simulationTimer?.invalidate()
        simulationTimer = nil
    }
    
    public func stepSimulation() {
        simulationIndex = (simulationIndex + 1) % demoScenarios.count
        let nextAlert = demoScenarios[simulationIndex]
        handleReceivedAlert(nextAlert)
    }
    
    // MARK: - Alert Processing
    private func handleReceivedAlert(_ alert: RiderAlertDTO?) {
        if let alert = alert {
            if alert.alertId != lastObservedAlertId {
                lastObservedAlertId = alert.alertId
                activeAlert = alert
                
                // Prepend to history without duplicates
                if !recentAlerts.contains(where: { $0.alertId == alert.alertId }) {
                    recentAlerts.insert(alert, at: 0)
                    if recentAlerts.count > 10 { recentAlerts.removeLast() }
                }
                
                // Play open-source sound and haptics
                audioService.playAlert(severity: alert.severity)
            }
        } else {
            activeAlert = nil
            lastObservedAlertId = nil
        }
    }
    
    deinit {
        pollingTimer?.invalidate()
        simulationTimer?.invalidate()
    }
}
