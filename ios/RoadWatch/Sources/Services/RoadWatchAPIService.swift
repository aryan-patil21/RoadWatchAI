import Foundation

public final class RoadWatchAPIService {
    public static let shared = RoadWatchAPIService()
    
    public var baseURL: URL = URL(string: "http://127.0.0.1:8000")!
    
    private let session: URLSession
    private let decoder: JSONDecoder
    
    public init(session: URLSession = .shared) {
        self.session = session
        self.decoder = JSONDecoder()
    }
    
    public func fetchHealth() async throws -> SystemHealthDTO {
        let url = baseURL.appendingPathComponent("health")
        let (data, response) = try await session.data(from: url)
        
        guard let httpResponse = response as? HTTPURLResponse, (200...299).contains(httpResponse.statusCode) else {
            throw URLError(.badServerResponse)
        }
        
        return try decoder.decode(SystemHealthDTO.self, from: data)
    }
    
    public func fetchActiveAlert() async throws -> RiderAlertDTO? {
        let url = baseURL.appendingPathComponent("api/v1/alerts/active")
        let (data, response) = try await session.data(from: url)
        
        guard let httpResponse = response as? HTTPURLResponse, (200...299).contains(httpResponse.statusCode) else {
            throw URLError(.badServerResponse)
        }
        
        let activeResponse = try decoder.decode(ActiveAlertResponse.self, from: data)
        return activeResponse.alert
    }
    
    // MARK: - Interactive Simulation Scenarios (For testing without running server)
    public static func makeDemoScenarios() -> [RiderAlertDTO?] {
        let alert1 = RiderAlertDTO(
            alertId: "DEMO-0001",
            timestampSec: 1.2,
            frameIdx: 30,
            vehicleId: 1,
            level: "CAUTION",
            title: "CAUTION: Truck from your left",
            message: "Truck nearby (closing distance).",
            suggestedAction: "Monitor vehicle and maintain safe buffer.",
            riskScore: 42.0,
            direction: "left",
            soundCue: "assets/sounds/chime_caution.wav"
        )
        
        let alert2 = RiderAlertDTO(
            alertId: "DEMO-0002",
            timestampSec: 2.8,
            frameIdx: 70,
            vehicleId: 1,
            level: "CRITICAL",
            title: "DANGER: High-Risk Truck from your left",
            message: "Truck closing distance rapidly (from your left).",
            suggestedAction: "Slow down and give vehicle wide berth.",
            riskScore: 78.5,
            direction: "left",
            soundCue: "assets/sounds/chime_critical.wav"
        )
        
        let alert3 = RiderAlertDTO(
            alertId: "DEMO-0003",
            timestampSec: 4.5,
            frameIdx: 110,
            vehicleId: 29,
            level: "CAUTION",
            title: "CAUTION: Car from your right",
            message: "Car nearby (rapid approach).",
            suggestedAction: "Check right mirror and hold lane.",
            riskScore: 48.0,
            direction: "right",
            soundCue: "assets/sounds/chime_caution.wav"
        )
        
        return [nil, alert1, alert2, alert3, nil]
    }
}
