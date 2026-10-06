import XCTest
@testable import RoadWatch

final class RoadWatchTests: XCTestCase {
    
    func testRiderAlertDTODecoding() throws {
        let jsonString = """
        {
            "alert_id": "ALERT-00042",
            "timestamp_sec": 3.75,
            "frame_idx": 95,
            "vehicle_id": 7,
            "level": "CRITICAL",
            "title": "DANGER: High-Risk Truck from your left",
            "message": "Truck closing distance rapidly (from your left).",
            "suggested_action": "Slow down and give vehicle wide berth.",
            "risk_score": 82.5,
            "direction": "left",
            "sound_cue": "assets/sounds/chime_critical.wav"
        }
        """
        
        let data = try XCTUnwrap(jsonString.data(using: .utf8))
        let alert = try JSONDecoder().decode(RiderAlertDTO.self, from: data)
        
        XCTAssertEqual(alert.alertId, "ALERT-00042")
        XCTAssertEqual(alert.frameIdx, 95)
        XCTAssertEqual(alert.vehicleId, 7)
        XCTAssertEqual(alert.level, "CRITICAL")
        XCTAssertEqual(alert.severity, .critical)
        XCTAssertEqual(alert.direction, "left")
        XCTAssertEqual(alert.riskScore, 82.5)
        XCTAssertEqual(alert.soundCue, "assets/sounds/chime_critical.wav")
    }
    
    func testActiveAlertResponseDecoding() throws {
        let jsonString = """
        {
            "has_active_alert": true,
            "alert": {
                "alert_id": "ALERT-00010",
                "timestamp_sec": 1.20,
                "frame_idx": 30,
                "vehicle_id": 3,
                "level": "CAUTION",
                "title": "CAUTION: Car from your right",
                "message": "Car nearby (closing distance).",
                "suggested_action": "Monitor vehicle and maintain safe buffer.",
                "risk_score": 45.0,
                "direction": "right",
                "sound_cue": "assets/sounds/chime_caution.wav"
            },
            "timestamp": 1720000000.0
        }
        """
        
        let data = try XCTUnwrap(jsonString.data(using: .utf8))
        let response = try JSONDecoder().decode(ActiveAlertResponse.self, from: data)
        
        XCTAssertTrue(response.hasActiveAlert)
        XCTAssertNotNil(response.alert)
        XCTAssertEqual(response.alert?.level, "CAUTION")
        XCTAssertEqual(response.alert?.severity, .caution)
    }
    
    func testSystemHealthDTODecoding() throws {
        let jsonString = """
        {
            "status": "healthy",
            "service": "RoadWatch AI Perception & Warning API",
            "version": "1.0.0",
            "device": "mps",
            "frames_processed": 1895,
            "alerts_fired": 4
        }
        """
        
        let data = try XCTUnwrap(jsonString.data(using: .utf8))
        let health = try JSONDecoder().decode(SystemHealthDTO.self, from: data)
        
        XCTAssertEqual(health.status, "healthy")
        XCTAssertEqual(health.device, "mps")
        XCTAssertEqual(health.framesProcessed, 1895)
        XCTAssertEqual(health.alertsFired, 4)
    }
    
    func testDemoScenariosProvideValidSequence() {
        let scenarios = RoadWatchAPIService.makeDemoScenarios()
        XCTAssertGreaterThan(scenarios.count, 2)
        
        // Ensure nil (normal) and caution/critical scenarios are present
        let nonNilScenarios = scenarios.compactMap { $0 }
        XCTAssertFalse(nonNilScenarios.isEmpty)
        
        let severities = Set(nonNilScenarios.map { $0.severity })
        XCTAssertTrue(severities.contains(.caution))
        XCTAssertTrue(severities.contains(.critical))
    }
}
