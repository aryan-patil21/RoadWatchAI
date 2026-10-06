import Foundation

public enum AlertSeverity: String, Codable, CaseIterable {
    case normal = "NORMAL"
    case info = "INFO"
    case caution = "CAUTION"
    case critical = "CRITICAL"
    
    public var titleText: String {
        switch self {
        case .normal, .info: return "ROAD CONDITIONS NORMAL"
        case .caution: return "CAUTION: HAZARD NEARBY"
        case .critical: return "EMERGENCY: HIGH RISK DETECTED"
        }
    }
}

public struct RiderAlertDTO: Codable, Identifiable, Equatable {
    public var id: String { alertId }
    public let alertId: String
    public let timestampSec: Double
    public let frameIdx: Int
    public let vehicleId: Int
    public let level: String
    public let title: String
    public let message: String
    public let suggestedAction: String
    public let riskScore: Double
    public let direction: String // "left", "center", "right", "ahead"
    public let soundCue: String
    
    enum CodingKeys: String, CodingKey {
        case alertId = "alert_id"
        case timestampSec = "timestamp_sec"
        case frameIdx = "frame_idx"
        case vehicleId = "vehicle_id"
        case level
        case title
        case message
        case suggestedAction = "suggested_action"
        case riskScore = "risk_score"
        case direction
        case soundCue = "sound_cue"
    }
    
    public var severity: AlertSeverity {
        if level.uppercased() == "CRITICAL" { return .critical }
        if level.uppercased() == "CAUTION" { return .caution }
        return .normal
    }
    
    public init(
        alertId: String,
        timestampSec: Double,
        frameIdx: Int,
        vehicleId: Int,
        level: String,
        title: String,
        message: String,
        suggestedAction: String,
        riskScore: Double,
        direction: String,
        soundCue: String
    ) {
        self.alertId = alertId
        self.timestampSec = timestampSec
        self.frameIdx = frameIdx
        self.vehicleId = vehicleId
        self.level = level
        self.title = title
        self.message = message
        self.suggestedAction = suggestedAction
        self.riskScore = riskScore
        self.direction = direction
        self.soundCue = soundCue
    }
}

public struct ActiveAlertResponse: Codable {
    public let hasActiveAlert: Bool
    public let alert: RiderAlertDTO?
    public let timestamp: Double
    
    enum CodingKeys: String, CodingKey {
        case hasActiveAlert = "has_active_alert"
        case alert
        case timestamp
    }
}

public struct SystemHealthDTO: Codable {
    public let status: String
    public let service: String
    public let version: String
    public let device: String
    public let framesProcessed: Int
    public let alertsFired: Int
    
    enum CodingKeys: String, CodingKey {
        case status
        case service
        case version
        case device
        case framesProcessed = "frames_processed"
        case alertsFired = "alerts_fired"
    }
}
