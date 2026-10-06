import Foundation
import AVFoundation

#if canImport(UIKit)
import UIKit
#endif

@MainActor
public final class AudioAlertService: NSObject, AVAudioPlayerDelegate {
    public static let shared = AudioAlertService()
    
    private var audioPlayer: AVAudioPlayer?
    
    private override init() {
        super.init()
        setupAudioSession()
    }
    
    private func setupAudioSession() {
        #if os(iOS)
        do {
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
            try AVAudioSession.sharedInstance().setActive(true)
        } catch {
            print("[AudioAlertService] Could not configure audio session: \(error)")
        }
        #endif
    }
    
    public func playAlert(severity: AlertSeverity) {
        let soundFileName = (severity == .critical) ? "chime_critical" : "chime_caution"
        
        // Try loading from Bundle.module (SwiftPM resource)
        if let url = Bundle.module.url(forResource: soundFileName, withExtension: "wav") {
            playWavFile(at: url)
        } else if let mainUrl = Bundle.main.url(forResource: soundFileName, withExtension: "wav") {
            playWavFile(at: mainUrl)
        } else {
            print("[AudioAlertService] Sound file '\(soundFileName).wav' not found in bundle.")
        }
        
        // Trigger haptic vibration on physical iOS devices
        triggerHaptic(for: severity)
    }
    
    private func playWavFile(at url: URL) {
        do {
            audioPlayer = try AVAudioPlayer(contentsOf: url)
            audioPlayer?.delegate = self
            audioPlayer?.prepareToPlay()
            audioPlayer?.play()
        } catch {
            print("[AudioAlertService] Audio playback error: \(error)")
        }
    }
    
    private func triggerHaptic(for severity: AlertSeverity) {
        #if os(iOS)
        DispatchQueue.main.async {
            let generator = UINotificationFeedbackGenerator()
            generator.prepare()
            switch severity {
            case .critical:
                generator.notificationOccurred(.error)
            case .caution:
                generator.notificationOccurred(.warning)
            case .normal, .info:
                break
            }
        }
        #endif
    }
}
