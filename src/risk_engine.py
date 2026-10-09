"""
RoadWatch AI - Milestone 6: Interpretable Risk Estimation Engine
-----------------------------------------------------------------
Translates numerical behavioural features into road-safety risk scores [0 - 100]
and discrete safety levels (LOW, MEDIUM, HIGH) using a transparent, rule-based approach.

Core Design Philosophy:
Risk is computed solely from observable dynamics (proximity, approach rate, TTC,
lane cutting, trajectory conflict), NEVER from vehicle brand, model, or driver identity.
"""

from typing import Dict, Any, Tuple


class RiskEngine:
    """
    Computes road safety risk scores and levels from vehicle behavioural features.
    """

    def __init__(
        self,
        low_threshold: float = 40.0,
        high_threshold: float = 70.0,
    ):
        """
        Args:
            low_threshold: Scores below this are classified as LOW_RISK.
            high_threshold: Scores at or above this are classified as HIGH_RISK.
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def evaluate_risk(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates risk score (0-100) and risk level for a single vehicle observation.

        Args:
            features: Dictionary of behavioural features from BehaviouralFeatureExtractor.

        Returns:
            Dictionary with:
            - risk_score: float [0.0 - 100.0]
            - risk_level: str ('LOW', 'MEDIUM', 'HIGH')
            - primary_factor: str (explaining the main source of danger)
            - alert_message: str (human-readable warning recommendation)
        """
        proximity = features.get("proximity_score", 0.0)
        approach_rate = features.get("approach_rate", 0.0)
        v_lat = features.get("v_lat_proxy", 0.0)
        ttc = features.get("ttc_proxy_sec", 99.9)
        in_ego = features.get("in_ego_corridor", 0)
        is_cutting = features.get("is_cutting_lane", 0)
        sudden_brake = features.get("sudden_braking_flag", 0)

        score = 0.0
        factors = []

        # 1. Proximity component (up to 30 points)
        # Closer objects naturally have higher baseline risk
        prox_points = min(30.0, proximity * 30.0)
        score += prox_points
        if proximity > 0.65:
            factors.append("very close proximity")

        # 2. Approach Rate & Time-To-Collision (up to 35 points)
        # Objects far on the horizon (proximity < 0.30) naturally expand slowly and pose no immediate collision threat
        if proximity > 0.30:
            if approach_rate > 0.15:
                score += 25.0
                factors.append("rapid approach")
            elif approach_rate > 0.05:
                score += 15.0
                factors.append("closing distance")

            if ttc < 2.5:
                score += 20.0
                factors.append(f"critical TTC ({ttc:.1f}s)")
            elif ttc < 5.0:
                score += 10.0
                factors.append(f"low TTC ({ttc:.1f}s)")

        # 3. Trajectory Conflict & Lane Cutting (up to 30 points)
        # Check whether vehicle is diverging away from rider's path (e.g. opposing traffic across divider)
        is_diverging_away = (
            (not in_ego)
            and not is_cutting
            and ((v_lat <= 5.0 and features.get("cx", 0.0) < 350.0) or (v_lat >= -5.0 and features.get("cx", 0.0) > 550.0))
        )

        if is_cutting:
            score += 30.0
            factors.append("aggressive lane cutting")
        elif in_ego:
            score += 15.0
            factors.append("directly in rider path")
        elif is_diverging_away:
            # Vehicle is in adjacent or opposing lane (e.g. across a divider) and NOT moving into rider path.
            # Its lateral trajectory is completely non-conflicting.
            # Neutralize rapid approach and critical TTC points since it will pass safely on the other side.
            score = min(score, 25.0)  # Cap score well below CAUTION threshold (40.0)
            factors = [f for f in factors if "TTC" not in f and "approach" not in f and "closing" not in f]
            if factors:
                factors.append("safe opposing/adjacent traffic")
        else:
            # Adjacent vehicle with minor lateral drift
            if proximity < 0.60 and approach_rate <= 0.05:
                score = max(0.0, score - 10.0)


        # 4. Sudden Deceleration / Braking Event (up to 15 points)
        # Only relevant for vehicles directly in front or cutting into our lane
        if sudden_brake and proximity > 0.35 and (in_ego or is_cutting):
            score += 15.0
            factors.append("sudden braking ahead")

        # Diverging vehicle across divider / separate lane: hard cap to SAFE baseline
        if is_diverging_away:
            score = min(score, 20.0)

        # Clamp score between 0.0 and 100.0
        final_score = min(100.0, max(0.0, round(score, 1)))

        # Categorize into discrete risk levels
        if final_score >= self.high_threshold:
            risk_level = "HIGH"
            alert_message = f"DANGER: {', '.join(factors) if factors else 'high collision risk'}"
        elif final_score >= self.low_threshold:
            risk_level = "MEDIUM"
            alert_message = f"CAUTION: {', '.join(factors) if factors else 'monitor vehicle'}"
        else:
            risk_level = "LOW"
            alert_message = "Normal road conditions"

        primary_factor = factors[0] if factors else "none"

        return {
            "vehicle_id": features.get("vehicle_id"),
            "risk_score": final_score,
            "risk_level": risk_level,
            "primary_factor": primary_factor,
            "alert_message": alert_message,
            "all_factors": factors,
        }
