#!/usr/bin/env python3
"""
Pattern Matching Agent
Matches transactions against known fraud patterns.
"""

from crewai import Agent
from crewai.tools import tool
from typing import Dict, List, Optional
import json
from datetime import datetime
from enum import Enum


class FraudPatternType(Enum):
    """Known fraud pattern types"""
    CARD_NOT_PRESENT = "card_not_present"
    CARD_TESTING = "card_testing"
    ACCOUNT_TAKEOVER = "account_takeover"
    BUST_OUT = "bust_out"
    FRIENDLY_FRAUD = "friendly_fraud"
    SYNTHETIC_IDENTITY = "synthetic_identity"
    VELOCITY_ABUSE = "velocity_abuse"
    GEOGRAPHIC_IMPOSSIBLE = "geographic_impossible"
    HIGH_RISK_MERCHANT = "high_risk_merchant"
    UNUSUAL_TIMING = "unusual_timing"


# Known fraud pattern definitions
FRAUD_PATTERNS = {
    FraudPatternType.CARD_NOT_PRESENT.value: {
        "name": "Card Not Present Fraud",
        "description": "Fraudulent online transactions using stolen card details",
        "indicators": [
            "Online transaction",
            "New merchant for user",
            "High-value purchase",
            "Electronics or gift cards",
            "Different shipping address"
        ],
        "risk_weight": 0.8,
        "common_mccs": [5732, 5944, 5311, 5999]
    },
    FraudPatternType.CARD_TESTING.value: {
        "name": "Card Testing",
        "description": "Multiple small transactions to test if card is valid",
        "indicators": [
            "Multiple small transactions",
            "Rapid succession (< 5 min apart)",
            "Different merchants",
            "Often followed by larger purchase",
            "Low-value amounts ($1-$10)"
        ],
        "risk_weight": 0.9,
        "amount_threshold": 10.00
    },
    FraudPatternType.ACCOUNT_TAKEOVER.value: {
        "name": "Account Takeover",
        "description": "Legitimate account compromised by fraudster",
        "indicators": [
            "Sudden change in spending pattern",
            "New device or location",
            "Password/email recently changed",
            "High-value purchases after dormancy",
            "Different geographic location"
        ],
        "risk_weight": 0.95
    },
    FraudPatternType.BUST_OUT.value: {
        "name": "Bust Out Fraud",
        "description": "Building credit then maxing out with no intent to pay",
        "indicators": [
            "Gradual credit limit increase usage",
            "Multiple cash advances",
            "Balance transfer attempts",
            "Recent credit limit increase",
            "Maxing out all available credit"
        ],
        "risk_weight": 0.85
    },
    FraudPatternType.VELOCITY_ABUSE.value: {
        "name": "Velocity Abuse",
        "description": "Abnormally high transaction frequency",
        "indicators": [
            "More than 5 transactions per hour",
            "More than 20 transactions per day",
            "Increasing amounts over short period",
            "Multiple merchants same category",
            "Geographic dispersion"
        ],
        "risk_weight": 0.75,
        "thresholds": {
            "per_hour": 5,
            "per_day": 20
        }
    },
    FraudPatternType.GEOGRAPHIC_IMPOSSIBLE.value: {
        "name": "Geographic Impossibility",
        "description": "Transactions from locations impossible to reach in time",
        "indicators": [
            "Transactions in different cities/countries within hours",
            "Distance impossible to travel",
            "Card-present at multiple locations",
            "Different time zones same day"
        ],
        "risk_weight": 0.95
    },
    FraudPatternType.HIGH_RISK_MERCHANT.value: {
        "name": "High Risk Merchant",
        "description": "Transaction at merchant known for fraud",
        "indicators": [
            "Merchant on watch list",
            "High chargeback rate merchant",
            "Recently reported compromised",
            "Unusual merchant category for user"
        ],
        "risk_weight": 0.7,
        "high_risk_mccs": [5944, 7995, 5732, 6011, 4829]
    },
    FraudPatternType.UNUSUAL_TIMING.value: {
        "name": "Unusual Timing",
        "description": "Transaction at unusual time for user",
        "indicators": [
            "Late night transaction (1-5 AM)",
            "Outside user's typical hours",
            "Holiday/weekend unusual pattern",
            "First transaction in weeks at odd time"
        ],
        "risk_weight": 0.5,
        "suspicious_hours": [1, 2, 3, 4, 5]
    }
}


class PatternMatchingAgent:
    """
    Agent responsible for matching transactions against known fraud patterns.
    Uses pattern recognition to identify potential fraud types.
    """

    def __init__(self, llm=None):
        self.llm = llm
        self.patterns = FRAUD_PATTERNS

    def create_agent(self) -> Agent:
        """Create the CrewAI agent"""
        return Agent(
            role="Fraud Pattern Analyst",
            goal="Identify which known fraud patterns match the suspicious transaction",
            backstory="""You are a fraud pattern recognition expert with years of experience
            analyzing credit card fraud. You know all the common fraud patterns and their
            indicators. Your job is to compare the flagged transaction against known fraud
            patterns and identify the most likely type of fraud being attempted.
            You are analytical, thorough, and always provide confidence scores.""",
            tools=[
                self.match_fraud_patterns,
                self.get_pattern_details,
                self.calculate_pattern_similarity,
                self.get_historical_pattern_matches,
                self.analyze_velocity_pattern
            ],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    @tool("Match Fraud Patterns")
    def match_fraud_patterns(transaction_data: str) -> str:
        """
        Match a transaction against all known fraud patterns.

        Args:
            transaction_data: JSON string with transaction details

        Returns:
            JSON string with pattern matches and confidence scores
        """
        try:
            transaction = json.loads(transaction_data)
        except json.JSONDecodeError:
            transaction = {"raw": transaction_data}

        matches = []

        # Check each pattern
        for pattern_type, pattern in FRAUD_PATTERNS.items():
            match_score = 0
            matched_indicators = []

            # Check amount-based patterns
            amount = transaction.get('amount', transaction.get('Amount_clean', 0))
            if isinstance(amount, str):
                amount = float(amount.replace('$', '').replace(',', ''))

            # Card testing pattern
            if pattern_type == FraudPatternType.CARD_TESTING.value:
                if amount <= pattern.get('amount_threshold', 10):
                    match_score += 0.3
                    matched_indicators.append("Low transaction amount")

            # Velocity pattern
            if pattern_type == FraudPatternType.VELOCITY_ABUSE.value:
                trans_count_1h = transaction.get('trans_count_1h', 0)
                trans_count_24h = transaction.get('trans_count_24h', 0)
                thresholds = pattern.get('thresholds', {})
                if trans_count_1h >= thresholds.get('per_hour', 5):
                    match_score += 0.5
                    matched_indicators.append(f"High velocity: {trans_count_1h} transactions/hour")
                if trans_count_24h >= thresholds.get('per_day', 20):
                    match_score += 0.3
                    matched_indicators.append(f"High daily volume: {trans_count_24h} transactions")

            # High risk merchant
            if pattern_type == FraudPatternType.HIGH_RISK_MERCHANT.value:
                mcc = transaction.get('MCC', transaction.get('mcc', 0))
                if mcc in pattern.get('high_risk_mccs', []):
                    match_score += 0.4
                    matched_indicators.append(f"High-risk MCC: {mcc}")

            # Card not present
            if pattern_type == FraudPatternType.CARD_NOT_PRESENT.value:
                is_online = transaction.get('is_online', 0)
                use_chip = transaction.get('Use Chip', '')
                if is_online or use_chip == 'Online Transaction':
                    match_score += 0.3
                    matched_indicators.append("Online transaction")
                if amount > 500:
                    match_score += 0.2
                    matched_indicators.append("High-value online purchase")
                mcc = transaction.get('MCC', transaction.get('mcc', 0))
                if mcc in pattern.get('common_mccs', []):
                    match_score += 0.2
                    matched_indicators.append(f"Common fraud MCC: {mcc}")

            # Unusual timing
            if pattern_type == FraudPatternType.UNUSUAL_TIMING.value:
                hour = transaction.get('hour', 12)
                if hour in pattern.get('suspicious_hours', []):
                    match_score += 0.6
                    matched_indicators.append(f"Suspicious hour: {hour}:00")

            # Geographic impossibility
            if pattern_type == FraudPatternType.GEOGRAPHIC_IMPOSSIBLE.value:
                is_different_state = transaction.get('is_different_state', 0)
                if is_different_state:
                    match_score += 0.3
                    matched_indicators.append("Different state from home")

            # Only include if there's a match
            if match_score > 0:
                matches.append({
                    "pattern_type": pattern_type,
                    "pattern_name": pattern["name"],
                    "confidence": min(match_score, 1.0),
                    "matched_indicators": matched_indicators,
                    "risk_weight": pattern["risk_weight"],
                    "description": pattern["description"]
                })

        # Sort by confidence
        matches.sort(key=lambda x: x['confidence'], reverse=True)

        result = {
            "transaction_analyzed": transaction.get('transaction_id', 'unknown'),
            "patterns_checked": len(FRAUD_PATTERNS),
            "matches_found": len(matches),
            "top_matches": matches[:5],
            "highest_confidence": matches[0]['confidence'] if matches else 0,
            "recommended_action": _get_recommended_action(matches)
        }

        return json.dumps(result, indent=2)

    @tool("Get Pattern Details")
    def get_pattern_details(pattern_type: str) -> str:
        """
        Get detailed information about a specific fraud pattern.

        Args:
            pattern_type: The type of fraud pattern

        Returns:
            JSON string with pattern details
        """
        if pattern_type in FRAUD_PATTERNS:
            pattern = FRAUD_PATTERNS[pattern_type]
            return json.dumps({
                "pattern_type": pattern_type,
                **pattern
            }, indent=2)
        else:
            return json.dumps({
                "error": f"Unknown pattern type: {pattern_type}",
                "available_patterns": list(FRAUD_PATTERNS.keys())
            }, indent=2)

    @tool("Calculate Pattern Similarity")
    def calculate_pattern_similarity(
        transaction_data: str,
        pattern_type: str
    ) -> str:
        """
        Calculate detailed similarity score between transaction and a specific pattern.

        Args:
            transaction_data: JSON string with transaction details
            pattern_type: The fraud pattern to compare against

        Returns:
            JSON string with similarity analysis
        """
        try:
            transaction = json.loads(transaction_data)
        except json.JSONDecodeError:
            transaction = {}

        if pattern_type not in FRAUD_PATTERNS:
            return json.dumps({"error": f"Unknown pattern: {pattern_type}"})

        pattern = FRAUD_PATTERNS[pattern_type]

        analysis = {
            "pattern_type": pattern_type,
            "pattern_name": pattern["name"],
            "indicators_analysis": [],
            "overall_similarity": 0,
            "risk_assessment": "Low"
        }

        # Analyze each indicator
        matched = 0
        for indicator in pattern["indicators"]:
            indicator_matched = False
            # Simple keyword matching for demo
            indicator_lower = indicator.lower()

            if "online" in indicator_lower and transaction.get('is_online', 0):
                indicator_matched = True
            elif "high-value" in indicator_lower and transaction.get('Amount_clean', 0) > 500:
                indicator_matched = True
            elif "late night" in indicator_lower and transaction.get('hour', 12) in [1, 2, 3, 4, 5]:
                indicator_matched = True

            if indicator_matched:
                matched += 1

            analysis["indicators_analysis"].append({
                "indicator": indicator,
                "matched": indicator_matched
            })

        # Calculate similarity
        total_indicators = len(pattern["indicators"])
        analysis["overall_similarity"] = matched / total_indicators if total_indicators > 0 else 0

        # Risk assessment
        if analysis["overall_similarity"] >= 0.7:
            analysis["risk_assessment"] = "High"
        elif analysis["overall_similarity"] >= 0.4:
            analysis["risk_assessment"] = "Medium"
        else:
            analysis["risk_assessment"] = "Low"

        return json.dumps(analysis, indent=2)

    @tool("Get Historical Pattern Matches")
    def get_historical_pattern_matches(pattern_type: str, days: int = 30) -> str:
        """
        Get historical data on pattern matches.

        Args:
            pattern_type: The fraud pattern type
            days: Number of days of history

        Returns:
            JSON string with historical pattern data
        """
        # In production, this would query a database
        historical = {
            "pattern_type": pattern_type,
            "period_days": days,
            "total_matches": 0,
            "confirmed_fraud": 0,
            "false_positives": 0,
            "precision": 0,
            "trend": "stable",
            "notes": "Connect to historical database for real data"
        }

        return json.dumps(historical, indent=2)

    @tool("Analyze Velocity Pattern")
    def analyze_velocity_pattern(user_id: str, hours: int = 24) -> str:
        """
        Analyze transaction velocity for a user.

        Args:
            user_id: The user identifier
            hours: Hours to analyze (default 24)

        Returns:
            JSON string with velocity analysis
        """
        velocity = {
            "user_id": user_id,
            "analysis_period_hours": hours,
            "transaction_count": 0,
            "average_amount": 0,
            "unique_merchants": 0,
            "geographic_spread": 0,
            "velocity_score": 0,
            "anomaly_detected": False,
            "comparison_to_baseline": {
                "typical_daily_count": 0,
                "current_vs_typical_ratio": 0
            }
        }

        return json.dumps(velocity, indent=2)


def _get_recommended_action(matches: List[Dict]) -> str:
    """Determine recommended action based on pattern matches"""
    if not matches:
        return "APPROVE - No fraud patterns matched"

    max_confidence = matches[0]['confidence']
    max_risk = max(m['risk_weight'] for m in matches)

    combined_score = (max_confidence + max_risk) / 2

    if combined_score >= 0.8:
        return "DECLINE - High confidence fraud pattern match"
    elif combined_score >= 0.6:
        return "MANUAL_REVIEW - Moderate fraud pattern match"
    elif combined_score >= 0.4:
        return "STEP_UP_AUTH - Some fraud indicators present"
    else:
        return "APPROVE_WITH_MONITORING - Minor indicators detected"


def get_all_pattern_types() -> List[str]:
    """Get list of all available fraud pattern types"""
    return [p.value for p in FraudPatternType]
