#!/usr/bin/env python3
"""
Transaction Rating System for Fraud Detection
Rates transactions based on fraud probability and risk factors.
"""

import numpy as np
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json


class RiskLevel(Enum):
    """Risk level categories"""
    VERY_LOW = "Very Low"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"
    CRITICAL = "Critical"


class TransactionRating(Enum):
    """Transaction rating grades"""
    A_PLUS = "A+"   # Excellent - Very low risk
    A = "A"         # Good - Low risk
    B = "B"         # Acceptable - Moderate risk
    C = "C"         # Caution - Elevated risk
    D = "D"         # Warning - High risk
    F = "F"         # Decline - Critical risk


@dataclass
class RiskFactor:
    """Individual risk factor"""
    name: str
    score: float  # 0-100 contribution to risk
    description: str
    weight: float = 1.0


class TransactionRatingEngine:
    """
    Comprehensive transaction rating engine that combines:
    - ML fraud probability
    - Rule-based risk factors
    - Contextual analysis

    Output: A letter grade (A+ to F) and detailed risk breakdown
    """

    # Risk factor weights
    WEIGHTS = {
        'fraud_probability': 0.40,    # ML model output
        'amount_risk': 0.15,          # Transaction amount risk
        'velocity_risk': 0.10,        # Transaction velocity
        'time_risk': 0.08,            # Time-based risk
        'geographic_risk': 0.08,      # Geographic anomaly
        'merchant_risk': 0.07,        # Merchant category risk
        'device_risk': 0.07,          # Transaction type risk
        'behavioral_risk': 0.05,      # Behavioral anomaly
    }

    # High-risk MCCs
    HIGH_RISK_MCCS = {
        6011: "ATM Cash Advance",
        5944: "Jewelry",
        7995: "Gambling",
        5732: "Electronics",
        4829: "Wire Transfer",
        6051: "Quasi Cash",
        5999: "Misc Retail"
    }

    def __init__(self):
        self.rating_thresholds = {
            TransactionRating.A_PLUS: (0, 10),
            TransactionRating.A: (10, 25),
            TransactionRating.B: (25, 45),
            TransactionRating.C: (45, 65),
            TransactionRating.D: (65, 85),
            TransactionRating.F: (85, 100),
        }

        self.risk_thresholds = {
            RiskLevel.VERY_LOW: (0, 10),
            RiskLevel.LOW: (10, 30),
            RiskLevel.MEDIUM: (30, 50),
            RiskLevel.HIGH: (50, 75),
            RiskLevel.VERY_HIGH: (75, 90),
            RiskLevel.CRITICAL: (90, 100),
        }

    def rate_transaction(
        self,
        transaction: Dict,
        fraud_probability: float,
        user_profile: Optional[Dict] = None,
        historical_stats: Optional[Dict] = None
    ) -> Dict:
        """
        Rate a transaction and provide comprehensive risk assessment.

        Args:
            transaction: Transaction details
            fraud_probability: ML model fraud probability (0-1)
            user_profile: User spending profile (optional)
            historical_stats: Historical transaction stats (optional)

        Returns:
            Rating result with grade, score, risk level, and factors
        """

        risk_factors = []

        # 1. ML Fraud Probability (primary signal)
        fraud_score = fraud_probability * 100
        risk_factors.append(RiskFactor(
            name="ML Fraud Score",
            score=fraud_score,
            description=f"Model prediction: {fraud_probability:.2%} fraud probability",
            weight=self.WEIGHTS['fraud_probability']
        ))

        # 2. Amount Risk
        amount_factor = self._assess_amount_risk(transaction, user_profile)
        risk_factors.append(amount_factor)

        # 3. Velocity Risk
        velocity_factor = self._assess_velocity_risk(transaction, historical_stats)
        risk_factors.append(velocity_factor)

        # 4. Time Risk
        time_factor = self._assess_time_risk(transaction)
        risk_factors.append(time_factor)

        # 5. Geographic Risk
        geo_factor = self._assess_geographic_risk(transaction, user_profile)
        risk_factors.append(geo_factor)

        # 6. Merchant Risk
        merchant_factor = self._assess_merchant_risk(transaction)
        risk_factors.append(merchant_factor)

        # 7. Device/Channel Risk
        device_factor = self._assess_device_risk(transaction)
        risk_factors.append(device_factor)

        # 8. Behavioral Risk
        behavioral_factor = self._assess_behavioral_risk(transaction, user_profile)
        risk_factors.append(behavioral_factor)

        # Calculate composite score
        composite_score = self._calculate_composite_score(risk_factors)

        # Determine rating and risk level
        rating = self._get_rating(composite_score)
        risk_level = self._get_risk_level(composite_score)

        # Generate recommendation
        recommendation = self._generate_recommendation(rating, risk_level, risk_factors)

        # Build result
        result = {
            'transaction_rating': rating.value,
            'rating_score': round(composite_score, 2),
            'risk_level': risk_level.value,
            'fraud_probability': fraud_probability,
            'risk_factors': [
                {
                    'name': f.name,
                    'score': round(f.score, 2),
                    'weighted_contribution': round(f.score * f.weight, 2),
                    'description': f.description
                }
                for f in risk_factors
            ],
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat(),
            'should_approve': rating in [TransactionRating.A_PLUS, TransactionRating.A, TransactionRating.B],
            'requires_review': rating in [TransactionRating.C, TransactionRating.D],
            'should_decline': rating == TransactionRating.F
        }

        return result

    def _assess_amount_risk(self, transaction: Dict, user_profile: Optional[Dict]) -> RiskFactor:
        """Assess risk based on transaction amount"""
        amount = self._parse_amount(transaction.get('Amount', '$0'))

        score = 0
        description = f"Transaction amount: ${amount:.2f}"

        # Base amount thresholds
        if amount > 5000:
            score = 80
            description += " - Very high amount"
        elif amount > 2000:
            score = 60
            description += " - High amount"
        elif amount > 1000:
            score = 40
            description += " - Elevated amount"
        elif amount > 500:
            score = 20
            description += " - Moderate amount"
        else:
            score = 5
            description += " - Normal amount"

        # Compare to user profile if available
        if user_profile:
            avg = user_profile.get('user_avg_amount_30d', amount)
            std = user_profile.get('user_std_amount_30d', 1)
            zscore = (amount - avg) / (std + 0.01)

            if zscore > 4:
                score = min(100, score + 30)
                description += f" | {zscore:.1f}x std deviation from user average"
            elif zscore > 2:
                score = min(100, score + 15)
                description += f" | {zscore:.1f}x std deviation from user average"

        return RiskFactor(
            name="Amount Risk",
            score=score,
            description=description,
            weight=self.WEIGHTS['amount_risk']
        )

    def _assess_velocity_risk(self, transaction: Dict, historical_stats: Optional[Dict]) -> RiskFactor:
        """Assess risk based on transaction velocity"""
        score = 0
        description = "Transaction velocity assessment"

        if historical_stats:
            # Check transactions in last hour
            count_1h = historical_stats.get('trans_count_1h', 0)
            count_24h = historical_stats.get('trans_count_24h', 0)
            time_since_last = historical_stats.get('time_since_last', 24)

            if count_1h > 5:
                score = 80
                description = f"{count_1h} transactions in last hour - Very high velocity"
            elif count_1h > 3:
                score = 60
                description = f"{count_1h} transactions in last hour - High velocity"
            elif count_24h > 15:
                score = 40
                description = f"{count_24h} transactions in last 24h - Elevated daily activity"
            elif time_since_last < 0.1:  # Less than 6 minutes
                score = 50
                description = f"Only {time_since_last*60:.1f} minutes since last transaction"
            else:
                score = 5
                description = f"Normal velocity - {count_24h} transactions in 24h"
        else:
            score = 10
            description = "No velocity data available"

        return RiskFactor(
            name="Velocity Risk",
            score=score,
            description=description,
            weight=self.WEIGHTS['velocity_risk']
        )

    def _assess_time_risk(self, transaction: Dict) -> RiskFactor:
        """Assess risk based on transaction time"""
        time_str = transaction.get('Time', '12:00')
        hour = int(time_str.split(':')[0])

        # Determine day of week
        year = transaction.get('Year', datetime.now().year)
        month = transaction.get('Month', datetime.now().month)
        day = transaction.get('Day', datetime.now().day)
        try:
            dt = datetime(year, month, day)
            day_of_week = dt.weekday()
            is_weekend = day_of_week >= 5
        except:
            is_weekend = False

        score = 0
        description = f"Transaction at {time_str}"

        # Late night transactions (1 AM - 5 AM) are higher risk
        if 1 <= hour <= 5:
            score = 70
            description += " - Late night hours (highest risk)"
        elif hour < 6 or hour > 22:
            score = 40
            description += " - Off-hours transaction"
        elif is_weekend:
            score = 15
            description += " - Weekend transaction"
        else:
            score = 5
            description += " - Normal business hours"

        return RiskFactor(
            name="Time Risk",
            score=score,
            description=description,
            weight=self.WEIGHTS['time_risk']
        )

    def _assess_geographic_risk(self, transaction: Dict, user_profile: Optional[Dict]) -> RiskFactor:
        """Assess risk based on geographic factors"""
        state = transaction.get('Merchant State', '')
        trans_type = transaction.get('Use Chip', '')

        score = 0
        description = f"Location: {state if state else 'Online'}"

        # Online transaction
        if not state or trans_type == 'Online Transaction':
            score = 25
            description = "Online transaction - moderate risk"
        elif user_profile:
            home_state = user_profile.get('user_home_state', '')
            if home_state and state != home_state:
                score = 45
                description = f"Out-of-state transaction (home: {home_state}, current: {state})"
            else:
                score = 5
                description = f"In-state transaction ({state})"
        else:
            score = 15
            description = f"Transaction in {state} (no profile)"

        return RiskFactor(
            name="Geographic Risk",
            score=score,
            description=description,
            weight=self.WEIGHTS['geographic_risk']
        )

    def _assess_merchant_risk(self, transaction: Dict) -> RiskFactor:
        """Assess risk based on merchant category"""
        mcc = transaction.get('MCC', 0)

        score = 0
        description = f"MCC: {mcc}"

        if mcc in self.HIGH_RISK_MCCS:
            score = 70
            description = f"High-risk merchant: {self.HIGH_RISK_MCCS[mcc]} (MCC {mcc})"
        elif mcc in [5411, 5541, 5812, 5912]:  # Common low-risk
            score = 5
            description = f"Low-risk merchant category (MCC {mcc})"
        else:
            score = 15
            description = f"Standard merchant category (MCC {mcc})"

        return RiskFactor(
            name="Merchant Risk",
            score=score,
            description=description,
            weight=self.WEIGHTS['merchant_risk']
        )

    def _assess_device_risk(self, transaction: Dict) -> RiskFactor:
        """Assess risk based on transaction type/device"""
        trans_type = transaction.get('Use Chip', '')

        score = 0
        description = f"Transaction type: {trans_type}"

        if trans_type == 'Online Transaction':
            score = 35
            description = "Online transaction - Card-not-present risk"
        elif trans_type == 'Swipe Transaction':
            score = 25
            description = "Swipe transaction - Possible cloned card"
        elif trans_type == 'Chip Transaction':
            score = 5
            description = "Chip transaction - Most secure"
        else:
            score = 20
            description = "Unknown transaction type"

        return RiskFactor(
            name="Device/Channel Risk",
            score=score,
            description=description,
            weight=self.WEIGHTS['device_risk']
        )

    def _assess_behavioral_risk(self, transaction: Dict, user_profile: Optional[Dict]) -> RiskFactor:
        """Assess risk based on behavioral patterns"""
        score = 0
        descriptions = []

        if user_profile:
            # Check MCC preference deviation
            pref_mccs = user_profile.get('preferred_mccs', [])
            if pref_mccs and transaction.get('MCC') not in pref_mccs:
                score += 25
                descriptions.append("Unusual merchant category for user")

            # Check online preference deviation
            online_ratio = user_profile.get('user_online_ratio', 0.5)
            is_online = transaction.get('Use Chip') == 'Online Transaction'
            if is_online and online_ratio < 0.2:
                score += 20
                descriptions.append("User rarely shops online")
            elif not is_online and online_ratio > 0.8:
                score += 15
                descriptions.append("User typically shops online")

            # Check new merchant
            if user_profile.get('is_new_merchant', False):
                score += 15
                descriptions.append("First time at this merchant")

        if not descriptions:
            descriptions.append("No behavioral anomalies detected")
            score = 5

        return RiskFactor(
            name="Behavioral Risk",
            score=min(score, 100),
            description=" | ".join(descriptions),
            weight=self.WEIGHTS['behavioral_risk']
        )

    def _calculate_composite_score(self, risk_factors: List[RiskFactor]) -> float:
        """Calculate weighted composite risk score"""
        total_weight = sum(f.weight for f in risk_factors)
        weighted_sum = sum(f.score * f.weight for f in risk_factors)

        return weighted_sum / total_weight if total_weight > 0 else 0

    def _get_rating(self, score: float) -> TransactionRating:
        """Convert score to letter grade"""
        for rating, (low, high) in self.rating_thresholds.items():
            if low <= score < high:
                return rating
        return TransactionRating.F

    def _get_risk_level(self, score: float) -> RiskLevel:
        """Convert score to risk level"""
        for level, (low, high) in self.risk_thresholds.items():
            if low <= score < high:
                return level
        return RiskLevel.CRITICAL

    def _generate_recommendation(
        self,
        rating: TransactionRating,
        risk_level: RiskLevel,
        risk_factors: List[RiskFactor]
    ) -> Dict:
        """Generate action recommendation"""

        # Sort factors by weighted contribution
        sorted_factors = sorted(
            risk_factors,
            key=lambda f: f.score * f.weight,
            reverse=True
        )
        top_factors = [f.name for f in sorted_factors[:3]]

        recommendations = {
            TransactionRating.A_PLUS: {
                'action': 'APPROVE',
                'message': 'Transaction appears safe. Auto-approve recommended.',
                'review_required': False
            },
            TransactionRating.A: {
                'action': 'APPROVE',
                'message': 'Low risk transaction. Approve.',
                'review_required': False
            },
            TransactionRating.B: {
                'action': 'APPROVE_WITH_MONITORING',
                'message': 'Moderate risk. Approve but flag for monitoring.',
                'review_required': False
            },
            TransactionRating.C: {
                'action': 'REVIEW',
                'message': 'Elevated risk. Manual review recommended.',
                'review_required': True
            },
            TransactionRating.D: {
                'action': 'STEP_UP_AUTH',
                'message': 'High risk. Require additional authentication.',
                'review_required': True
            },
            TransactionRating.F: {
                'action': 'DECLINE',
                'message': 'Critical risk. Decline transaction.',
                'review_required': True
            }
        }

        rec = recommendations.get(rating, recommendations[TransactionRating.F])
        rec['primary_risk_factors'] = top_factors

        return rec

    def _parse_amount(self, amount) -> float:
        """Parse amount from various formats"""
        if isinstance(amount, (int, float)):
            return float(amount)
        if isinstance(amount, str):
            return float(amount.replace('$', '').replace(',', ''))
        return 0.0


def rate_transaction(
    transaction: Dict,
    fraud_probability: float,
    user_profile: Optional[Dict] = None,
    historical_stats: Optional[Dict] = None
) -> Dict:
    """
    Convenience function to rate a transaction.

    Args:
        transaction: Transaction details
        fraud_probability: ML model fraud probability (0-1)
        user_profile: User spending profile (optional)
        historical_stats: Historical transaction stats (optional)

    Returns:
        Rating result dictionary
    """
    engine = TransactionRatingEngine()
    return engine.rate_transaction(
        transaction,
        fraud_probability,
        user_profile,
        historical_stats
    )


# Example usage
if __name__ == "__main__":
    # Test transactions
    test_cases = [
        {
            'name': 'Normal Transaction',
            'transaction': {
                'User': 1,
                'Card': 0,
                'Year': 2024,
                'Month': 1,
                'Day': 15,
                'Time': '14:30',
                'Amount': '$45.00',
                'Use Chip': 'Chip Transaction',
                'Merchant State': 'CA',
                'MCC': 5411
            },
            'fraud_probability': 0.02
        },
        {
            'name': 'Suspicious High Amount',
            'transaction': {
                'User': 1,
                'Card': 0,
                'Year': 2024,
                'Month': 1,
                'Day': 15,
                'Time': '03:15',
                'Amount': '$3500.00',
                'Use Chip': 'Swipe Transaction',
                'Merchant State': 'NV',
                'MCC': 5944
            },
            'fraud_probability': 0.78
        },
        {
            'name': 'Online Electronics',
            'transaction': {
                'User': 1,
                'Card': 0,
                'Year': 2024,
                'Month': 1,
                'Day': 15,
                'Time': '22:45',
                'Amount': '$899.99',
                'Use Chip': 'Online Transaction',
                'Merchant State': '',
                'MCC': 5732
            },
            'fraud_probability': 0.45
        }
    ]

    engine = TransactionRatingEngine()

    for test in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {test['name']}")
        print('='*60)

        result = engine.rate_transaction(
            test['transaction'],
            test['fraud_probability']
        )

        print(f"\nRating: {result['transaction_rating']} (Score: {result['rating_score']})")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Fraud Probability: {result['fraud_probability']:.2%}")

        print(f"\nRisk Factors:")
        for factor in result['risk_factors']:
            print(f"  - {factor['name']}: {factor['score']} (weighted: {factor['weighted_contribution']})")
            print(f"    {factor['description']}")

        print(f"\nRecommendation: {result['recommendation']['action']}")
        print(f"  {result['recommendation']['message']}")
        print(f"  Top factors: {', '.join(result['recommendation']['primary_risk_factors'])}")
