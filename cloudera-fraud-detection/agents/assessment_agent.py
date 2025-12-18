#!/usr/bin/env python3
"""
Assessment Writer Agent
Writes comprehensive fraud assessment reports.
"""

from crewai import Agent
from crewai.tools import tool
from typing import Dict, List, Optional
import json
from datetime import datetime
from enum import Enum


class AssessmentDecision(Enum):
    """Final assessment decisions"""
    CONFIRMED_FRAUD = "confirmed_fraud"
    LIKELY_FRAUD = "likely_fraud"
    SUSPICIOUS = "suspicious"
    INCONCLUSIVE = "inconclusive"
    LIKELY_LEGITIMATE = "likely_legitimate"
    CONFIRMED_LEGITIMATE = "confirmed_legitimate"


class RiskCategory(Enum):
    """Risk categories for assessment"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


class AssessmentWriterAgent:
    """
    Agent responsible for writing comprehensive fraud assessment reports.
    Synthesizes information from other agents into actionable reports.
    """

    def __init__(self, llm=None):
        self.llm = llm

    def create_agent(self) -> Agent:
        """Create the CrewAI agent"""
        return Agent(
            role="Fraud Assessment Analyst",
            goal="Write comprehensive, actionable fraud assessment reports",
            backstory="""You are a senior fraud analyst with expertise in writing clear,
            comprehensive assessment reports. Your reports are used by fraud investigators
            and risk managers to make decisions. You excel at synthesizing complex
            information into clear recommendations. Your reports are always well-structured,
            evidence-based, and actionable. You never make unfounded claims and always
            clearly state confidence levels.""",
            tools=[
                self.generate_assessment_report,
                self.calculate_risk_score,
                self.generate_recommendation,
                self.format_evidence_summary,
                self.create_timeline
            ],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    @tool("Generate Assessment Report")
    def generate_assessment_report(
        transaction_data: str,
        pattern_matches: str,
        merchant_research: str
    ) -> str:
        """
        Generate a comprehensive fraud assessment report.

        Args:
            transaction_data: JSON string with transaction details
            pattern_matches: JSON string with pattern matching results
            merchant_research: JSON string with merchant research findings

        Returns:
            JSON string with the complete assessment report
        """
        try:
            transaction = json.loads(transaction_data)
            patterns = json.loads(pattern_matches)
            merchant = json.loads(merchant_research)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON input: {e}"})

        # Calculate overall risk score
        risk_score = _calculate_composite_risk(transaction, patterns, merchant)

        # Determine decision
        decision = _determine_decision(risk_score, patterns, merchant)

        # Build report
        report = {
            "report_id": f"FAR-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "transaction_id": transaction.get('transaction_id', 'unknown'),

            "executive_summary": {
                "decision": decision.value,
                "risk_category": _get_risk_category(risk_score).value,
                "risk_score": risk_score,
                "confidence_level": _get_confidence_level(patterns),
                "recommended_action": _get_recommended_action(decision, risk_score)
            },

            "transaction_details": {
                "amount": transaction.get('Amount', transaction.get('amount', 'N/A')),
                "timestamp": transaction.get('Time', 'N/A'),
                "merchant": transaction.get('Merchant Name', transaction.get('merchant', {}).get('name', 'N/A')),
                "location": transaction.get('Merchant State', 'N/A'),
                "transaction_type": transaction.get('Use Chip', 'N/A'),
                "user_id": transaction.get('User', 'N/A')
            },

            "pattern_analysis": {
                "patterns_matched": patterns.get('matches_found', 0),
                "highest_confidence_pattern": patterns.get('top_matches', [{}])[0].get('pattern_name', 'None') if patterns.get('top_matches') else 'None',
                "pattern_confidence": patterns.get('highest_confidence', 0),
                "all_matched_patterns": [
                    {
                        "pattern": m.get('pattern_name'),
                        "confidence": m.get('confidence'),
                        "indicators": m.get('matched_indicators', [])
                    }
                    for m in patterns.get('top_matches', [])
                ]
            },

            "merchant_analysis": {
                "merchant_name": merchant.get('merchant_name', 'N/A'),
                "compromise_status": merchant.get('compromise_status', 'Unknown'),
                "risk_indicators": merchant.get('risk_indicators', []),
                "breach_history": merchant.get('breach_history', []),
                "trust_score": merchant.get('trust_score', 'N/A')
            },

            "risk_factors": _identify_risk_factors(transaction, patterns, merchant),

            "evidence_summary": _compile_evidence(transaction, patterns, merchant),

            "recommended_actions": _get_action_items(decision, risk_score, patterns),

            "analyst_notes": _generate_analyst_notes(decision, patterns, merchant),

            "metadata": {
                "report_version": "2.0",
                "analysis_model": "CrewAI Fraud Analysis",
                "review_required": decision.value in ['suspicious', 'inconclusive']
            }
        }

        return json.dumps(report, indent=2)

    @tool("Calculate Risk Score")
    def calculate_risk_score(
        fraud_probability: float,
        pattern_confidence: float,
        merchant_risk: float
    ) -> str:
        """
        Calculate composite risk score from multiple factors.

        Args:
            fraud_probability: ML model fraud probability (0-1)
            pattern_confidence: Pattern match confidence (0-1)
            merchant_risk: Merchant risk score (0-1)

        Returns:
            JSON string with risk score breakdown
        """
        # Weights for different factors
        weights = {
            'fraud_probability': 0.4,
            'pattern_confidence': 0.35,
            'merchant_risk': 0.25
        }

        # Calculate weighted score
        composite = (
            fraud_probability * weights['fraud_probability'] +
            pattern_confidence * weights['pattern_confidence'] +
            merchant_risk * weights['merchant_risk']
        )

        result = {
            "composite_score": round(composite, 3),
            "components": {
                "fraud_probability": {
                    "value": fraud_probability,
                    "weight": weights['fraud_probability'],
                    "contribution": round(fraud_probability * weights['fraud_probability'], 3)
                },
                "pattern_confidence": {
                    "value": pattern_confidence,
                    "weight": weights['pattern_confidence'],
                    "contribution": round(pattern_confidence * weights['pattern_confidence'], 3)
                },
                "merchant_risk": {
                    "value": merchant_risk,
                    "weight": weights['merchant_risk'],
                    "contribution": round(merchant_risk * weights['merchant_risk'], 3)
                }
            },
            "risk_category": _get_risk_category(composite).value,
            "percentile": f"Top {max(1, int((1-composite) * 100))}% risk"
        }

        return json.dumps(result, indent=2)

    @tool("Generate Recommendation")
    def generate_recommendation(risk_score: float, context: str) -> str:
        """
        Generate actionable recommendation based on risk score.

        Args:
            risk_score: Composite risk score (0-1)
            context: Additional context about the case

        Returns:
            JSON string with detailed recommendation
        """
        if risk_score >= 0.85:
            action = "BLOCK_IMMEDIATELY"
            urgency = "CRITICAL"
            steps = [
                "Block the transaction immediately",
                "Flag the card for review",
                "Contact cardholder for verification",
                "Escalate to fraud investigation team",
                "Document all evidence"
            ]
        elif risk_score >= 0.7:
            action = "DECLINE_AND_REVIEW"
            urgency = "HIGH"
            steps = [
                "Decline the transaction",
                "Send alert to cardholder",
                "Queue for manual review within 2 hours",
                "Monitor account for additional activity"
            ]
        elif risk_score >= 0.5:
            action = "STEP_UP_AUTHENTICATION"
            urgency = "MEDIUM"
            steps = [
                "Request additional authentication (OTP, biometric)",
                "Verify transaction details with cardholder",
                "If verified, approve with monitoring flag",
                "If not verified, decline and review"
            ]
        elif risk_score >= 0.3:
            action = "APPROVE_WITH_MONITORING"
            urgency = "LOW"
            steps = [
                "Approve the transaction",
                "Add to monitoring queue",
                "Alert if similar transactions follow",
                "Review after 24 hours if no issues"
            ]
        else:
            action = "APPROVE"
            urgency = "MINIMAL"
            steps = [
                "Approve the transaction",
                "Standard monitoring applies"
            ]

        recommendation = {
            "action": action,
            "urgency": urgency,
            "risk_score": risk_score,
            "steps": steps,
            "sla": _get_sla(urgency),
            "escalation_path": _get_escalation_path(urgency),
            "context_considered": context[:200] if context else "None provided"
        }

        return json.dumps(recommendation, indent=2)

    @tool("Format Evidence Summary")
    def format_evidence_summary(evidence_items: str) -> str:
        """
        Format evidence items into a structured summary.

        Args:
            evidence_items: JSON string with evidence items

        Returns:
            JSON string with formatted evidence summary
        """
        try:
            items = json.loads(evidence_items)
        except json.JSONDecodeError:
            items = []

        formatted = {
            "evidence_count": len(items) if isinstance(items, list) else 0,
            "summary": "Evidence compilation for fraud assessment",
            "items": items if isinstance(items, list) else [items],
            "strength_assessment": "Moderate" if items else "Insufficient"
        }

        return json.dumps(formatted, indent=2)

    @tool("Create Timeline")
    def create_timeline(events: str) -> str:
        """
        Create a timeline of relevant events.

        Args:
            events: JSON string with event data

        Returns:
            JSON string with formatted timeline
        """
        try:
            event_list = json.loads(events)
        except json.JSONDecodeError:
            event_list = []

        timeline = {
            "timeline_generated": datetime.now().isoformat(),
            "total_events": len(event_list) if isinstance(event_list, list) else 0,
            "events": event_list,
            "span": "Unable to determine" if not event_list else "Calculated"
        }

        return json.dumps(timeline, indent=2)


def _calculate_composite_risk(transaction: Dict, patterns: Dict, merchant: Dict) -> float:
    """Calculate composite risk score"""
    fraud_prob = transaction.get('fraud_probability', 0.5)
    pattern_conf = patterns.get('highest_confidence', 0)
    merchant_risk = merchant.get('risk_score', 0.5)

    # Weighted average
    return 0.4 * fraud_prob + 0.35 * pattern_conf + 0.25 * merchant_risk


def _determine_decision(risk_score: float, patterns: Dict, merchant: Dict) -> AssessmentDecision:
    """Determine assessment decision"""
    if risk_score >= 0.85:
        return AssessmentDecision.CONFIRMED_FRAUD
    elif risk_score >= 0.7:
        return AssessmentDecision.LIKELY_FRAUD
    elif risk_score >= 0.5:
        return AssessmentDecision.SUSPICIOUS
    elif risk_score >= 0.3:
        return AssessmentDecision.INCONCLUSIVE
    elif risk_score >= 0.15:
        return AssessmentDecision.LIKELY_LEGITIMATE
    else:
        return AssessmentDecision.CONFIRMED_LEGITIMATE


def _get_risk_category(score: float) -> RiskCategory:
    """Get risk category from score"""
    if score >= 0.8:
        return RiskCategory.CRITICAL
    elif score >= 0.6:
        return RiskCategory.HIGH
    elif score >= 0.4:
        return RiskCategory.MEDIUM
    elif score >= 0.2:
        return RiskCategory.LOW
    else:
        return RiskCategory.MINIMAL


def _get_confidence_level(patterns: Dict) -> str:
    """Determine confidence level"""
    conf = patterns.get('highest_confidence', 0)
    if conf >= 0.8:
        return "High"
    elif conf >= 0.5:
        return "Medium"
    else:
        return "Low"


def _get_recommended_action(decision: AssessmentDecision, risk_score: float) -> str:
    """Get recommended action"""
    actions = {
        AssessmentDecision.CONFIRMED_FRAUD: "Block transaction, freeze account, investigate",
        AssessmentDecision.LIKELY_FRAUD: "Decline transaction, contact cardholder",
        AssessmentDecision.SUSPICIOUS: "Step-up authentication required",
        AssessmentDecision.INCONCLUSIVE: "Manual review recommended",
        AssessmentDecision.LIKELY_LEGITIMATE: "Approve with monitoring",
        AssessmentDecision.CONFIRMED_LEGITIMATE: "Approve transaction"
    }
    return actions.get(decision, "Manual review required")


def _identify_risk_factors(transaction: Dict, patterns: Dict, merchant: Dict) -> List[Dict]:
    """Identify specific risk factors"""
    factors = []

    # Transaction amount
    amount = transaction.get('Amount_clean', transaction.get('amount', 0))
    if isinstance(amount, str):
        amount = float(amount.replace('$', '').replace(',', ''))
    if amount > 1000:
        factors.append({
            "factor": "High transaction amount",
            "severity": "Medium",
            "details": f"Amount ${amount:.2f} exceeds typical threshold"
        })

    # Pattern matches
    for match in patterns.get('top_matches', [])[:3]:
        factors.append({
            "factor": f"Pattern match: {match.get('pattern_name', 'Unknown')}",
            "severity": "High" if match.get('confidence', 0) > 0.7 else "Medium",
            "details": ", ".join(match.get('matched_indicators', []))
        })

    # Merchant risk
    if merchant.get('compromise_status') == 'compromised':
        factors.append({
            "factor": "Compromised merchant",
            "severity": "Critical",
            "details": "Merchant has been reported as compromised"
        })

    return factors


def _compile_evidence(transaction: Dict, patterns: Dict, merchant: Dict) -> List[str]:
    """Compile evidence summary"""
    evidence = []

    if patterns.get('matches_found', 0) > 0:
        evidence.append(f"Matched {patterns['matches_found']} fraud patterns")

    if merchant.get('breach_history'):
        evidence.append("Merchant has history of data breaches")

    if transaction.get('is_online'):
        evidence.append("Card-not-present transaction")

    return evidence if evidence else ["No significant evidence collected"]


def _get_action_items(decision: AssessmentDecision, risk_score: float, patterns: Dict) -> List[str]:
    """Get action items based on decision"""
    items = []

    if decision in [AssessmentDecision.CONFIRMED_FRAUD, AssessmentDecision.LIKELY_FRAUD]:
        items.extend([
            "Immediately block the transaction",
            "Flag card for potential compromise",
            "Initiate cardholder contact",
            "Create fraud case for investigation"
        ])
    elif decision == AssessmentDecision.SUSPICIOUS:
        items.extend([
            "Request step-up authentication",
            "Queue for expedited review",
            "Monitor for follow-up transactions"
        ])
    else:
        items.extend([
            "Standard processing with monitoring",
            "No immediate action required"
        ])

    return items


def _generate_analyst_notes(decision: AssessmentDecision, patterns: Dict, merchant: Dict) -> str:
    """Generate analyst notes"""
    notes = []

    if patterns.get('top_matches'):
        top = patterns['top_matches'][0]
        notes.append(f"Primary pattern identified: {top.get('pattern_name', 'Unknown')}")

    if merchant.get('compromise_status'):
        notes.append(f"Merchant status: {merchant['compromise_status']}")

    notes.append(f"Final decision: {decision.value}")

    return " | ".join(notes) if notes else "No additional notes"


def _get_sla(urgency: str) -> str:
    """Get SLA based on urgency"""
    slas = {
        "CRITICAL": "Immediate action required (< 5 minutes)",
        "HIGH": "Action required within 2 hours",
        "MEDIUM": "Review within 24 hours",
        "LOW": "Review within 72 hours",
        "MINIMAL": "Standard processing queue"
    }
    return slas.get(urgency, "Standard processing")


def _get_escalation_path(urgency: str) -> str:
    """Get escalation path based on urgency"""
    paths = {
        "CRITICAL": "Fraud Manager → VP Risk → CISO",
        "HIGH": "Senior Fraud Analyst → Fraud Manager",
        "MEDIUM": "Fraud Analyst → Senior Fraud Analyst",
        "LOW": "Automated monitoring",
        "MINIMAL": "No escalation needed"
    }
    return paths.get(urgency, "Standard escalation")
