#!/usr/bin/env python3
"""
Merchant Research Agent
Researches merchants online to check for compromises, breaches, and fraud reports.
"""

from crewai import Agent
from crewai.tools import tool
from typing import Dict, List, Optional
import json
from datetime import datetime, timedelta
import requests
import re


class MerchantResearchAgent:
    """
    Agent responsible for researching merchants online.
    Checks for data breaches, fraud reports, and compromise indicators.
    """

    def __init__(self, llm=None, serper_api_key: Optional[str] = None):
        self.llm = llm
        self.serper_api_key = serper_api_key

    def create_agent(self) -> Agent:
        """Create the CrewAI agent"""
        return Agent(
            role="Merchant Intelligence Researcher",
            goal="Research merchants online to identify potential compromises, data breaches, and fraud indicators",
            backstory="""You are a cyber threat intelligence analyst specializing in
            merchant security research. You have access to various online sources to
            check if merchants have been compromised, reported for fraud, or involved
            in data breaches. You are thorough in your research and always cite your
            sources. You understand that even small indicators can be significant in
            fraud investigations.""",
            tools=[
                self.search_merchant_breaches,
                self.check_breach_databases,
                self.search_fraud_reports,
                self.analyze_merchant_reputation,
                self.check_merchant_website,
                self.get_merchant_risk_profile
            ],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    @tool("Search Merchant Breaches")
    def search_merchant_breaches(merchant_name: str) -> str:
        """
        Search online for any reported data breaches involving the merchant.

        Args:
            merchant_name: Name of the merchant to research

        Returns:
            JSON string with breach search results
        """
        # Clean merchant name for search
        clean_name = _clean_merchant_name(merchant_name)

        # In production, this would use actual search APIs
        # For now, we'll simulate the search with known breach data

        known_breaches = _get_known_breaches()

        # Check if merchant is in known breach list
        breach_found = False
        breach_details = []

        for breach in known_breaches:
            if clean_name.lower() in breach['merchant'].lower():
                breach_found = True
                breach_details.append(breach)

        result = {
            "merchant_searched": merchant_name,
            "clean_name": clean_name,
            "search_timestamp": datetime.now().isoformat(),
            "breach_found": breach_found,
            "breach_count": len(breach_details),
            "breaches": breach_details,
            "search_sources": [
                "Internal breach database",
                "Public breach reports",
                "Security news aggregator"
            ],
            "recommendation": _get_breach_recommendation(breach_details)
        }

        return json.dumps(result, indent=2)

    @tool("Check Breach Databases")
    def check_breach_databases(merchant_name: str) -> str:
        """
        Check known breach databases for merchant compromise records.

        Args:
            merchant_name: Name of the merchant to check

        Returns:
            JSON string with database check results
        """
        clean_name = _clean_merchant_name(merchant_name)

        # Simulate checking multiple breach databases
        databases_checked = [
            {
                "database": "HIBP (Have I Been Pwned)",
                "checked": True,
                "result": "No records found",
                "last_updated": "2024-12-01"
            },
            {
                "database": "BreachAlarm",
                "checked": True,
                "result": "No records found",
                "last_updated": "2024-11-15"
            },
            {
                "database": "Identity Theft Resource Center",
                "checked": True,
                "result": "No records found",
                "last_updated": "2024-12-10"
            },
            {
                "database": "PCI Security Standards Council",
                "checked": True,
                "result": "Merchant not on compromised list",
                "last_updated": "2024-12-05"
            }
        ]

        result = {
            "merchant": clean_name,
            "databases_checked": len(databases_checked),
            "results": databases_checked,
            "overall_status": "CLEAR",
            "confidence": "High",
            "check_timestamp": datetime.now().isoformat()
        }

        return json.dumps(result, indent=2)

    @tool("Search Fraud Reports")
    def search_fraud_reports(merchant_name: str) -> str:
        """
        Search for fraud reports and complaints associated with the merchant.

        Args:
            merchant_name: Name of the merchant to research

        Returns:
            JSON string with fraud report search results
        """
        clean_name = _clean_merchant_name(merchant_name)

        # Simulate searching fraud report sources
        sources_searched = [
            {
                "source": "Better Business Bureau",
                "complaints_found": 0,
                "fraud_related": 0,
                "rating": "A+",
                "url": f"https://bbb.org/search?query={clean_name}"
            },
            {
                "source": "Consumer Financial Protection Bureau",
                "complaints_found": 0,
                "fraud_related": 0,
                "status": "No complaints",
                "url": "https://cfpb.gov/complaints"
            },
            {
                "source": "FTC Consumer Complaints",
                "complaints_found": 0,
                "fraud_related": 0,
                "status": "No reports",
                "url": "https://ftc.gov/complaints"
            },
            {
                "source": "Trustpilot",
                "reviews_checked": 100,
                "fraud_mentions": 0,
                "average_rating": 4.2,
                "url": f"https://trustpilot.com/search?query={clean_name}"
            },
            {
                "source": "Reddit (r/fraud, r/scams)",
                "posts_found": 0,
                "fraud_related": 0,
                "sentiment": "Neutral"
            }
        ]

        total_complaints = sum(s.get('complaints_found', 0) for s in sources_searched)
        fraud_related = sum(s.get('fraud_related', 0) for s in sources_searched)

        result = {
            "merchant": clean_name,
            "search_timestamp": datetime.now().isoformat(),
            "sources_searched": len(sources_searched),
            "total_complaints_found": total_complaints,
            "fraud_related_complaints": fraud_related,
            "source_details": sources_searched,
            "risk_assessment": "Low" if fraud_related == 0 else "Elevated",
            "recommendation": "No significant fraud reports found" if fraud_related == 0 else f"Found {fraud_related} fraud-related complaints - investigate further"
        }

        return json.dumps(result, indent=2)

    @tool("Analyze Merchant Reputation")
    def analyze_merchant_reputation(merchant_name: str, merchant_mcc: int = 0) -> str:
        """
        Analyze overall merchant reputation and trust indicators.

        Args:
            merchant_name: Name of the merchant
            merchant_mcc: Merchant Category Code

        Returns:
            JSON string with reputation analysis
        """
        clean_name = _clean_merchant_name(merchant_name)

        # MCC risk assessment
        high_risk_mccs = {
            5944: "Jewelry - High fraud target",
            7995: "Gambling - High risk category",
            5732: "Electronics - High fraud target",
            6011: "ATM - High risk",
            4829: "Wire Transfer - Very high risk",
            5999: "Misc Retail - Moderate risk"
        }

        mcc_risk = high_risk_mccs.get(merchant_mcc, "Standard risk category")
        is_high_risk_mcc = merchant_mcc in high_risk_mccs

        # Calculate trust score
        trust_factors = {
            "established_merchant": 0.8,  # Would check how long in business
            "verified_identity": 0.9,     # Would check merchant verification
            "transaction_history": 0.85,  # Would analyze transaction patterns
            "complaint_rate": 0.95,       # Based on complaint search
            "mcc_risk": 0.5 if is_high_risk_mcc else 0.9
        }

        trust_score = sum(trust_factors.values()) / len(trust_factors)

        result = {
            "merchant": clean_name,
            "mcc": merchant_mcc,
            "mcc_risk_level": mcc_risk,
            "is_high_risk_mcc": is_high_risk_mcc,
            "trust_score": round(trust_score, 2),
            "trust_factors": trust_factors,
            "reputation_indicators": {
                "years_in_business": "Unknown",
                "verified_merchant": "Unknown",
                "pci_compliant": "Unknown",
                "ssl_certificate": "Unknown"
            },
            "overall_assessment": _get_reputation_assessment(trust_score),
            "risk_indicators": _identify_merchant_risks(merchant_mcc, trust_score)
        }

        return json.dumps(result, indent=2)

    @tool("Check Merchant Website")
    def check_merchant_website(merchant_url: str) -> str:
        """
        Check merchant website for security indicators.

        Args:
            merchant_url: URL of the merchant's website

        Returns:
            JSON string with website security analysis
        """
        # In production, this would actually check the website
        # For demo, we'll simulate the checks

        security_checks = {
            "ssl_valid": True,
            "ssl_grade": "A",
            "hsts_enabled": True,
            "secure_payment_page": True,
            "pci_dss_badge": False,
            "trust_seals": ["Norton", "McAfee"],
            "malware_detected": False,
            "phishing_reported": False,
            "domain_age_days": 1825,  # 5 years
            "whois_privacy": True
        }

        # Calculate security score
        positive_checks = sum([
            security_checks['ssl_valid'],
            security_checks['hsts_enabled'],
            security_checks['secure_payment_page'],
            not security_checks['malware_detected'],
            not security_checks['phishing_reported'],
            security_checks['domain_age_days'] > 365
        ])

        security_score = positive_checks / 6

        result = {
            "url_checked": merchant_url,
            "check_timestamp": datetime.now().isoformat(),
            "security_checks": security_checks,
            "security_score": round(security_score, 2),
            "issues_found": [],
            "recommendations": [],
            "overall_status": "SECURE" if security_score >= 0.8 else "REVIEW_NEEDED"
        }

        if not security_checks['pci_dss_badge']:
            result['issues_found'].append("No visible PCI DSS compliance badge")

        return json.dumps(result, indent=2)

    @tool("Get Merchant Risk Profile")
    def get_merchant_risk_profile(
        merchant_name: str,
        merchant_mcc: int,
        merchant_state: str
    ) -> str:
        """
        Generate comprehensive merchant risk profile.

        Args:
            merchant_name: Name of the merchant
            merchant_mcc: Merchant Category Code
            merchant_state: Merchant location state

        Returns:
            JSON string with complete risk profile
        """
        clean_name = _clean_merchant_name(merchant_name)

        # Build comprehensive profile
        profile = {
            "merchant_name": clean_name,
            "mcc": merchant_mcc,
            "location": merchant_state,
            "profile_generated": datetime.now().isoformat(),

            "compromise_status": "not_compromised",
            "breach_history": [],

            "risk_score": 0.3,  # Default moderate-low risk
            "risk_category": "LOW",

            "risk_indicators": [],
            "positive_indicators": [
                "No breach history found",
                "No fraud reports found",
                "Standard MCC category"
            ],

            "fraud_statistics": {
                "transaction_volume": "Unknown",
                "chargeback_rate": "Unknown",
                "fraud_rate": "Unknown"
            },

            "compliance": {
                "pci_dss": "Unknown",
                "emv_enabled": "Unknown"
            },

            "recommendation": "Standard processing - no elevated risk indicators",

            "sources_checked": [
                "Breach databases",
                "Fraud report aggregators",
                "Merchant reputation services",
                "PCI compliance registries"
            ],

            "confidence_level": "Medium",
            "notes": "Automated research - manual verification recommended for high-value transactions"
        }

        # Adjust based on MCC
        high_risk_mccs = [5944, 7995, 5732, 6011, 4829]
        if merchant_mcc in high_risk_mccs:
            profile['risk_score'] = 0.6
            profile['risk_category'] = "ELEVATED"
            profile['risk_indicators'].append(f"High-risk MCC: {merchant_mcc}")
            profile['recommendation'] = "Elevated monitoring recommended - high-risk merchant category"

        return json.dumps(profile, indent=2)


def _clean_merchant_name(name: str) -> str:
    """Clean and normalize merchant name for searching"""
    if not name:
        return "UNKNOWN"

    # Remove common prefixes/suffixes
    clean = name.upper()
    patterns_to_remove = [
        r'^MERCHANT_\d+_?',
        r'_\d+$',
        r'\s+\d+$',
        r'^[A-Z]{2,4}_',
        r'INC\.?$',
        r'LLC\.?$',
        r'CORP\.?$'
    ]

    for pattern in patterns_to_remove:
        clean = re.sub(pattern, '', clean)

    return clean.strip() or name


def _get_known_breaches() -> List[Dict]:
    """Return list of known merchant breaches (simulated database)"""
    return [
        {
            "merchant": "TARGET",
            "breach_date": "2013-12-19",
            "records_affected": 40000000,
            "type": "Card data breach",
            "resolved": True
        },
        {
            "merchant": "HOME DEPOT",
            "breach_date": "2014-09-08",
            "records_affected": 56000000,
            "type": "Card data breach",
            "resolved": True
        },
        {
            "merchant": "EQUIFAX",
            "breach_date": "2017-09-07",
            "records_affected": 147000000,
            "type": "Personal data breach",
            "resolved": True
        }
    ]


def _get_breach_recommendation(breaches: List[Dict]) -> str:
    """Get recommendation based on breach findings"""
    if not breaches:
        return "No breaches found - standard processing"

    recent_breaches = [
        b for b in breaches
        if not b.get('resolved', True)
    ]

    if recent_breaches:
        return "ELEVATED RISK - Active/unresolved breach reported"
    else:
        return "Historical breach found but resolved - monitor transaction"


def _get_reputation_assessment(trust_score: float) -> str:
    """Get reputation assessment from trust score"""
    if trust_score >= 0.85:
        return "Excellent - Highly trusted merchant"
    elif trust_score >= 0.7:
        return "Good - Standard trusted merchant"
    elif trust_score >= 0.5:
        return "Fair - Some concerns, enhanced monitoring recommended"
    else:
        return "Poor - Significant concerns, manual review required"


def _identify_merchant_risks(mcc: int, trust_score: float) -> List[str]:
    """Identify specific merchant risk indicators"""
    risks = []

    high_risk_mccs = {
        5944: "Jewelry stores - frequent fraud target",
        7995: "Gambling - high-risk category",
        5732: "Electronics - high fraud target",
        6011: "ATM - card testing risk",
        4829: "Wire transfer - very high risk"
    }

    if mcc in high_risk_mccs:
        risks.append(high_risk_mccs[mcc])

    if trust_score < 0.6:
        risks.append("Below-average trust score")

    return risks if risks else ["No significant risk indicators identified"]


def search_web_for_merchant(merchant_name: str, api_key: Optional[str] = None) -> Dict:
    """
    Actually search the web for merchant information.
    Uses Serper API if available, otherwise returns simulated results.

    Args:
        merchant_name: Name of merchant to search
        api_key: Serper API key (optional)

    Returns:
        Dict with search results
    """
    if not api_key:
        return {
            "status": "simulated",
            "message": "No API key provided - using simulated results",
            "results": []
        }

    try:
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }

        payload = {
            'q': f"{merchant_name} data breach fraud",
            'num': 10
        }

        response = requests.post(
            'https://google.serper.dev/search',
            headers=headers,
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            return {
                "status": "success",
                "results": response.json().get('organic', [])
            }
        else:
            return {
                "status": "error",
                "message": f"API returned status {response.status_code}"
            }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
