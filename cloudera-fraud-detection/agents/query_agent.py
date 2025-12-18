#!/usr/bin/env python3
"""
Transaction Query Agent
Gathers and enriches transaction data for fraud analysis.
"""

from crewai import Agent
from crewai.tools import tool
from typing import Dict, List, Optional
import json
from datetime import datetime, timedelta


class TransactionQueryAgent:
    """
    Agent responsible for querying and gathering transaction data.
    Retrieves transaction details, user history, and contextual information.
    """

    def __init__(self, llm=None):
        self.llm = llm
        self.transaction_store = {}
        self.user_history = {}

    def create_agent(self) -> Agent:
        """Create the CrewAI agent"""
        return Agent(
            role="Transaction Data Analyst",
            goal="Gather comprehensive transaction data and user history for fraud analysis",
            backstory="""You are an expert data analyst specializing in financial transactions.
            Your job is to gather all relevant information about a flagged transaction,
            including the user's transaction history, spending patterns, and any anomalies.
            You are meticulous and thorough in your data gathering.""",
            tools=[
                self.get_transaction_details,
                self.get_user_history,
                self.get_user_profile,
                self.get_similar_transactions,
                self.get_merchant_history
            ],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    @tool("Get Transaction Details")
    def get_transaction_details(transaction_id: str) -> str:
        """
        Retrieve detailed information about a specific transaction.

        Args:
            transaction_id: The unique identifier of the transaction

        Returns:
            JSON string with transaction details
        """
        # In production, this would query a database
        # For demo, we'll return structured mock data or stored data

        transaction = {
            "transaction_id": transaction_id,
            "timestamp": datetime.now().isoformat(),
            "amount": 0,
            "currency": "USD",
            "merchant": {
                "name": "Unknown",
                "mcc": 0,
                "city": "Unknown",
                "state": "Unknown",
                "country": "US"
            },
            "card": {
                "last_four": "****",
                "type": "Unknown",
                "entry_mode": "Unknown"
            },
            "user_id": 0,
            "risk_score": 0,
            "fraud_probability": 0
        }

        return json.dumps(transaction, indent=2)

    @tool("Get User Transaction History")
    def get_user_history(user_id: str, days: int = 30) -> str:
        """
        Retrieve the transaction history for a user.

        Args:
            user_id: The user identifier
            days: Number of days of history to retrieve (default 30)

        Returns:
            JSON string with transaction history summary
        """
        history = {
            "user_id": user_id,
            "period_days": days,
            "total_transactions": 0,
            "total_amount": 0,
            "average_amount": 0,
            "max_amount": 0,
            "min_amount": 0,
            "unique_merchants": 0,
            "transaction_types": {
                "chip": 0,
                "swipe": 0,
                "online": 0
            },
            "geographic_spread": [],
            "recent_transactions": []
        }

        return json.dumps(history, indent=2)

    @tool("Get User Profile")
    def get_user_profile(user_id: str) -> str:
        """
        Retrieve the user's spending profile and patterns.

        Args:
            user_id: The user identifier

        Returns:
            JSON string with user profile data
        """
        profile = {
            "user_id": user_id,
            "account_age_days": 0,
            "average_monthly_spend": 0,
            "typical_transaction_amount": {
                "mean": 0,
                "std": 0,
                "median": 0
            },
            "preferred_merchants": [],
            "home_state": "Unknown",
            "typical_transaction_times": {
                "peak_hours": [],
                "weekend_ratio": 0
            },
            "online_shopping_ratio": 0,
            "international_transactions": 0,
            "previous_fraud_cases": 0,
            "risk_tier": "standard"
        }

        return json.dumps(profile, indent=2)

    @tool("Get Similar Transactions")
    def get_similar_transactions(
        user_id: str,
        amount: float,
        merchant_mcc: int,
        tolerance: float = 0.2
    ) -> str:
        """
        Find similar transactions in user's history.

        Args:
            user_id: The user identifier
            amount: Transaction amount to match
            merchant_mcc: Merchant category code
            tolerance: Amount tolerance (default 20%)

        Returns:
            JSON string with similar transactions
        """
        similar = {
            "query": {
                "user_id": user_id,
                "amount": amount,
                "mcc": merchant_mcc,
                "tolerance": tolerance
            },
            "matches": [],
            "match_count": 0,
            "similarity_assessment": "No similar transactions found"
        }

        return json.dumps(similar, indent=2)

    @tool("Get Merchant Transaction History")
    def get_merchant_history(merchant_name: str) -> str:
        """
        Retrieve transaction history for a specific merchant.

        Args:
            merchant_name: The merchant name or ID

        Returns:
            JSON string with merchant history
        """
        merchant_data = {
            "merchant_name": merchant_name,
            "total_transactions": 0,
            "fraud_rate": 0,
            "average_transaction": 0,
            "risk_score": 0,
            "recent_fraud_reports": 0,
            "category": "Unknown",
            "first_seen": None,
            "trust_score": 0
        }

        return json.dumps(merchant_data, indent=2)

    def load_transaction_data(self, transaction: Dict):
        """Load transaction data for the tools to access"""
        self.transaction_store[transaction.get('transaction_id', 'unknown')] = transaction

    def load_user_history(self, user_id: int, history: List[Dict]):
        """Load user history for the tools to access"""
        self.user_history[user_id] = history


def create_query_tools():
    """Create standalone tools for the query agent"""

    @tool("Get Transaction Details")
    def get_transaction_details(transaction_id: str) -> str:
        """Retrieve detailed information about a specific transaction."""
        return json.dumps({
            "transaction_id": transaction_id,
            "status": "Retrieved",
            "note": "Connect to actual data source in production"
        })

    @tool("Get User History")
    def get_user_history(user_id: str, days: int = 30) -> str:
        """Retrieve user transaction history."""
        return json.dumps({
            "user_id": user_id,
            "period_days": days,
            "status": "Retrieved"
        })

    return [get_transaction_details, get_user_history]
