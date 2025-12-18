#!/usr/bin/env python3
"""
Fraud Analysis Crew
Orchestrates multiple AI agents to analyze fraud alerts.
"""

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
import os

from .query_agent import TransactionQueryAgent
from .pattern_agent import PatternMatchingAgent, FRAUD_PATTERNS
from .assessment_agent import AssessmentWriterAgent
from .merchant_agent import MerchantResearchAgent


class FraudAnalysisCrew:
    """
    Orchestrates a crew of AI agents to analyze fraud alerts.

    Workflow:
    1. Query Agent: Gathers transaction data and user history
    2. Pattern Agent: Matches transaction against known fraud patterns
    3. Merchant Agent: Researches merchant online for compromises
    4. Assessment Agent: Writes comprehensive assessment report

    Usage:
        crew = FraudAnalysisCrew()
        result = crew.analyze_alert(alert_data)
    """

    def __init__(
        self,
        llm=None,
        verbose: bool = True,
        serper_api_key: Optional[str] = None
    ):
        """
        Initialize the Fraud Analysis Crew.

        Args:
            llm: Language model to use (defaults to CrewAI default)
            verbose: Whether to print detailed logs
            serper_api_key: API key for web search (optional)
        """
        self.llm = llm
        self.verbose = verbose
        self.serper_api_key = serper_api_key or os.getenv('SERPER_API_KEY')

        # Initialize agent factories
        self.query_agent_factory = TransactionQueryAgent(llm=llm)
        self.pattern_agent_factory = PatternMatchingAgent(llm=llm)
        self.merchant_agent_factory = MerchantResearchAgent(
            llm=llm,
            serper_api_key=self.serper_api_key
        )
        self.assessment_agent_factory = AssessmentWriterAgent(llm=llm)

    def create_agents(self) -> Dict[str, Agent]:
        """Create all agents for the crew"""
        return {
            'query_agent': self.query_agent_factory.create_agent(),
            'pattern_agent': self.pattern_agent_factory.create_agent(),
            'merchant_agent': self.merchant_agent_factory.create_agent(),
            'assessment_agent': self.assessment_agent_factory.create_agent()
        }

    def create_tasks(
        self,
        agents: Dict[str, Agent],
        alert_data: Dict
    ) -> List[Task]:
        """
        Create tasks for each agent based on the alert data.

        Args:
            agents: Dictionary of created agents
            alert_data: The fraud alert to analyze

        Returns:
            List of tasks in execution order
        """

        # Prepare alert context
        alert_json = json.dumps(alert_data, indent=2)
        transaction_id = alert_data.get('transaction_id', 'unknown')
        merchant_name = alert_data.get('Merchant Name', alert_data.get('merchant', 'Unknown'))
        merchant_mcc = alert_data.get('MCC', alert_data.get('mcc', 0))

        # Task 1: Query Agent - Gather transaction data
        query_task = Task(
            description=f"""
            Gather comprehensive information about the flagged transaction.

            Transaction Alert Data:
            {alert_json}

            Your tasks:
            1. Get detailed transaction information for transaction ID: {transaction_id}
            2. Get the user's transaction history for the past 30 days
            3. Get the user's spending profile
            4. Find any similar transactions in the user's history
            5. Get merchant transaction history

            Compile all this information into a comprehensive data package for analysis.
            """,
            expected_output="""
            A JSON object containing:
            - Complete transaction details
            - User transaction history summary
            - User spending profile
            - Similar past transactions
            - Merchant history
            """,
            agent=agents['query_agent']
        )

        # Task 2: Pattern Agent - Match fraud patterns
        pattern_task = Task(
            description=f"""
            Analyze the transaction against known fraud patterns.

            Transaction to analyze:
            {alert_json}

            Your tasks:
            1. Match the transaction against all known fraud patterns
            2. Calculate confidence scores for each matching pattern
            3. Identify the most likely fraud type(s)
            4. Analyze velocity patterns for the user
            5. Provide detailed analysis of matched indicators

            Use the pattern matching tools to perform thorough analysis.
            """,
            expected_output="""
            A JSON object containing:
            - List of matched fraud patterns with confidence scores
            - Most likely fraud type
            - Matched indicators for each pattern
            - Velocity analysis results
            - Overall pattern-based risk assessment
            """,
            agent=agents['pattern_agent'],
            context=[query_task]  # Depends on query results
        )

        # Task 3: Merchant Agent - Research merchant online
        merchant_task = Task(
            description=f"""
            Research the merchant online to check for compromises and fraud reports.

            Merchant to research: {merchant_name}
            Merchant Category Code (MCC): {merchant_mcc}

            Your tasks:
            1. Search for any data breaches involving this merchant
            2. Check breach databases for compromise records
            3. Search for fraud reports and complaints
            4. Analyze the merchant's overall reputation
            5. Generate a comprehensive merchant risk profile

            Be thorough in your research and document all findings.
            """,
            expected_output="""
            A JSON object containing:
            - Breach search results
            - Database check results
            - Fraud report findings
            - Merchant reputation analysis
            - Complete merchant risk profile
            - Recommendation based on findings
            """,
            agent=agents['merchant_agent']
        )

        # Task 4: Assessment Agent - Write final report
        assessment_task = Task(
            description=f"""
            Write a comprehensive fraud assessment report based on all gathered intelligence.

            Original Alert:
            {alert_json}

            Your tasks:
            1. Synthesize findings from transaction analysis, pattern matching, and merchant research
            2. Calculate the overall risk score
            3. Make a clear fraud/not-fraud determination
            4. Provide actionable recommendations
            5. Document all evidence and reasoning

            The report should be clear, professional, and actionable.
            Include confidence levels for all assessments.
            """,
            expected_output="""
            A complete Fraud Assessment Report containing:
            - Executive summary with decision and risk score
            - Detailed transaction analysis
            - Pattern matching results
            - Merchant research findings
            - Risk factors identified
            - Evidence summary
            - Recommended actions
            - Analyst notes
            """,
            agent=agents['assessment_agent'],
            context=[query_task, pattern_task, merchant_task]  # Depends on all previous tasks
        )

        return [query_task, pattern_task, merchant_task, assessment_task]

    def create_crew(self, agents: Dict[str, Agent], tasks: List[Task]) -> Crew:
        """
        Create the fraud analysis crew.

        Args:
            agents: Dictionary of agents
            tasks: List of tasks

        Returns:
            Configured Crew object
        """
        return Crew(
            agents=list(agents.values()),
            tasks=tasks,
            process=Process.sequential,  # Tasks run in order
            verbose=self.verbose,
            memory=True,  # Enable memory for context sharing
            planning=True  # Enable planning for better coordination
        )

    def analyze_alert(self, alert_data: Dict) -> Dict:
        """
        Analyze a fraud alert using the full crew.

        Args:
            alert_data: Dictionary containing the fraud alert

        Returns:
            Dictionary with analysis results
        """
        print("\n" + "="*70)
        print("  FRAUD ANALYSIS CREW - Starting Investigation")
        print("="*70 + "\n")

        start_time = datetime.now()

        try:
            # Create agents
            print("[1/4] Initializing agents...")
            agents = self.create_agents()

            # Create tasks
            print("[2/4] Creating analysis tasks...")
            tasks = self.create_tasks(agents, alert_data)

            # Create crew
            print("[3/4] Assembling crew...")
            crew = self.create_crew(agents, tasks)

            # Execute
            print("[4/4] Executing analysis...\n")
            result = crew.kickoff()

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Parse and enhance result
            analysis_result = {
                "status": "completed",
                "analysis_id": f"FRAUD-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "alert_analyzed": alert_data.get('transaction_id', 'unknown'),
                "duration_seconds": duration,
                "timestamp": end_time.isoformat(),
                "crew_output": str(result),
                "agents_used": list(agents.keys()),
                "tasks_completed": len(tasks)
            }

            # Try to parse the final assessment
            try:
                if hasattr(result, 'raw'):
                    analysis_result['assessment'] = json.loads(result.raw)
                else:
                    analysis_result['assessment'] = str(result)
            except:
                analysis_result['assessment'] = str(result)

            print("\n" + "="*70)
            print(f"  Analysis Complete - Duration: {duration:.2f} seconds")
            print("="*70 + "\n")

            return analysis_result

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            error_result = {
                "status": "error",
                "error": str(e),
                "analysis_id": f"FRAUD-{datetime.now().strftime('%Y%m%d%H%M%S')}-ERR",
                "alert_analyzed": alert_data.get('transaction_id', 'unknown'),
                "duration_seconds": duration,
                "timestamp": end_time.isoformat()
            }

            print(f"\n[ERROR] Analysis failed: {e}\n")

            return error_result

    def quick_analyze(self, alert_data: Dict) -> Dict:
        """
        Perform a quick analysis without full crew orchestration.
        Useful for high-volume, lower-priority alerts.

        Args:
            alert_data: Dictionary containing the fraud alert

        Returns:
            Dictionary with quick analysis results
        """
        print("\n[QUICK ANALYSIS] Starting rapid assessment...")

        start_time = datetime.now()

        # Direct tool calls without agent orchestration
        from .pattern_agent import PatternMatchingAgent
        from .merchant_agent import MerchantResearchAgent

        pattern_agent = PatternMatchingAgent()
        merchant_agent = MerchantResearchAgent()

        # Pattern matching
        pattern_result = pattern_agent.match_fraud_patterns.__wrapped__(
            json.dumps(alert_data)
        )

        # Merchant research
        merchant_name = alert_data.get('Merchant Name', 'Unknown')
        merchant_mcc = alert_data.get('MCC', 0)
        merchant_state = alert_data.get('Merchant State', '')

        merchant_result = merchant_agent.get_merchant_risk_profile.__wrapped__(
            merchant_name, merchant_mcc, merchant_state
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Parse results
        patterns = json.loads(pattern_result)
        merchant = json.loads(merchant_result)

        # Calculate quick risk score
        pattern_risk = patterns.get('highest_confidence', 0)
        merchant_risk = merchant.get('risk_score', 0.3)
        fraud_prob = alert_data.get('fraud_probability', 0.5)

        quick_risk = (pattern_risk * 0.35 + merchant_risk * 0.25 + fraud_prob * 0.4)

        quick_result = {
            "status": "completed",
            "analysis_type": "quick",
            "analysis_id": f"QUICK-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "duration_seconds": duration,
            "timestamp": end_time.isoformat(),
            "risk_score": round(quick_risk, 3),
            "risk_level": _get_risk_level(quick_risk),
            "patterns_matched": patterns.get('matches_found', 0),
            "top_pattern": patterns.get('top_matches', [{}])[0].get('pattern_name', 'None') if patterns.get('top_matches') else 'None',
            "merchant_status": merchant.get('compromise_status', 'unknown'),
            "recommendation": patterns.get('recommended_action', 'Review'),
            "details": {
                "pattern_analysis": patterns,
                "merchant_analysis": merchant
            }
        }

        print(f"[QUICK ANALYSIS] Complete - Risk: {quick_result['risk_level']} ({quick_risk:.2%})")

        return quick_result


def _get_risk_level(score: float) -> str:
    """Convert risk score to risk level"""
    if score >= 0.8:
        return "CRITICAL"
    elif score >= 0.6:
        return "HIGH"
    elif score >= 0.4:
        return "MEDIUM"
    elif score >= 0.2:
        return "LOW"
    else:
        return "MINIMAL"


def analyze_fraud_alert(alert_data: Dict, quick: bool = False) -> Dict:
    """
    Convenience function to analyze a fraud alert.

    Args:
        alert_data: The fraud alert to analyze
        quick: If True, perform quick analysis without full crew

    Returns:
        Analysis results
    """
    crew = FraudAnalysisCrew()

    if quick:
        return crew.quick_analyze(alert_data)
    else:
        return crew.analyze_alert(alert_data)


# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("  Fraud Analysis Crew - Demo")
    print("="*70)

    # Sample alert data
    sample_alert = {
        "transaction_id": "TXN_20241218143052_8472",
        "User": 123,
        "Card": 0,
        "Year": 2024,
        "Month": 12,
        "Day": 18,
        "Time": "03:15",
        "Amount": "$2,500.00",
        "Use Chip": "Online Transaction",
        "Merchant Name": "ELECTRONICS_STORE_5732",
        "Merchant City": "",
        "Merchant State": "",
        "MCC": 5732,
        "fraud_probability": 0.78,
        "risk_level": "High",
        "transaction_rating": "D"
    }

    print("\n[DEMO] Sample Alert:")
    print(json.dumps(sample_alert, indent=2))

    # Run quick analysis (doesn't require LLM)
    print("\n" + "-"*70)
    print("Running Quick Analysis (no LLM required)...")
    print("-"*70)

    crew = FraudAnalysisCrew(verbose=True)
    quick_result = crew.quick_analyze(sample_alert)

    print("\n[RESULT] Quick Analysis Result:")
    print(json.dumps(quick_result, indent=2))

    # Full analysis would require an LLM
    print("\n" + "-"*70)
    print("Full Analysis requires an LLM (OpenAI, Anthropic, etc.)")
    print("Set OPENAI_API_KEY or configure CrewAI with your preferred LLM")
    print("-"*70)

    # Example of how to run full analysis:
    # result = crew.analyze_alert(sample_alert)
    # print(json.dumps(result, indent=2))
