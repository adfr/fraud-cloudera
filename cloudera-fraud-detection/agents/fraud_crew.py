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
import sys

from .query_agent import TransactionQueryAgent
from .pattern_agent import PatternMatchingAgent, FRAUD_PATTERNS
from .assessment_agent import AssessmentWriterAgent
from .merchant_agent import MerchantResearchAgent


def _is_jupyter() -> bool:
    """Check if running in a Jupyter notebook environment."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        # Check for various Jupyter kernel types
        shell_name = shell.__class__.__name__
        return shell_name in ('ZMQInteractiveShell', 'TerminalInteractiveShell')
    except (ImportError, NameError):
        return False


def _disable_rich_jupyter_mode():
    """
    Disable Rich's Jupyter integration to prevent recursion issues.

    Rich's Jupyter mode intercepts stdout and can cause infinite recursion
    when combined with CrewAI's Rich-based output system.
    """
    try:
        from rich.console import Console

        # Create a non-Jupyter console and patch the default
        # This prevents the infinite recursion when Rich tries to display
        # through Jupyter's display system
        import rich
        rich.reconfigure(force_terminal=True, force_interactive=False)

    except ImportError:
        pass
    except Exception:
        # If this fails for any reason, just continue
        pass


# Disable Rich Jupyter mode at import time to prevent recursion
if _is_jupyter():
    _disable_rich_jupyter_mode()


def _safe_print(*args, **kwargs):
    """
    Print function that bypasses Rich's console to avoid recursion in Jupyter.

    Rich intercepts sys.stdout in Jupyter environments, which can cause
    infinite recursion when Rich tries to display output and triggers
    stdout flush, which triggers Rich again.

    This function writes directly to the original stdout to avoid the issue.
    """
    # Use the original stdout to bypass any interceptors (like Rich)
    original_stdout = sys.__stdout__

    # Convert args to string
    message = ' '.join(str(arg) for arg in args)
    end = kwargs.get('end', '\n')

    try:
        original_stdout.write(message + end)
        original_stdout.flush()
    except Exception:
        # Fallback to regular print if something goes wrong
        pass


def _check_llm_configuration() -> tuple[bool, str]:
    """
    Check if an LLM is properly configured for CrewAI.

    Returns:
        Tuple of (is_configured, message)
    """
    # Check for common LLM API keys
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    azure_key = os.getenv('AZURE_OPENAI_API_KEY')

    if openai_key and openai_key.startswith('sk-'):
        return True, "OpenAI API key configured"
    if anthropic_key:
        return True, "Anthropic API key configured"
    if azure_key:
        return True, "Azure OpenAI API key configured"

    return False, """
No LLM API key found. CrewAI requires an LLM to orchestrate agents.

To fix this, set one of these environment variables:
  - OPENAI_API_KEY: For OpenAI GPT models (recommended)
  - ANTHROPIC_API_KEY: For Anthropic Claude models
  - AZURE_OPENAI_API_KEY: For Azure OpenAI

Example:
  export OPENAI_API_KEY='sk-your-key-here'

Or create a .env file with:
  OPENAI_API_KEY=sk-your-key-here

For testing without an LLM, use quick_analyze() which doesn't require LLM.
"""


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
        serper_api_key: Optional[str] = None,
        skip_llm_check: bool = False
    ):
        """
        Initialize the Fraud Analysis Crew.

        Args:
            llm: Language model to use (defaults to CrewAI default)
            verbose: Whether to print detailed logs
            serper_api_key: API key for web search (optional)
            skip_llm_check: Skip LLM configuration check (for quick_analyze only)
        """
        self.llm = llm
        self.verbose = verbose
        self.serper_api_key = serper_api_key or os.getenv('SERPER_API_KEY')
        self._llm_configured = None
        self._llm_message = None

        # Check LLM configuration (but don't fail yet - wait for analyze_alert)
        if not skip_llm_check and llm is None:
            self._llm_configured, self._llm_message = _check_llm_configuration()

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
        # Check LLM configuration before attempting analysis
        if self._llm_configured is False:
            _safe_print("\n" + "="*70)
            _safe_print("  FRAUD ANALYSIS CREW - Configuration Error")
            _safe_print("="*70)
            _safe_print(self._llm_message)

            return {
                "status": "configuration_error",
                "error": "LLM not configured",
                "message": self._llm_message.strip(),
                "analysis_id": f"FRAUD-{datetime.now().strftime('%Y%m%d%H%M%S')}-CFG",
                "alert_analyzed": alert_data.get('transaction_id', 'unknown'),
                "timestamp": datetime.now().isoformat(),
                "suggestion": "Use quick_analyze() for testing without LLM, or configure an API key."
            }

        _safe_print("\n" + "="*70)
        _safe_print("  FRAUD ANALYSIS CREW - Starting Investigation")
        _safe_print("="*70 + "\n")

        start_time = datetime.now()

        try:
            # Create agents
            _safe_print("[1/4] Initializing agents...")
            agents = self.create_agents()

            # Create tasks
            _safe_print("[2/4] Creating analysis tasks...")
            tasks = self.create_tasks(agents, alert_data)

            # Create crew
            _safe_print("[3/4] Assembling crew...")
            crew = self.create_crew(agents, tasks)

            # Execute
            _safe_print("[4/4] Executing analysis...\n")
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

            _safe_print("\n" + "="*70)
            _safe_print(f"  Analysis Complete - Duration: {duration:.2f} seconds")
            _safe_print("="*70 + "\n")

            return analysis_result

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Provide more helpful error messages for common issues
            error_msg = str(e)
            suggestion = ""

            if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                suggestion = "Check that your OPENAI_API_KEY is valid and has sufficient credits."
            elif "rate limit" in error_msg.lower():
                suggestion = "Rate limit exceeded. Wait a moment and try again."
            elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                suggestion = "Network error. Check your internet connection and try again."

            error_result = {
                "status": "error",
                "error": error_msg,
                "analysis_id": f"FRAUD-{datetime.now().strftime('%Y%m%d%H%M%S')}-ERR",
                "alert_analyzed": alert_data.get('transaction_id', 'unknown'),
                "duration_seconds": duration,
                "timestamp": end_time.isoformat()
            }

            if suggestion:
                error_result["suggestion"] = suggestion

            _safe_print(f"\n[ERROR] Analysis failed: {e}")
            if suggestion:
                _safe_print(f"[SUGGESTION] {suggestion}")
            _safe_print("")

            return error_result

    def quick_analyze(self, alert_data: Dict) -> Dict:
        """
        Perform a quick analysis without full crew orchestration.
        Useful for high-volume, lower-priority alerts.

        This method does NOT require an LLM - it uses direct tool calls
        for pattern matching and merchant research.

        Args:
            alert_data: Dictionary containing the fraud alert

        Returns:
            Dictionary with quick analysis results
        """
        _safe_print("\n[QUICK ANALYSIS] Starting rapid assessment...")

        start_time = datetime.now()

        try:
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

            _safe_print(f"[QUICK ANALYSIS] Complete - Risk: {quick_result['risk_level']} ({quick_risk:.2%})")

            return quick_result

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            _safe_print(f"[QUICK ANALYSIS] Error: {e}")

            return {
                "status": "error",
                "analysis_type": "quick",
                "error": str(e),
                "analysis_id": f"QUICK-{datetime.now().strftime('%Y%m%d%H%M%S')}-ERR",
                "duration_seconds": duration,
                "timestamp": end_time.isoformat()
            }


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
    _safe_print("="*70)
    _safe_print("  Fraud Analysis Crew - Demo")
    _safe_print("="*70)

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

    _safe_print("\n[DEMO] Sample Alert:")
    _safe_print(json.dumps(sample_alert, indent=2))

    # Run quick analysis (doesn't require LLM)
    _safe_print("\n" + "-"*70)
    _safe_print("Running Quick Analysis (no LLM required)...")
    _safe_print("-"*70)

    crew = FraudAnalysisCrew(verbose=True)
    quick_result = crew.quick_analyze(sample_alert)

    _safe_print("\n[RESULT] Quick Analysis Result:")
    _safe_print(json.dumps(quick_result, indent=2))

    # Full analysis would require an LLM
    _safe_print("\n" + "-"*70)
    _safe_print("Full Analysis requires an LLM (OpenAI, Anthropic, etc.)")
    _safe_print("Set OPENAI_API_KEY or configure CrewAI with your preferred LLM")
    _safe_print("-"*70)

    # Example of how to run full analysis:
    # result = crew.analyze_alert(sample_alert)
    # _safe_print(json.dumps(result, indent=2))
