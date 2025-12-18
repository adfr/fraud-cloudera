#!/usr/bin/env python3
"""
Alert Pipeline Integration
Connects fraud alerts to the CrewAI analysis system.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue
import time


class AlertPriority(Enum):
    """Alert priority levels"""
    CRITICAL = "critical"     # Immediate full analysis
    HIGH = "high"             # Full analysis within 5 minutes
    MEDIUM = "medium"         # Quick analysis + queue for full
    LOW = "low"               # Quick analysis only
    INFO = "info"             # Log only


class AlertStatus(Enum):
    """Alert processing status"""
    PENDING = "pending"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    DISMISSED = "dismissed"
    ERROR = "error"


@dataclass
class FraudAlert:
    """Fraud alert data structure"""
    alert_id: str
    transaction_id: str
    transaction_data: Dict
    fraud_probability: float
    risk_level: str
    transaction_rating: str
    priority: AlertPriority
    status: AlertStatus
    created_at: str
    analysis_result: Optional[Dict] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['priority'] = self.priority.value
        data['status'] = self.status.value
        return data

    @classmethod
    def from_transaction(cls, transaction: Dict, model_result: Dict) -> 'FraudAlert':
        """Create alert from transaction and model result"""
        alert_id = f"ALERT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{transaction.get('transaction_id', 'UNK')[-4:]}"

        fraud_prob = model_result.get('fraud_probability', 0)
        risk_level = model_result.get('risk_level', 'Unknown')
        rating = model_result.get('transaction_rating', 'N/A')

        # Determine priority
        if fraud_prob >= 0.85 or risk_level == 'Very High':
            priority = AlertPriority.CRITICAL
        elif fraud_prob >= 0.7 or risk_level == 'High':
            priority = AlertPriority.HIGH
        elif fraud_prob >= 0.5 or risk_level == 'Medium':
            priority = AlertPriority.MEDIUM
        elif fraud_prob >= 0.3:
            priority = AlertPriority.LOW
        else:
            priority = AlertPriority.INFO

        return cls(
            alert_id=alert_id,
            transaction_id=transaction.get('transaction_id', 'unknown'),
            transaction_data=transaction,
            fraud_probability=fraud_prob,
            risk_level=risk_level,
            transaction_rating=rating,
            priority=priority,
            status=AlertStatus.PENDING,
            created_at=datetime.now().isoformat()
        )


class AlertPipeline:
    """
    Pipeline for processing fraud alerts with CrewAI analysis.

    Features:
    - Priority-based alert queuing
    - Automatic analysis triggering
    - Configurable thresholds
    - Async processing support
    - Alert storage and retrieval
    """

    def __init__(
        self,
        analysis_threshold: float = 0.5,
        auto_analyze: bool = True,
        max_queue_size: int = 1000,
        output_dir: str = "output/alerts"
    ):
        """
        Initialize the alert pipeline.

        Args:
            analysis_threshold: Min fraud probability to trigger analysis
            auto_analyze: Whether to automatically analyze alerts
            max_queue_size: Maximum alerts to queue
            output_dir: Directory for storing alert results
        """
        self.analysis_threshold = analysis_threshold
        self.auto_analyze = auto_analyze
        self.max_queue_size = max_queue_size
        self.output_dir = output_dir

        # Alert queues by priority
        self.critical_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self.standard_queue = queue.Queue(maxsize=max_queue_size)

        # Alert storage
        self.alerts: Dict[str, FraudAlert] = {}

        # Callbacks
        self.on_alert_created: Optional[Callable] = None
        self.on_analysis_complete: Optional[Callable] = None
        self.on_escalation: Optional[Callable] = None

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Analysis crew (lazy loaded)
        self._crew = None

    @property
    def crew(self):
        """Lazy load the fraud analysis crew"""
        if self._crew is None:
            from .fraud_crew import FraudAnalysisCrew
            self._crew = FraudAnalysisCrew(verbose=True)
        return self._crew

    def process_transaction_result(
        self,
        transaction: Dict,
        model_result: Dict
    ) -> Optional[FraudAlert]:
        """
        Process a transaction result from the ML model.
        Creates alert if above threshold and queues for analysis.

        Args:
            transaction: Original transaction data
            model_result: Result from fraud detection model

        Returns:
            FraudAlert if created, None otherwise
        """
        fraud_prob = model_result.get('fraud_probability', 0)

        # Check if alert should be created
        if fraud_prob < self.analysis_threshold:
            return None

        # Create alert
        alert = FraudAlert.from_transaction(transaction, model_result)
        self.alerts[alert.alert_id] = alert

        print(f"\n[ALERT CREATED] {alert.alert_id}")
        print(f"  Transaction: {alert.transaction_id}")
        print(f"  Fraud Probability: {alert.fraud_probability:.2%}")
        print(f"  Risk Level: {alert.risk_level}")
        print(f"  Priority: {alert.priority.value.upper()}")

        # Trigger callback
        if self.on_alert_created:
            self.on_alert_created(alert)

        # Queue for analysis based on priority
        if alert.priority == AlertPriority.CRITICAL:
            self.critical_queue.put((0, alert.alert_id))  # Priority 0 = highest
        elif alert.priority in [AlertPriority.HIGH, AlertPriority.MEDIUM]:
            self.critical_queue.put((1, alert.alert_id))
        else:
            self.standard_queue.put(alert.alert_id)

        # Auto-analyze if enabled and critical
        if self.auto_analyze and alert.priority == AlertPriority.CRITICAL:
            self.analyze_alert(alert.alert_id)

        return alert

    def analyze_alert(self, alert_id: str, full_analysis: bool = None) -> Dict:
        """
        Analyze an alert using the CrewAI system.

        Args:
            alert_id: ID of the alert to analyze
            full_analysis: Force full/quick analysis (None = auto based on priority)

        Returns:
            Analysis result
        """
        alert = self.alerts.get(alert_id)
        if not alert:
            return {"error": f"Alert {alert_id} not found"}

        # Update status
        alert.status = AlertStatus.ANALYZING

        print(f"\n[ANALYZING] {alert_id}")

        # Determine analysis type
        if full_analysis is None:
            full_analysis = alert.priority in [AlertPriority.CRITICAL, AlertPriority.HIGH]

        # Prepare alert data for analysis
        alert_data = {
            **alert.transaction_data,
            "fraud_probability": alert.fraud_probability,
            "risk_level": alert.risk_level,
            "transaction_rating": alert.transaction_rating,
            "alert_id": alert.alert_id,
            "alert_priority": alert.priority.value
        }

        try:
            # Run analysis
            if full_analysis:
                result = self.crew.analyze_alert(alert_data)
            else:
                result = self.crew.quick_analyze(alert_data)

            # Update alert
            alert.analysis_result = result
            alert.status = AlertStatus.COMPLETED
            alert.completed_at = datetime.now().isoformat()

            # Save result
            self._save_alert_result(alert)

            # Trigger callback
            if self.on_analysis_complete:
                self.on_analysis_complete(alert, result)

            # Check for escalation
            if self._should_escalate(result):
                self._escalate_alert(alert)

            print(f"[ANALYSIS COMPLETE] {alert_id}")
            print(f"  Risk Score: {result.get('risk_score', 'N/A')}")
            print(f"  Risk Level: {result.get('risk_level', 'N/A')}")

            return result

        except Exception as e:
            alert.status = AlertStatus.ERROR
            alert.error_message = str(e)
            print(f"[ERROR] Analysis failed for {alert_id}: {e}")
            return {"error": str(e)}

    def analyze_pending_alerts(self, max_alerts: int = 10) -> List[Dict]:
        """
        Process pending alerts from the queue.

        Args:
            max_alerts: Maximum number of alerts to process

        Returns:
            List of analysis results
        """
        results = []
        processed = 0

        # Process critical queue first
        while processed < max_alerts and not self.critical_queue.empty():
            try:
                _, alert_id = self.critical_queue.get_nowait()
                result = self.analyze_alert(alert_id)
                results.append(result)
                processed += 1
            except queue.Empty:
                break

        # Then standard queue
        while processed < max_alerts and not self.standard_queue.empty():
            try:
                alert_id = self.standard_queue.get_nowait()
                result = self.analyze_alert(alert_id, full_analysis=False)
                results.append(result)
                processed += 1
            except queue.Empty:
                break

        return results

    def get_alert(self, alert_id: str) -> Optional[Dict]:
        """Get alert by ID"""
        alert = self.alerts.get(alert_id)
        return alert.to_dict() if alert else None

    def get_pending_alerts(self) -> List[Dict]:
        """Get all pending alerts"""
        return [
            alert.to_dict()
            for alert in self.alerts.values()
            if alert.status == AlertStatus.PENDING
        ]

    def get_alert_summary(self) -> Dict:
        """Get summary of all alerts"""
        by_status = {}
        by_priority = {}

        for alert in self.alerts.values():
            # Count by status
            status = alert.status.value
            by_status[status] = by_status.get(status, 0) + 1

            # Count by priority
            priority = alert.priority.value
            by_priority[priority] = by_priority.get(priority, 0) + 1

        return {
            "total_alerts": len(self.alerts),
            "by_status": by_status,
            "by_priority": by_priority,
            "critical_queue_size": self.critical_queue.qsize(),
            "standard_queue_size": self.standard_queue.qsize()
        }

    def _should_escalate(self, result: Dict) -> bool:
        """Determine if alert should be escalated"""
        risk_score = result.get('risk_score', 0)
        risk_level = result.get('risk_level', '').upper()

        return risk_score >= 0.85 or risk_level in ['CRITICAL', 'HIGH']

    def _escalate_alert(self, alert: FraudAlert):
        """Escalate an alert"""
        alert.status = AlertStatus.ESCALATED

        print(f"\n[ESCALATION] Alert {alert.alert_id} escalated!")
        print(f"  Fraud Probability: {alert.fraud_probability:.2%}")
        print(f"  Requires immediate attention")

        if self.on_escalation:
            self.on_escalation(alert)

    def _save_alert_result(self, alert: FraudAlert):
        """Save alert result to file"""
        filename = f"{alert.alert_id}.json"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(alert.to_dict(), f, indent=2)


class AlertProcessor:
    """
    Background processor for handling alerts asynchronously.
    """

    def __init__(self, pipeline: AlertPipeline, interval_seconds: float = 5.0):
        self.pipeline = pipeline
        self.interval = interval_seconds
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the background processor"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        print("[ALERT PROCESSOR] Started background processing")

    def stop(self):
        """Stop the background processor"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        print("[ALERT PROCESSOR] Stopped")

    def _process_loop(self):
        """Main processing loop"""
        while self._running:
            try:
                # Process pending alerts
                results = self.pipeline.analyze_pending_alerts(max_alerts=5)

                if results:
                    print(f"[ALERT PROCESSOR] Processed {len(results)} alerts")

            except Exception as e:
                print(f"[ALERT PROCESSOR] Error: {e}")

            time.sleep(self.interval)


def create_alert_handler(
    analysis_threshold: float = 0.5,
    output_dir: str = "output/alerts"
) -> AlertPipeline:
    """
    Create and configure an alert handler.

    Args:
        analysis_threshold: Min fraud probability to create alerts
        output_dir: Directory for alert results

    Returns:
        Configured AlertPipeline
    """
    pipeline = AlertPipeline(
        analysis_threshold=analysis_threshold,
        auto_analyze=True,
        output_dir=output_dir
    )

    # Set up callbacks
    def on_alert(alert: FraudAlert):
        print(f"[CALLBACK] New alert: {alert.alert_id} - Priority: {alert.priority.value}")

    def on_complete(alert: FraudAlert, result: Dict):
        print(f"[CALLBACK] Analysis complete: {alert.alert_id}")

    def on_escalate(alert: FraudAlert):
        print(f"[CALLBACK] ESCALATION: {alert.alert_id} - Immediate action required!")

    pipeline.on_alert_created = on_alert
    pipeline.on_analysis_complete = on_complete
    pipeline.on_escalation = on_escalate

    return pipeline


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("  Alert Pipeline Demo")
    print("="*70)

    # Create pipeline
    pipeline = create_alert_handler(analysis_threshold=0.3)

    # Sample transactions and model results
    test_cases = [
        {
            "transaction": {
                "transaction_id": "TXN_001",
                "User": 1,
                "Amount": "$45.00",
                "Use Chip": "Chip Transaction",
                "Merchant Name": "GROCERY_STORE",
                "MCC": 5411
            },
            "model_result": {
                "fraud_probability": 0.15,
                "risk_level": "Low",
                "transaction_rating": "A"
            }
        },
        {
            "transaction": {
                "transaction_id": "TXN_002",
                "User": 2,
                "Amount": "$1,500.00",
                "Use Chip": "Online Transaction",
                "Merchant Name": "ELECTRONICS_STORE",
                "MCC": 5732
            },
            "model_result": {
                "fraud_probability": 0.72,
                "risk_level": "High",
                "transaction_rating": "D"
            }
        },
        {
            "transaction": {
                "transaction_id": "TXN_003",
                "User": 3,
                "Amount": "$3,500.00",
                "Time": "03:15",
                "Use Chip": "Swipe Transaction",
                "Merchant Name": "JEWELRY_STORE",
                "MCC": 5944
            },
            "model_result": {
                "fraud_probability": 0.89,
                "risk_level": "Very High",
                "transaction_rating": "F"
            }
        }
    ]

    # Process transactions
    print("\n[DEMO] Processing transactions...\n")

    for case in test_cases:
        alert = pipeline.process_transaction_result(
            case["transaction"],
            case["model_result"]
        )
        if alert:
            print(f"  Created: {alert.alert_id}")
        else:
            print(f"  Skipped: {case['transaction']['transaction_id']} (below threshold)")
        print()

    # Show summary
    print("\n" + "-"*70)
    print("Alert Summary:")
    print("-"*70)
    summary = pipeline.get_alert_summary()
    print(json.dumps(summary, indent=2))

    # Run quick analysis on pending alerts
    print("\n" + "-"*70)
    print("Running Quick Analysis on Pending Alerts...")
    print("-"*70)

    results = pipeline.analyze_pending_alerts(max_alerts=10)
    print(f"\nProcessed {len(results)} alerts")
