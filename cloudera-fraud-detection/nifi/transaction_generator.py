#!/usr/bin/env python3
"""
Transaction Generator for NiFi Integration
Generates transactions manually or automatically and sends them to NiFi/ML model.
"""

import json
import random
import requests
import argparse
from datetime import datetime
import time
import sys


class TransactionGenerator:
    """Generate credit card transactions for testing fraud detection"""

    def __init__(self, model_endpoint=None, nifi_endpoint=None):
        self.model_endpoint = model_endpoint
        self.nifi_endpoint = nifi_endpoint

        # Configuration
        self.mccs = [5411, 5541, 5812, 5912, 5311, 5999, 7011, 4111, 5732, 5944]
        self.states = ['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
        self.trans_types = ['Chip Transaction', 'Swipe Transaction', 'Online Transaction']

    def generate_normal_transaction(self, user_id=None, card_id=None):
        """Generate a normal-looking transaction"""
        now = datetime.now()
        user_id = user_id or random.randint(1, 500)
        card_id = card_id or random.randint(0, 2)

        trans_type = random.choice(self.trans_types)
        is_online = trans_type == 'Online Transaction'

        return {
            'User': user_id,
            'Card': card_id,
            'Year': now.year,
            'Month': now.month,
            'Day': now.day,
            'Time': f'{random.randint(8, 20):02d}:{random.randint(0, 59):02d}',
            'Amount': f'${random.uniform(10, 200):.2f}',
            'Use Chip': trans_type,
            'Merchant Name': f'MERCHANT_{random.randint(1000, 9999)}',
            'Merchant City': '' if is_online else f'CITY_{random.randint(1, 50)}',
            'Merchant State': '' if is_online else random.choice(self.states),
            'Zip': '' if is_online else f'{random.randint(10000, 99999)}',
            'MCC': random.choice([5411, 5541, 5812, 5912]),  # Common MCCs
            'transaction_id': f'TXN_{now.strftime("%Y%m%d%H%M%S")}_{random.randint(1000, 9999)}'
        }

    def generate_suspicious_transaction(self, user_id=None, card_id=None, fraud_type='random'):
        """Generate a suspicious/fraudulent transaction"""
        now = datetime.now()
        user_id = user_id or random.randint(1, 500)
        card_id = card_id or random.randint(0, 2)

        fraud_types = ['high_amount', 'unusual_time', 'online_burst', 'unusual_merchant']
        if fraud_type == 'random':
            fraud_type = random.choice(fraud_types)

        trans = {
            'User': user_id,
            'Card': card_id,
            'Year': now.year,
            'Month': now.month,
            'Day': now.day,
            'transaction_id': f'TXN_{now.strftime("%Y%m%d%H%M%S")}_{random.randint(1000, 9999)}'
        }

        if fraud_type == 'high_amount':
            trans.update({
                'Time': f'{random.randint(8, 20):02d}:{random.randint(0, 59):02d}',
                'Amount': f'${random.uniform(2000, 10000):.2f}',  # Very high amount
                'Use Chip': 'Swipe Transaction',
                'Merchant Name': f'JEWELRY_{random.randint(1000, 9999)}',
                'Merchant City': f'CITY_{random.randint(1, 50)}',
                'Merchant State': random.choice(self.states),
                'Zip': f'{random.randint(10000, 99999)}',
                'MCC': 5944  # Jewelry
            })

        elif fraud_type == 'unusual_time':
            trans.update({
                'Time': f'{random.randint(1, 5):02d}:{random.randint(0, 59):02d}',  # Late night
                'Amount': f'${random.uniform(100, 500):.2f}',
                'Use Chip': 'Swipe Transaction',
                'Merchant Name': f'ATM_{random.randint(1000, 9999)}',
                'Merchant City': f'CITY_{random.randint(1, 50)}',
                'Merchant State': random.choice(self.states),
                'Zip': f'{random.randint(10000, 99999)}',
                'MCC': 6011  # ATM
            })

        elif fraud_type == 'online_burst':
            trans.update({
                'Time': f'{random.randint(0, 23):02d}:{random.randint(0, 59):02d}',
                'Amount': f'${random.uniform(500, 2000):.2f}',
                'Use Chip': 'Online Transaction',
                'Merchant Name': f'ELECTRONICS_{random.randint(1000, 9999)}',
                'Merchant City': '',
                'Merchant State': '',
                'Zip': '',
                'MCC': 5732  # Electronics
            })

        elif fraud_type == 'unusual_merchant':
            trans.update({
                'Time': f'{random.randint(8, 20):02d}:{random.randint(0, 59):02d}',
                'Amount': f'${random.uniform(300, 1500):.2f}',
                'Use Chip': random.choice(['Chip Transaction', 'Swipe Transaction']),
                'Merchant Name': f'CASINO_{random.randint(1000, 9999)}',
                'Merchant City': 'LAS VEGAS',
                'Merchant State': 'NV',
                'Zip': '89101',
                'MCC': 7995  # Gambling
            })

        return trans

    def create_interactive_transaction(self):
        """Create a transaction interactively from user input"""
        print("\n=== Create Transaction Manually ===")
        print("Press Enter to use default values\n")

        now = datetime.now()

        try:
            user_id = input(f"User ID [1-500, default={random.randint(1, 500)}]: ").strip()
            user_id = int(user_id) if user_id else random.randint(1, 500)

            card_id = input(f"Card ID [0-2, default=0]: ").strip()
            card_id = int(card_id) if card_id else 0

            amount = input("Amount (e.g., 150.00) [default=random]: ").strip()
            amount = f"${float(amount):.2f}" if amount else f"${random.uniform(10, 500):.2f}"

            print("\nTransaction types:")
            print("  1. Chip Transaction")
            print("  2. Swipe Transaction")
            print("  3. Online Transaction")
            trans_type_choice = input("Transaction type [1-3, default=1]: ").strip()
            trans_type_map = {'1': 'Chip Transaction', '2': 'Swipe Transaction', '3': 'Online Transaction'}
            trans_type = trans_type_map.get(trans_type_choice, 'Chip Transaction')

            hour = input(f"Hour [0-23, default={now.hour}]: ").strip()
            hour = int(hour) if hour else now.hour
            minute = input(f"Minute [0-59, default={now.minute}]: ").strip()
            minute = int(minute) if minute else now.minute

            state = ''
            if trans_type != 'Online Transaction':
                state = input(f"Merchant State (e.g., CA) [default={random.choice(self.states)}]: ").strip()
                state = state.upper() if state else random.choice(self.states)

            mcc = input(f"MCC [default=5411 (Grocery)]: ").strip()
            mcc = int(mcc) if mcc else 5411

            trans = {
                'User': user_id,
                'Card': card_id,
                'Year': now.year,
                'Month': now.month,
                'Day': now.day,
                'Time': f'{hour:02d}:{minute:02d}',
                'Amount': amount,
                'Use Chip': trans_type,
                'Merchant Name': f'MERCHANT_{random.randint(1000, 9999)}',
                'Merchant City': '' if trans_type == 'Online Transaction' else f'CITY_{random.randint(1, 50)}',
                'Merchant State': state,
                'Zip': '' if trans_type == 'Online Transaction' else f'{random.randint(10000, 99999)}',
                'MCC': mcc,
                'transaction_id': f'TXN_{now.strftime("%Y%m%d%H%M%S")}_{random.randint(1000, 9999)}'
            }

            return trans

        except KeyboardInterrupt:
            print("\nCancelled")
            return None
        except ValueError as e:
            print(f"\nInvalid input: {e}")
            return None

    def send_to_model(self, transaction):
        """Send transaction to ML model endpoint"""
        if not self.model_endpoint:
            print("Error: Model endpoint not configured")
            return None

        try:
            # Convert transaction to model input format
            model_input = self._convert_to_model_input(transaction)

            response = requests.post(
                self.model_endpoint,
                json=model_input,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error calling model: {e}")
            return None

    def send_to_nifi(self, transaction):
        """Send transaction to NiFi HTTP endpoint"""
        if not self.nifi_endpoint:
            print("Error: NiFi endpoint not configured")
            return None

        try:
            response = requests.post(
                self.nifi_endpoint,
                json=transaction,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            response.raise_for_status()
            return {'status': 'success', 'nifi_response': response.text}

        except requests.exceptions.RequestException as e:
            print(f"Error sending to NiFi: {e}")
            return None

    def _convert_to_model_input(self, transaction):
        """Convert raw transaction to model input features"""
        # Parse amount
        amount = transaction.get('Amount', '$0')
        if isinstance(amount, str):
            amount = float(amount.replace('$', '').replace(',', ''))

        # Parse time
        time_str = transaction.get('Time', '12:00')
        hour = int(time_str.split(':')[0])

        # Determine transaction type
        use_chip = transaction.get('Use Chip', '')
        is_online = 1 if use_chip == 'Online Transaction' else 0
        is_chip = 1 if use_chip == 'Chip Transaction' else 0
        is_swipe = 1 if use_chip == 'Swipe Transaction' else 0

        # Date features
        year = transaction.get('Year', datetime.now().year)
        month = transaction.get('Month', datetime.now().month)
        day = transaction.get('Day', datetime.now().day)
        dt = datetime(year, month, day)

        return {
            'Amount_clean': amount,
            'hour': hour,
            'day_of_week': dt.weekday(),
            'day_of_month': day,
            'is_weekend': 1 if dt.weekday() >= 5 else 0,
            'is_online': is_online,
            'is_chip': is_chip,
            'is_swipe': is_swipe,
            'is_online_state': 1 if transaction.get('Merchant State', '') == '' else 0,
            'mcc_encoded': transaction.get('MCC', 5411),
            'state_encoded': 0,  # Would need lookup table
            'time_since_last': 1.0,  # Default
            'amount_mean_5': amount,
            'amount_std_5': 0,
            'amount_mean_10': amount,
            'amount_std_10': 0,
            'amount_mean_30': amount,
            'amount_std_30': 0,
            'amount_deviation': 0,
            'amount_zscore': 0,
            'trans_count_day': 1,
            # Additional features from transaction
            'User': transaction.get('User', 0),
            'Card': transaction.get('Card', 0),
            'transaction_id': transaction.get('transaction_id', '')
        }

    def display_transaction(self, transaction):
        """Display transaction details"""
        print("\n--- Transaction Details ---")
        print(json.dumps(transaction, indent=2))

    def display_result(self, result):
        """Display model/NiFi response"""
        if result:
            print("\n--- Response ---")
            print(json.dumps(result, indent=2))

            if 'risk_level' in result:
                risk = result.get('risk_level', 'Unknown')
                prob = result.get('fraud_probability', 0)
                rating = result.get('transaction_rating', 'N/A')

                if risk in ['High', 'Very High']:
                    print(f"\n⚠️  ALERT: {risk} RISK - Probability: {prob:.2%} - Rating: {rating}")
                else:
                    print(f"\n✓ {risk} Risk - Probability: {prob:.2%} - Rating: {rating}")


def main():
    parser = argparse.ArgumentParser(description='Generate and test transactions for fraud detection')
    parser.add_argument('--model-endpoint', '-m', help='Cloudera ML model endpoint URL')
    parser.add_argument('--nifi-endpoint', '-n', help='NiFi HTTP input endpoint URL')
    parser.add_argument('--mode', '-M', choices=['interactive', 'normal', 'suspicious', 'batch'],
                        default='interactive', help='Generation mode')
    parser.add_argument('--count', '-c', type=int, default=1, help='Number of transactions (for batch mode)')
    parser.add_argument('--delay', '-d', type=float, default=1.0, help='Delay between transactions in seconds')
    parser.add_argument('--user', '-u', type=int, help='Specific user ID')
    parser.add_argument('--fraud-type', '-f', choices=['high_amount', 'unusual_time', 'online_burst', 'unusual_merchant', 'random'],
                        default='random', help='Fraud type for suspicious transactions')
    parser.add_argument('--output', '-o', help='Output file for transactions (JSON)')

    args = parser.parse_args()

    generator = TransactionGenerator(
        model_endpoint=args.model_endpoint,
        nifi_endpoint=args.nifi_endpoint
    )

    transactions = []
    results = []

    print("=" * 60)
    print("Transaction Generator for Fraud Detection")
    print("=" * 60)

    try:
        if args.mode == 'interactive':
            while True:
                print("\n--- Menu ---")
                print("1. Create normal transaction")
                print("2. Create suspicious transaction")
                print("3. Create custom transaction")
                print("4. Send to ML model")
                print("5. Send to NiFi")
                print("6. Exit")

                choice = input("\nChoice [1-6]: ").strip()

                if choice == '1':
                    trans = generator.generate_normal_transaction(user_id=args.user)
                    generator.display_transaction(trans)
                    transactions.append(trans)

                elif choice == '2':
                    trans = generator.generate_suspicious_transaction(
                        user_id=args.user,
                        fraud_type=args.fraud_type
                    )
                    generator.display_transaction(trans)
                    transactions.append(trans)

                elif choice == '3':
                    trans = generator.create_interactive_transaction()
                    if trans:
                        generator.display_transaction(trans)
                        transactions.append(trans)

                elif choice == '4':
                    if not transactions:
                        print("No transactions created yet")
                    elif not args.model_endpoint:
                        print("Model endpoint not specified. Use --model-endpoint")
                    else:
                        trans = transactions[-1]
                        print(f"\nSending to model: {args.model_endpoint}")
                        result = generator.send_to_model(trans)
                        generator.display_result(result)
                        if result:
                            results.append(result)

                elif choice == '5':
                    if not transactions:
                        print("No transactions created yet")
                    elif not args.nifi_endpoint:
                        print("NiFi endpoint not specified. Use --nifi-endpoint")
                    else:
                        trans = transactions[-1]
                        print(f"\nSending to NiFi: {args.nifi_endpoint}")
                        result = generator.send_to_nifi(trans)
                        generator.display_result(result)

                elif choice == '6':
                    break

        elif args.mode in ['normal', 'suspicious']:
            for i in range(args.count):
                if args.mode == 'normal':
                    trans = generator.generate_normal_transaction(user_id=args.user)
                else:
                    trans = generator.generate_suspicious_transaction(
                        user_id=args.user,
                        fraud_type=args.fraud_type
                    )

                generator.display_transaction(trans)
                transactions.append(trans)

                if args.model_endpoint:
                    result = generator.send_to_model(trans)
                    generator.display_result(result)
                    if result:
                        results.append(result)

                if args.nifi_endpoint:
                    generator.send_to_nifi(trans)

                if i < args.count - 1:
                    time.sleep(args.delay)

        elif args.mode == 'batch':
            print(f"Generating {args.count} mixed transactions...")
            for i in range(args.count):
                if random.random() < 0.1:  # 10% suspicious
                    trans = generator.generate_suspicious_transaction(fraud_type=args.fraud_type)
                else:
                    trans = generator.generate_normal_transaction()

                transactions.append(trans)

                if args.model_endpoint:
                    result = generator.send_to_model(trans)
                    if result:
                        results.append({**trans, **result})

                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{args.count} transactions")

                if i < args.count - 1:
                    time.sleep(args.delay)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    # Save output if requested
    if args.output and transactions:
        output_data = {
            'transactions': transactions,
            'results': results,
            'generated_at': datetime.now().isoformat()
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved {len(transactions)} transactions to {args.output}")

    print(f"\nTotal transactions generated: {len(transactions)}")
    print("Done!")


if __name__ == "__main__":
    main()
