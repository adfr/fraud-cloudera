#!/usr/bin/env python3
"""
Simple Hello World job for Cloudera AI
"""
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Simple Hello World script")
    parser.add_argument("--name", type=str, default="World", help="Name to greet")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times to repeat the greeting")
    
    args = parser.parse_args()
    
    for i in range(args.repeat):
        print(f"Hello, {args.name}! (#{i+1})")

if __name__ == "__main__":
    main() 