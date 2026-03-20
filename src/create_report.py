import os
import json
import pandas as pd
import quantstats as qs
import argparse
import glob
from .config import Config

def get_latest_results_file(directory=Config.ResultsDir):
    """Finds the most recent JSON file in the results directory."""
    files = glob.glob(os.path.join(directory, "results_*.json"))
    if not files:
        return None
    # Sort by modification time
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def create_report(file_path):
    """Parses JSON backtest results and produces a quantstats report."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    print(f"Loading results from: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)

    if 'equity_curve' not in data or 'equity_times' not in data:
        print("Error: JSON file does not contain 'equity_curve' or 'equity_times'.")
        return

    # Create a Series from equity curve
    equity = pd.Series(data['equity_curve'], index=pd.to_datetime(data['equity_times']))
    
    # Calculate returns (percentage change)
    returns = equity.pct_change().dropna()
    
    # Generate report path
    timestamp = data.get('timestamp', 'latest')
    output_path = os.path.join(Config.ResultsDir, f"report_{timestamp}.html")
    
    print(f"Generating Quantstats report: {output_path}")
    
    # Standard format for quantstats
    qs.reports.html(returns, output=output_path, title=f"GPT Trader Performance - {timestamp}")
    
    print(f"Report successfully saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Quantstats report from backtest results.")
    parser.add_argument("--file", type=str, help="Path to a specific results JSON file.")
    
    args = parser.parse_args()
    
    target_file = args.file
    if not target_file:
        target_file = get_latest_results_file()
        if not target_file:
            print(f"No results found in the {Config.ResultsDir} folder.")
            exit(1)
            
    create_report(target_file)
