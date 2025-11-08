"""
Quick Test: Verify Integration Works

This is a minimal test to verify the integration is working correctly.
Runs just a few environment steps to test the callback mechanism.

Usage:
    python integration/quick_test.py
"""

import sys
import os
from pathlib import Path

# Add paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SmartFolio"))

# Optional: Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_file = ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✓ Loaded environment from {env_file}")
except ImportError:
    pass  # python-dotenv not installed, skip

# Check if API key is set
if not os.getenv('GEMINI_API_KEY') and not os.getenv('GOOGLE_API_KEY'):
    print("\n[WARNING] No API key found!")
    print("   To enable LLM explanations, set your API key:")
    print("   PowerShell: $env:GEMINI_API_KEY = 'your-key-here'")
    print("   Or add to .env file: GEMINI_API_KEY=your-key-here")
    print("\n   Continuing with deterministic explanations...\n")
else:
    key_name = 'GEMINI_API_KEY' if os.getenv('GEMINI_API_KEY') else 'GOOGLE_API_KEY'
    print(f"[OK] API key found: {key_name}")
    print(f"  LLM-enhanced explanations enabled!\n")

import numpy as np
import torch
from env.portfolio_env import StockPortfolioEnv
from integration.smartfolio_explainer import create_callback_from_master_tickers

# Minimal args
class Args:
    def __init__(self):
        self.num_stocks = 8
        self.input_dim = 6
        self.lookback_len = 10
        self.initial_capital = 100000
        self.transaction_cost = 0.001
        self.risk_score = 0.5
        self.dd_base_weight = 1.0
        self.dd_risk_factor = 1.0

print("Loading tickers...")
callback_fn, tickers = create_callback_from_master_tickers(
    which='tickers',
    report_dir='reports_quick_test',
    top_k=5,
    use_llm = True,
    llm_model ='gemini-2.0-flash',
    min_weight_threshold=0.01,
    async_mode=False,  # Synchronous for testing
    base_date='2025-11-08'  # Today's date - will get fresh news!
)
print(f"✓ Loaded {len(tickers)} tickers: {tickers[:8]}")

# Use only first 8 tickers for quick test
tickers = tickers[:8]

print("\nCreating dummy environment...")
args = Args()
args.num_stocks = len(tickers)

# Dummy data (convert to tensors) - need all required tensors
num_stocks = len(tickers)
num_days = 50

# Create dummy tensors for all required data
corr = torch.eye(num_stocks, dtype=torch.float32).unsqueeze(0).expand(num_days, -1, -1)
features = torch.randn(num_days, num_stocks, 6, dtype=torch.float32) * 0.1
returns = torch.randn(num_days, num_stocks, dtype=torch.float32) * 0.02

# Create dummy graph matrices (industry, momentum, reversal)
ind = torch.eye(num_stocks, dtype=torch.float32).unsqueeze(0).expand(num_days, -1, -1)
pos = torch.eye(num_stocks, dtype=torch.float32).unsqueeze(0).expand(num_days, -1, -1)
neg = torch.eye(num_stocks, dtype=torch.float32).unsqueeze(0).expand(num_days, -1, -1)

env = StockPortfolioEnv(
    args=args,
    corr=corr,
    features=features,
    returns=returns,
    ind=ind,  # Industry graph
    pos=pos,  # Momentum graph
    neg=neg,  # Reversal graph
    ticker_list=tickers,
    report_callback=callback_fn
)
print(f"✓ Environment created")

print("\nRunning 3 test steps...")
obs = env.reset()
for i in range(3):
    # Random action
    action = np.random.randn(num_stocks)
    obs, reward, done, info = env.step(action)
    print(f"  Step {i+1}: reward={reward:.6f}, done={done}, portfolio_value={info.get('net_value', 'N/A')}")
    if done:
        break

print("\n✓ Test complete!")
print("Check reports_quick_test/ directory for generated reports")
