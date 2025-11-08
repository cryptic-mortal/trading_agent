"""
Demo: Training SmartFolio with Real-time Explanations

This script demonstrates how to train SmartFolio with the Trading Agent
generating explanations for portfolio decisions in real-time.

Usage:
    python integration/demo_training_with_explanations.py

Options:
    - Modify WHICH_TICKERS to use 'tickers' (106 stocks) or 'tickers1' (254 stocks)
    - Modify NUM_TRAIN_STEPS to control training duration
    - Modify TOP_K to limit explanations to top holdings
"""

import sys
import os
from pathlib import Path

# Add workspace root to Python path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SmartFolio"))
sys.path.insert(0, str(ROOT / "explainable-rl-through-causality"))

import numpy as np
import torch
from stable_baselines3 import PPO

# SmartFolio imports
from env.portfolio_env import StockPortfolioEnv
from dataloader.data_loader import AllGraphDataSampler
from torch_geometric.loader import DataLoader

# Integration imports
from integration.smartfolio_explainer import create_callback_from_master_tickers

# =============================================================================
# Configuration
# =============================================================================

WHICH_TICKERS = 'tickers'  # Choose 'tickers' (106) or 'tickers1' (254)
NUM_TRAIN_STEPS = 5000     # Number of RL training steps (reduce for quick test)
TOP_K = 10                 # Generate reports for top K holdings only
MIN_WEIGHT = 0.02          # Only explain holdings >= 2%
REPORT_DIR = 'reports_demo'
ASYNC_MODE = True          # Run explanations in background
USE_REAL_DATA = True       # Set False for quick test with dummy data

# =============================================================================
# Helper: Create dummy args object
# =============================================================================

class Args:
    """Dummy args object for SmartFolio environment"""
    def __init__(self):
        self.num_stocks = 8  # Will be updated
        self.input_dim = 6
        self.lookback_len = 10
        self.initial_capital = 100000
        self.transaction_cost = 0.001
        self.risk_score = 0.5
        self.dd_base_weight = 1.0
        self.dd_risk_factor = 1.0

# =============================================================================
# Main Training Function
# =============================================================================

def train_with_explanations():
    """Train SmartFolio with real-time explanations from Trading Agent"""
    
    print("="*80)
    print("SmartFolio + Trading Agent Integration Demo")
    print("="*80)
    
    # Step 1: Load tickers and create callback
    print(f"\n[1/5] Loading tickers from {WHICH_TICKERS}.csv...")
    try:
        callback_fn, tickers = create_callback_from_master_tickers(
            which=WHICH_TICKERS,
            report_dir=REPORT_DIR,
            top_k=TOP_K,
            min_weight_threshold=MIN_WEIGHT,
            async_mode=ASYNC_MODE
        )
        print(f"✓ Loaded {len(tickers)} tickers")
        print(f"  First 5: {tickers[:5]}")
        print(f"  Reports will be saved to: {REPORT_DIR}/")
    except Exception as e:
        print(f"✗ Failed to load tickers: {e}")
        print("  Make sure SmartFolio/tickers.csv exists")
        return
    
    # Step 2: Create environment
    print(f"\n[2/5] Creating SmartFolio environment...")
    
    if USE_REAL_DATA:
        # Load real data from SmartFolio's dataset
        try:
            data_dir = 'SmartFolio/dataset_default/data_train_predict_custom/1_hy/'
            if not os.path.exists(data_dir):
                print(f"  Dataset not found at {data_dir}")
                print(f"  Falling back to dummy data...")
                USE_REAL_DATA = False
            else:
                dataset = AllGraphDataSampler(
                    base_dir=data_dir,
                    date=True,
                    train_start_date='2020-01-06',
                    train_end_date='2023-01-31',
                    mode="train"
                )
                loader = DataLoader(dataset, batch_size=len(dataset), pin_memory=True)
                batch = next(iter(loader))
                data_sample = batch[0]
                
                corr = data_sample['correlation'].numpy()
                features = data_sample['features'].numpy()
                returns = data_sample['return'].numpy()
                
                print(f"✓ Loaded real data:")
                print(f"  Correlation shape: {corr.shape}")
                print(f"  Features shape: {features.shape}")
                print(f"  Returns shape: {returns.shape}")
        except Exception as e:
            print(f"  Failed to load real data: {e}")
            print(f"  Falling back to dummy data...")
            USE_REAL_DATA = False
    
    if not USE_REAL_DATA:
        # Create dummy data for quick testing
        num_stocks = len(tickers)
        num_days = 252  # 1 year
        
        corr = np.eye(num_stocks, dtype=np.float32)
        features = np.random.randn(num_days, num_stocks, 6).astype(np.float32)
        returns = np.random.randn(num_days, num_stocks).astype(np.float32) * 0.01
        
        print(f"✓ Created dummy data for {num_stocks} stocks, {num_days} days")
    
    # Update args with actual number of stocks
    args = Args()
    args.num_stocks = len(tickers)
    
    # Create environment WITH callback
    env = StockPortfolioEnv(
        args=args,
        corr=corr,
        features=features,
        returns=returns,
        ticker_list=tickers,           # Pass ticker names
        report_callback=callback_fn     # Pass explanation callback
    )
    print(f"✓ Environment created with {args.num_stocks} stocks")
    print(f"  Callback enabled: {env.report_callback is not None}")
    
    # Step 3: Create RL model
    print(f"\n[3/5] Creating PPO model...")
    model = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"✓ PPO model created on {model.device}")
    
    # Step 4: Train model
    print(f"\n[4/5] Training model for {NUM_TRAIN_STEPS} steps...")
    print("  (Explanations will be generated in background)")
    print("  Press Ctrl+C to stop early\n")
    
    try:
        model.learn(total_timesteps=NUM_TRAIN_STEPS)
        print(f"\n✓ Training complete!")
    except KeyboardInterrupt:
        print(f"\n✓ Training interrupted by user")
    
    # Step 5: Check generated reports
    print(f"\n[5/5] Checking generated reports...")
    report_path = Path(REPORT_DIR)
    if report_path.exists():
        reports = list(report_path.glob("*.md"))
        print(f"✓ Found {len(reports)} explanation reports in {REPORT_DIR}/")
        
        if reports:
            print(f"\n  Sample reports:")
            for report in sorted(reports)[:5]:
                size = report.stat().st_size
                print(f"    - {report.name} ({size:,} bytes)")
            
            if len(reports) > 5:
                print(f"    ... and {len(reports) - 5} more")
            
            # Show snippet from first report
            first_report = sorted(reports)[0]
            print(f"\n  Preview of {first_report.name}:")
            print("  " + "-"*76)
            with open(first_report, 'r') as f:
                lines = f.readlines()[:15]
                for line in lines:
                    print(f"  {line.rstrip()}")
            if len(lines) >= 15:
                print("  ...")
            print("  " + "-"*76)
    else:
        print(f"✗ No reports directory found")
    
    print("\n" + "="*80)
    print("Demo complete!")
    print(f"Check {REPORT_DIR}/ for explanation reports")
    print("="*80)

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    try:
        train_with_explanations()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
