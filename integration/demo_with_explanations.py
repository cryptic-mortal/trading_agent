"""
Demo: SmartFolio with Real-Time Trading Agent Explanations

This script demonstrates the callback integration pattern where SmartFolio
generates portfolio decisions and the Trading Agent immediately produces
explanation reports.

Usage:
    python integration/demo_with_explanations.py
"""
import sys
import os
import logging
from pathlib import Path

# Add project roots to path
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "SmartFolio"))
sys.path.insert(0, str(workspace_root / "trading_agent"))
sys.path.insert(0, str(workspace_root))

import torch
import numpy as np
from integration.smartfolio_explainer import SmartFolioExplainer, create_callback_from_master_tickers
from integration.utils import load_tickers_from_master

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_fake_env_for_demo(num_stocks=8, num_steps=10):
    """Create minimal fake environment for smoke testing."""
    from SmartFolio.env.portfolio_env import StockPortfolioEnv
    
    # Minimal args object
    class Args:
        input_dim = 6
    
    args = Args()
    
    # Create fake tensors (all zeros for demo)
    returns = torch.zeros((num_steps, num_stocks))
    corr = torch.zeros((num_steps, num_stocks, num_stocks))
    ts_features = torch.zeros((num_steps, num_stocks, args.input_dim))
    features = ts_features
    ind = torch.zeros((num_steps, num_stocks, num_stocks))
    pos = torch.zeros((num_steps, num_stocks, num_stocks))
    neg = torch.zeros((num_steps, num_stocks, num_stocks))
    
    # Fake ticker list
    fake_tickers = [f"FAKE{i:02d}.SZ" for i in range(num_stocks)]
    
    logger.info(f"Created fake environment: {num_stocks} stocks, {num_steps} steps")
    logger.info(f"Fake tickers: {fake_tickers}")
    
    return args, returns, corr, ts_features, features, ind, pos, neg, fake_tickers


def run_demo_with_fake_data():
    """Run a quick demo with fake data to test integration."""
    logger.info("=" * 60)
    logger.info("DEMO: SmartFolio + Trading Agent Integration (Fake Data)")
    logger.info("=" * 60)
    
    # Create fake environment components
    args, returns, corr, ts_features, features, ind, pos, neg, tickers = \
        create_fake_env_for_demo(num_stocks=8, num_steps=5)
    
    # Import environment
    from SmartFolio.env.portfolio_env import StockPortfolioEnv
    
    # Create explainer
    explainer = SmartFolioExplainer(
        report_dir="reports/demo_fake",
        top_k=3,  # Only explain top 3 holdings
        async_mode=False,  # Synchronous for demo clarity
        lookback_days=7
    )
    
    # Create environment WITH callback
    env = StockPortfolioEnv(
        args,
        corr=corr,
        ts_features=ts_features,
        features=features,
        ind=ind,
        pos=pos,
        neg=neg,
        returns=returns,
        mode="train",
        device='cpu',
        ind_yn=False,
        pos_yn=False,
        neg_yn=False,
        ticker_list=tickers,
        report_callback=explainer.handle_portfolio_decision  # ← THE MAGIC
    )
    
    logger.info("Environment created with explainer callback attached")
    
    # Run a few steps
    obs = env.reset()
    logger.info("Environment reset successful")
    
    for step in range(3):
        # Create a fake action (continuous weights)
        action = np.random.rand(8)
        action = action / action.sum()  # Normalize to sum to 1
        
        logger.info(f"\n--- Step {step + 1} ---")
        logger.info(f"Action (weights): {action}")
        
        obs, reward, done, info = env.step(action)
        logger.info(f"Reward: {reward:.6f}, Done: {done}")
        
        if done:
            break
    
    # Shutdown explainer
    explainer.shutdown()
    
    logger.info("\n" + "=" * 60)
    logger.info("Demo completed! Check reports/demo_fake/ for generated reports")
    logger.info("=" * 60)


def run_demo_with_real_data():
    """Run demo with real SmartFolio dataset (if available)."""
    logger.info("=" * 60)
    logger.info("DEMO: SmartFolio + Trading Agent Integration (Real Data)")
    logger.info("=" * 60)
    
    # Try to load real dataset
    dataset_path = workspace_root / "SmartFolio" / "dataset_default"
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        logger.error("Please run with fake data mode or set correct dataset path")
        return
    
    try:
        # Load tickers from top-level tickers.csv in the SmartFolio repo
        tickers = load_tickers_from_master(root_path=workspace_root, which='tickers')
        logger.info(f"Loaded {len(tickers)} tickers from SmartFolio/tickers.csv")
        logger.info(f"First 5 tickers: {tickers[:5]}")

        # TODO: Load actual SmartFolio data (features, returns, etc.)
        # For now, this is a placeholder - you would load your trained data here
        logger.warning("Real data loading not fully implemented - use your data loader")

        # You would continue similar to fake data demo but with real data:
        # 1. Load features, returns, correlation matrices
        # 2. Create explainer
        # 3. Create env with ticker_list and report_callback
        # 4. Run episodes

    except Exception as e:
        logger.error(f"Error loading real data: {e}")
        logger.info("Falling back to fake data demo")
        run_demo_with_fake_data()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Demo SmartFolio + Trading Agent integration"
    )
    parser.add_argument(
        "--mode",
        choices=["fake", "real"],
        default="fake",
        help="Run with fake or real data"
    )
    
    args = parser.parse_args()
    
    if args.mode == "fake":
        run_demo_with_fake_data()
    else:
        run_demo_with_real_data()


if __name__ == "__main__":
    main()
