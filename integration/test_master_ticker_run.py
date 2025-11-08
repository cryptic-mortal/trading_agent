import sys
from pathlib import Path
# ensure workspace root on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from integration.utils import load_tickers_from_master
from integration.smartfolio_explainer import create_callback_from_master_tickers
import numpy as np
import torch
from SmartFolio.env.portfolio_env import StockPortfolioEnv

# Load first N tickers from master tickers.csv
tickers = load_tickers_from_master(which='tickers')[:8]
print('Using tickers:', tickers)
# Create callback and explainer
callback_fn, _ = create_callback_from_master_tickers(which='tickers', report_dir='reports/test_master', async_mode=False, top_k=3)

# Create minimal fake env
class Args:
    input_dim = 6
args = Args()
num_steps = 3
num_stocks = 8
returns = torch.zeros((num_steps, num_stocks))
corr = torch.zeros((num_steps, num_stocks, num_stocks))
ts_features = torch.zeros((num_steps, num_stocks, args.input_dim))
features = ts_features
ind = torch.zeros((num_steps, num_stocks, num_stocks))
pos = torch.zeros((num_steps, num_stocks, num_stocks))
neg = torch.zeros((num_steps, num_stocks, num_stocks))

env = StockPortfolioEnv(args, corr=corr, ts_features=ts_features, features=features, ind=ind, pos=pos, neg=neg, returns=returns, mode='train', device='cpu', ind_yn=False, pos_yn=False, neg_yn=False, ticker_list=tickers, report_callback=callback_fn)
obs = env.reset()
action = np.random.rand(num_stocks)
action = action / action.sum()
obs, reward, done, info = env.step(action)
print('Step returned reward=', reward, 'done=', done)
