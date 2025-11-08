"""
Visual representation of the SmartFolio + Trading Agent integration.
Run this to see the ASCII architecture diagram.
"""

def print_integration_flow():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         SmartFolio + Trading Agent Integration Architecture          ║
╚══════════════════════════════════════════════════════════════════════╝

Step 1: Setup
─────────────
┌─────────────────────────┐
│  Your Training Script   │
│                         │
│  1. Load tickers from   │
│     dataset CSV         │
│                         │
│  2. Create explainer    │
│     with config         │
│                         │
│  3. Create SmartFolio   │
│     env with:           │
│     • ticker_list       │
│     • report_callback   │
└─────────────────────────┘


Step 2: Training Loop
─────────────────────
┌──────────────────────────────────────────────────────────┐
│  for episode in range(num_episodes):                     │
│      obs = env.reset()                                   │
│      while not done:                                     │
│          action = agent.get_action(obs)                  │
│          obs, reward, done, info = env.step(action)      │
│                                           ↓              │
│                                    [CALLBACK TRIGGERED]  │
└──────────────────────────────────────────────────────────┘


Step 3: Inside env.step() - SmartFolio
───────────────────────────────────────
┌────────────────────────────────────────────────────────┐
│  def step(self, actions):                              │
│      # 1. Process actions (agent's decisions)          │
│      weights = normalize(actions)  # [0.3, 0.4, 0.3]   │
│                                                         │
│      # 2. Calculate reward from market returns         │
│      reward = dot(weights, returns)                    │
│      net_value *= (1 + reward)                         │
│                                                         │
│      # 3. INVOKE CALLBACK (if provided)                │
│      if self.report_callback is not None:              │
│          # Map indices to tickers                      │
│          tickers = [ticker_list[i] for i in indices]   │
│          # Call explainer                              │
│          self.report_callback(                         │
│              as_of=current_step,                       │
│              tickers=["AAPL", "MSFT", "GOOGL"],        │
│              weights=[0.35, 0.40, 0.25],               │
│              net_value=1.05                            │
│          )                                             │
│                                                         │
│      return observation, reward, done, {}              │
└────────────────────────────────────────────────────────┘
                            ↓
                            ↓ callback invocation
                            ↓

Step 4: SmartFolioExplainer.handle_portfolio_decision()
────────────────────────────────────────────────────────
┌────────────────────────────────────────────────────────┐
│  def handle_portfolio_decision(as_of, tickers,         │
│                                 weights, net_value):   │
│      # 1. Filter by weight threshold                   │
│      filtered = [(t,w) for t,w in zip(tickers,weights) │
│                  if w > 0.01]                          │
│                                                         │
│      # 2. Optional: Keep only top K                    │
│      if top_k:                                         │
│          filtered = sorted(...)[:top_k]                │
│                                                         │
│      # 3. For each ticker, generate report             │
│      for ticker, weight in filtered:                   │
│          if async_mode:                                │
│              executor.submit(generate_report, ...)     │
│          else:                                         │
│              generate_report(ticker, weight, as_of)    │
└────────────────────────────────────────────────────────┘
                            ↓
                            ↓ calls Trading Agent
                            ↓

Step 5: Trading Agent (WeightSynthesisAgent)
─────────────────────────────────────────────
┌────────────────────────────────────────────────────────┐
│  For each ticker (e.g., "AAPL"):                       │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │  1. Fundamental Agent                            │  │
│  │     • Fetch financial metrics                    │  │
│  │     • P/E ratio, ROE, margins, growth, D/E       │  │
│  │     • Generate rationale points                  │  │
│  └──────────────────────────────────────────────────┘  │
│                        +                               │
│  ┌──────────────────────────────────────────────────┐  │
│  │  2. News Agent                                   │  │
│  │     • Scrape recent news (last 7 days)           │  │
│  │     • Analyze sentiment (pos/neg/neutral)        │  │
│  │     • Extract key headlines                      │  │
│  └──────────────────────────────────────────────────┘  │
│                        ↓                               │
│  ┌──────────────────────────────────────────────────┐  │
│  │  3. Synthesis                                    │  │
│  │     • Combine fundamental + news insights        │  │
│  │     • Generate unified rationale                 │  │
│  │     • Create WeightSynthesisReport               │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
                            ↓
                            ↓ return report
                            ↓

Step 6: Save Report to Disk
────────────────────────────
┌────────────────────────────────────────────────────────┐
│  Filename: reports/step_000042_AAPL.md                │
│                                                         │
│  # Combined Weight Review: AAPL                        │
│                                                         │
│  - **As of:** 42                                       │
│  - **Assigned Weight:** 35.00%                         │
│  - **News Lookback:** 7 day(s)                         │
│                                                         │
│  ## Unified Summary                                    │
│  - Strong revenue growth of 12.3%                      │
│  - Positive news sentiment (5 pos / 1 neg)             │
│  - ROE of 45.2% shows excellent efficiency             │
│                                                         │
│  ## Fundamental Agent Detail                           │
│  [P/E, ROE, margins, debt ratios...]                   │
│                                                         │
│  ## News Agent Detail                                  │
│  [Recent articles with sentiment scores...]            │
└────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════

Data Flow Timeline (Single Step)
═════════════════════════════════

T = 0.000s  │ Agent decides action [0.35, 0.40, 0.25]
            │
T = 0.001s  │ SmartFolio.step() normalizes weights
            │
T = 0.002s  │ Calculates reward = 0.015 (1.5% return)
            │
T = 0.003s  │ Updates net_value = 1.00 → 1.015
            │
T = 0.004s  │ ★ CALLBACK INVOKED ★
            │ → explainer.handle_portfolio_decision()
            │
T = 0.005s  │ Explainer filters tickers (removes < 1%)
            │
T = 0.006s  │ [If async] Submits 3 tasks to thread pool
            │             Returns immediately
            │ [If sync]  Blocks and generates all 3 reports
            │
            │ ┌─ Background Thread 1 ────────────────┐
            │ │ T=0.010s  Fetch AAPL fundamentals    │
            │ │ T=0.500s  Scrape AAPL news           │
            │ │ T=1.200s  Generate report            │
            │ │ T=1.250s  Save to disk               │
            │ └──────────────────────────────────────┘
            │
            │ ┌─ Background Thread 2 ────────────────┐
            │ │ T=0.011s  Fetch MSFT fundamentals    │
            │ │ ...                                  │
            │ └──────────────────────────────────────┘
            │
T = 0.007s  │ env.step() returns to training loop
            │ (async mode: reports still generating)
            │
T = 0.008s  │ Training continues with next step
            │ (no blocking!)
            │
T = 2.500s  │ All background reports completed

═══════════════════════════════════════════════════════════════

Key Design Decisions
════════════════════

1. Callback Pattern (Not Polling)
   ✅ SmartFolio actively notifies explainer
   ✅ No need to read log files or poll
   ✅ Real-time, event-driven

2. Minimal Changes to SmartFolio
   ✅ Only ~25 lines added
   ✅ Fully backward compatible
   ✅ Optional parameters (ticker_list, callback)

3. Async by Default
   ✅ Reports generated in background
   ✅ Training loop not blocked
   ✅ ThreadPoolExecutor manages concurrency

4. Error Isolation
   ✅ try-except in callback invocation
   ✅ Explainer errors don't crash training
   ✅ Warnings logged, training continues

5. Flexible Filtering
   ✅ min_weight_threshold (skip tiny allocations)
   ✅ top_k (only explain largest holdings)
   ✅ Configurable at runtime

═══════════════════════════════════════════════════════════════
""")


def print_file_structure():
    print("""
File Structure After Integration
═════════════════════════════════

workspace/
│
├── SmartFolio/
│   ├── env/
│   │   └── portfolio_env.py          ← MODIFIED (~25 lines)
│   ├── dataset_default/
│   │   ├── hs300_org.csv             (ticker source)
│   │   └── ...
│   └── ...
│
├── trading_agent/
│   ├── tradingagents/
│   │   ├── combined_weight_agent.py  (unchanged, used by integration)
│   │   ├── fundamental_agent.py      (unchanged)
│   │   └── news_weight_agent.py      (unchanged)
│   └── ...
│
└── integration/                       ← NEW DIRECTORY
    ├── __init__.py
    ├── smartfolio_explainer.py        ← MAIN CONNECTOR
    ├── utils.py                       ← HELPER FUNCTIONS
    ├── demo_with_explanations.py      ← DEMO SCRIPT
    ├── README.md                      ← DOCUMENTATION
    ├── IMPLEMENTATION_SUMMARY.md      ← THIS SUMMARY
    └── architecture_diagram.py        ← YOU ARE HERE


reports/                               ← GENERATED BY INTEGRATION
├── demo_fake/
│   └── (test reports)
└── your_experiment/
    ├── step_000001_AAPL.md
    ├── step_000001_MSFT.md
    ├── step_000002_GOOGL.md
    └── ...

═══════════════════════════════════════════════════════════════
""")


def print_usage_quick_reference():
    print("""
Quick Reference: Using the Integration
═══════════════════════════════════════

Minimal Example (Copy-Paste Ready)
───────────────────────────────────
```python
from integration.smartfolio_explainer import SmartFolioExplainer
from integration.utils import load_ticker_list
from SmartFolio.env.portfolio_env import StockPortfolioEnv

# 1. Load tickers
tickers = load_ticker_list("SmartFolio/dataset_default", "hs300")

# 2. Create explainer
explainer = SmartFolioExplainer(
    report_dir="reports/my_experiment",
    top_k=10,           # Only explain top 10 holdings
    async_mode=True     # Background generation
)

# 3. Create SmartFolio env WITH callback
env = StockPortfolioEnv(
    args, corr=corr, features=features, returns=returns,
    # ... your existing params ...
    ticker_list=tickers,                              # ← ADD
    report_callback=explainer.handle_portfolio_decision  # ← ADD
)

# 4. Train as usual - reports auto-generated!
obs = env.reset()
for episode in range(num_episodes):
    action = agent.get_action(obs)
    obs, reward, done, info = env.step(action)  # ← Explanations here!
    if done:
        break

# 5. Cleanup
explainer.shutdown()
```

Run Demo
────────
```powershell
# Test with fake data
python integration/demo_with_explanations.py --mode fake

# With real data
python integration/demo_with_explanations.py --mode real
```

Common Configurations
─────────────────────
# Explain everything (slow for many stocks)
SmartFolioExplainer(report_dir="reports")

# Only top 5 holdings
SmartFolioExplainer(top_k=5)

# Skip small allocations (< 2%)
SmartFolioExplainer(min_weight_threshold=0.02)

# Synchronous (for debugging)
SmartFolioExplainer(async_mode=False)

# More news history
SmartFolioExplainer(lookback_days=14, max_articles=15)

═══════════════════════════════════════════════════════════════
""")


if __name__ == "__main__":
    print_integration_flow()
    print("\n" * 2)
    print_file_structure()
    print("\n" * 2)
    print_usage_quick_reference()
