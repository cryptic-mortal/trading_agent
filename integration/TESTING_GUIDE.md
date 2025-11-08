# Testing the SmartFolio + Trading Agent Integration

This guide shows you how to test the integration between SmartFolio and the Trading Agent.

## Quick Start

### Option 1: Quick Test (Recommended First)
Run a minimal test with dummy data to verify the integration works:

```powershell
cd C:\Users\manna
python integration\quick_test.py
```

**What it does:**
- Loads 8 real tickers from `tickers.csv`
- Creates a simple environment with dummy data
- Runs 3 steps with random actions
- Generates explanation reports for portfolio decisions
- Takes ~30 seconds

**Expected output:**
```
Loading tickers...
✓ Loaded 106 tickers: ['RELIANCE.NS', 'TCS.NS', ...]
Creating dummy environment...
✓ Environment created
Running 3 test steps...
  Step 1: reward=0.001234, done=False, portfolio_value=100123.45
  Step 2: reward=-0.000567, done=False, portfolio_value=100067.89
  Step 3: reward=0.002134, done=False, portfolio_value=100281.23
✓ Test complete!
Check reports_quick_test/ directory for generated reports
```

Check `reports_quick_test/` for markdown reports explaining each decision.

---

### Option 2: Full Training Demo
Run a complete training session with explanations:

```powershell
cd C:\Users\manna
python integration\demo_training_with_explanations.py
```

**What it does:**
- Loads real Indian stock tickers
- Creates SmartFolio environment (attempts to use real data if available)
- Trains a PPO agent for 5000 steps
- Generates explanations for portfolio decisions in background
- Takes ~10-30 minutes depending on hardware

**Expected output:**
```
================================================================================
SmartFolio + Trading Agent Integration Demo
================================================================================

[1/5] Loading tickers from tickers.csv...
✓ Loaded 106 tickers
  First 5: ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS']
  Reports will be saved to: reports_demo/

[2/5] Creating SmartFolio environment...
✓ Environment created with 106 stocks
  Callback enabled: True

[3/5] Creating PPO model...
✓ PPO model created on cuda

[4/5] Training model for 5000 steps...
  (Explanations will be generated in background)
  Press Ctrl+C to stop early

[PPO training logs...]

✓ Training complete!

[5/5] Checking generated reports...
✓ Found 234 explanation reports in reports_demo/

  Sample reports:
    - step_000001_RELIANCE.NS.md (12,456 bytes)
    - step_000001_TCS.NS.md (11,234 bytes)
    - step_000002_HDFCBANK.NS.md (13,567 bytes)
    ...
```

Check `reports_demo/` for detailed markdown reports.

---

### Option 3: Use Your Own SmartFolio Training Script

Modify your existing training code to add explanations:

```python
from integration.smartfolio_explainer import create_callback_from_master_tickers

# Load tickers and create callback
callback_fn, tickers = create_callback_from_master_tickers(
    which='tickers',           # Or 'tickers1' for more stocks
    report_dir='reports',      # Where to save reports
    top_k=10,                  # Explain only top 10 holdings
    min_weight_threshold=0.02, # Ignore holdings < 2%
    async_mode=True            # Run in background (recommended)
)

# Create environment WITH callback
env = StockPortfolioEnv(
    args=args,
    corr=corr,
    features=features,
    returns=returns,
    ticker_list=tickers,        # Pass ticker names
    report_callback=callback_fn  # Enable explanations
)

# Train as usual - explanations happen automatically!
model.learn(total_timesteps=10000)
```

---

## Configuration Options

### Ticker Selection
- `'tickers'`: 106 Indian stocks (default)
- `'tickers1'`: 254 Indian stocks (more comprehensive)

### Report Configuration
- `top_k=10`: Generate reports for top K holdings by weight
- `min_weight_threshold=0.02`: Ignore holdings below 2%
- `async_mode=True`: Run explanations in background (recommended for training)
- `async_mode=False`: Wait for each explanation (slower, but synchronous)

### Custom Report Directory
```python
callback_fn, tickers = create_callback_from_master_tickers(
    report_dir='my_custom_reports',
    ...
)
```

---

## Understanding the Reports

Each report is a markdown file named: `step_{N:06d}_{TICKER}.md`

Example: `step_000042_RELIANCE.NS.md`

**Report Structure:**
```markdown
# Portfolio Explanation Report
**Ticker:** RELIANCE.NS
**Weight:** 15.3%
**Date:** 42
**Portfolio Value:** $102,345.67

## Fundamental Analysis
- PE Ratio: 25.3
- Revenue Growth: 12.5%
- Profit Margins: 8.9%
...

## News Sentiment Analysis
- Recent news: "Reliance announces new green energy initiative"
- Sentiment: Positive (0.78)
...

## Rationale
This stock was allocated 15.3% based on strong fundamentals
and positive market sentiment...
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'tradingagents'"
**Solution:** The Trading Agent repo needs to be at `C:\Users\manna\explainable-rl-through-causality`

Check if it exists:
```powershell
cd C:\Users\manna\explainable-rl-through-causality
ls
```

### Issue: "as_of must be in YYYY-MM-DD format"
**Expected behavior:** The integration currently passes step numbers (0, 1, 2...) instead of dates. Reports are still generated but with this warning.

**Fix (optional):** Modify the callback to convert step numbers to dates:
```python
# In smartfolio_explainer.py, modify _generate_and_save_report:
from datetime import datetime, timedelta

base_date = datetime(2023, 1, 1)
actual_date = base_date + timedelta(days=as_of)
date_str = actual_date.strftime('%Y-%m-%d')

report = self.agent.generate_report(
    ticker=ticker,
    weight=weight,
    as_of=date_str  # Use actual date string
)
```

### Issue: "No reports generated"
**Check:**
1. Is `async_mode=True`? Reports may still be processing. Wait a moment.
2. Are weights too small? Lower `min_weight_threshold` or check actual weights
3. Check terminal for errors during callback execution

### Issue: Training is slow
**Solution:** Make sure `async_mode=True` so explanations don't block training

### Issue: Too many reports
**Solution:** Increase `min_weight_threshold` or decrease `top_k`:
```python
callback_fn, tickers = create_callback_from_master_tickers(
    top_k=5,                    # Only top 5 holdings
    min_weight_threshold=0.05,  # Only holdings >= 5%
    ...
)
```

---

## Next Steps

1. **Run quick test** to verify everything works
2. **Examine generated reports** to understand the explanations
3. **Integrate into your training pipeline** using Option 3 above
4. **Tune configuration** based on your needs (top_k, thresholds, etc.)
5. **Convert step numbers to dates** if you need accurate date-based analysis

---

## Files Created

- `integration/quick_test.py` - Minimal test script
- `integration/demo_training_with_explanations.py` - Full training demo
- `integration/smartfolio_explainer.py` - Main connector class
- `integration/utils.py` - Helper utilities
- `integration/INTEGRATION_COMPLETE.md` - Technical documentation
- `SmartFolio/env/portfolio_env.py` - Modified to support callbacks

---

## Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Review `integration/INTEGRATION_COMPLETE.md` for technical details
3. Ensure both repos (SmartFolio and explainable-rl-through-causality) are present
4. Verify tickers.csv exists in SmartFolio root directory
