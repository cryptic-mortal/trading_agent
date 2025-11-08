"""
SmartFolio -> Trading Agent explainer connector.
Loads tickers from master top-level files by default.
"""
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from datetime import datetime, timedelta

# try to import trading agent from sibling folder
import sys
ROOT = Path(__file__).resolve().parents[1]
TRA_PATH = ROOT / 'trading_agent'
if str(TRA_PATH) not in sys.path:
    sys.path.insert(0, str(TRA_PATH))

try:
    from tradingagents.combined_weight_agent import WeightSynthesisAgent
    TRADING_AGENT_AVAILABLE = True
except Exception as e:
    logging.warning(f"Trading Agent import failed: {e}")
    WeightSynthesisAgent = None
    TRADING_AGENT_AVAILABLE = False


class SmartFolioExplainer:
    def __init__(
        self,
        report_dir: str = 'reports',
        lookback_days: int = 7,
        max_articles: int = 8,
        min_weight_threshold: float = 0.01,
        top_k: Optional[int] = None,
        async_mode: bool = True,
        max_workers: int = 3,
        base_date: Optional[str] = None,  # Base date for step-to-date conversion
        use_llm: bool = False,  # NEW: Enable LLM-enhanced explanations
        llm_model: Optional[str] = None  # NEW: Override default LLM model
    ):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.lookback_days = lookback_days
        self.max_articles = max_articles
        self.min_weight_threshold = min_weight_threshold
        self.top_k = top_k
        self.async_mode = async_mode
        self.use_llm = use_llm
        self.llm_model = llm_model
        
        # Date conversion: if base_date provided, convert step numbers to dates
        if base_date:
            try:
                self.base_date = datetime.strptime(base_date, '%Y-%m-%d')
            except ValueError:
                logging.warning(f"Invalid base_date format: {base_date}, using default")
                self.base_date = datetime(2023, 1, 1)
        else:
            self.base_date = datetime(2023, 1, 1)  # Default base date
        
        if not TRADING_AGENT_AVAILABLE:
            logging.error('Trading Agent not available; explainer inactive')
            self.agent = None
            self.executor = None
            return
        self.agent = WeightSynthesisAgent()
        self.executor = ThreadPoolExecutor(max_workers=max_workers) if async_mode else None
        logging.info(f"SmartFolioExplainer initialized; reports -> {self.report_dir}")

    def handle_portfolio_decision(self, as_of: int, tickers: List[str], weights: List[float], net_value: float):
        if self.agent is None:
            return
        # pair and filter
        pairs = [(t, w) for t, w in zip(tickers, weights) if w >= self.min_weight_threshold]
        if not pairs:
            return
        if self.top_k is not None:
            pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:self.top_k]
        for ticker, weight in pairs:
            if self.async_mode and self.executor is not None:
                self.executor.submit(self._generate_and_save_report, ticker, weight, as_of)
            else:
                self._generate_and_save_report(ticker, weight, as_of)

    def _generate_and_save_report(self, ticker: str, weight: float, as_of: int):
        try:
            # Convert step number to date string
            actual_date = self.base_date + timedelta(days=int(as_of))
            date_str = actual_date.strftime('%Y-%m-%d')
            
            report = self.agent.generate_report(
                ticker=ticker, 
                weight=weight, 
                as_of=date_str,  # Use date string instead of step number
                lookback_days=self.lookback_days, 
                max_articles=self.max_articles,
                use_llm=self.use_llm,  # Enable LLM if configured
                llm_model=self.llm_model  # Use specified model or default
            )
            fname = self.report_dir / f"step_{as_of:06d}_{ticker}.md"
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(report.to_markdown(include_components=True, include_metrics=True, include_articles=True))
            logging.info(f"Saved report: {fname}")
        except Exception as e:
            logging.error(f"Failed to generate report for {ticker}: {e}")

    def shutdown(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True)


def create_callback_from_master_tickers(which: str = 'tickers', **kwargs):
    """Utility to quickly create a callback using master tickers file.
    which: 'tickers' or 'tickers1'
    Returns: (callback_function, ticker_list)
    """
    from .utils import load_tickers_from_master
    tickers = load_tickers_from_master(which=which)
    expl = SmartFolioExplainer(**kwargs)
    return expl.handle_portfolio_decision, tickers
