"""
Agent Stock Screener - Uses trading_agent to screen and rank stocks.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from trading_agent.tradingagents.graph.trading_graph import TradingAgentsGraph
from trading_agent.tradingagents.default_config import DEFAULT_CONFIG
from typing import List, Dict
import re


class AgentStockScreener:
    """Use trading_agent to screen and rank stocks."""
    
    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG.copy()
        self.agent = TradingAgentsGraph(debug=False, config=self.config)
    
    def screen_stocks(self, tickers: List[str], date: str) -> Dict[str, dict]:
        """
        Run trading agent on multiple stocks.
        
        Args:
            tickers: List of stock ticker symbols
            date: Date string in format "YYYY-MM-DD"
        
        Returns:
            Dict mapping ticker -> {
                decision: 'BUY'/'SELL'/'HOLD', 
                reasoning: str, 
                confidence: float
            }
        """
        results = {}
        
        for ticker in tickers:
            try:
                print(f"Analyzing {ticker}...")
                _, decision_text = self.agent.propagate(ticker, date)
                
                # Parse decision from text
                decision = self._parse_decision(decision_text)
                results[ticker] = {
                    'decision': decision,
                    'reasoning': decision_text,
                    'confidence': self._extract_confidence(decision_text)
                }
                print(f"  → {decision} (confidence: {results[ticker]['confidence']:.2f})")
                
            except Exception as e:
                print(f"Error analyzing {ticker}: {e}")
                results[ticker] = {
                    'decision': 'HOLD', 
                    'reasoning': f"Error: {str(e)}", 
                    'confidence': 0.0
                }
        
        return results
    
    def get_buy_recommendations(self, tickers: List[str], date: str, 
                               min_confidence: float = 0.5) -> List[str]:
        """
        Get list of stocks recommended for BUY.
        
        Args:
            tickers: List of stock ticker symbols
            date: Date string
            min_confidence: Minimum confidence threshold (0-1)
        
        Returns:
            List of ticker symbols with BUY recommendation
        """
        results = self.screen_stocks(tickers, date)
        
        buy_stocks = [
            ticker for ticker, info in results.items()
            if info['decision'] == 'BUY' and info['confidence'] >= min_confidence
        ]
        
        return buy_stocks
    
    def _parse_decision(self, text: str) -> str:
        """Extract BUY/SELL/HOLD from agent output."""
        text_upper = text.upper()
        
        # Look for explicit markers
        if 'FINAL TRANSACTION PROPOSAL:' in text_upper:
            match = re.search(r'FINAL TRANSACTION PROPOSAL:\s*\*\*(\w+)\*\*', text_upper)
            if match:
                return match.group(1)
        
        # Fallback: count keywords
        buy_count = text_upper.count('BUY')
        sell_count = text_upper.count('SELL')
        hold_count = text_upper.count('HOLD')
        
        if buy_count > max(sell_count, hold_count):
            return 'BUY'
        elif sell_count > max(buy_count, hold_count):
            return 'SELL'
        else:
            return 'HOLD'
    
    def _extract_confidence(self, text: str) -> float:
        """Heuristic confidence based on language strength."""
        strong_words = ['strongly', 'highly', 'confident', 'excellent', 'significant', 
                       'compelling', 'robust', 'solid', 'favorable']
        weak_words = ['maybe', 'uncertain', 'cautious', 'mixed', 'moderate',
                     'concern', 'risk', 'volatility', 'unclear']
        
        text_lower = text.lower()
        strong_count = sum(word in text_lower for word in strong_words)
        weak_count = sum(word in text_lower for word in weak_words)
        
        # Base confidence + adjustments
        confidence = 0.5 + (strong_count * 0.08) - (weak_count * 0.08)
        return max(0.0, min(1.0, confidence))


if __name__ == "__main__":
    # Quick test
    screener = AgentStockScreener()
    
    test_tickers = ['AAPL', 'MSFT']
    date = "2024-11-08"
    
    print(f"\nTesting AgentStockScreener with {test_tickers} on {date}\n")
    print("=" * 60)
    
    results = screener.screen_stocks(test_tickers, date)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    for ticker, info in results.items():
        print(f"\n{ticker}:")
        print(f"  Decision: {info['decision']}")
        print(f"  Confidence: {info['confidence']:.2f}")
        print(f"  Reasoning: {info['reasoning'][:200]}...")
