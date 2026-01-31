"""
Prediction Validation System
Generates trading signals and tracks their accuracy over time

This system:
1. Generates daily trading signals (predictions)
2. Stores predictions in a JSON file
3. Validates predictions against actual market outcomes
4. Calculates rolling accuracy metrics
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import argparse

# Import from improved strategy
from improved_strategy import (
    ImprovedBacktestConfig, 
    calculate_ema, calculate_atr,
    calculate_supertrend, calculate_stochastic_rsi,
    calculate_sma
)


@dataclass
class Prediction:
    """A single trading signal prediction."""
    timestamp: str
    symbol: str
    signal: str  # 'long', 'short', 'neutral'
    confidence: float  # 0-100
    price_at_prediction: float
    target_price: float
    stop_loss: float
    expected_direction: str  # 'up', 'down', 'sideways'
    indicators: Dict
    # Validation fields (filled later)
    actual_outcome: Optional[str] = None  # 'correct', 'incorrect', 'pending'
    actual_price: Optional[float] = None
    validated_at: Optional[str] = None
    pnl_percent: Optional[float] = None


class PredictionValidator:
    """
    Validates trading predictions against actual market outcomes.
    """
    
    def __init__(self, storage_path: str = "predictions.json"):
        self.storage_path = storage_path
        self.predictions: List[Prediction] = []
        self.load_predictions()
    
    def load_predictions(self):
        """Load existing predictions from storage."""
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.predictions = [
                    Prediction(**p) for p in data.get('predictions', [])
                ]
    
    def save_predictions(self):
        """Save predictions to storage."""
        with open(self.storage_path, 'w') as f:
            json.dump({
                'last_updated': datetime.now().isoformat(),
                'predictions': [asdict(p) for p in self.predictions]
            }, f, indent=2)
    
    def generate_prediction(self, data: List[Dict], symbol: str, 
                            config: ImprovedBacktestConfig) -> Optional[Prediction]:
        """
        Generate a trading prediction based on current market data.
        
        Args:
            data: OHLCV data (most recent at end)
            symbol: Trading symbol
            config: Strategy configuration
        
        Returns:
            Prediction object or None if no signal
        """
        min_bars = config.ema_period + 10
        if len(data) < min_bars:
            print(f"Insufficient data: need {min_bars} bars")
            return None
        
        # Extract data
        closes = [d['close'] for d in data]
        highs = [d['high'] for d in data]
        lows = [d['low'] for d in data]
        volumes = [d['volume'] for d in data]
        
        # Calculate indicators on most recent data
        ema200 = calculate_ema(closes, config.ema_period)
        st1, dir1 = calculate_supertrend(highs, lows, closes, config.st1_period, config.st1_mult)
        st2, dir2 = calculate_supertrend(highs, lows, closes, config.st2_period, config.st2_mult)
        st3, dir3 = calculate_supertrend(highs, lows, closes, config.st3_period, config.st3_mult)
        stoch_k, stoch_d = calculate_stochastic_rsi(closes, config.rsi_period, 
                                                      config.stoch_period, config.stoch_smooth)
        atr = calculate_atr(highs, lows, closes, config.atr_period)
        vol_ma = calculate_sma(volumes, config.volume_ma_period)
        
        # Get latest values
        i = len(data) - 1
        
        if any(x is None for x in [ema200[i], st1[i], st2[i], st3[i], 
                                    stoch_k[i], stoch_d[i], atr[i]]):
            print("Indicators not ready")
            return None
        
        # Calculate conditions
        current_price = closes[i]
        above_ema = current_price > ema200[i]
        below_ema = current_price < ema200[i]
        
        bullish_st = sum(1 for d in [dir1[i], dir2[i], dir3[i]] if d == 1)
        bearish_st = sum(1 for d in [dir1[i], dir2[i], dir3[i]] if d == -1)
        
        # Check for crossover (need previous values)
        k_crossed_up = (stoch_k[i-1] is not None and stoch_d[i-1] is not None and
                       stoch_k[i-1] < stoch_d[i-1] and stoch_k[i] > stoch_d[i])
        k_crossed_down = (stoch_k[i-1] is not None and stoch_d[i-1] is not None and
                         stoch_k[i-1] > stoch_d[i-1] and stoch_k[i] < stoch_d[i])
        
        oversold = stoch_k[i] < config.oversold_level
        overbought = stoch_k[i] > config.overbought_level
        
        volume_confirm = volumes[i] > vol_ma[i] if vol_ma[i] else True
        
        # Generate signal
        signal = 'neutral'
        confidence = 0
        expected_direction = 'sideways'
        
        # LONG signal
        if (above_ema and bullish_st >= 2 and k_crossed_up and oversold and volume_confirm):
            signal = 'long'
            confidence = min(bullish_st * 25 + 25, 100)  # Max 100%
            expected_direction = 'up'
        
        # SHORT signal  
        elif (below_ema and bearish_st >= 2 and k_crossed_down and overbought and volume_confirm):
            signal = 'short'
            confidence = min(bearish_st * 25 + 25, 100)
            expected_direction = 'down'
        
        # Calculate targets
        if signal == 'long':
            stop_loss = current_price - (atr[i] * config.stop_loss_atr_mult)
            target_price = current_price + (atr[i] * config.take_profit_atr_mult)
        elif signal == 'short':
            stop_loss = current_price + (atr[i] * config.stop_loss_atr_mult)
            target_price = current_price - (atr[i] * config.take_profit_atr_mult)
        else:
            stop_loss = 0
            target_price = 0
        
        prediction = Prediction(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            price_at_prediction=current_price,
            target_price=round(target_price, 2),
            stop_loss=round(stop_loss, 2),
            expected_direction=expected_direction,
            indicators={
                'ema200': round(ema200[i], 2),
                'supertrend_bullish_count': bullish_st,
                'stoch_k': round(stoch_k[i], 2),
                'stoch_d': round(stoch_d[i], 2),
                'atr': round(atr[i], 2),
                'price_vs_ema': 'above' if above_ema else 'below'
            }
        )
        
        return prediction
    
    def add_prediction(self, prediction: Prediction):
        """Add a new prediction."""
        self.predictions.append(prediction)
        self.save_predictions()
    
    def validate_predictions(self, current_data: Dict) -> List[Prediction]:
        """
        Validate pending predictions against current market data.
        
        Args:
            current_data: Current market data with 'close' price and 'date'
        
        Returns:
            List of validated predictions
        """
        validated = []
        current_price = current_data['close']
        current_date = current_data.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        for pred in self.predictions:
            if pred.actual_outcome is not None:
                continue  # Already validated
            
            if pred.signal == 'neutral':
                continue  # No position to validate
            
            # Check if prediction is old enough (at least 1 day)
            pred_time = datetime.fromisoformat(pred.timestamp)
            if datetime.now() - pred_time < timedelta(hours=6):
                continue  # Too recent
            
            # Validate based on signal type
            if pred.signal == 'long':
                if current_price >= pred.target_price:
                    pred.actual_outcome = 'correct'
                    pred.pnl_percent = (current_price - pred.price_at_prediction) / pred.price_at_prediction * 100
                elif current_price <= pred.stop_loss:
                    pred.actual_outcome = 'incorrect'
                    pred.pnl_percent = (current_price - pred.price_at_prediction) / pred.price_at_prediction * 100
                else:
                    # Check if price moved in expected direction
                    if current_price > pred.price_at_prediction * 1.005:  # 0.5% move up
                        pred.actual_outcome = 'correct'
                    elif current_price < pred.price_at_prediction * 0.995:  # 0.5% move down
                        pred.actual_outcome = 'incorrect'
                    pred.pnl_percent = (current_price - pred.price_at_prediction) / pred.price_at_prediction * 100
            
            elif pred.signal == 'short':
                if current_price <= pred.target_price:
                    pred.actual_outcome = 'correct'
                    pred.pnl_percent = (pred.price_at_prediction - current_price) / pred.price_at_prediction * 100
                elif current_price >= pred.stop_loss:
                    pred.actual_outcome = 'incorrect'
                    pred.pnl_percent = (pred.price_at_prediction - current_price) / pred.price_at_prediction * 100
                else:
                    if current_price < pred.price_at_prediction * 0.995:
                        pred.actual_outcome = 'correct'
                    elif current_price > pred.price_at_prediction * 1.005:
                        pred.actual_outcome = 'incorrect'
                    pred.pnl_percent = (pred.price_at_prediction - current_price) / pred.price_at_prediction * 100
            
            if pred.actual_outcome:
                pred.actual_price = current_price
                pred.validated_at = current_date
                validated.append(pred)
        
        self.save_predictions()
        return validated
    
    def get_performance_stats(self) -> Dict:
        """Calculate performance statistics."""
        validated = [p for p in self.predictions if p.actual_outcome and p.signal != 'neutral']
        
        if not validated:
            return {
                "total_predictions": len(self.predictions),
                "validated": 0,
                "pending": len([p for p in self.predictions if not p.actual_outcome]),
                "accuracy": 0,
                "avg_pnl_percent": 0,
                "correct": 0,
                "incorrect": 0,
                "long_predictions": 0,
                "long_accuracy": 0,
                "short_predictions": 0,
                "short_accuracy": 0
            }
        
        correct = sum(1 for p in validated if p.actual_outcome == 'correct')
        total_pnl = sum(p.pnl_percent for p in validated if p.pnl_percent)
        
        # Breakdown by signal type
        long_preds = [p for p in validated if p.signal == 'long']
        short_preds = [p for p in validated if p.signal == 'short']
        
        return {
            "total_predictions": len(self.predictions),
            "validated": len(validated),
            "pending": len([p for p in self.predictions if not p.actual_outcome and p.signal != 'neutral']),
            "correct": correct,
            "incorrect": len(validated) - correct,
            "accuracy": round(correct / len(validated) * 100, 2) if validated else 0,
            "avg_pnl_percent": round(total_pnl / len(validated), 2) if validated else 0,
            "long_predictions": len(long_preds),
            "long_accuracy": round(sum(1 for p in long_preds if p.actual_outcome == 'correct') / len(long_preds) * 100, 2) if long_preds else 0,
            "short_predictions": len(short_preds),
            "short_accuracy": round(sum(1 for p in short_preds if p.actual_outcome == 'correct') / len(short_preds) * 100, 2) if short_preds else 0
        }
    
    def generate_report(self) -> str:
        """Generate a markdown report of prediction performance."""
        stats = self.get_performance_stats()
        
        report = f"""# Prediction Validation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Predictions | {stats['total_predictions']} |
| Validated | {stats['validated']} |
| Pending | {stats['pending']} |
| Correct | {stats.get('correct', 0)} |
| Incorrect | {stats.get('incorrect', 0)} |
| **Overall Accuracy** | **{stats['accuracy']}%** |
| Average P&L | {stats['avg_pnl_percent']}% |

## Breakdown by Signal Type

| Type | Count | Accuracy |
|------|-------|----------|
| Long | {stats['long_predictions']} | {stats['long_accuracy']}% |
| Short | {stats['short_predictions']} | {stats['short_accuracy']}% |

## Research Benchmark Comparison

| Metric | Our Result | Research Target |
|--------|------------|-----------------|
| Accuracy | {stats['accuracy']}% | 42-48% |
| Avg P&L | {stats['avg_pnl_percent']}% | Positive |

"""
        
        # Add recent predictions
        recent = sorted(self.predictions, key=lambda x: x.timestamp, reverse=True)[:10]
        
        report += "\n## Recent Predictions\n\n"
        report += "| Date | Symbol | Signal | Confidence | Entry | Target | Outcome |\n"
        report += "|------|--------|--------|------------|-------|--------|--------|\n"
        
        for p in recent:
            outcome = p.actual_outcome or 'pending'
            outcome_emoji = '✓' if outcome == 'correct' else ('✗' if outcome == 'incorrect' else '⏳')
            report += f"| {p.timestamp[:10]} | {p.symbol} | {p.signal.upper()} | {p.confidence}% | {p.price_at_prediction} | {p.target_price} | {outcome_emoji} {outcome} |\n"
        
        return report


def main():
    """Main entry point for prediction validation."""
    parser = argparse.ArgumentParser(description='Prediction Validation System')
    parser.add_argument('--mode', choices=['generate', 'validate', 'report'], required=True)
    parser.add_argument('--symbol', default='NIFTY50')
    parser.add_argument('--storage', default='predictions.json')
    parser.add_argument('--output', help='Output file for report')
    
    args = parser.parse_args()
    
    validator = PredictionValidator(args.storage)
    config = ImprovedBacktestConfig()
    
    if args.mode == 'generate':
        # Generate a prediction using sample data
        from improved_strategy import generate_sample_data
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=300)).strftime('%Y-%m-%d')
        
        data = generate_sample_data(args.symbol, start_date, end_date)
        
        prediction = validator.generate_prediction(data, args.symbol, config)
        
        if prediction and prediction.signal != 'neutral':
            validator.add_prediction(prediction)
            print(f"\n{'='*60}")
            print(f"NEW PREDICTION GENERATED")
            print(f"{'='*60}")
            print(f"Symbol:     {prediction.symbol}")
            print(f"Signal:     {prediction.signal.upper()}")
            print(f"Confidence: {prediction.confidence}%")
            print(f"Entry:      {prediction.price_at_prediction}")
            print(f"Target:     {prediction.target_price}")
            print(f"Stop Loss:  {prediction.stop_loss}")
            print(f"{'='*60}")
        else:
            print("No trading signal generated (neutral market conditions)")
    
    elif args.mode == 'validate':
        # Validate pending predictions
        from improved_strategy import generate_sample_data
        
        data = generate_sample_data(args.symbol, 
                                    (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                                    datetime.now().strftime('%Y-%m-%d'))
        
        if data:
            current_data = data[-1]
            validated = validator.validate_predictions(current_data)
            
            print(f"Validated {len(validated)} predictions")
            for p in validated:
                print(f"  {p.symbol}: {p.signal} -> {p.actual_outcome} (P&L: {p.pnl_percent:.2f}%)")
    
    elif args.mode == 'report':
        report = validator.generate_report()
        print(report)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
