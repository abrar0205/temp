"""
Machine Learning Signal Filter
Uses Random Forest and LSTM for improved signal classification

Features:
- Random Forest for quick classification of signal quality
- LSTM for price direction prediction
- Feature importance analysis
- Ensemble combining both models
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
import pickle
import os


@dataclass
class MLPrediction:
    """ML model prediction result."""
    signal: str  # 'long', 'short', 'neutral'
    confidence: float  # 0-100
    rf_probability: float
    lstm_probability: float
    ensemble_score: float
    features_used: List[str]


class FeatureExtractor:
    """Extract features from price data for ML models."""
    
    @staticmethod
    def extract_features(data: List[Dict], lookback: int = 20) -> np.ndarray:
        """
        Extract features for ML model.
        
        Features:
        1. Price momentum (ROC at multiple periods)
        2. Volatility (ATR ratio)
        3. Trend strength (ADX)
        4. RSI levels
        5. MACD histogram
        6. Volume ratio
        7. Price relative to EMAs
        8. Bollinger Band position
        """
        if len(data) < lookback + 50:
            return None
        
        features_list = []
        
        closes = [d['close'] for d in data]
        highs = [d['high'] for d in data]
        lows = [d['low'] for d in data]
        volumes = [d['volume'] for d in data]
        
        for i in range(lookback + 50, len(data)):
            features = []
            
            # 1. Rate of Change (momentum) at multiple periods
            for period in [5, 10, 20]:
                roc = (closes[i] - closes[i - period]) / closes[i - period] * 100
                features.append(roc)
            
            # 2. Volatility ratio (current ATR / 20-period average ATR)
            tr_current = max(highs[i] - lows[i], 
                           abs(highs[i] - closes[i-1]), 
                           abs(lows[i] - closes[i-1]))
            tr_avg = np.mean([max(highs[j] - lows[j], 
                                 abs(highs[j] - closes[j-1]), 
                                 abs(lows[j] - closes[j-1])) 
                            for j in range(i-20, i)])
            vol_ratio = tr_current / tr_avg if tr_avg > 0 else 1
            features.append(vol_ratio)
            
            # 3. ADX (trend strength) - simplified calculation
            adx = FeatureExtractor._calculate_adx(highs[i-30:i+1], lows[i-30:i+1], closes[i-30:i+1])
            features.append(adx)
            
            # 4. RSI
            rsi = FeatureExtractor._calculate_rsi(closes[i-15:i+1], 14)
            features.append(rsi)
            
            # 5. MACD histogram
            ema12 = FeatureExtractor._ema(closes[i-30:i+1], 12)[-1]
            ema26 = FeatureExtractor._ema(closes[i-30:i+1], 26)[-1]
            macd = ema12 - ema26
            signal_line = FeatureExtractor._ema([macd], 9)[-1] if len([macd]) >= 9 else macd
            macd_hist = macd - signal_line
            features.append(macd_hist / closes[i] * 100)  # Normalize
            
            # 6. Volume ratio
            vol_ratio = volumes[i] / np.mean(volumes[i-20:i]) if np.mean(volumes[i-20:i]) > 0 else 1
            features.append(vol_ratio)
            
            # 7. Price relative to EMAs
            ema20 = FeatureExtractor._ema(closes[i-25:i+1], 20)[-1]
            ema50 = FeatureExtractor._ema(closes[i-55:i+1], 50)[-1] if i >= 55 else ema20
            features.append((closes[i] - ema20) / ema20 * 100)
            features.append((closes[i] - ema50) / ema50 * 100)
            
            # 8. Bollinger Band position
            bb_mid = np.mean(closes[i-20:i])
            bb_std = np.std(closes[i-20:i])
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
            bb_position = (closes[i] - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
            features.append(bb_position)
            
            # 9. Consecutive candle direction
            up_candles = sum(1 for j in range(i-5, i+1) if closes[j] > closes[j-1])
            features.append(up_candles / 6)
            
            # 10. High-Low range position
            period_high = max(highs[i-20:i+1])
            period_low = min(lows[i-20:i+1])
            hl_position = (closes[i] - period_low) / (period_high - period_low) if (period_high - period_low) > 0 else 0.5
            features.append(hl_position)
            
            features_list.append(features)
        
        return np.array(features_list)
    
    @staticmethod
    def _ema(data: List[float], period: int) -> List[float]:
        """Calculate EMA."""
        if len(data) < period:
            return data
        
        ema = [sum(data[:period]) / period]
        multiplier = 2 / (period + 1)
        
        for price in data[period:]:
            ema.append((price * multiplier) + (ema[-1] * (1 - multiplier)))
        
        return ema
    
    @staticmethod
    def _calculate_rsi(closes: List[float], period: int = 14) -> float:
        """Calculate RSI."""
        if len(closes) < period + 1:
            return 50
        
        changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [max(c, 0) for c in changes]
        losses = [abs(min(c, 0)) for c in changes]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _calculate_adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """Calculate ADX (simplified)."""
        if len(highs) < period + 1:
            return 25
        
        plus_dm = []
        minus_dm = []
        tr = []
        
        for i in range(1, len(highs)):
            high_diff = highs[i] - highs[i-1]
            low_diff = lows[i-1] - lows[i]
            
            plus_dm.append(max(high_diff, 0) if high_diff > low_diff else 0)
            minus_dm.append(max(low_diff, 0) if low_diff > high_diff else 0)
            tr.append(max(highs[i] - lows[i], 
                         abs(highs[i] - closes[i-1]), 
                         abs(lows[i] - closes[i-1])))
        
        if len(tr) < period:
            return 25
        
        atr = sum(tr[-period:]) / period
        plus_di = (sum(plus_dm[-period:]) / period) / atr * 100 if atr > 0 else 0
        minus_di = (sum(minus_dm[-period:]) / period) / atr * 100 if atr > 0 else 0
        
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0
        
        return dx  # Simplified - actual ADX is smoothed DX


class RandomForestFilter:
    """
    Random Forest model for signal quality classification.
    
    Labels:
    - 0: Bad signal (price moves against prediction)
    - 1: Neutral (small move or sideways)
    - 2: Good signal (price moves in predicted direction)
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
        self.feature_importance = None
        self.is_trained = False
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Dict:
        """Build a decision tree recursively."""
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < 10:
            return {'leaf': True, 'prediction': np.bincount(y.astype(int)).argmax() if len(y) > 0 else 1}
        
        # Random feature selection
        n_features = int(np.sqrt(X.shape[1]))
        feature_indices = np.random.choice(X.shape[1], n_features, replace=False)
        
        best_gain = -1
        best_split = None
        
        for feat_idx in feature_indices:
            thresholds = np.percentile(X[:, feat_idx], [25, 50, 75])
            
            for threshold in thresholds:
                left_mask = X[:, feat_idx] <= threshold
                right_mask = ~left_mask
                
                if sum(left_mask) < 5 or sum(right_mask) < 5:
                    continue
                
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feat_idx, threshold, left_mask, right_mask)
        
        if best_split is None:
            return {'leaf': True, 'prediction': np.bincount(y.astype(int)).argmax()}
        
        feat_idx, threshold, left_mask, right_mask = best_split
        
        return {
            'leaf': False,
            'feature': feat_idx,
            'threshold': threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }
    
    def _information_gain(self, parent: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
        """Calculate information gain."""
        def entropy(y):
            if len(y) == 0:
                return 0
            probs = np.bincount(y.astype(int)) / len(y)
            probs = probs[probs > 0]
            return -np.sum(probs * np.log2(probs))
        
        n = len(parent)
        n_left, n_right = len(left), len(right)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        return entropy(parent) - (n_left/n * entropy(left) + n_right/n * entropy(right))
    
    def _predict_tree(self, tree: Dict, x: np.ndarray) -> int:
        """Predict using a single tree."""
        if tree['leaf']:
            return tree['prediction']
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_tree(tree['left'], x)
        else:
            return self._predict_tree(tree['right'], x)
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the Random Forest model."""
        print(f"Training Random Forest with {self.n_estimators} trees...")
        self.trees = []
        
        for i in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(len(X), len(X), replace=True)
            X_boot, y_boot = X[indices], y[indices]
            
            tree = self._build_tree(X_boot, y_boot)
            self.trees.append(tree)
            
            if (i + 1) % 20 == 0:
                print(f"  Built {i + 1}/{self.n_estimators} trees")
        
        # Calculate feature importance (simplified)
        self.feature_importance = np.zeros(X.shape[1])
        for tree in self.trees:
            self._accumulate_importance(tree, 1.0)
        self.feature_importance /= self.n_estimators
        
        self.is_trained = True
        print("Random Forest training complete!")
    
    def _accumulate_importance(self, tree: Dict, weight: float):
        """Accumulate feature importance from tree."""
        if tree['leaf']:
            return
        
        self.feature_importance[tree['feature']] += weight
        self._accumulate_importance(tree['left'], weight * 0.5)
        self._accumulate_importance(tree['right'], weight * 0.5)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict class and probability."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        predictions = np.zeros((len(X), 3))  # 3 classes
        
        for tree in self.trees:
            for i, x in enumerate(X):
                pred = self._predict_tree(tree, x)
                predictions[i, pred] += 1
        
        predictions /= self.n_estimators
        classes = np.argmax(predictions, axis=1)
        probabilities = np.max(predictions, axis=1)
        
        return classes, probabilities


class SimpleLSTM:
    """
    Simple LSTM implementation for price direction prediction.
    Uses numpy only (no TensorFlow/PyTorch dependency).
    
    This is a simplified version - for production, use TensorFlow/PyTorch.
    """
    
    def __init__(self, input_size: int = 10, hidden_size: int = 32, output_size: int = 3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.is_trained = False
        self.weights_initialized = False
    
    def _init_weights(self, actual_input_size: int):
        """Initialize weights with actual input size."""
        self.input_size = actual_input_size
        
        # Initialize weights (Xavier initialization)
        scale = np.sqrt(2.0 / (self.input_size + self.hidden_size))
        
        # LSTM weights
        self.Wf = np.random.randn(self.hidden_size, self.input_size + self.hidden_size) * scale
        self.Wi = np.random.randn(self.hidden_size, self.input_size + self.hidden_size) * scale
        self.Wc = np.random.randn(self.hidden_size, self.input_size + self.hidden_size) * scale
        self.Wo = np.random.randn(self.hidden_size, self.input_size + self.hidden_size) * scale
        
        self.bf = np.zeros((self.hidden_size, 1))
        self.bi = np.zeros((self.hidden_size, 1))
        self.bc = np.zeros((self.hidden_size, 1))
        self.bo = np.zeros((self.hidden_size, 1))
        
        # Output layer
        self.Wy = np.random.randn(self.output_size, self.hidden_size) * np.sqrt(2.0 / self.hidden_size)
        self.by = np.zeros((self.output_size, 1))
        
        self.weights_initialized = True
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh activation."""
        return np.tanh(x)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through LSTM."""
        if not self.weights_initialized:
            self._init_weights(X.shape[1])
        
        seq_len = X.shape[0]
        
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        
        for t in range(seq_len):
            x = X[t].reshape(-1, 1)
            concat = np.vstack([h, x])
            
            f = self._sigmoid(self.Wf @ concat + self.bf)
            i = self._sigmoid(self.Wi @ concat + self.bi)
            c_tilde = self._tanh(self.Wc @ concat + self.bc)
            c = f * c + i * c_tilde
            o = self._sigmoid(self.Wo @ concat + self.bo)
            h = o * self._tanh(c)
        
        y = self.Wy @ h + self.by
        return self._softmax(y)
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, lr: float = 0.001):
        """
        Train LSTM with simple gradient descent.
        X: (n_samples, seq_len, features)
        y: (n_samples,) class labels
        """
        print(f"Training LSTM for {epochs} epochs...")
        
        # Store the last hidden state for each sample for gradient calculation
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            
            for i in range(len(X)):
                # Forward pass
                output = self.forward(X[i])
                
                # Cross-entropy loss
                target = np.zeros((self.output_size, 1))
                target[int(y[i])] = 1
                loss = -np.sum(target * np.log(output + 1e-8))
                total_loss += loss
                
                if np.argmax(output) == y[i]:
                    correct += 1
                
                # Simplified gradient update (output layer only)
                # This is a very basic update - real LSTM uses BPTT
                error = output - target  # (output_size, 1)
                
                # Get the last hidden state by running forward again
                h = np.zeros((self.hidden_size, 1))
                c = np.zeros((self.hidden_size, 1))
                
                for t in range(X[i].shape[0]):
                    x = X[i][t].reshape(-1, 1)
                    concat = np.vstack([h, x])
                    
                    f = self._sigmoid(self.Wf @ concat + self.bf)
                    inp = self._sigmoid(self.Wi @ concat + self.bi)
                    c_tilde = self._tanh(self.Wc @ concat + self.bc)
                    c = f * c + inp * c_tilde
                    o = self._sigmoid(self.Wo @ concat + self.bo)
                    h = o * self._tanh(c)
                
                # Update output weights: dL/dWy = error @ h.T
                self.Wy -= lr * (error @ h.T)
                self.by -= lr * error
            
            if (epoch + 1) % 10 == 0:
                accuracy = correct / len(X) * 100
                print(f"  Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(X):.4f}, Accuracy: {accuracy:.1f}%")
        
        self.is_trained = True
        print("LSTM training complete!")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict class and probability."""
        if not self.is_trained:
            # Return neutral predictions if not trained
            return np.ones(len(X)).astype(int), np.ones(len(X)) * 0.33
        
        predictions = []
        probabilities = []
        
        for x in X:
            output = self.forward(x)
            predictions.append(np.argmax(output))
            probabilities.append(np.max(output))
        
        return np.array(predictions), np.array(probabilities)


class MLSignalFilter:
    """
    Ensemble ML filter combining Random Forest and LSTM.
    
    Usage:
    1. Extract features from historical data
    2. Generate labels based on future price movement
    3. Train models
    4. Use for signal filtering
    """
    
    def __init__(self, model_path: str = "ml_models"):
        self.model_path = model_path
        self.rf_model = RandomForestFilter(n_estimators=100, max_depth=10)
        self.lstm_model = SimpleLSTM(input_size=12, hidden_size=32, output_size=3)
        self.feature_extractor = FeatureExtractor()
        self.is_trained = False
        
        # Feature names for interpretability
        self.feature_names = [
            'ROC_5', 'ROC_10', 'ROC_20',
            'Volatility_Ratio', 'ADX', 'RSI',
            'MACD_Hist', 'Volume_Ratio',
            'Price_vs_EMA20', 'Price_vs_EMA50',
            'BB_Position', 'Consecutive_Up_Pct',
            'HL_Range_Position'
        ]
    
    def generate_labels(self, data: List[Dict], lookahead: int = 5, 
                        threshold_pct: float = 1.0) -> np.ndarray:
        """
        Generate labels based on future price movement.
        
        Labels:
        - 0: Price drops > threshold (bad for long signals)
        - 1: Price within threshold (neutral)
        - 2: Price rises > threshold (good for long signals)
        """
        closes = [d['close'] for d in data]
        labels = []
        
        for i in range(len(closes) - lookahead):
            future_return = (closes[i + lookahead] - closes[i]) / closes[i] * 100
            
            if future_return > threshold_pct:
                labels.append(2)  # Good signal
            elif future_return < -threshold_pct:
                labels.append(0)  # Bad signal
            else:
                labels.append(1)  # Neutral
        
        return np.array(labels)
    
    def prepare_training_data(self, data: List[Dict], lookahead: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for training."""
        features = self.feature_extractor.extract_features(data)
        
        if features is None or len(features) == 0:
            raise ValueError("Insufficient data for feature extraction")
        
        # Generate labels
        labels = self.generate_labels(data, lookahead)
        
        # Align features and labels
        # Features start from index (lookback + 50) in original data
        # Labels start from index 0 but need to match features
        feature_start_idx = len(data) - len(features)
        
        # Trim to match
        min_len = min(len(features), len(labels) - feature_start_idx)
        features = features[:min_len]
        labels = labels[feature_start_idx:feature_start_idx + min_len]
        
        return features, labels
    
    def train(self, data: List[Dict], lookahead: int = 5):
        """Train both Random Forest and LSTM models."""
        print("Preparing training data...")
        X, y = self.prepare_training_data(data, lookahead)
        
        print(f"Training data: {len(X)} samples")
        print(f"Label distribution: {np.bincount(y.astype(int))}")
        
        # Train Random Forest
        self.rf_model.train(X, y)
        
        # Prepare sequence data for LSTM (use last 10 samples as sequence)
        seq_len = 10
        X_lstm = []
        y_lstm = []
        
        for i in range(seq_len, len(X)):
            X_lstm.append(X[i-seq_len:i])
            y_lstm.append(y[i])
        
        X_lstm = np.array(X_lstm)
        y_lstm = np.array(y_lstm)
        
        # Train LSTM
        self.lstm_model.train(X_lstm, y_lstm, epochs=30)
        
        self.is_trained = True
        print("ML Signal Filter training complete!")
    
    def predict(self, data: List[Dict]) -> List[MLPrediction]:
        """
        Generate ML-filtered predictions for the data.
        """
        features = self.feature_extractor.extract_features(data)
        
        if features is None or len(features) == 0:
            return []
        
        # Random Forest predictions
        rf_classes, rf_probs = self.rf_model.predict(features)
        
        # LSTM predictions (if enough data)
        seq_len = 10
        if len(features) >= seq_len:
            X_lstm = []
            for i in range(seq_len, len(features)):
                X_lstm.append(features[i-seq_len:i])
            X_lstm = np.array(X_lstm)
            
            lstm_classes, lstm_probs = self.lstm_model.predict(X_lstm)
            
            # Pad LSTM results to match RF length
            lstm_classes = np.concatenate([np.ones(seq_len) * 1, lstm_classes])
            lstm_probs = np.concatenate([np.ones(seq_len) * 0.33, lstm_probs])
        else:
            lstm_classes = np.ones(len(features)) * 1
            lstm_probs = np.ones(len(features)) * 0.33
        
        # Generate predictions
        predictions = []
        
        for i in range(len(features)):
            # Ensemble: weighted average of RF and LSTM
            rf_weight = 0.6
            lstm_weight = 0.4
            
            ensemble_score = rf_weight * rf_probs[i] + lstm_weight * lstm_probs[i]
            
            # Determine signal
            rf_signal = ['short', 'neutral', 'long'][int(rf_classes[i])]
            lstm_signal = ['short', 'neutral', 'long'][int(lstm_classes[i])]
            
            # Agreement determines final signal
            if rf_classes[i] == lstm_classes[i]:
                signal = rf_signal
                confidence = ensemble_score * 100
            elif rf_classes[i] == 1 or lstm_classes[i] == 1:
                # One is neutral, use the other
                signal = lstm_signal if rf_classes[i] == 1 else rf_signal
                confidence = ensemble_score * 80
            else:
                # Disagreement - go neutral
                signal = 'neutral'
                confidence = 30
            
            predictions.append(MLPrediction(
                signal=signal,
                confidence=min(confidence, 100),
                rf_probability=rf_probs[i],
                lstm_probability=lstm_probs[i],
                ensemble_score=ensemble_score,
                features_used=self.feature_names
            ))
        
        return predictions
    
    def save_models(self):
        """Save trained models to disk."""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
        # Save model parameters as JSON (simplified)
        model_info = {
            'rf_trained': self.rf_model.is_trained,
            'lstm_trained': self.lstm_model.is_trained,
            'feature_names': self.feature_names
        }
        
        with open(os.path.join(self.model_path, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Models saved to {self.model_path}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from Random Forest."""
        if not self.rf_model.is_trained:
            return {}
        
        importance = {}
        for i, name in enumerate(self.feature_names):
            if i < len(self.rf_model.feature_importance):
                importance[name] = float(self.rf_model.feature_importance[i])
        
        # Sort by importance
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Command line interface for ML Signal Filter."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Signal Filter for Trading')
    parser.add_argument('--mode', choices=['train', 'predict', 'demo'], default='demo',
                       help='Operation mode')
    parser.add_argument('--data', type=str, help='Path to price data JSON')
    parser.add_argument('--model-path', type=str, default='ml_models',
                       help='Path to save/load models')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        print("=" * 60)
        print("ML SIGNAL FILTER DEMO")
        print("=" * 60)
        
        # Generate synthetic data for demo
        print("\nGenerating synthetic price data...")
        np.random.seed(42)
        
        data = []
        price = 100
        
        for i in range(500):
            # Simulate trending market with noise
            trend = 0.1 if i < 250 else -0.1
            noise = np.random.randn() * 0.5
            change = trend + noise
            
            open_price = price
            close_price = price * (1 + change / 100)
            high_price = max(open_price, close_price) * (1 + abs(np.random.randn() * 0.2) / 100)
            low_price = min(open_price, close_price) * (1 - abs(np.random.randn() * 0.2) / 100)
            
            data.append({
                'date': f'2024-{(i//30)+1:02d}-{(i%30)+1:02d}',
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': int(np.random.uniform(100000, 500000))
            })
            
            price = close_price
        
        print(f"Generated {len(data)} bars of synthetic data")
        
        # Train ML filter
        ml_filter = MLSignalFilter(model_path=args.model_path)
        
        print("\n" + "=" * 60)
        print("TRAINING ML MODELS")
        print("=" * 60)
        
        ml_filter.train(data, lookahead=5)
        
        # Get predictions
        print("\n" + "=" * 60)
        print("GENERATING PREDICTIONS")
        print("=" * 60)
        
        predictions = ml_filter.predict(data[-100:])  # Last 100 bars
        
        # Show last 10 predictions
        print("\nLast 10 predictions:")
        print("-" * 60)
        
        for i, pred in enumerate(predictions[-10:]):
            print(f"Bar {len(predictions)-10+i+1}: {pred.signal.upper():8s} | "
                  f"Confidence: {pred.confidence:5.1f}% | "
                  f"RF: {pred.rf_probability:.2f} | "
                  f"LSTM: {pred.lstm_probability:.2f}")
        
        # Feature importance
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE")
        print("=" * 60)
        
        importance = ml_filter.get_feature_importance()
        for name, imp in list(importance.items())[:5]:
            print(f"  {name}: {imp:.4f}")
        
        print("\n✅ ML Signal Filter demo complete!")


if __name__ == "__main__":
    main()
