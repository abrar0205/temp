# UMCI Trading Manual

## Universal Multi-Confirmation Indicator - Complete Trading Guide

---

## Table of Contents

1. [Entry Rules](#1-entry-rules)
2. [Exit Rules](#2-exit-rules)
3. [Risk Management](#3-risk-management)
4. [Position Sizing](#4-position-sizing)
5. [Daily Trading Routine](#5-daily-trading-routine)
6. [Paper Trading Guide](#6-paper-trading-guide)
7. [Live Trading Transition](#7-live-trading-transition)
8. [Common Mistakes](#8-common-mistakes)
9. [FAQ](#9-faq)

---

## 1. Entry Rules

### 1.1 Long Entry (Buy Signal)

A valid BUY signal requires ALL of the following conditions:

| Condition | Indicator | Requirement |
|-----------|-----------|-------------|
| 1. Trend Up | Fast EMA > Slow EMA | Yes |
| 2. Trend Confirm | SuperTrend = Bullish | Yes |
| 3. Momentum | RSI between 40-60, rising | Yes |
| 4. Volatility | ATR Ratio 0.5-3.0 | Yes |
| 5. Volume | OBV > OBV EMA | Yes (if enabled) |
| 6. Fresh Signal | Not already in uptrend | Yes |

**Visual Confirmation**:
- Green triangle with "BUY" text appears below the candle
- Background turns light green
- SuperTrend line is green and below price
- EMAs are in bullish configuration (fast above slow)

### 1.2 Short Entry (Sell Signal)

A valid SELL signal requires ALL of the following conditions:

| Condition | Indicator | Requirement |
|-----------|-----------|-------------|
| 1. Trend Down | Fast EMA < Slow EMA | Yes |
| 2. Trend Confirm | SuperTrend = Bearish | Yes |
| 3. Momentum | RSI between 40-60, falling | Yes |
| 4. Volatility | ATR Ratio 0.5-3.0 | Yes |
| 5. Volume | OBV < OBV EMA | Yes (if enabled) |
| 6. Fresh Signal | Not already in downtrend | Yes |

**Visual Confirmation**:
- Red triangle with "SELL" text appears above the candle
- Background turns light red
- SuperTrend line is red and above price
- EMAs are in bearish configuration (fast below slow)

### 1.3 Entry Best Practices

1. **Wait for candle close**: Enter only after the signal candle closes
2. **Check the table**: Verify all conditions show green in the info table
3. **Check broader timeframe**: Ensure the higher timeframe agrees (optional but recommended)
4. **Avoid entries near**:
   - Market open (first 15 minutes)
   - Market close (last 15 minutes)
   - Major news events
   - Expiry day (last 2 hours)

---

## 2. Exit Rules

### 2.1 Stop Loss Exit

**Automatic Stop Loss** is calculated as:
```
Long Stop Loss = Entry Price - (ATR × Stop Loss Multiplier)
Short Stop Loss = Entry Price + (ATR × Stop Loss Multiplier)

Default: 2.0 × ATR
```

**When hit**: Exit immediately at stop loss price

### 2.2 Take Profit Exit

**Automatic Take Profit** is calculated as:
```
Long Take Profit = Entry Price + (ATR × Take Profit Multiplier)
Short Take Profit = Entry Price - (ATR × Take Profit Multiplier)

Default: 3.0 × ATR (Risk:Reward = 1:1.5)
```

**When hit**: Exit at take profit price

### 2.3 Signal Exit

Exit when opposite conditions appear:

**Exit Long (Orange X)**:
- Strong downtrend develops (EMA cross down + SuperTrend flip)
- OR RSI becomes overbought (> 60)

**Exit Short (Orange X)**:
- Strong uptrend develops (EMA cross up + SuperTrend flip)
- OR RSI becomes oversold (< 40)

### 2.4 Time-Based Exit (Optional)

For intraday traders:
- Exit all positions 15 minutes before market close
- No overnight positions (unless swing trading)

### 2.5 Exit Priority

1. **Stop Loss** (highest priority - protect capital)
2. **Take Profit** (lock in gains)
3. **Signal Exit** (follow the system)
4. **Time Exit** (intraday discipline)

---

## 3. Risk Management

### 3.1 The 2% Rule

**Never risk more than 2% of your capital on a single trade**

Example:
```
Capital: ₹1,00,000
Max Risk per Trade: 2% = ₹2,000

If Stop Loss = ₹100 from entry
Max Position Size = ₹2,000 / ₹100 = 20 shares
```

### 3.2 Daily Loss Limit

**Stop trading for the day if you lose 3% of capital**

Example:
```
Capital: ₹1,00,000
Daily Loss Limit: 3% = ₹3,000

After reaching ₹3,000 loss → Stop trading
```

### 3.3 Weekly Loss Limit

**Reduce position size by 50% if weekly loss exceeds 5%**

Example:
```
Capital: ₹1,00,000
Weekly Loss Limit: 5% = ₹5,000

If weekly loss > ₹5,000:
- Reduce position size from 10% to 5%
- Review what went wrong
- Consider paper trading for rest of week
```

### 3.4 Consecutive Loss Management

| Consecutive Losses | Action |
|-------------------|--------|
| 3 losses | Review trades, no action needed |
| 5 losses | Reduce position size by 50% |
| 7 losses | Stop trading, paper trade only |
| 10 losses | Full review, consider parameter adjustment |

### 3.5 Never Do This

- Move stop loss further away after entry
- Average down on a losing position
- Remove stop loss entirely
- Risk more to "recover" losses
- Trade during high-impact news
- Trade on margin beyond your comfort level

---

## 4. Position Sizing

### 4.1 Fixed Percentage Method

**Default: 10% of capital per trade**

Example:
```
Capital: ₹1,00,000
Position Size: 10% = ₹10,000 per trade
```

### 4.2 ATR-Based Position Sizing (Advanced)

Calculate position size based on volatility:

```
Position Size = (Capital × Risk%) / (ATR × Stop Loss Multiplier)

Example:
Capital: ₹1,00,000
Risk per trade: 2%
ATR: ₹50
Stop Loss Multiplier: 2.0

Position Size = (1,00,000 × 0.02) / (50 × 2.0)
             = 2,000 / 100
             = 20 shares
```

### 4.3 Position Sizing Table

| Capital | 2% Risk | Max Position (10% capital) |
|---------|---------|---------------------------|
| ₹50,000 | ₹1,000 | ₹5,000 |
| ₹1,00,000 | ₹2,000 | ₹10,000 |
| ₹2,00,000 | ₹4,000 | ₹20,000 |
| ₹5,00,000 | ₹10,000 | ₹50,000 |
| ₹10,00,000 | ₹20,000 | ₹1,00,000 |

---

## 5. Daily Trading Routine

### 5.1 Pre-Market (8:30 AM - 9:15 AM)

1. **Check global cues** (5 mins)
   - US markets overnight performance
   - SGX NIFTY futures
   - Asian markets opening

2. **Review watchlist** (10 mins)
   - NIFTY 50 levels
   - Bank NIFTY levels
   - Key stocks on radar

3. **Check economic calendar** (5 mins)
   - Any major news today?
   - RBI announcements?
   - Global events?

4. **Mark key levels** (10 mins)
   - Previous day high/low
   - Support/resistance
   - Round numbers (e.g., 22000, 22500)

5. **Set alerts** (5 mins)
   - Verify TradingView alerts are active
   - Check notification settings

### 5.2 Market Hours (9:15 AM - 3:30 PM)

**9:15 - 9:30 AM**: Observe opening, no trades
**9:30 AM - 3:15 PM**: Active trading zone
**3:15 - 3:30 PM**: Close positions, no new trades

**During trading**:
- Monitor only your watchlist (avoid distraction)
- Follow signals strictly (no second-guessing)
- Log each trade in your journal
- Take breaks (5 mins every 2 hours)

### 5.3 Post-Market (3:30 PM - 4:30 PM)

1. **Close all intraday positions** (by 3:25 PM)

2. **Review trades** (15 mins)
   - What went right?
   - What went wrong?
   - Any patterns noticed?

3. **Update trading journal** (15 mins)
   - Entry/exit prices
   - P&L
   - Emotional state
   - Lessons learned

4. **Prepare for tomorrow** (15 mins)
   - Any earnings/news tomorrow?
   - Update watchlist
   - Set overnight alerts

---

## 6. Paper Trading Guide

### 6.1 Minimum Paper Trading Period

**30 days minimum before live trading**

### 6.2 Paper Trading Checklist

Week 1-2: Learn the signals
- [ ] Identify 20+ buy signals correctly
- [ ] Identify 20+ sell signals correctly
- [ ] Understand when signals fail

Week 2-3: Practice execution
- [ ] Paper trade at least 30 trades
- [ ] Track every trade in journal
- [ ] Calculate win rate

Week 3-4: Validate performance
- [ ] Achieve 55%+ win rate
- [ ] Achieve 1.5+ profit factor
- [ ] Maximum 5 consecutive losses

### 6.3 Paper Trading Rules

1. **Treat it like real money**
2. **Use realistic capital** (what you'll trade with)
3. **Include commissions** (0.02% each way)
4. **Include slippage** (0.01% each way)
5. **Follow all rules** (no cheating)

### 6.4 Paper Trading Journal Template

```
Date: ___________
Symbol: ___________
Timeframe: ___________

Entry:
- Time: ___________
- Price: ___________
- Direction: Long / Short
- Position Size: ___________
- Stop Loss: ___________
- Take Profit: ___________

Exit:
- Time: ___________
- Price: ___________
- Reason: SL / TP / Signal / Time

Result:
- P&L (₹): ___________
- P&L (%): ___________
- Notes: ___________
```

---

## 7. Live Trading Transition

### 7.1 Go-Live Checklist

Before trading real money:

- [ ] 30 days paper trading completed
- [ ] Win rate > 55% achieved
- [ ] Profit factor > 1.5 achieved
- [ ] Max drawdown < 15%
- [ ] Comfortable with the system
- [ ] Risk capital allocated (money you can lose)
- [ ] Broker account ready
- [ ] Alerts configured
- [ ] Emergency contacts ready

### 7.2 Gradual Transition

**Week 1**: Trade with 25% of planned position size
**Week 2**: Trade with 50% of planned position size
**Week 3**: Trade with 75% of planned position size
**Week 4**: Trade with 100% of planned position size

### 7.3 First Month Rules

1. Maximum 3 trades per day
2. Maximum 2% risk per trade (not 10%)
3. Stop trading after 2 consecutive losses
4. Daily review mandatory
5. Weekly mentor/peer review (if possible)

---

## 8. Common Mistakes

### 8.1 Entry Mistakes

| Mistake | Solution |
|---------|----------|
| Entering before candle close | Wait for full candle |
| Ignoring volume filter | Keep it enabled |
| Trading in sideways market | Wait for trend to develop |
| FOMO on missed signals | Next signal will come |
| Entering on partial signals | All 5 conditions must be met |

### 8.2 Exit Mistakes

| Mistake | Solution |
|---------|----------|
| Moving stop loss | Never move against you |
| Removing stop loss | Never remove it |
| Taking profit too early | Trust the system |
| Holding losers too long | Exit at stop loss |
| Ignoring exit signals | Follow the system |

### 8.3 Emotional Mistakes

| Mistake | Solution |
|---------|----------|
| Revenge trading | Daily loss limit stops this |
| Overconfidence after wins | Stick to position sizing |
| Fear after losses | Trust the system stats |
| Boredom trading | Only trade valid signals |
| Greed (removing take profit) | Trust the risk:reward |

---

## 9. FAQ

### Q: What is the best timeframe?
**A**: 15-minute for intraday trading on NIFTY/Bank NIFTY. 1-hour for swing trading.

### Q: How many trades per day can I expect?
**A**: 2-5 trades on 15-minute NIFTY 50. Varies by market conditions.

### Q: What is the expected win rate?
**A**: 55-60% with default settings. Varies by asset and market conditions.

### Q: Can I use this for options trading?
**A**: The indicator generates direction signals. For options, combine with proper options strategy (buying ATM/ITM, appropriate expiry).

### Q: Should I trade Bank NIFTY or NIFTY 50?
**A**: NIFTY 50 is more stable for beginners. Bank NIFTY is more volatile (higher profit potential but higher risk).

### Q: What if I miss a signal?
**A**: Do NOT chase. Wait for the next signal. There will always be another opportunity.

### Q: Can I modify the indicator settings?
**A**: Yes, but only after extensive paper trading with new settings. Default settings are optimized.

### Q: What broker should I use?
**A**: Any reputable broker with low commissions (Zerodha, Angel One, Upstox, etc.). Execution speed matters.

### Q: Is this indicator guaranteed to make money?
**A**: NO. No indicator can guarantee profits. This is a tool to help identify high-probability setups. Risk management and discipline are equally important.

### Q: How do I handle news events?
**A**: Avoid trading 30 minutes before and after major news (RBI policy, US Fed, earnings, etc.).

---

## Summary: The 10 Commandments

1. **Wait for all conditions** - Never enter on partial signals
2. **Use stop loss always** - Capital protection is #1 priority
3. **Follow position sizing** - Never risk more than 2% per trade
4. **Paper trade first** - 30 days minimum before live
5. **Keep a journal** - Track every trade
6. **Follow the system** - No second-guessing
7. **Accept losses** - They are part of trading
8. **Stay disciplined** - Emotions are your enemy
9. **Continuous learning** - Review and improve weekly
10. **Be patient** - Wealth is built over years, not days

---

*Manual Version: 1.0*  
*Last Updated: January 2026*
