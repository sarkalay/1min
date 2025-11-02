import os
import requests
import json
import time
import re
import numpy as np
from binance.client import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MultiPairScalpingTrader:
    def __init__(self):
        # Load config from .env file
        self.binance_api_key = os.getenv('BINANCE_API_KEY')
        self.binance_secret = os.getenv('BINANCE_SECRET_KEY')
        self.deepseek_key = os.getenv('DEEPSEEK_API_KEY')
        
        # SCALPING parameters
        self.trade_size_usd = 50  # Base size, will be adjusted dynamically
        self.leverage = 10
        self.risk_percentage = 1.0
        self.scalp_take_profit = 0.008  # 0.8% for scalping
        self.scalp_stop_loss = 0.005    # 0.5% for scalping
        
        # Multi-pair parameters
        self.max_concurrent_trades = 3
        self.available_pairs = []
        self.active_trades = {}  # Dictionary to track multiple trades
        self.blacklisted_pairs = ["BTCUSDT"]  # BTC ကိုထည့်မထားဘူး
        
        # Precision settings for different pairs
        self.quantity_precision = {}
        self.price_precision = {}
        
        # Auto pair selection parameters
        self.pair_rotation_hours = 6
        self.last_rotation_time = 0
        
        # Initialize Binance client
        self.binance = Client(self.binance_api_key, self.binance_secret)
        
        print("🤖 MULTI-PAIR SCALPING BOT ACTIVATED!")
        print(f"💵 Base Trade Size: ${self.trade_size_usd} (will adjust dynamically)")
        print(f"📈 Max Concurrent Trades: {self.max_concurrent_trades}")
        print(f"🎯 Take Profit: {self.scalp_take_profit*100}%")
        print(f"🛡️ Stop Loss: {self.scalp_stop_loss*100}%")
        print(f"🚫 Blacklisted: {self.blacklisted_pairs}")
        
        self.validate_config()
        self.setup_futures()
        self.load_symbol_precision()
    
    def validate_config(self):
        """Check API keys"""
        if not all([self.binance_api_key, self.binance_secret, self.deepseek_key]):
            print("❌ Missing API keys in .env file!")
            return False
        
        # Test Binance connection
        try:
            self.binance.futures_exchange_info()
            print("✅ Binance connection successful!")
        except Exception as e:
            print(f"❌ Binance connection failed: {e}")
            return False
            
        print("✅ Configuration loaded successfully!")
        return True
    
    def load_symbol_precision(self):
        """Load quantity and price precision for all trading pairs"""
        try:
            exchange_info = self.binance.futures_exchange_info()
            for symbol in exchange_info['symbols']:
                pair = symbol['symbol']
                
                # Get quantity precision from LOT_SIZE filter
                for f in symbol['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        step_size = f['stepSize']
                        # Calculate precision from step size
                        if '1' in step_size:
                            qty_precision = 0
                        else:
                            qty_precision = len(step_size.split('.')[1].rstrip('0'))
                        self.quantity_precision[pair] = qty_precision
                    
                    # Get price precision from PRICE_FILTER
                    elif f['filterType'] == 'PRICE_FILTER':
                        tick_size = f['tickSize']
                        if '1' in tick_size:
                            price_precision = 0
                        else:
                            price_precision = len(tick_size.split('.')[1].rstrip('0'))
                        self.price_precision[pair] = price_precision
            
            print("✅ Symbol precision loaded for all pairs")
        except Exception as e:
            print(f"❌ Error loading symbol precision: {e}")
    
    def get_dynamic_trade_size(self, pair, price):
        """Smart trade sizing based on pair price"""
        # High price pairs need larger trade size to meet Binance minimums
        if price > 3000:  # ETH
            return 200
        elif price > 500:   # BNB
            return 100
        elif price > 100:   # SOL, AVAX, LINK
            return 75
        else:               # ADA, XRP, DOGE, MATIC, etc.
            return 50

    def get_quantity(self, pair, price):
        """FIXED quantity calculation for high-priced pairs"""
        try:
            # Get dynamic trade size
            trade_size = self.get_dynamic_trade_size(pair, price)
            print(f"🎯 {pair} target size: ${trade_size}")
            
            # Calculate base quantity
            quantity = trade_size / price
            precision = self.quantity_precision.get(pair, 3)
            
            # Round to correct precision
            quantity = round(quantity, precision)
            if precision == 0:
                quantity = int(quantity)
            
            # SPECIAL HANDLING FOR HIGH-PRICED PAIRS
            if pair == "ETHUSDT" and quantity == 0:
                # Force minimum quantity for ETH
                quantity = 0.01
                trade_value = quantity * price
                print(f"🚀 ETHUSDT forced quantity: {quantity} = ${trade_value:.2f}")
                return quantity
            
            # BINANCE MINIMUM ENFORCEMENT
            min_order_value = 20
            trade_value = quantity * price
            
            print(f"🔢 Initial: {quantity} {pair} = ${trade_value:.2f}")
            
            # If below minimum, calculate proper quantity
            if trade_value < min_order_value:
                required_quantity = min_order_value / price
                quantity = round(required_quantity, precision)
                if precision == 0:
                    quantity = int(quantity)
                
                trade_value = quantity * price
                print(f"🚀 Adjusted for Binance minimum: {quantity} = ${trade_value:.2f}")
            
            # Final validation
            if quantity <= 0:
                # Emergency calculation - use higher minimum
                emergency_size = 50  # Increased from 25
                quantity = round(emergency_size / price, precision)
                if precision == 0:
                    quantity = int(quantity)
                trade_value = quantity * price
                print(f"🆘 Emergency quantity: {quantity} = ${trade_value:.2f}")
            
            print(f"💰 FINAL: {quantity} {pair} = ${trade_value:.2f}")
            
            # Final safety check
            if trade_value < 15:
                print(f"❌ CRITICAL: Still below safe minimum: ${trade_value:.2f}")
                return None
            
            return quantity
            
        except Exception as e:
            print(f"❌ Quantity calculation failed: {e}")
            return None
    
    def format_price(self, pair, price):
        """Format price according to symbol precision"""
        precision = self.price_precision.get(pair, 4)
        return round(price, precision)
    
    def setup_futures(self):
        """Setup futures trading for initial pairs"""
        try:
            # Initial pairs without BTC
            initial_pairs = ["ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT"]
            for pair in initial_pairs:
                try:
                    self.binance.futures_change_leverage(
                        symbol=pair,
                        leverage=self.leverage
                    )
                    print(f"✅ Leverage set for {pair}")
                except Exception as e:
                    print(f"⚠️ Leverage setup failed for {pair}: {e}")
            print("✅ Futures setup completed!")
        except Exception as e:
            print(f"❌ Futures setup failed: {e}")
    
    def get_ai_recommended_pairs(self):
        """AI ကနေ BTC မပါတဲ့ scalping pairs တွေရွေးခိုင်းခြင်း"""
        print("🤖 AI က BTC မပါတဲ့ scalping pairs တွေရွေးနေပါတယ်...")
        
        prompt = """
        BINANCE FUTURES SCALPING PAIR RECOMMENDATIONS (EXCLUDE BTCUSDT):
        
        SCALPING CRITERIA:
        - High liquidity but NOT BTCUSDT
        - Good volatility (2-10% daily moves)
        - Tight spreads
        - Popular altcoin pairs
        - USDT pairs only
        - Suitable for 0.5-1% quick scalps
        - Exclude BTC completely
        
        Recommend 6-10 best altcoin pairs for scalping from Binance futures.
        Focus on ETH, BNB, SOL, ADA, XRP, DOT, MATIC, AVAX, LINK, etc.
        
        RESPONSE (JSON only):
        {
            "recommended_pairs": ["ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", ...],
            "reason": "These altcoin pairs have high liquidity and volatility suitable for scalping, excluding BTC",
            "expected_volatility": "high/medium",
            "market_sentiment": "bullish/bearish/neutral"
        }
        """
        
        try:
            headers = {
                "Authorization": f"Bearer {self.deepseek_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 500
            }
            
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    recommendation = json.loads(json_match.group())
                    pairs = recommendation.get("recommended_pairs", [])
                    
                    # Remove BTC if AI accidentally includes it
                    pairs = [p for p in pairs if p != "BTCUSDT"]
                    
                    print(f"✅ AI Recommended Pairs (No BTC): {pairs}")
                    print(f"📝 Reason: {recommendation.get('reason', '')}")
                    
                    # Validate if pairs exist in Binance
                    valid_pairs = self.validate_ai_pairs(pairs)
                    return valid_pairs
            
        except Exception as e:
            print(f"❌ AI pair selection error: {e}")
        
        # Fallback to default pairs without BTC
        fallback_pairs = ["ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT", "MATICUSDT"]
        print(f"🔄 Using fallback pairs (No BTC): {fallback_pairs}")
        return fallback_pairs
    
    def validate_ai_pairs(self, ai_pairs):
        """AI ရွေးတဲ့ pairs တွေ Binance မှာရှိမရှိစစ်ဆေးခြင်း"""
        valid_pairs = []
        
        try:
            # Get all available futures pairs from Binance
            exchange_info = self.binance.futures_exchange_info()
            all_symbols = [symbol['symbol'] for symbol in exchange_info['symbols']]
            
            for pair in ai_pairs:
                if pair in all_symbols and pair not in self.blacklisted_pairs:
                    # Check if pair is trading and has required leverage
                    for symbol in exchange_info['symbols']:
                        if symbol['symbol'] == pair and symbol['status'] == 'TRADING':
                            valid_pairs.append(pair)
                            print(f"✅ {pair} is available for trading")
                            break
                    else:
                        print(f"⚠️ {pair} exists but not trading")
                else:
                    print(f"❌ {pair} not available or blacklisted")
        
        except Exception as e:
            print(f"❌ Pair validation error: {e}")
            # Fallback to first 8 AI pairs assuming they're valid
            return ai_pairs[:8]
        
        print(f"🎯 Final Validated Pairs: {valid_pairs}")
        return valid_pairs[:10]  # Maximum 10 pairs for selection pool
    
    def auto_rotate_pairs(self):
        """Auto rotate pairs based on time interval"""
        current_time = time.time()
        
        # Rotate every 6 hours or if no pairs available
        if (current_time - self.last_rotation_time > self.pair_rotation_hours * 3600 or 
            not self.available_pairs):
            
            print(f"🕒 Time for pair rotation...")
            self.available_pairs = self.get_ai_recommended_pairs()
            self.last_rotation_time = current_time
    
    def get_detailed_market_data(self):
        """Get market data for all active pairs"""
        market_data = {}
        
        if not self.available_pairs:
            print("⚠️ No pairs available, getting new pairs...")
            self.available_pairs = self.get_ai_recommended_pairs()
        
        for pair in self.available_pairs:
            try:
                # Skip if this pair already has active trade
                if pair in self.active_trades:
                    continue
                    
                # Get current price
                ticker = self.binance.futures_symbol_ticker(symbol=pair)
                price = float(ticker['price'])
                
                # Get klines for analysis
                klines = self.binance.futures_klines(
                    symbol=pair,
                    interval=Client.KLINE_INTERVAL_15MINUTE,
                    limit=20
                )
                
                if len(klines) > 0:
                    closes = [float(k[4]) for k in klines]
                    volumes = [float(k[5]) for k in klines]
                    highs = [float(k[2]) for k in klines]
                    lows = [float(k[3]) for k in klines]
                    
                    # Calculate metrics
                    current_volume = volumes[-1] if volumes else 0
                    avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else current_volume
                    
                    # Price change calculations
                    price_change_1h = ((closes[-1] - closes[-4]) / closes[-4]) * 100 if len(closes) >= 4 else 0
                    price_change_4h = ((closes[-1] - closes[-16]) / closes[-16]) * 100 if len(closes) >= 16 else 0
                    
                    # Volatility (ATR-like calculation)
                    true_ranges = []
                    for i in range(1, min(14, len(klines))):
                        high = highs[i]
                        low = lows[i]
                        prev_close = closes[i-1]
                        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                        true_ranges.append(tr)
                    
                    atr = np.mean(true_ranges) if true_ranges else 0
                    volatility = (atr / price) * 100 if price > 0 else 0
                    
                    market_data[pair] = {
                        'price': price,
                        'change_1h': price_change_1h,
                        'change_4h': price_change_4h,
                        'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1,
                        'volatility': volatility,
                        'high_1h': max(highs[-4:]) if len(highs) >= 4 else price,
                        'low_1h': min(lows[-4:]) if len(lows) >= 4 else price
                    }
                
            except Exception as e:
                print(f"❌ Market data error for {pair}: {e}")
                continue
                
        return market_data
    
    def get_scalping_decision(self, market_data):
        """BALANCED AI VERSION - REALISTIC DECISIONS"""
        pair = list(market_data.keys())[0]
        data = market_data[pair]
        price = data['price']
        
        # Check market conditions first
        volume_ratio = data.get('volume_ratio', 1)
        volatility = data.get('volatility', 0)
        change_1h = data.get('change_1h', 0)
        
        # Auto-skip poor conditions
        if volume_ratio < 0.8 or volatility < 0.2:
            return {
                "action": "SKIP",
                "confidence": 30,
                "reason": f"Poor conditions: volume {volume_ratio:.2f}x, volatility {volatility:.2f}%"
            }
        
        prompt = f"""
        REALISTIC SCALPING ANALYSIS FOR {pair}:

        CURRENT MARKET DATA:
        - Price: ${price}
        - 1H Change: {change_1h:.2f}%
        - Volume Ratio: {volume_ratio:.2f}x
        - Volatility: {volatility:.2f}%
        - 1H Range: ${data.get('low_1h', price):.2f} - ${data.get('high_1h', price):.2f}

        BE HONEST AND SELECTIVE:
        - Only trade if clear setup exists
        - Skip if conditions are mediocre
        - Consider both LONG and SHORT opportunities equally

        RESPONSE (JSON only):
        {{
            "action": "TRADE/SKIP",
            "pair": "{pair}",
            "direction": "LONG/SHORT",
            "entry_price": {price},
            "stop_loss": number,
            "take_profit": number,
            "confidence": 0-100,
            "reason": "Brief explanation of the setup",
            "urgency": "high/medium/low"
        }}
        """
        
        try:
            headers = {
                "Authorization": f"Bearer {self.deepseek_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
                "max_tokens": 400
            }
            
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    decision = json.loads(json_match.group())
                    print(f"🤖 {pair}: {decision['action']} ({decision['confidence']}% confidence)")
                    if decision['action'] == 'TRADE':
                        print(f"   📈 Direction: {decision['direction']}")
                        print(f"   🎯 Reason: {decision['reason']}")
                    else:
                        print(f"   🚫 Reason: {decision['reason']}")
                    return decision
    
        except Exception as e:
            print(f"❌ AI API Error for {pair}: {e}")
        
        # Fallback based on market conditions
        if volume_ratio > 1.2 and volatility > 0.3:
            import random
            direction = "LONG" if random.random() < 0.5 else "SHORT"
            return {
                "action": "TRADE",
                "pair": pair,
                "direction": direction,
                "entry_price": price,
                "stop_loss": price * (0.995 if direction == "LONG" else 1.005),
                "take_profit": price * (1.008 if direction == "LONG" else 0.992),
                "confidence": 65,
                "reason": f"Fallback: Good volume & volatility",
                "urgency": "medium"
            }
        else:
            return {
                "action": "SKIP", 
                "confidence": 40,
                "reason": f"Fallback: Mediocre conditions"
            }

    def execute_scalping_trade(self, decision):
        """STABLE VERSION - PROPER ORDER MANAGEMENT"""
        try:
            pair = decision["pair"]
            direction = decision["direction"]
            
            print(f"🎯 EXECUTING TRADE: {pair} {direction}")
            
            # Check if we can open new trade
            if len(self.active_trades) >= self.max_concurrent_trades:
                print(f"⚠️ Maximum trades reached ({self.max_concurrent_trades}), skipping {pair}")
                return False
            
            # Check if this pair already has active trade
            if pair in self.active_trades:
                print(f"⚠️ Already have active trade for {pair}, skipping")
                return False
            
            # Get current price
            ticker = self.binance.futures_symbol_ticker(symbol=pair)
            current_price = float(ticker['price'])
            print(f"💰 Current {pair} price: ${current_price}")
            
            # Calculate quantity
            quantity = self.get_quantity(pair, current_price)
            if quantity is None:
                print(f"❌ Quantity calculation failed for {pair}")
                return False
            
            print(f"⚡ QUANTITY: {quantity} {pair}")
            
            # Use current price as entry
            entry_price = current_price
            
            # MARKET ENTRY
            try:
                if direction == "LONG":
                    order = self.binance.futures_create_order(
                        symbol=pair,
                        side='BUY',
                        type='MARKET',
                        quantity=quantity
                    )
                    print(f"✅ LONG ENTRY: {quantity} {pair}")
                else:  # SHORT
                    order = self.binance.futures_create_order(
                        symbol=pair,
                        side='SELL', 
                        type='MARKET',
                        quantity=quantity
                    )
                    print(f"✅ SHORT ENTRY: {quantity} {pair}")
                    
            except Exception as order_error:
                print(f"❌ Entry order failed: {order_error}")
                return False
            
            # Calculate TP/SL
            if direction == "LONG":
                stop_loss = entry_price * (1 - self.scalp_stop_loss)
                take_profit = entry_price * (1 + self.scalp_take_profit)
            else:  # SHORT
                stop_loss = entry_price * (1 + self.scalp_stop_loss)
                take_profit = entry_price * (1 - self.scalp_take_profit)
            
            # Format prices
            stop_loss = self.format_price(pair, stop_loss)
            take_profit = self.format_price(pair, take_profit)
            
            print(f"🎯 {direction}: Entry=${entry_price}, TP=${take_profit}, SL=${stop_loss}")
            
            # Place TP/SL orders with proper error handling
            try:
                if direction == "LONG":
                    # STOP LOSS
                    self.binance.futures_create_order(
                        symbol=pair,
                        side='SELL',
                        type='STOP_MARKET',
                        quantity=quantity,
                        stopPrice=stop_loss,
                        timeInForce='GTC',
                        reduceOnly=True
                    )
                    # TAKE PROFIT  
                    self.binance.futures_create_order(
                        symbol=pair,
                        side='SELL',
                        type='TAKE_PROFIT_MARKET',
                        quantity=quantity,
                        stopPrice=take_profit,
                        timeInForce='GTC',
                        reduceOnly=True
                    )
                else:  # SHORT
                    # STOP LOSS
                    self.binance.futures_create_order(
                        symbol=pair,
                        side='BUY',
                        type='STOP_MARKET',
                        quantity=quantity,
                        stopPrice=stop_loss,
                        timeInForce='GTC',
                        reduceOnly=True
                    )
                    # TAKE PROFIT
                    self.binance.futures_create_order(
                        symbol=pair,
                        side='BUY',
                        type='TAKE_PROFIT_MARKET',
                        quantity=quantity,
                        stopPrice=take_profit,
                        timeInForce='GTC',
                        reduceOnly=True
                    )
                    
                print(f"✅ TP/SL ORDERS PLACED")
                
            except Exception as sl_tp_error:
                print(f"❌ TP/SL order failed: {sl_tp_error}")
                return False
            
            # Store trade info
            self.active_trades[pair] = {
                "pair": pair,
                "direction": direction,
                "entry_price": entry_price,
                "quantity": quantity,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "entry_time": time.time(),
                "confidence": decision["confidence"]
            }
            
            print(f"🚀 TRADE ACTIVATED: {pair} {direction}")
            print(f"📊 Active Trades: {list(self.active_trades.keys())}")
            
            # Verify position was created
            time.sleep(2)
            try:
                positions = self.binance.futures_position_information(symbol=pair)
                for pos in positions:
                    if pos['symbol'] == pair:
                        position_amt = float(pos['positionAmt'])
                        if position_amt != 0:
                            print(f"✅ POSITION CONFIRMED: {position_amt} contracts")
                        else:
                            print(f"❌ POSITION NOT FOUND")
                        break
            except Exception as e:
                print(f"⚠️ Position verification failed: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Trade execution failed: {e}")
            return False

    def check_scalping_trades(self):
        """PROPER POSITION CHECKING"""
        if not self.active_trades:
            return
        
        completed_trades = []
        
        for pair, trade_info in self.active_trades.items():
            try:
                # Get position information
                positions = self.binance.futures_position_information(symbol=pair)
                
                position_found = False
                for pos in positions:
                    if pos['symbol'] == pair:
                        position_amt = float(pos['positionAmt'])
                        unrealized_pnl = float(pos['unRealizedProfit'])
                        
                        if position_amt != 0:
                            position_found = True
                            print(f"📊 {pair} active: {position_amt} contracts, PnL: ${unrealized_pnl:.2f}")
                            break
                
                if not position_found:
                    # Trade completed
                    exit_time = time.time()
                    trade_duration = (exit_time - trade_info["entry_time"]) / 60
                    
                    print(f"💰 TRADE COMPLETED: {pair}!")
                    print(f"   Direction: {trade_info['direction']}")
                    print(f"   Duration: {trade_duration:.1f} minutes")
                    print(f"   Confidence: {trade_info['confidence']}%")
                    
                    completed_trades.append(pair)
                    
            except Exception as e:
                print(f"❌ Trade check error for {pair}: {e}")
        
        # Remove completed trades
        for pair in completed_trades:
            del self.active_trades[pair]
        
        if completed_trades:
            print(f"📊 Remaining Active Trades: {list(self.active_trades.keys())}")

    def run_scalping_cycle(self):
        """STABLE SCALPING CYCLE"""
        try:
            # Auto rotate pairs if needed
            self.auto_rotate_pairs()
            
            # Get market data
            market_data = self.get_detailed_market_data()
            
            if not market_data:
                print("⚠️ No market data available")
                return
            
            # Display status
            print(f"\n📊 CURRENT STATUS:")
            print(f"   Available Pairs: {len(self.available_pairs)}")
            print(f"   Active Trades: {len(self.active_trades)}/{self.max_concurrent_trades}")
            if self.active_trades:
                print(f"   Trading: {list(self.active_trades.keys())}")
            
            # Check if we already have maximum trades
            if len(self.active_trades) >= self.max_concurrent_trades:
                print(f"⏸️  Maximum trades reached, skipping analysis")
                self.check_scalping_trades()
                return
            
            # Get AI decisions
            trade_opportunities = []
            
            for pair in self.available_pairs:
                if len(self.active_trades) >= self.max_concurrent_trades:
                    break
                    
                if pair in self.active_trades:
                    continue
                    
                if pair in market_data:
                    pair_data = {pair: market_data[pair]}
                    decision = self.get_scalping_decision(pair_data)
                    
                    # 60% confidence threshold
                    if decision["action"] == "TRADE" and decision["confidence"] >= 60:
                        trade_opportunities.append((decision, decision["confidence"]))
                        print(f"✅ QUALIFIED: {pair} - {decision['confidence']}% confidence")
            
            # Sort by confidence and execute
            trade_opportunities.sort(key=lambda x: x[1], reverse=True)
            
            # Take only available slots
            available_slots = self.max_concurrent_trades - len(self.active_trades)
            trade_opportunities = trade_opportunities[:available_slots]
            
            print(f"🎯 Trade Opportunities: {len(trade_opportunities)}")
            
            executed_count = 0
            for decision, confidence in trade_opportunities:
                if len(self.active_trades) >= self.max_concurrent_trades:
                    break
                    
                print(f"🚀 EXECUTING: {decision['pair']} {decision['direction']}")
                success = self.execute_scalping_trade(decision)
                if success:
                    executed_count += 1
                    time.sleep(3)  # Wait between executions
            
            print(f"📈 Executed {executed_count} trades this cycle")
            
            # Check active trades
            self.check_scalping_trades()
            
        except Exception as e:
            print(f"❌ Scalping cycle error: {e}")

    def start_auto_trading(self):
        """Main auto trading loop"""
        print("🚀 STARTING MULTI-PAIR SCALPING BOT!")
        
        # Initial pair selection
        self.available_pairs = self.get_ai_recommended_pairs()
        self.last_rotation_time = time.time()
        
        cycle_count = 0
        
        while True:
            try:
                cycle_count += 1
                print(f"\n{'='*60}")
                print(f"🔄 CYCLE {cycle_count} - {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*60}")
                
                self.run_scalping_cycle()
                
                # Status update
                if cycle_count % 10 == 0:
                    print(f"\n📈 BOT STATUS:")
                    print(f"   Total Cycles: {cycle_count}")
                    print(f"   Available Pairs: {len(self.available_pairs)}")
                    print(f"   Active Trades: {len(self.active_trades)}/{self.max_concurrent_trades}")
                
                time.sleep(60)  # 1 minute between cycles
                
            except KeyboardInterrupt:
                print(f"\n🛑 BOT STOPPED BY USER")
                break
            except Exception as e:
                print(f"❌ Main loop error: {e}")
                time.sleep(30)

# START BOT
if __name__ == "__main__":
    try:
        bot = MultiPairScalpingTrader()
        bot.start_auto_trading()
    except Exception as e:
        print(f"❌ Failed to start bot: {e}")
