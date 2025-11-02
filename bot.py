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
        
        # SCALPING parameters - FIXED
        self.trade_size_usd = 50   # $50 ပြန်ချိန်းထားတယ်
        self.leverage = 5
        self.risk_percentage = 1.0
        self.scalp_take_profit = 0.010  # 1.0% for better execution
        self.scalp_stop_loss = 0.006    # 0.6% 
        
        # Multi-pair parameters
        self.max_concurrent_trades = 2
        self.available_pairs = []
        self.active_trades = {}
        
        # ✅ PROBLEMATIC PAIRS ကို ခွဲထားတယ် - MATIC ဖြုတ်, POL ထည့်
        self.problematic_pairs = ["ADAUSDT", "DOGEUSDT"]
        self.reliable_pairs = ["SOLUSDT", "AVAXUSDT", "XRPUSDT", "LINKUSDT", "DOTUSDT", "POLUSDT"]
        
        # Precision settings
        self.quantity_precision = {}
        self.price_precision = {}
        
        # Pair-specific settings - UPDATED
        self.pair_settings = {
            "ADAUSDT": {"min_qty": 1, "price_precision": 4},
            "DOGEUSDT": {"min_qty": 1, "price_precision": 5},
            "XRPUSDT": {"min_qty": 1, "price_precision": 4},
            "SOLUSDT": {"min_qty": 0.1, "price_precision": 2},
            "AVAXUSDT": {"min_qty": 0.1, "price_precision": 3},
            "LINKUSDT": {"min_qty": 0.1, "price_precision": 3},
            "DOTUSDT": {"min_qty": 0.1, "price_precision": 3},
            "POLUSDT": {"min_qty": 0.1, "price_precision": 3}
        }
        
        # Initialize Binance client
        self.binance = Client(self.binance_api_key, self.binance_secret)
        
        print("🤖 FIXED SCALPING BOT ACTIVATED!")
        print(f"💵 Trade Size: ${self.trade_size_usd} per trade")
        print(f"🚫 Problematic Pairs: {self.problematic_pairs}")
        print(f"✅ Reliable Pairs: {self.reliable_pairs}")
        
        self.validate_config()
        self.setup_futures()
        self.load_symbol_precision()
        self.available_pairs = self.get_safe_recommended_pairs()
    
    def validate_config(self):
        if not all([self.binance_api_key, self.binance_secret, self.deepseek_key]):
            print("❌ Missing API keys!")
            return False
        try:
            self.binance.futures_exchange_info()
            print("✅ Binance connection successful!")
        except Exception as e:
            print(f"❌ Binance connection failed: {e}")
            return False
        return True
    
    def load_symbol_precision(self):
        try:
            exchange_info = self.binance.futures_exchange_info()
            for symbol in exchange_info['symbols']:
                pair = symbol['symbol']
                for f in symbol['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        step_size = f['stepSize']
                        if '1' in step_size:
                            qty_precision = 0
                        else:
                            qty_precision = len(step_size.split('.')[1].rstrip('0'))
                        self.quantity_precision[pair] = qty_precision
                    elif f['filterType'] == 'PRICE_FILTER':
                        tick_size = f['tickSize']
                        if '1' in tick_size:
                            price_precision = 0
                        else:
                            price_precision = len(tick_size.split('.')[1].rstrip('0'))
                        self.price_precision[pair] = price_precision
            print("✅ Symbol precision loaded")
        except Exception as e:
            print(f"❌ Error loading symbol precision: {e}")
    
    def get_quantity(self, pair, price):
        try:
            trade_size = self.trade_size_usd
            quantity = trade_size / price
            
            # Get precision and minimum quantity
            precision = self.quantity_precision.get(pair, 3)
            min_qty = self.pair_settings.get(pair, {}).get("min_qty", 1)
            
            # Round to correct precision
            quantity = round(quantity, precision)
            if precision == 0:
                quantity = int(quantity)
            
            # Ensure minimum quantity
            if quantity < min_qty:
                quantity = min_qty
            
            trade_value = quantity * price
            
            print(f"🎯 {pair} target size: ${trade_size}")
            print(f"🔢 Calculated: {quantity} {pair} = ${trade_value:.2f}")
            
            # Ensure minimum order value
            if trade_value < 20:
                required_quantity = 25 / price
                quantity = round(required_quantity, precision)
                if precision == 0:
                    quantity = int(quantity)
                if quantity < min_qty:
                    quantity = min_qty
                trade_value = quantity * price
                print(f"🚀 Auto-adjusted: {quantity} = ${trade_value:.2f}")
            
            print(f"💰 FINAL: {quantity} {pair} = ${trade_value:.2f}")
            return quantity
            
        except Exception as e:
            print(f"❌ Quantity calculation failed for {pair}: {e}")
            return None
    
    def format_price(self, pair, price):
        precision = self.price_precision.get(pair, 4)
        return round(price, precision)
    
    def setup_futures(self):
        try:
            # Reliable pairs ပဲ leverage set လုပ်မယ်
            for pair in self.reliable_pairs:
                try:
                    self.binance.futures_change_leverage(symbol=pair, leverage=self.leverage)
                    print(f"✅ Leverage set for {pair}")
                except Exception as e:
                    print(f"⚠️ Leverage setup failed for {pair}: {e}")
            print("✅ Futures setup completed!")
        except Exception as e:
            print(f"❌ Futures setup failed: {e}")
    
    def get_safe_recommended_pairs(self):
        """ADA, DOGE ပြဿနာရှိတာတွေကို ရှောင်ပြီး pairs recommend"""
        print("🔧 Safe pairs selection...")
        
        # Reliable pairs ပဲသုံးမယ်
        safe_pairs = self.reliable_pairs.copy()
        
        # AI recommendation ယူမယ်၊ ဒါပေမယ့် problematic pairs ဖြတ်မယ်
        try:
            prompt = "Recommend 6 best Binance futures pairs EXCLUDING ADAUSDT, DOGEUSDT for scalping. Include SOL, AVAX, XRP, LINK, DOT, POL"
            headers = {"Authorization": f"Bearer {self.deepseek_key}", "Content-Type": "application/json"}
            payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "temperature": 0.3, "max_tokens": 300}
            
            response = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                json_match = re.search(r'\[.*\]', content)
                if json_match:
                    ai_pairs = json.loads(json_match.group())
                    # Problematic pairs filter
                    safe_ai_pairs = [p for p in ai_pairs if p not in self.problematic_pairs]
                    safe_pairs = safe_ai_pairs[:6]  # Take top 6
        except Exception as e:
            print(f"⚠️ AI pair selection failed, using reliable pairs: {e}")
            
        print(f"✅ Safe Pairs Selected: {safe_pairs}")
        return safe_pairs

    def validate_ai_decision(self, decision, current_price):
        """AI decision ကို validate လုပ်တဲ့ function အသစ်"""
        try:
            sl = decision["stop_loss"]
            tp = decision["take_profit"]
            pair = decision["pair"]
            
            # 1. Check if prices are different
            if sl == tp:
                print("❌ TP/SL are same - generating proper prices")
                return self.generate_proper_prices(decision, current_price)
            
            # 2. Check minimum distance (0.5% minimum)
            min_distance = current_price * 0.005
            if abs(sl - current_price) < min_distance or abs(tp - current_price) < min_distance:
                print("❌ TP/SL too close - regenerating")
                return self.generate_proper_prices(decision, current_price)
                
            # 3. Check logical direction
            if decision["direction"] == "LONG":
                if tp <= current_price or sl >= current_price:
                    print("❌ LONG direction wrong - regenerating")
                    return self.generate_proper_prices(decision, current_price)
            else:  # SHORT
                if tp >= current_price or sl <= current_price:
                    print("❌ SHORT direction wrong - regenerating")  
                    return self.generate_proper_prices(decision, current_price)
                    
            print(f"✅ AI Decision Validated: SL=${sl}, TP=${tp}")
            return decision  # All good
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return self.generate_proper_prices(decision, current_price)
    
    def generate_proper_prices(self, decision, current_price):
        """AI မှားရင် proper prices generate လုပ်မယ်"""
        pair = decision["pair"]
        direction = decision["direction"]
        
        if direction == "LONG":
            stop_loss = current_price * (1 - self.scalp_stop_loss)
            take_profit = current_price * (1 + self.scalp_take_profit)
        else:
            stop_loss = current_price * (1 + self.scalp_stop_loss) 
            take_profit = current_price * (1 - self.scalp_take_profit)
        
        # Format with proper precision
        stop_loss = self.format_price(pair, stop_loss)
        take_profit = self.format_price(pair, take_profit)
        
        decision["stop_loss"] = stop_loss
        decision["take_profit"] = take_profit
        decision["reason"] = "Auto-corrected prices"
        
        print(f"🔧 Auto-corrected: SL=${stop_loss}, TP=${take_profit}")
        return decision

    def get_detailed_market_data(self):
        market_data = {}
        
        if not self.available_pairs:
            self.available_pairs = self.get_safe_recommended_pairs()
        
        for pair in self.available_pairs:
            try:
                if pair in self.active_trades:
                    continue
                    
                ticker = self.binance.futures_symbol_ticker(symbol=pair)
                if 'price' not in ticker or not ticker['price']:
                    continue
                    
                price = float(ticker['price'])
                
                klines = self.binance.futures_klines(symbol=pair, interval=Client.KLINE_INTERVAL_15MINUTE, limit=20)
                
                if len(klines) > 0:
                    closes = [float(k[4]) for k in klines]
                    volumes = [float(k[5]) for k in klines]
                    
                    current_volume = volumes[-1] if volumes else 0
                    avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else current_volume
                    
                    price_change_1h = ((closes[-1] - closes[-4]) / closes[-4]) * 100 if len(closes) >= 4 else 0
                    
                    market_data[pair] = {
                        'price': price,
                        'change_1h': price_change_1h,
                        'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1,
                    }
                
            except Exception as e:
                print(f"❌ Market data error for {pair}: {e}")
                continue
                
        return market_data

    def get_scalping_decision(self, market_data):
        """Fixed version with validation"""
        pair = list(market_data.keys())[0]
        data = market_data[pair]
        price = data['price']
        
        # Better prompt with exact calculation instructions
        prompt = f"""
        BINANCE FUTURES SCALPING for {pair}:
        Current Price: ${price}
        1H Change: {data['change_1h']:.2f}%
        Volume Ratio: {data['volume_ratio']:.2f}
        
        CALCULATE EXACT PRICES:
        - For LONG: SL = {price} * 0.994 = {price * 0.994:.6f}, TP = {price} * 1.010 = {price * 1.01:.6f}
        - For SHORT: SL = {price} * 1.006 = {price * 1.006:.6f}, TP = {price} * 0.990 = {price * 0.99:.6f}
        
        MUST: 
        - Prices DIFFERENT from current price
        - Logical for direction (LONG: TP > Price > SL, SHORT: TP < Price < SL)
        - Use proper precision
        
        RESPONSE (JSON only):
        {{
            "action": "TRADE", 
            "pair": "{pair}",
            "direction": "LONG/SHORT",
            "entry_price": {price},
            "stop_loss": {price * 0.994:.6f},
            "take_profit": {price * 1.01:.6f},
            "confidence": 75,
            "reason": "Valid setup"
        }}
        """
        
        try:
            headers = {"Authorization": f"Bearer {self.deepseek_key}", "Content-Type": "application/json"}
            payload = {
                "model": "deepseek-chat", 
                "messages": [{"role": "user", "content": prompt}], 
                "temperature": 0.2,  # Lower temperature for consistency
                "max_tokens": 500
            }
            
            response = requests.post("https://api.deepseek.com/v1/chat/completions", 
                                   headers=headers, json=payload, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    decision = json.loads(json_match.group())
                    print(f"🤖 AI Raw Decision: {decision}")
                    
                    # ✅ VALIDATE THE DECISION
                    validated_decision = self.validate_ai_decision(decision, price)
                    return validated_decision
                    
        except Exception as e:
            print(f"❌ AI decision error: {e}")
        
        # Fallback with auto-generated prices
        fallback_decision = {
            "action": "TRADE",
            "pair": pair,
            "direction": "LONG" if data['change_1h'] > 0 else "SHORT",
            "entry_price": price,
            "stop_loss": price * 0.994,
            "take_profit": price * 1.01, 
            "confidence": 65,
            "reason": "Fallback decision"
        }
        return self.validate_ai_decision(fallback_decision, price)

    def execute_scalping_trade(self, decision):
        """Fixed execution with better error handling"""
        try:
            pair = decision["pair"]
            direction = decision["direction"]
            
            print(f"🎯 EXECUTING TRADE: {pair} {direction}")
            
            if len(self.active_trades) >= self.max_concurrent_trades or pair in self.active_trades:
                return False
            
            ticker = self.binance.futures_symbol_ticker(symbol=pair)
            current_price = float(ticker['price'])
            print(f"💰 Current {pair} price: ${current_price}")
            
            quantity = self.get_quantity(pair, current_price)
            if quantity is None:
                return False
            
            print(f"⚡ Quantity: {quantity} {pair}")
            
            entry_price = current_price
            
            # MARKET ENTRY
            try:
                if direction == "LONG":
                    order = self.binance.futures_create_order(symbol=pair, side='BUY', type='MARKET', quantity=quantity)
                    print(f"✅ LONG ENTRY: {quantity} {pair} @ ${entry_price}")
                else:
                    order = self.binance.futures_create_order(symbol=pair, side='SELL', type='MARKET', quantity=quantity)
                    print(f"✅ SHORT ENTRY: {quantity} {pair} @ ${entry_price}")
            except Exception as e:
                print(f"❌ Entry order failed: {e}")
                return False
            
            # ✅ AI ကပေးတဲ့ TP/SL ကိုသုံးပါ (already validated)
            stop_loss = decision["stop_loss"]
            take_profit = decision["take_profit"]
            
            # Format prices only
            stop_loss = self.format_price(pair, stop_loss)
            take_profit = self.format_price(pair, take_profit)
            
            print(f"🎯 AI SET: Entry=${entry_price}, TP=${take_profit}, SL=${stop_loss}")
            
            # UNIVERSAL TP/SL PLACEMENT
            try:
                if direction == "LONG":
                    self.binance.futures_create_order(
                        symbol=pair, side='SELL', type='STOP_MARKET', 
                        quantity=quantity, stopPrice=stop_loss, 
                        timeInForce='GTC', reduceOnly=True
                    )
                    self.binance.futures_create_order(
                        symbol=pair, side='SELL', type='LIMIT', 
                        quantity=quantity, price=take_profit, 
                        timeInForce='GTC', reduceOnly=True
                    )
                else:
                    self.binance.futures_create_order(
                        symbol=pair, side='BUY', type='STOP_MARKET', 
                        quantity=quantity, stopPrice=stop_loss, 
                        timeInForce='GTC', reduceOnly=True
                    )
                    self.binance.futures_create_order(
                        symbol=pair, side='BUY', type='LIMIT', 
                        quantity=quantity, price=take_profit, 
                        timeInForce='GTC', reduceOnly=True
                    )
                    
                print(f"✅ TP/SL ORDERS PLACED")
                
            except Exception as e:
                print(f"❌ TP/SL order failed: {e}")
                
                # ALTERNATIVE METHOD FOR PROBLEMATIC PAIRS
                try:
                    print(f"🔄 Trying alternative TP/SL method...")
                    if direction == "LONG":
                        self.binance.futures_create_order(
                            symbol=pair, side='SELL', type='TAKE_PROFIT_MARKET',
                            quantity=quantity, stopPrice=take_profit,
                            timeInForce='GTC', reduceOnly=True
                        )
                        self.binance.futures_create_order(
                            symbol=pair, side='SELL', type='STOP_MARKET',
                            quantity=quantity, stopPrice=stop_loss,
                            timeInForce='GTC', reduceOnly=True
                        )
                    else:
                        self.binance.futures_create_order(
                            symbol=pair, side='BUY', type='TAKE_PROFIT_MARKET',
                            quantity=quantity, stopPrice=take_profit,
                            timeInForce='GTC', reduceOnly=True
                        )
                        self.binance.futures_create_order(
                            symbol=pair, side='BUY', type='STOP_MARKET',
                            quantity=quantity, stopPrice=stop_loss,
                            timeInForce='GTC', reduceOnly=True
                        )
                    print(f"✅ ALTERNATIVE TP/SL SUCCESSFUL")
                except Exception as e2:
                    print(f"❌ Alternative also failed: {e2}")
                    return False
            
            self.active_trades[pair] = {
                "pair": pair,
                "direction": direction,
                "entry_price": entry_price,
                "quantity": quantity,
                "entry_time": time.time(),
                "confidence": decision["confidence"]
            }
            
            print(f"🚀 TRADE ACTIVATED: {pair} {direction}")
            print(f"📊 Active Trades: {list(self.active_trades.keys())}")
            return True
            
        except Exception as e:
            print(f"❌ Trade execution failed for {pair}: {e}")
            return False

    def check_scalping_trades(self):
        if not self.active_trades:
            return
        
        completed_trades = []
        
        for pair, trade_info in self.active_trades.items():
            try:
                positions = self.binance.futures_position_information(symbol=pair)
                position_found = False
                for pos in positions:
                    if pos['symbol'] == pair and float(pos['positionAmt']) != 0:
                        position_found = True
                        break
                
                if not position_found:
                    exit_time = time.time()
                    trade_duration = (exit_time - trade_info["entry_time"]) / 60
                    print(f"💰 TRADE COMPLETED: {pair}!")
                    print(f"   Direction: {trade_info['direction']}")
                    print(f"   Duration: {trade_duration:.1f} minutes")
                    completed_trades.append(pair)
                    
            except Exception as e:
                print(f"❌ Trade check error for {pair}: {e}")
        
        for pair in completed_trades:
            del self.active_trades[pair]
        
        if completed_trades:
            print(f"📊 Remaining Active Trades: {list(self.active_trades.keys())}")

    def run_scalping_cycle(self):
        try:
            market_data = self.get_detailed_market_data()
            
            if not market_data:
                return
            
            print(f"\n📊 CURRENT STATUS:")
            print(f"   Available Pairs: {len(self.available_pairs)}")
            print(f"   Active Trades: {len(self.active_trades)}/{self.max_concurrent_trades}")
            
            if len(self.active_trades) >= self.max_concurrent_trades:
                self.check_scalping_trades()
                return
            
            trade_opportunities = []
            
            for pair in self.available_pairs:
                if len(self.active_trades) >= self.max_concurrent_trades:
                    break
                    
                if pair in self.active_trades:
                    continue
                    
                if pair in market_data:
                    pair_data = {pair: market_data[pair]}
                    decision = self.get_scalping_decision(pair_data)
                    
                    if decision["action"] == "TRADE":
                        trade_opportunities.append((decision, decision["confidence"]))
                        print(f"✅ QUALIFIED: {pair} - {decision['confidence']}% confidence")
            
            trade_opportunities.sort(key=lambda x: x[1], reverse=True)
            
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
                    time.sleep(2)
            
            print(f"📈 Executed {executed_count} trades this cycle")
            
            self.check_scalping_trades()
            
        except Exception as e:
            print(f"❌ Scalping cycle error: {e}")

    def start_auto_trading(self):
        print("🚀 STARTING FIXED SCALPING BOT!")
        
        cycle_count = 0
        
        while True:
            try:
                cycle_count += 1
                print(f"\n{'='*60}")
                print(f"🔄 CYCLE {cycle_count} - {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*60}")
                
                self.run_scalping_cycle()
                
                if cycle_count % 10 == 0:
                    print(f"\n📈 BOT STATUS:")
                    print(f"   Total Cycles: {cycle_count}")
                    print(f"   Active Trades: {len(self.active_trades)}/{self.max_concurrent_trades}")
                
                time.sleep(60)
                
            except KeyboardInterrupt:
                print(f"\n🛑 BOT STOPPED BY USER")
                break
            except Exception as e:
                print(f"❌ Main loop error: {e}")
                time.sleep(30)

if __name__ == "__main__":
    try:
        bot = MultiPairScalpingTrader()
        bot.start_auto_trading()
    except Exception as e:
        print(f"❌ Failed to start bot: {e}")
