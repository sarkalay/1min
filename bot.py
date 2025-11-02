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
        self.trade_size_usd = 50
        self.leverage = 5
        self.risk_percentage = 1.0
        
        # Multi-pair parameters
        self.max_concurrent_trades = 2
        self.available_pairs = []
        self.active_trades = {}
        
        # Trading pairs
        self.reliable_pairs = ["SOLUSDT", "AVAXUSDT", "XRPUSDT", "LINKUSDT", "DOTUSDT"]
        
        # Precision settings
        self.quantity_precision = {}
        self.price_precision = {}
        
        # Minimum quantities for each pair
        self.min_quantities = {
            "SOLUSDT": 0.01,
            "AVAXUSDT": 0.1, 
            "XRPUSDT": 1,
            "LINKUSDT": 0.1,
            "DOTUSDT": 0.1
        }
        
        # Initialize Binance client
        self.binance = Client(self.binance_api_key, self.binance_secret)
        
        print("🤖 AI SCALPING BOT ACTIVATED!")
        print(f"💵 Trade Size: ${self.trade_size_usd}")
        print(f"✅ Trading Pairs: {self.reliable_pairs}")
        
        self.validate_config()
        self.setup_futures()
        self.load_symbol_precision()
        self.available_pairs = self.reliable_pairs
    
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
            
            # Get precision from Binance
            precision = self.quantity_precision.get(pair, 3)
            
            # Round to correct precision
            quantity = round(quantity, precision)
            
            # Ensure minimum quantity for each pair
            min_qty = self.min_quantities.get(pair, 0.1)
            if quantity < min_qty:
                quantity = min_qty
            
            # Calculate actual trade value
            trade_value = quantity * price
            
            print(f"🎯 {pair} target size: ${trade_size}")
            print(f"🔢 Calculated: {quantity} {pair} = ${trade_value:.2f}")
            
            # Better auto-adjustment
            if trade_value < 45:  # If too small compared to target
                required_quantity = trade_size / price
                quantity = round(required_quantity, precision)
                if quantity < min_qty:
                    quantity = min_qty
                trade_value = quantity * price
                print(f"🔄 Size adjusted: {quantity} = ${trade_value:.2f}")
            elif trade_value > 65:  # If too large
                # Reduce quantity to get closer to $50
                optimal_quantity = trade_size / price
                quantity = round(optimal_quantity, precision)
                if quantity < min_qty:
                    quantity = min_qty
                trade_value = quantity * price
                print(f"🔄 Size reduced: {quantity} = ${trade_value:.2f}")
            
            print(f"💰 FINAL: {quantity} {pair} = ${trade_value:.2f}")
            return quantity
            
        except Exception as e:
            print(f"❌ Quantity calculation failed for {pair}: {e}")
            return None
    
    def format_price(self, pair, price):
        precision = self.price_precision.get(pair, 4)
        
        # ✅ FIXED: Check if price is valid
        if price is None or price <= 0:
            print(f"❌ Invalid price for {pair}: {price}")
            return 0
        
        # ✅ FIXED: Use proper rounding
        formatted_price = round(price, precision)
        
        # ✅ FIXED: Ensure minimum price difference for low-priced pairs
        if pair in ["XRPUSDT", "ADAUSDT", "DOGEUSDT", "MATICUSDT"]:
            # Low-priced pairs need minimum tick size
            if pair == "XRPUSDT":
                tick_size = 0.0001
            else:
                tick_size = 0.001
            formatted_price = round(formatted_price / tick_size) * tick_size
        
        return formatted_price

    def setup_futures(self):
        try:
            for pair in self.reliable_pairs:
                try:
                    self.binance.futures_change_leverage(symbol=pair, leverage=self.leverage)
                    print(f"✅ Leverage set for {pair}")
                except Exception as e:
                    print(f"⚠️ Leverage setup failed for {pair}: {e}")
            print("✅ Futures setup completed!")
        except Exception as e:
            print(f"❌ Futures setup failed: {e}")
    
    def get_detailed_market_data(self):
        market_data = {}
        
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
        pair = list(market_data.keys())[0]
        data = market_data[pair]
        current_price = data['price']
        
        # ✅ FIXED: BETTER PROMPT - AI ကို NO TRADE မပြန်ခိုင်းတော့ဘူး
        prompt = f"""
        BINANCE FUTURES SCALPING DECISION for {pair}:
        Current Price: ${current_price}
        1H Change: {data['change_1h']:.2f}%
        Volume Ratio: {data['volume_ratio']:.2f}
        
        CRITICAL INSTRUCTIONS:
        1. You MUST provide a TRADE decision (LONG or SHORT) - DO NOT say "NO TRADE"
        2. Even in low volume conditions, find the better direction
        3. For LONG: stop_loss = current_price * 0.994 to 0.998, take_profit = current_price * 1.006 to 1.012
        4. For SHORT: stop_loss = current_price * 1.002 to 1.006, take_profit = current_price * 0.988 to 0.994
        5. Prices MUST be valid numbers, not None or zero
        6. STOP_LOSS and TAKE_PROFIT MUST be DIFFERENT prices

        Based on the data, choose either LONG or SHORT and provide exact prices.
        
        RESPONSE (JSON only):
        {{
            "action": "TRADE",
            "pair": "{pair}",
            "direction": "LONG",
            "entry_price": {current_price},
            "stop_loss": {current_price * 0.996},
            "take_profit": {current_price * 1.008},
            "confidence": 65,
            "reason": "Your analysis here"
        }}
        """
        
        try:
            headers = {"Authorization": f"Bearer {self.deepseek_key}", "Content-Type": "application/json"}
            payload = {
                "model": "deepseek-chat", 
                "messages": [{"role": "user", "content": prompt}], 
                "temperature": 0.4,
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
                    
                    # ✅ FIXED: BETTER VALIDATION
                    return self.validate_ai_decision(decision, current_price)
                    
        except Exception as e:
            print(f"❌ AI decision error: {e}")
        
        # Fallback to simple trend following
        return self.get_simple_decision(market_data)

    def validate_ai_decision(self, decision, current_price):
        """AI decision ကို validate လုပ်မယ် - IMPROVED"""
        try:
            # Check if decision is valid
            if not decision or decision.get("action") != "TRADE":
                print("❌ AI returned NO TRADE - Using simple logic")
                return self.get_simple_decision({"dummy": {"price": current_price}})
            
            pair = decision.get("pair", "UNKNOWN")
            direction = decision.get("direction")
            stop_loss = decision.get("stop_loss")
            take_profit = decision.get("take_profit")
            
            # Check if required fields exist
            if not all([direction, stop_loss, take_profit]):
                print("❌ AI missing required fields - Using simple logic")
                return self.get_simple_decision({"dummy": {"price": current_price}})
            
            # Convert to float if string
            if isinstance(stop_loss, str):
                stop_loss = float(stop_loss)
            if isinstance(take_profit, str):
                take_profit = float(take_profit)
            
            # Check if prices are valid numbers
            if stop_loss == 0 or take_profit == 0 or stop_loss is None or take_profit is None:
                print("❌ AI provided invalid prices - Using simple logic")
                return self.get_simple_decision({"dummy": {"price": current_price}})
            
            # ✅ FIXED: Better price adjustment with minimum distances
            min_distance_percent = 0.008  # 0.8% minimum distance
            
            if direction == "LONG":
                min_tp_distance = current_price * min_distance_percent
                min_sl_distance = current_price * min_distance_percent
                
                if take_profit <= current_price + min_tp_distance:
                    take_profit = current_price * 1.010  # 1.0% take profit
                    print(f"🔄 Adjusted TP for LONG: {take_profit:.4f}")
                
                if stop_loss >= current_price - min_sl_distance:
                    stop_loss = current_price * 0.990  # 1.0% stop loss
                    print(f"🔄 Adjusted SL for LONG: {stop_loss:.4f}")
                    
            elif direction == "SHORT":
                min_tp_distance = current_price * min_distance_percent
                min_sl_distance = current_price * min_distance_percent
                
                if take_profit >= current_price - min_tp_distance:
                    take_profit = current_price * 0.990  # 1.0% take profit
                    print(f"🔄 Adjusted TP for SHORT: {take_profit:.4f}")
                
                if stop_loss <= current_price + min_sl_distance:
                    stop_loss = current_price * 1.010  # 1.0% stop loss
                    print(f"🔄 Adjusted SL for SHORT: {stop_loss:.4f}")
            else:
                print("❌ Invalid direction - Using simple logic")
                return self.get_simple_decision({"dummy": {"price": current_price}})
            
            # ✅ FIXED: Format prices
            original_sl = stop_loss
            original_tp = take_profit
            stop_loss = self.format_price(pair, stop_loss)
            take_profit = self.format_price(pair, take_profit)
            
            print(f"🔧 Price formatting: SL {original_sl:.4f} -> {stop_loss:.4f}, TP {original_tp:.4f} -> {take_profit:.4f}")
            
            # ✅ FIXED: Final validation to ensure prices are different
            if abs(stop_loss - take_profit) < current_price * 0.005:  # Less than 0.5% difference
                print("❌ CRITICAL: SL and TP too close after formatting!")
                if direction == "LONG":
                    take_profit = current_price * 1.012
                    stop_loss = current_price * 0.988
                else:
                    take_profit = current_price * 0.988
                    stop_loss = current_price * 1.012
                
                stop_loss = self.format_price(pair, stop_loss)
                take_profit = self.format_price(pair, take_profit)
                print(f"🔧 Re-adjusted: SL {stop_loss:.4f}, TP {take_profit:.4f}")
            
            decision.update({
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "confidence": decision.get("confidence", 60)
            })
            
            print(f"✅ AI Decision: {direction} | Entry: ${current_price:.4f} | SL: ${stop_loss:.4f} | TP: ${take_profit:.4f}")
            return decision
            
        except Exception as e:
            print(f"❌ AI validation failed: {e}")
            return self.get_simple_decision({"dummy": {"price": current_price}})

    def get_simple_decision(self, market_data):
        """Simple but reliable decision making - IMPROVED"""
        try:
            pair = list(market_data.keys())[0]
            data = market_data[pair]
            current_price = data['price']
            
            # Simple trend following logic
            if data.get('change_1h', 0) > 0:
                direction = "LONG"
                stop_loss = current_price * 0.988  # 1.2% stop loss
                take_profit = current_price * 1.012  # 1.2% take profit
                confidence = 60
                reason = "Simple: Bullish trend"
            else:
                direction = "SHORT"
                stop_loss = current_price * 1.012  # 1.2% stop loss
                take_profit = current_price * 0.988  # 1.2% take profit
                confidence = 55
                reason = "Simple: Bearish trend"
            
            # ✅ FIXED: Ensure prices are different
            if abs(stop_loss - take_profit) < current_price * 0.01:  # Less than 1% difference
                if direction == "LONG":
                    take_profit = current_price * 1.015
                    stop_loss = current_price * 0.985
                else:
                    take_profit = current_price * 0.985
                    stop_loss = current_price * 1.015
            
            # Format prices
            stop_loss = self.format_price(pair, stop_loss)
            take_profit = self.format_price(pair, take_profit)
            
            decision = {
                "action": "TRADE",
                "pair": pair,
                "direction": direction,
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "confidence": confidence,
                "reason": reason
            }
            
            print(f"🔧 Simple Decision: {direction} | SL: ${stop_loss:.4f} | TP: ${take_profit:.4f}")
            return decision
            
        except Exception as e:
            print(f"❌ Simple decision failed: {e}")
            # Ultimate fallback
            return {
                "action": "WAIT", 
                "reason": "All decision methods failed"
            }

    def execute_scalping_trade(self, decision):
        try:
            if decision.get("action") != "TRADE":
                print(f"⏸️ No trade: {decision.get('reason', 'Unknown reason')}")
                return False
                
            pair = decision["pair"]
            direction = decision["direction"]
            stop_loss = decision["stop_loss"]
            take_profit = decision["take_profit"]
            
            print(f"🎯 EXECUTING TRADE: {pair} {direction}")
            
            if len(self.active_trades) >= self.max_concurrent_trades or pair in self.active_trades:
                return False
            
            # Get current price for quantity calculation
            ticker = self.binance.futures_symbol_ticker(symbol=pair)
            current_price = float(ticker['price'])
            
            quantity = self.get_quantity(pair, current_price)
            if quantity is None:
                return False
            
            print(f"💰 Current Price: ${current_price}")
            print(f"⚡ Quantity: {quantity}")
            print(f"🎯 TP: ${take_profit} | SL: ${stop_loss}")
            
            # Final validation
            if direction == "LONG":
                if take_profit <= current_price or stop_loss >= current_price:
                    print("❌ CRITICAL: Invalid TP/SL prices - ABORTING")
                    return False
            else:
                if take_profit >= current_price or stop_loss <= current_price:
                    print("❌ CRITICAL: Invalid TP/SL prices - ABORTING")
                    return False
            
            # MARKET ENTRY
            try:
                if direction == "LONG":
                    order = self.binance.futures_create_order(
                        symbol=pair, side='BUY', type='MARKET', quantity=quantity
                    )
                    print(f"✅ LONG ENTRY: {quantity} {pair} @ ${current_price}")
                else:
                    order = self.binance.futures_create_order(
                        symbol=pair, side='SELL', type='MARKET', quantity=quantity
                    )
                    print(f"✅ SHORT ENTRY: {quantity} {pair} @ ${current_price}")
            except Exception as e:
                print(f"❌ Entry order failed: {e}")
                return False
            
            # TP/SL ORDERS
            try:
                if direction == "LONG":
                    # Stop Loss
                    self.binance.futures_create_order(
                        symbol=pair, side='SELL', type='STOP_MARKET',
                        quantity=quantity, stopPrice=stop_loss,
                        timeInForce='GTC', reduceOnly=True
                    )
                    # Take Profit  
                    self.binance.futures_create_order(
                        symbol=pair, side='SELL', type='TAKE_PROFIT_MARKET',
                        quantity=quantity, stopPrice=take_profit,
                        timeInForce='GTC', reduceOnly=True
                    )
                else:
                    # Stop Loss
                    self.binance.futures_create_order(
                        symbol=pair, side='BUY', type='STOP_MARKET',
                        quantity=quantity, stopPrice=stop_loss, 
                        timeInForce='GTC', reduceOnly=True
                    )
                    # Take Profit
                    self.binance.futures_create_order(
                        symbol=pair, side='BUY', type='TAKE_PROFIT_MARKET',
                        quantity=quantity, stopPrice=take_profit,
                        timeInForce='GTC', reduceOnly=True
                    )
                    
                print(f"✅ TP/SL ORDERS PLACED SUCCESSFULLY")
                
            except Exception as e:
                print(f"❌ TP/SL order failed: {e}")
                return False
            
            self.active_trades[pair] = {
                "pair": pair,
                "direction": direction,
                "entry_price": current_price,
                "quantity": quantity,
                "entry_time": time.time(),
                "confidence": decision["confidence"]
            }
            
            print(f"🚀 TRADE ACTIVATED: {pair} {direction}")
            print(f"📊 Active Trades: {list(self.active_trades.keys())}")
            return True
            
        except Exception as e:
            print(f"❌ Trade execution failed: {e}")
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
                print("📊 No market data available")
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
                    
                    if decision.get("action") == "TRADE":
                        trade_opportunities.append((decision, decision.get("confidence", 50)))
                        print(f"✅ QUALIFIED: {pair} - {decision.get('confidence', 50)}% confidence")
            
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
        print("🚀 STARTING ULTIMATE AI BOT!")
        
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
