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
        self.leverage = 10
        self.risk_percentage = 1.0
        self.scalp_take_profit = 0.008
        self.scalp_stop_loss = 0.005
        
        # Multi-pair parameters
        self.max_concurrent_trades = 3
        self.available_pairs = []
        self.active_trades = {}
        self.blacklisted_pairs = ["BTCUSDT"]
        
        # Precision settings
        self.quantity_precision = {}
        self.price_precision = {}
        
        # Initialize Binance client
        self.binance = Client(self.binance_api_key, self.binance_secret)
        
        print("🤖 MULTI-PAIR SCALPING BOT ACTIVATED!")
        
        self.validate_config()
        self.setup_futures()
        self.load_symbol_precision()
        self.available_pairs = self.get_ai_recommended_pairs()
    
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
    
    def get_dynamic_trade_size(self, pair, price):
        if price > 3000:
            return 200
        elif price > 500:
            return 100
        elif price > 100:
            return 75
        else:
            return 50

    def get_quantity(self, pair, price):
        try:
            trade_size = self.get_dynamic_trade_size(pair, price)
            quantity = trade_size / price
            precision = self.quantity_precision.get(pair, 3)
            quantity = round(quantity, precision)
            if precision == 0:
                quantity = int(quantity)
            
            if pair == "ETHUSDT" and quantity == 0:
                quantity = 0.01
                return quantity
            
            min_order_value = 20
            trade_value = quantity * price
            
            if trade_value < min_order_value:
                required_quantity = min_order_value / price
                quantity = round(required_quantity, precision)
                if precision == 0:
                    quantity = int(quantity)
            
            if quantity <= 0:
                quantity = round(50 / price, precision)
                if precision == 0:
                    quantity = int(quantity)
            
            trade_value = quantity * price
            if trade_value < 15:
                return None
            
            return quantity
            
        except Exception as e:
            print(f"❌ Quantity calculation failed: {e}")
            return None
    
    def format_price(self, pair, price):
        precision = self.price_precision.get(pair, 4)
        return round(price, precision)
    
    def setup_futures(self):
        try:
            initial_pairs = ["ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT"]
            for pair in initial_pairs:
                try:
                    self.binance.futures_change_leverage(symbol=pair, leverage=self.leverage)
                except Exception as e:
                    print(f"⚠️ Leverage setup failed for {pair}: {e}")
            print("✅ Futures setup completed!")
        except Exception as e:
            print(f"❌ Futures setup failed: {e}")
    
    def get_ai_recommended_pairs(self):
        print("🤖 AI က BTC မပါတဲ့ scalping pairs တွေရွေးနေပါတယ်...")
        
        prompt = """
        BINANCE FUTURES SCALPING PAIR RECOMMENDATIONS (EXCLUDE BTCUSDT):
        Recommend 6-10 best altcoin pairs for scalping from Binance futures.
        Focus on ETH, BNB, SOL, ADA, XRP, DOT, MATIC, AVAX, LINK, etc.
        
        RESPONSE (JSON only):
        {
            "recommended_pairs": ["ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", ...],
            "reason": "These altcoin pairs have high liquidity and volatility suitable for scalping, excluding BTC"
        }
        """
        
        try:
            headers = {"Authorization": f"Bearer {self.deepseek_key}", "Content-Type": "application/json"}
            payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "temperature": 0.3, "max_tokens": 500}
            
            response = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload, timeout=20)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    recommendation = json.loads(json_match.group())
                    pairs = recommendation.get("recommended_pairs", [])
                    pairs = [p for p in pairs if p != "BTCUSDT"]
                    print(f"✅ AI Recommended Pairs: {pairs}")
                    return pairs[:8]
        except Exception as e:
            print(f"❌ AI pair selection error: {e}")
        
        fallback_pairs = ["ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT", "MATICUSDT"]
        print(f"🔄 Using fallback pairs: {fallback_pairs}")
        return fallback_pairs
    
    def get_detailed_market_data(self):
        market_data = {}
        
        if not self.available_pairs:
            self.available_pairs = self.get_ai_recommended_pairs()
        
        for pair in self.available_pairs:
            try:
                if pair in self.active_trades:
                    continue
                    
                ticker = self.binance.futures_symbol_ticker(symbol=pair)
                price = float(ticker['price'])
                
                klines = self.binance.futures_klines(symbol=pair, interval=Client.KLINE_INTERVAL_15MINUTE, limit=20)
                
                if len(klines) > 0:
                    closes = [float(k[4]) for k in klines]
                    volumes = [float(k[5]) for k in klines]
                    highs = [float(k[2]) for k in klines]
                    lows = [float(k[3]) for k in klines]
                    
                    current_volume = volumes[-1] if volumes else 0
                    avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else current_volume
                    
                    price_change_1h = ((closes[-1] - closes[-4]) / closes[-4]) * 100 if len(closes) >= 4 else 0
                    
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
        pair = list(market_data.keys())[0]
        data = market_data[pair]
        price = data['price']
        
        prompt = f"""
        URGENT: CONFIDENT SCALPING ANALYSIS NEEDED FOR {pair}
        
        CURRENT MARKET DATA:
        - Price: ${price}
        - 1H Change: {data.get('change_1h', 0):.2f}%
        - Volume Ratio: {data.get('volume_ratio', 1):.2f}x
        - Volatility: {data.get('volatility', 0):.2f}%
        
        BE MORE CONFIDENT - this is scalping, not long-term investing
        Look for ANY reasonable 0.5-1% move opportunity
        Minimum confidence MUST be 60%+
        
        RESPONSE (JSON):
        {{
            "action": "TRADE",
            "pair": "{pair}",
            "direction": "LONG/SHORT",
            "entry_price": {price},
            "stop_loss": {price * 0.995},
            "take_profit": {price * 1.008},
            "confidence": 65,
            "reason": "Confident scalping setup",
            "urgency": "high"
        }}
        """
        
        try:
            headers = {"Authorization": f"Bearer {self.deepseek_key}", "Content-Type": "application/json"}
            payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "temperature": 0.8, "max_tokens": 500}
            
            response = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    decision = json.loads(json_match.group())
                    print(f"🤖 {pair}: {decision['action']} ({decision['confidence']}% confidence)")
                    if decision['action'] == 'TRADE':
                        print(f"   📈 Direction: {decision['direction']}")
                    return decision
        
        except Exception as e:
            print(f"❌ AI API Error for {pair}: {e}")
        
        # Fallback
        import random
        direction = "LONG" if random.random() < 0.5 else "SHORT"
        return {
            "action": "TRADE",
            "pair": pair,
            "direction": direction,
            "entry_price": price,
            "stop_loss": price * (0.995 if direction == "LONG" else 1.005),
            "take_profit": price * (1.008 if direction == "LONG" else 0.992),
            "confidence": 70,
            "reason": "Confident fallback",
            "urgency": "high"
        }

    def execute_scalping_trade(self, decision):
        try:
            pair = decision["pair"]
            direction = decision["direction"]
            
            print(f"🎯 EXECUTING TRADE: {pair} {direction}")
            
            # Trade limit checks
            if len(self.active_trades) >= self.max_concurrent_trades:
                print(f"⚠️ Maximum trades reached, skipping {pair}")
                return False
            
            if pair in self.active_trades:
                print(f"⚠️ Already trading {pair}, skipping")
                return False
            
            # Get current price and quantity
            ticker = self.binance.futures_symbol_ticker(symbol=pair)
            current_price = float(ticker['price'])
            print(f"💰 Current {pair} price: ${current_price}")
            
            quantity = self.get_quantity(pair, current_price)
            if quantity is None:
                print(f"❌ Quantity calculation failed for {pair}")
                return False
            
            print(f"⚡ Quantity: {quantity} {pair}")
            
            entry_price = current_price
            
            # MARKET ENTRY
            try:
                if direction == "LONG":
                    order = self.binance.futures_create_order(
                        symbol=pair, side='BUY', type='MARKET', quantity=quantity
                    )
                    print(f"✅ LONG ENTRY: {quantity} {pair} @ ${entry_price}")
                else:
                    order = self.binance.futures_create_order(
                        symbol=pair, side='SELL', type='MARKET', quantity=quantity
                    )
                    print(f"✅ SHORT ENTRY: {quantity} {pair} @ ${entry_price}")
            except Exception as e:
                print(f"❌ Entry order failed: {e}")
                return False
            
            # UNIVERSAL TP/SL CALCULATION FOR ALL PAIRS
            if direction == "LONG":
                stop_loss_price = entry_price * (1 - self.scalp_stop_loss)
                take_profit_price = entry_price * (1 + self.scalp_take_profit)
            else:
                stop_loss_price = entry_price * (1 + self.scalp_stop_loss)
                take_profit_price = entry_price * (1 - self.scalp_take_profit)
            
            # Format prices according to pair precision
            stop_loss_price = self.format_price(pair, stop_loss_price)
            take_profit_price = self.format_price(pair, take_profit_price)
            
            # CRITICAL: VALIDATE PRICES FOR ALL PAIRS
            print(f"🔧 Validating prices for {pair}...")
            
            if direction == "LONG":
                if take_profit_price <= entry_price:
                    print(f"⚠️ TP adjustment needed: TP${take_profit_price} <= Entry${entry_price}")
                    take_profit_price = self.format_price(pair, entry_price * 1.01)
                if stop_loss_price >= entry_price:
                    print(f"⚠️ SL adjustment needed: SL${stop_loss_price} >= Entry${entry_price}")  
                    stop_loss_price = self.format_price(pair, entry_price * 0.99)
            else:
                if take_profit_price >= entry_price:
                    print(f"⚠️ TP adjustment needed: TP${take_profit_price} >= Entry${entry_price}")
                    take_profit_price = self.format_price(pair, entry_price * 0.99)
                if stop_loss_price <= entry_price:
                    print(f"⚠️ SL adjustment needed: SL${stop_loss_price} <= Entry${entry_price}")
                    stop_loss_price = self.format_price(pair, entry_price * 1.01)
            
            print(f"🎯 {direction}: Entry=${entry_price}, TP=${take_profit_price}, SL=${stop_loss_price}")
            
            # UNIVERSAL TP/SL PLACEMENT
            try:
                if direction == "LONG":
                    # STOP LOSS
                    self.binance.futures_create_order(
                        symbol=pair,
                        side='SELL',
                        type='STOP_MARKET',
                        quantity=quantity,
                        stopPrice=stop_loss_price,
                        timeInForce='GTC',
                        reduceOnly=True
                    )
                    # TAKE PROFIT
                    self.binance.futures_create_order(
                        symbol=pair,
                        side='SELL', 
                        type='LIMIT',
                        quantity=quantity,
                        price=take_profit_price,
                        timeInForce='GTC',
                        reduceOnly=True
                    )
                else:
                    # STOP LOSS
                    self.binance.futures_create_order(
                        symbol=pair,
                        side='BUY',
                        type='STOP_MARKET',
                        quantity=quantity,
                        stopPrice=stop_loss_price, 
                        timeInForce='GTC',
                        reduceOnly=True
                    )
                    # TAKE PROFIT
                    self.binance.futures_create_order(
                        symbol=pair,
                        side='BUY',
                        type='LIMIT',
                        quantity=quantity,
                        price=take_profit_price,
                        timeInForce='GTC',
                        reduceOnly=True
                    )
                    
                print(f"✅ TP/SL ORDERS PLACED SUCCESSFULLY")
                
            except Exception as e:
                print(f"❌ TP/SL order failed: {e}")
                
                # EMERGENCY: Close position if TP/SL fails
                try:
                    print(f"🔄 Closing position due to TP/SL failure...")
                    if direction == "LONG":
                        self.binance.futures_create_order(
                            symbol=pair, side='SELL', type='MARKET', quantity=quantity, reduceOnly=True
                        )
                    else:
                        self.binance.futures_create_order(
                            symbol=pair, side='BUY', type='MARKET', quantity=quantity, reduceOnly=True  
                        )
                    print(f"⚠️ Position closed due to TP/SL error")
                except Exception as close_error:
                    print(f"❌ Failed to close position: {close_error}")
                
                return False
            
            # Store trade info
            self.active_trades[pair] = {
                "pair": pair,
                "direction": direction,
                "entry_price": entry_price,
                "quantity": quantity,
                "stop_loss": stop_loss_price,
                "take_profit": take_profit_price,
                "entry_time": time.time(),
                "confidence": decision["confidence"]
            }
            
            print(f"🚀 TRADE SUCCESSFULLY ACTIVATED: {pair} {direction}")
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
                return
            
            print(f"\n📊 CURRENT STATUS:")
            print(f"   Available Pairs: {len(self.available_pairs)}")
            print(f"   Active Trades: {len(self.active_trades)}/{self.max_concurrent_trades}")
            
            if len(self.active_trades) >= self.max_concurrent_trades:
                print(f"⏸️ Maximum trades reached, skipping analysis")
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
                    
                    if decision["action"] == "TRADE" and decision["confidence"] >= 60:
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
        print("🚀 STARTING MULTI-PAIR SCALPING BOT!")
        
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
