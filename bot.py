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

class AIScalpingTrader:
    def __init__(self):
        # Load config from .env file
        self.binance_api_key = os.getenv('BINANCE_API_KEY')
        self.binance_secret = os.getenv('BINANCE_SECRET_KEY')
        self.deepseek_key = os.getenv('DEEPSEEK_API_KEY')
        
        # SCALPING parameters
        self.trade_size_usd = 50
        self.leverage = 5
        
        # ✅ FIXED: Manual TP/SL percentages
        self.take_profit_percent = 0.008  # 0.8%
        self.stop_loss_percent = 0.005    # 0.5%
        
        # Multi-pair parameters
        self.max_concurrent_trades = 1
        self.available_pairs = ["SOLUSDT", "AVAXUSDT", "XRPUSDT", "LINKUSDT", "DOTUSDT"]
        self.active_trades = {}
        
        # Precision settings
        self.quantity_precision = {}
        self.price_precision = {}
        
        # Initialize Binance client
        self.binance = Client(self.binance_api_key, self.binance_secret)
        
        print("🤖 AI DECISION + MANUAL TP/SL BOT ACTIVATED!")
        print(f"💵 Trade Size: ${self.trade_size_usd}")
        print(f"🎯 Fixed TP: {self.take_profit_percent*100}% | Fixed SL: {self.stop_loss_percent*100}%")
        
        self.validate_config()
        self.setup_futures()
        self.load_symbol_precision()
    
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
            
            precision = self.quantity_precision.get(pair, 3)
            quantity = round(quantity, precision)
            if precision == 0:
                quantity = int(quantity)
            
            min_qty = 0.1
            if quantity < min_qty:
                quantity = min_qty
            
            trade_value = quantity * price
            print(f"💰 {quantity} {pair} = ${trade_value:.2f}")
            return quantity
            
        except Exception as e:
            print(f"❌ Quantity calculation failed: {e}")
            return None
    
    def format_price(self, pair, price):
        precision = self.price_precision.get(pair, 4)
        return round(price, precision)
    
    def setup_futures(self):
        try:
            for pair in self.available_pairs:
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

    def get_ai_decision(self, market_data):
        """✅ AI က entry decision ပဲပေးမယ်"""
        pair = list(market_data.keys())[0]
        data = market_data[pair]
        current_price = data['price']
        
        prompt = f"""
        BINANCE FUTURES SCALPING ANALYSIS for {pair}:
        Current Price: ${current_price}
        1H Change: {data['change_1h']:.2f}%
        Volume Ratio: {data['volume_ratio']:.2f}
        
        Analyze the market condition and recommend ONLY the trade direction.
        DO NOT calculate any prices - just recommend LONG or SHORT.
        
        RESPONSE (JSON only):
        {{
            "action": "TRADE",
            "pair": "{pair}",
            "direction": "LONG/SHORT",
            "confidence": 75,
            "reason": "Your technical analysis here"
        }}
        """
        
        try:
            headers = {"Authorization": f"Bearer {self.deepseek_key}", "Content-Type": "application/json"}
            payload = {
                "model": "deepseek-chat", 
                "messages": [{"role": "user", "content": prompt}], 
                "temperature": 0.3,
                "max_tokens": 300
            }
            
            response = requests.post("https://api.deepseek.com/v1/chat/completions", 
                                   headers=headers, json=payload, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    decision = json.loads(json_match.group())
                    print(f"🤖 AI Decision: {decision}")
                    
                    # ✅ FIXED: AI က direction ပဲပေး၊ TP/SL က manual calculation
                    return self.apply_manual_tp_sl(decision, current_price)
                    
        except Exception as e:
            print(f"❌ AI decision error: {e}")
        
        # Fallback
        return self.get_fallback_decision(pair, current_price)

    def apply_manual_tp_sl(self, ai_decision, current_price):
        """✅ AI decision ကို manual TP/SL နဲ့ပေါင်းမယ်"""
        pair = ai_decision["pair"]
        direction = ai_decision["direction"]
        
        # ✅ FIXED: Manual TP/SL calculation
        if direction == "LONG":
            stop_loss = current_price * (1 - self.stop_loss_percent)
            take_profit = current_price * (1 + self.take_profit_percent)
        else:
            stop_loss = current_price * (1 + self.stop_loss_percent)
            take_profit = current_price * (1 - self.take_profit_percent)
        
        # Format prices
        stop_loss = self.format_price(pair, stop_loss)
        take_profit = self.format_price(pair, take_profit)
        
        # ✅ FIXED: Final check - တူမှာမဟုတ်ဘူး
        print(f"🔧 TP/SL Calculation:")
        print(f"   Direction: {direction}")
        print(f"   Current: ${current_price:.4f}")
        print(f"   SL: ${stop_loss:.4f} ({self.stop_loss_percent*100}%)")
        print(f"   TP: ${take_profit:.4f} ({self.take_profit_percent*100}%)")
        
        final_decision = {
            "action": "TRADE",
            "pair": pair,
            "direction": direction,
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "confidence": ai_decision.get("confidence", 65),
            "reason": f"AI Analysis: {ai_decision.get('reason', 'Market condition')}"
        }
        
        print(f"✅ FINAL: {direction} | Entry: ${current_price:.4f} | SL: ${stop_loss:.4f} | TP: ${take_profit:.4f}")
        return final_decision

    def get_fallback_decision(self, pair, current_price):
        """AI fail ရင် fallback"""
        import random
        
        direction = "LONG" if random.random() > 0.5 else "SHORT"
        
        decision = {
            "action": "TRADE",
            "pair": pair,
            "direction": direction,
            "confidence": 50,
            "reason": "Fallback: Random direction"
        }
        
        return self.apply_manual_tp_sl(decision, current_price)

    def execute_trade(self, decision):
        try:
            pair = decision["pair"]
            direction = decision["direction"]
            stop_loss = decision["stop_loss"]
            take_profit = decision["take_profit"]
            
            print(f"🎯 EXECUTING: {pair} {direction}")
            
            if len(self.active_trades) >= self.max_concurrent_trades or pair in self.active_trades:
                return False
            
            # Get current price
            ticker = self.binance.futures_symbol_ticker(symbol=pair)
            current_price = float(ticker['price'])
            
            quantity = self.get_quantity(pair, current_price)
            if quantity is None:
                return False
            
            print(f"⚡ Quantity: {quantity}")
            print(f"🎯 TP: ${take_profit} | SL: ${stop_loss}")
            
            # Final validation
            if direction == "LONG":
                if take_profit <= current_price or stop_loss >= current_price:
                    print("❌ Invalid prices for LONG")
                    return False
            else:
                if take_profit >= current_price or stop_loss <= current_price:
                    print("❌ Invalid prices for SHORT")
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
                        symbol=pair, side='SEL', type='MARKET', quantity=quantity
                    )
                    print(f"✅ SHORT ENTRY: {quantity} {pair} @ ${current_price}")
            except Exception as e:
                print(f"❌ Entry failed: {e}")
                return False
            
            # TP/SL ORDERS
            try:
                if direction == "LONG":
                    self.binance.futures_create_order(
                        symbol=pair, side='SELL', type='STOP_MARKET',
                        quantity=quantity, stopPrice=stop_loss,
                        timeInForce='GTC', reduceOnly=True
                    )
                    self.binance.futures_create_order(
                        symbol=pair, side='SELL', type='TAKE_PROFIT_MARKET',
                        quantity=quantity, stopPrice=take_profit,
                        timeInForce='GTC', reduceOnly=True
                    )
                else:
                    self.binance.futures_create_order(
                        symbol=pair, side='BUY', type='STOP_MARKET',
                        quantity=quantity, stopPrice=stop_loss,
                        timeInForce='GTC', reduceOnly=True
                    )
                    self.binance.futures_create_order(
                        symbol=pair, side='BUY', type='TAKE_PROFIT_MARKET',
                        quantity=quantity, stopPrice=take_profit,
                        timeInForce='GTC', reduceOnly=True
                    )
                    
                print(f"✅ TP/SL PLACED")
                
            except Exception as e:
                print(f"❌ TP/SL failed: {e}")
                return False
            
            self.active_trades[pair] = {
                "pair": pair,
                "direction": direction,
                "entry_price": current_price,
                "quantity": quantity,
                "entry_time": time.time()
            }
            
            print(f"🚀 TRADE ACTIVATED: {pair} {direction}")
            return True
            
        except Exception as e:
            print(f"❌ Trade execution failed: {e}")
            return False

    def check_active_trades(self):
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
                    print(f"💰 TRADE COMPLETED: {pair}")
                    completed_trades.append(pair)
                    
            except Exception as e:
                print(f"❌ Trade check error: {e}")
        
        for pair in completed_trades:
            del self.active_trades[pair]
        
        if completed_trades:
            print(f"📊 Active Trades: {list(self.active_trades.keys())}")

    def run_trading_cycle(self):
        try:
            market_data = self.get_detailed_market_data()
            
            if not market_data:
                return
            
            print(f"\n📊 STATUS: Active Trades: {len(self.active_trades)}/{self.max_concurrent_trades}")
            
            if len(self.active_trades) >= self.max_concurrent_trades:
                self.check_active_trades()
                return
            
            for pair in self.available_pairs:
                if len(self.active_trades) >= self.max_concurrent_trades:
                    break
                    
                if pair in self.active_trades:
                    continue
                    
                if pair in market_data:
                    pair_data = {pair: market_data[pair]}
                    decision = self.get_ai_decision(pair_data)
                    
                    if decision["action"] == "TRADE":
                        print(f"✅ QUALIFIED: {pair}")
                        success = self.execute_trade(decision)
                        if success:
                            time.sleep(2)
            
            self.check_active_trades()
            
        except Exception as e:
            print(f"❌ Trading cycle error: {e}")

    def start_trading(self):
        print("🚀 STARTING AI + MANUAL TP/SL BOT!")
        
        cycle_count = 0
        
        while True:
            try:
                cycle_count += 1
                print(f"\n{'='*50}")
                print(f"🔄 CYCLE {cycle_count} - {time.strftime('%H:%M:%S')}")
                print(f"{'='*50}")
                
                self.run_trading_cycle()
                time.sleep(60)
                
            except KeyboardInterrupt:
                print(f"\n🛑 BOT STOPPED")
                break
            except Exception as e:
                print(f"❌ Main loop error: {e}")
                time.sleep(30)

if __name__ == "__main__":
    try:
        bot = AIScalpingTrader()
        bot.start_trading()
    except Exception as e:
        print(f"❌ Failed to start bot: {e}")
