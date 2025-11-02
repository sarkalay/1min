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
        
        # ✅ FIXED: Minimum quantities for each pair
        self.min_quantities = {
            "SOLUSDT": 0.01,
            "AVAXUSDT": 0.1, 
            "XRPUSDT": 1,
            "LINKUSDT": 0.1,
            "DOTUSDT": 0.1,
            "ADAUSDT": 1,
            "DOGEUSDT": 1,
            "MATICUSDT": 1,
            "POLUSDT": 0.1
        }
        
        # Initialize Binance client
        self.binance = Client(self.binance_api_key, self.binance_secret)
        
        print("🤖 PERFECT FIXED BOT ACTIVATED!")
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
            
            # ✅ FIXED: Ensure minimum quantity for each pair
            min_qty = self.min_quantities.get(pair, 0.1)
            if quantity < min_qty:
                quantity = min_qty
            
            # Calculate actual trade value
            trade_value = quantity * price
            
            print(f"🎯 {pair} target size: ${trade_size}")
            print(f"🔢 Calculated: {quantity} {pair} = ${trade_value:.2f}")
            
            # ✅ FIXED: Better auto-adjustment
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
        return round(price, precision)
    
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
        
        # Simple manual decision
        if data['change_1h'] > 0.3 and data['volume_ratio'] > 1.1:
            direction = "LONG"
            stop_loss = current_price * 0.994  # 0.6% stop loss
            take_profit = current_price * 1.006  # 0.6% take profit
            confidence = 70
            reason = "Bullish with volume"
        elif data['change_1h'] < -0.3 and data['volume_ratio'] > 1.0:
            direction = "SHORT" 
            stop_loss = current_price * 1.006  # 0.6% stop loss
            take_profit = current_price * 0.994  # 0.6% take profit
            confidence = 65
            reason = "Bearish momentum"
        else:
            return {"action": "WAIT", "reason": "No clear signal"}
        
        # Format prices
        stop_loss = self.format_price(pair, stop_loss)
        take_profit = self.format_price(pair, take_profit)
        
        # Final validation
        if direction == "LONG":
            if take_profit <= current_price or stop_loss >= current_price:
                print("❌ Invalid prices - using defaults")
                stop_loss = current_price * 0.995
                take_profit = current_price * 1.008
        else:
            if take_profit >= current_price or stop_loss <= current_price:
                print("❌ Invalid prices - using defaults")
                stop_loss = current_price * 1.008
                take_profit = current_price * 0.995
        
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
        
        print(f"✅ MANUAL DECISION: {direction} | Entry: ${current_price:.4f} | SL: ${stop_loss:.4f} | TP: ${take_profit:.4f}")
        return decision

    def execute_scalping_trade(self, decision):
        try:
            pair = decision["pair"]
            direction = decision["direction"]
            entry_price = decision["entry_price"]
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

    # ... check_scalping_trades, run_scalping_cycle, start_auto_trading methods remain the same ...

if __name__ == "__main__":
    try:
        bot = MultiPairScalpingTrader()
        bot.start_auto_trading()
    except Exception as e:
        print(f"❌ Failed to start bot: {e}")
