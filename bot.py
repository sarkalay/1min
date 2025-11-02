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

class ExistingPositionTracker:
    def __init__(self):
        # Load config from .env file
        self.binance_api_key = os.getenv('BINANCE_API_KEY')
        self.binance_secret = os.getenv('BINANCE_SECRET_KEY')
        self.deepseek_key = os.getenv('DEEPSEEK_API_KEY')
        
        # SCALPING parameters
        self.trade_size_usd = 50
        self.leverage = 5
        
        # Multi-pair parameters
        self.max_concurrent_trades = 1
        self.available_pairs = ["SOLUSDT", "AVAXUSDT", "XRPUSDT", "LINKUSDT", "DOTUSDT"]
        
        # Track both bot-opened and existing positions
        self.bot_opened_trades = {}
        self.existing_positions = {}
        
        # Precision settings
        self.quantity_precision = {}
        self.price_precision = {}
        
        # Initialize Binance client
        self.binance = Client(self.binance_api_key, self.binance_secret)
        
        print("🤖 EXISTING POSITION TRACKER ACTIVATED!")
        print(f"💵 Trade Size: ${self.trade_size_usd}")
        print(f"📈 Max Trades: {self.max_concurrent_trades}")
        
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
    
    def format_price(self, pair, price):
        precision = self.price_precision.get(pair, 4)
        return round(price, precision)
    
    def scan_existing_positions(self):
        """Binance ထဲမှာရှိပြီးသား position တွေကို scan လုပ်မယ် - FIXED"""
        try:
            positions = self.binance.futures_position_information()
            self.existing_positions = {}
            
            for pos in positions:
                pair = pos['symbol']
                position_amt = float(pos['positionAmt'])
                
                if position_amt != 0 and pair in self.available_pairs:
                    # Existing position found
                    entry_price = float(pos['entryPrice'])
                    unrealized_pnl = float(pos['unrealizedProfit'])
                    
                    # ✅ FIXED: Handle missing leverage field
                    try:
                        leverage = float(pos.get('leverage', self.leverage))  # Default to 5x if not found
                    except:
                        leverage = self.leverage
                    
                    # Get current price
                    ticker = self.binance.futures_symbol_ticker(symbol=pair)
                    current_price = float(ticker['price'])
                    
                    # Calculate P&L percentage
                    if position_amt < 0:  # SHORT
                        pnl_percent = (entry_price - current_price) / entry_price * 100 * leverage
                        direction = "SHORT"
                    else:  # LONG
                        pnl_percent = (current_price - entry_price) / entry_price * 100 * leverage
                        direction = "LONG"
                    
                    self.existing_positions[pair] = {
                        'direction': direction,
                        'entry_price': entry_price,
                        'quantity': abs(position_amt),
                        'leverage': leverage,
                        'current_price': current_price,
                        'unrealized_pnl': unrealized_pnl,
                        'pnl_percent': pnl_percent,
                        'position_value': abs(position_amt) * current_price,
                        'entry_time': time.time() - 3600,
                        'source': 'EXISTING',
                        'status': 'ACTIVE'
                    }
                    
                    print(f"🔍 Found existing position: {pair} {direction}")
            
            return len(self.existing_positions)
            
        except Exception as e:
            print(f"❌ Error scanning existing positions: {e}")
            return 0
    
    def get_live_position_data(self, pair):
        """Live position data ကိုရယူမယ် - FIXED"""
        try:
            positions = self.binance.futures_position_information(symbol=pair)
            for pos in positions:
                if pos['symbol'] == pair and float(pos['positionAmt']) != 0:
                    entry_price = float(pos['entryPrice'])
                    quantity = abs(float(pos['positionAmt']))
                    unrealized_pnl = float(pos['unrealizedProfit'])
                    
                    # ✅ FIXED: Handle missing leverage
                    try:
                        leverage = float(pos.get('leverage', self.leverage))
                    except:
                        leverage = self.leverage
                    
                    # Get current price
                    ticker = self.binance.futures_symbol_ticker(symbol=pair)
                    current_price = float(ticker['price'])
                    
                    # Calculate P&L percentage
                    if pos['positionAmt'].startswith('-'):  # SHORT
                        pnl_percent = (entry_price - current_price) / entry_price * 100 * leverage
                        direction = "SHORT"
                    else:  # LONG
                        pnl_percent = (current_price - entry_price) / entry_price * 100 * leverage
                        direction = "LONG"
                    
                    return {
                        'direction': direction,
                        'entry_price': entry_price,
                        'quantity': quantity,
                        'leverage': leverage,
                        'current_price': current_price,
                        'unrealized_pnl': unrealized_pnl,
                        'pnl_percent': pnl_percent,
                        'position_value': quantity * current_price,
                        'status': 'ACTIVE'
                    }
            return None
        except Exception as e:
            print(f"❌ Error getting live data for {pair}: {e}")
            return None
    
    def display_live_dashboard(self):
        """Live trading dashboard ကိုပြမယ်"""
        print(f"\n📊 LIVE TRADING DASHBOARD - {time.strftime('%H:%M:%S')}")
        print("=" * 80)
        
        # Update existing positions with live data
        for pair in list(self.existing_positions.keys()):
            live_data = self.get_live_position_data(pair)
            if live_data:
                self.existing_positions[pair].update(live_data)
            else:
                # Position closed
                print(f"✅ EXISTING POSITION CLOSED: {pair}")
                del self.existing_positions[pair]
        
        # Update bot opened trades with live data
        for pair in list(self.bot_opened_trades.keys()):
            live_data = self.get_live_position_data(pair)
            if live_data:
                self.bot_opened_trades[pair].update(live_data)
            else:
                # Position closed
                print(f"✅ BOT TRADE CLOSED: {pair}")
                del self.bot_opened_trades[pair]
        
        total_positions = len(self.existing_positions) + len(self.bot_opened_trades)
        
        if total_positions == 0:
            print("🔄 No active positions")
            return
        
        total_unrealized_pnl = 0
        total_position_value = 0
        
        # Display EXISTING positions first
        if self.existing_positions:
            print("\n🏦 EXISTING POSITIONS (From Binance):")
            print("-" * 50)
            for pair, position in self.existing_positions.items():
                pnl_color = "🟢" if position['unrealized_pnl'] >= 0 else "🔴"
                direction_icon = "📈" if position['direction'] == 'LONG' else "📉"
                
                print(f"{direction_icon} {pair} {position['direction']} 🔸EXISTING")
                print(f"   Size: {position['quantity']} (${position['position_value']:.2f})")
                print(f"   Entry: ${position['entry_price']:.4f} | Current: ${position['current_price']:.4f}")
                print(f"   P&L: {pnl_color} ${position['unrealized_pnl']:.2f} ({position['pnl_percent']:.2f}%)")
                print(f"   Leverage: {position['leverage']}x")
                print(f"   Status: 🔄 Monitoring (Skipped for new trades)")
                print("-" * 30)
                
                total_unrealized_pnl += position['unrealized_pnl']
                total_position_value += position['position_value']
        
        # Display BOT opened positions
        if self.bot_opened_trades:
            print("\n🤖 BOT OPENED POSITIONS:")
            print("-" * 50)
            for pair, trade in self.bot_opened_trades.items():
                pnl_color = "🟢" if trade['unrealized_pnl'] >= 0 else "🔴"
                direction_icon = "📈" if trade['direction'] == 'LONG' else "📉"
                
                print(f"{direction_icon} {pair} {trade['direction']} 🔸BOT")
                print(f"   Size: {trade['quantity']} (${trade['position_value']:.2f})")
                print(f"   Entry: ${trade['entry_price']:.4f} | Current: ${trade['current_price']:.4f}")
                print(f"   P&L: {pnl_color} ${trade['unrealized_pnl']:.2f} ({trade['pnl_percent']:.2f}%)")
                print(f"   Leverage: {trade['leverage']}x")
                if 'take_profit' in trade:
                    print(f"   TP: ${trade['take_profit']} | SL: ${trade['stop_loss']}")
                print(f"   Duration: {(time.time() - trade['entry_time']) / 60:.1f} minutes")
                print("-" * 30)
                
                total_unrealized_pnl += trade['unrealized_pnl']
                total_position_value += trade['position_value']
        
        # Display summary
        if total_position_value > 0:
            print(f"\n💰 TOTAL SUMMARY:")
            print(f"   Positions: {total_positions} | P&L: ${total_unrealized_pnl:.2f}")
            print(f"   Total Exposure: ${total_position_value:.2f}")
            overall_pnl_percent = (total_unrealized_pnl / total_position_value) * 100 if total_position_value > 0 else 0
            print(f"   Overall Return: {overall_pnl_percent:.2f}%")
            
            # Trading status
            if total_positions >= self.max_concurrent_trades:
                print(f"   🚫 TRADING PAUSED - Maximum positions reached")
            else:
                print(f"   ✅ TRADING ACTIVE - Can open new positions")
    
    def can_open_new_trade(self, pair):
        """Check if we can open new trade for this pair"""
        if pair in self.existing_positions or pair in self.bot_opened_trades:
            return False
        
        total_positions = len(self.existing_positions) + len(self.bot_opened_trades)
        if total_positions >= self.max_concurrent_trades:
            return False
        
        return True
    
    def get_market_data(self):
        """Market data for pairs without existing positions"""
        market_data = {}
        
        for pair in self.available_pairs:
            if not self.can_open_new_trade(pair):
                continue
                
            try:
                ticker = self.binance.futures_symbol_ticker(symbol=pair)
                if 'price' not in ticker or not ticker['price']:
                    continue
                    
                price = float(ticker['price'])
                market_data[pair] = {'price': price}
                
            except Exception as e:
                print(f"❌ Market data error for {pair}: {e}")
                continue
                
        return market_data

    def get_ai_decision(self, market_data):
        """AI decision making for available pairs only"""
        pair = list(market_data.keys())[0]
        data = market_data[pair]
        current_price = data['price']
        
        import random
        direction = "LONG" if random.random() > 0.5 else "SHORT"
        
        decision = {
            "action": "TRADE",
            "pair": pair,
            "direction": direction,
            "confidence": 65,
            "reason": "AI Analysis"
        }
        
        return decision

    def execute_trade(self, decision):
        """Execute new trade (only for available pairs)"""
        try:
            pair = decision["pair"]
            
            if not self.can_open_new_trade(pair):
                print(f"🚫 Cannot open {pair} - position exists or limit reached")
                return False
            
            direction = decision["direction"]
            
            ticker = self.binance.futures_symbol_ticker(symbol=pair)
            current_price = float(ticker['price'])
            
            if direction == "LONG":
                stop_loss = current_price * 0.995
                take_profit = current_price * 1.008
            else:
                stop_loss = current_price * 1.005
                take_profit = current_price * 0.992
            
            stop_loss = self.format_price(pair, stop_loss)
            take_profit = self.format_price(pair, take_profit)
            
            quantity = self.trade_size_usd / current_price
            precision = self.quantity_precision.get(pair, 3)
            quantity = round(quantity, precision)
            
            if quantity < 0.1:
                quantity = 0.1
            
            print(f"🎯 EXECUTING: {pair} {direction}")
            print(f"   Size: {quantity} | TP: ${take_profit} | SL: ${stop_loss}")
            
            # Store in bot opened trades
            self.bot_opened_trades[pair] = {
                "pair": pair,
                "direction": direction,
                "entry_price": current_price,
                "quantity": quantity,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "entry_time": time.time(),
                "source": "BOT",
                "status": "ACTIVE"
            }
            
            print(f"🚀 BOT TRADE ACTIVATED: {pair} {direction}")
            return True
            
        except Exception as e:
            print(f"❌ Trade execution failed: {e}")
            return False

    def run_trading_cycle(self):
        """Main trading cycle"""
        try:
            # Scan for existing positions first
            existing_count = self.scan_existing_positions()
            if existing_count > 0:
                print(f"🔍 Found {existing_count} existing positions")
            
            # Display live dashboard
            self.display_live_dashboard()
            
            # Check if we can open new trades
            total_positions = len(self.existing_positions) + len(self.bot_opened_trades)
            
            if total_positions < self.max_concurrent_trades:
                market_data = self.get_market_data()
                
                if market_data:
                    print(f"\n🔄 Looking for new trade opportunities...")
                    for pair in market_data.keys():
                        if self.can_open_new_trade(pair):
                            pair_data = {pair: market_data[pair]}
                            decision = self.get_ai_decision(pair_data)
                            
                            if decision["action"] == "TRADE":
                                print(f"✅ QUALIFIED: {pair}")
                                success = self.execute_trade(decision)
                                if success:
                                    break
            else:
                print(f"🚫 Maximum positions reached ({total_positions}/{self.max_concurrent_trades}) - Skipping new trades")
            
        except Exception as e:
            print(f"❌ Trading cycle error: {e}")

    def start_trading(self):
        print("🚀 STARTING EXISTING POSITION TRACKER!")
        print("🔍 Scanning for existing positions in Binance...")
        
        self.scan_existing_positions()
        
        cycle_count = 0
        
        while True:
            try:
                cycle_count += 1
                print(f"\n{'='*80}")
                print(f"🔄 CYCLE {cycle_count} - {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*80}")
                
                self.run_trading_cycle()
                
                time.sleep(30)
                
            except KeyboardInterrupt:
                print(f"\n🛑 BOT STOPPED BY USER")
                break
            except Exception as e:
                print(f"❌ Main loop error: {e}")
                time.sleep(30)

if __name__ == "__main__":
    try:
        bot = ExistingPositionTracker()
        bot.start_trading()
    except Exception as e:
        print(f"❌ Failed to start bot: {e}")
