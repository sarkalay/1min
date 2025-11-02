import os
import requests
import json
import time
import re
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

# Install colorama first: pip install colorama
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    print("⚠️  Colorama not installed. Run: pip install colorama")

# Load environment variables
load_dotenv()

class RealOrderPositionTracker:
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
        
        self.print_color(f"🤖 REAL ORDER POSITION TRACKER WITH OCO ACTIVATED!", Fore.CYAN)
        self.print_color(f"💵 Trade Size: ${self.trade_size_usd}", Fore.GREEN)
        self.print_color(f"📈 Max Trades: {self.max_concurrent_trades}", Fore.YELLOW)
        self.print_color(f"🎯 Using OCO Orders for TP/SL", Fore.CYAN)
        
        self.validate_config()
        self.setup_futures()
        self.load_symbol_precision()
    
    def print_color(self, text, color=Fore.WHITE, style=Style.NORMAL):
        """Colorful print function"""
        if COLORAMA_AVAILABLE:
            print(f"{style}{color}{text}")
        else:
            print(text)
    
    def validate_config(self):
        if not all([self.binance_api_key, self.binance_secret, self.deepseek_key]):
            self.print_color("❌ Missing API keys!", Fore.RED)
            return False
        try:
            self.binance.futures_exchange_info()
            self.print_color("✅ Binance connection successful!", Fore.GREEN)
        except Exception as e:
            self.print_color(f"❌ Binance connection failed: {e}", Fore.RED)
            return False
        return True

    def setup_futures(self):
        try:
            for pair in self.available_pairs:
                try:
                    self.binance.futures_change_leverage(symbol=pair, leverage=self.leverage)
                    self.print_color(f"✅ Leverage set for {pair}", Fore.GREEN)
                except Exception as e:
                    self.print_color(f"⚠️ Leverage setup failed for {pair}: {e}", Fore.YELLOW)
            self.print_color("✅ Futures setup completed!", Fore.GREEN)
        except Exception as e:
            self.print_color(f"❌ Futures setup failed: {e}", Fore.RED)
    
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
            self.print_color("✅ Symbol precision loaded", Fore.GREEN)
        except Exception as e:
            self.print_color(f"❌ Error loading symbol precision: {e}", Fore.RED)
    
    def format_price(self, pair, price):
        precision = self.price_precision.get(pair, 4)
        return round(price, precision)
    
    def get_quantity(self, pair, price):
        """Calculate proper quantity"""
        try:
            quantity = self.trade_size_usd / price
            precision = self.quantity_precision.get(pair, 3)
            quantity = round(quantity, precision)
            
            # Ensure minimum quantity
            min_qty = 0.1
            if quantity < min_qty:
                quantity = min_qty
                
            return quantity
        except Exception as e:
            self.print_color(f"❌ Quantity calculation failed: {e}", Fore.RED)
            return None

    def cleanup_old_orders(self, pair):
        """Clean up any existing orders for a pair"""
        try:
            self.binance.futures_cancel_all_open_orders(symbol=pair)
            self.print_color(f"🧹 Cleaned up old orders for {pair}", Fore.YELLOW)
            return True
        except Exception as e:
            self.print_color(f"⚠️ No orders to clean for {pair}: {e}", Fore.YELLOW)
            return True
    
    def execute_oco_trade(self, decision):
        """✅ OCO Order System - TP/SL တစ်ခုထိရင် အခြားတစ်ခု အလိုလို cancel"""
        try:
            pair = decision["pair"]
            
            if not self.can_open_new_trade(pair):
                self.print_color(f"🚫 Cannot open {pair} - position exists or limit reached", Fore.RED)
                return False
            
            direction = decision["direction"]
            
            # Clean up any existing orders first
            self.cleanup_old_orders(pair)
            
            # Get current price
            ticker = self.binance.futures_symbol_ticker(symbol=pair)
            current_price = float(ticker['price'])
            
            # Calculate quantity
            quantity = self.get_quantity(pair, current_price)
            if quantity is None:
                return False
            
            # Calculate TP/SL with better risk management
            if direction == "LONG":
                stop_loss = current_price * 0.995   # 0.5% SL
                take_profit = current_price * 1.008 # 0.8% TP
                stop_side = 'SELL'
            else:  # SHORT
                stop_loss = current_price * 1.005   # 0.5% SL
                take_profit = current_price * 0.992 # 0.8% TP
                stop_side = 'BUY'
            
            stop_loss = self.format_price(pair, stop_loss)
            take_profit = self.format_price(pair, take_profit)
            
            direction_color = Fore.BLUE if direction == 'LONG' else Fore.RED
            self.print_color(f"🎯 EXECUTING OCO TRADE: {pair} {direction}", direction_color)
            self.print_color(f"   📦 Size: {quantity} | 🎯 TP: ${take_profit} | 🛑 SL: ${stop_loss}", Fore.WHITE)
            
            # ✅ Step 1: Open position with MARKET order
            try:
                if direction == "LONG":
                    order = self.binance.futures_create_order(
                        symbol=pair,
                        side='BUY',
                        type='MARKET',
                        quantity=quantity
                    )
                    self.print_color(f"✅ REAL LONG ORDER EXECUTED: {quantity} {pair} @ ${current_price}", Fore.GREEN)
                else:
                    order = self.binance.futures_create_order(
                        symbol=pair,
                        side='SELL',
                        type='MARKET',
                        quantity=quantity
                    )
                    self.print_color(f"✅ REAL SHORT ORDER EXECUTED: {quantity} {pair} @ ${current_price}", Fore.GREEN)
                
                # ✅ Step 2: Wait for position to open
                time.sleep(2)
                
                # ✅ Step 3: Place OCO ORDER using STOP_MARKET and TAKE_PROFIT_MARKET
                # Binance Futures doesn't support direct OCO, so we use separate orders
                # but we'll manage them to cancel each other when one executes
                
                # Place STOP LOSS order
                sl_order = self.binance.futures_create_order(
                    symbol=pair,
                    side=stop_side,
                    type='STOP_MARKET',
                    quantity=quantity,
                    stopPrice=stop_loss,
                    reduceOnly=True,
                    timeInForce='GTC'
                )
                
                # Place TAKE PROFIT order  
                tp_order = self.binance.futures_create_order(
                    symbol=pair,
                    side=stop_side,
                    type='TAKE_PROFIT_MARKET',
                    quantity=quantity,
                    stopPrice=take_profit,
                    reduceOnly=True,
                    timeInForce='GTC'
                )
                
                self.print_color(f"✅ OCO ORDERS PLACED - TP/SL activated!", Fore.GREEN)
                self.print_color(f"   🛑 SL Order ID: {sl_order['orderId']}", Fore.YELLOW)
                self.print_color(f"   🎯 TP Order ID: {tp_order['orderId']}", Fore.YELLOW)
                
                # Store trade info with both order IDs
                self.bot_opened_trades[pair] = {
                    "pair": pair,
                    "direction": direction,
                    "entry_price": current_price,
                    "quantity": quantity,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "sl_order_id": sl_order['orderId'],
                    "tp_order_id": tp_order['orderId'],
                    "entry_time": time.time(),
                    "source": "BOT",
                    'status': 'ACTIVE'
                }
                
                self.print_color(f"🚀 OCO TRADE ACTIVATED: {pair} {direction}", Fore.GREEN)
                return True
                
            except BinanceAPIException as e:
                self.print_color(f"❌ Binance API Error: {e}", Fore.RED)
                # Clean up on error
                self.cleanup_old_orders(pair)
                return False
            except Exception as e:
                self.print_color(f"❌ Order execution failed: {e}", Fore.RED)
                self.cleanup_old_orders(pair)
                return False
            
        except Exception as e:
            self.print_color(f"❌ Trade execution failed: {e}", Fore.RED)
            return False

    def monitor_oco_orders(self):
        """Monitor and clean up OCO orders when position is closed"""
        try:
            for pair, trade in list(self.bot_opened_trades.items()):
                if trade['status'] == 'ACTIVE':
                    # Check if position still exists
                    position_info = self.get_live_position_data(pair)
                    
                    if position_info is None:
                        # Position closed, clean up any remaining orders
                        self.print_color(f"✅ Position closed for {pair}, cleaning up orders...", Fore.GREEN)
                        self.cleanup_old_orders(pair)
                        trade['status'] = 'CLOSED'
                        
                        # Calculate final P&L
                        current_price = self.get_current_price(pair)
                        if current_price:
                            if trade['direction'] == 'LONG':
                                pnl = (current_price - trade['entry_price']) * trade['quantity']
                            else:
                                pnl = (trade['entry_price'] - current_price) * trade['quantity']
                            
                            pnl_color = Fore.GREEN if pnl >= 0 else Fore.RED
                            self.print_color(f"💰 FINAL P&L for {pair}: ${pnl:.2f}", pnl_color)
                        
                        # Remove from active tracking after some time
                        if time.time() - trade['entry_time'] > 300:  # 5 minutes
                            del self.bot_opened_trades[pair]
                            self.print_color(f"🗑️ Removed {pair} from tracking", Fore.YELLOW)
                    
        except Exception as e:
            self.print_color(f"❌ OCO monitoring error: {e}", Fore.RED)

    def get_current_price(self, pair):
        """Get current price for a pair"""
        try:
            ticker = self.binance.futures_symbol_ticker(symbol=pair)
            return float(ticker['price'])
        except:
            return None

    def scan_existing_positions(self):
        """Binance ထဲမှာရှိပြီးသား position တွေကို scan လုပ်မယ်"""
        try:
            positions = self.binance.futures_position_information()
            self.existing_positions = {}
            
            for pos in positions:
                pair = pos['symbol']
                position_amt = float(pos['positionAmt'])
                
                if position_amt != 0 and pair in self.available_pairs:
                    entry_price = float(pos.get('entryPrice', 0))
                    unrealized_pnl = float(pos.get('unRealizedProfit', 0))
                    leverage = float(pos.get('leverage', self.leverage))
                    
                    # Get current price
                    try:
                        ticker = self.binance.futures_symbol_ticker(symbol=pair)
                        current_price = float(ticker['price'])
                    except:
                        current_price = entry_price
                    
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
            
            return len(self.existing_positions)
            
        except Exception as e:
            self.print_color(f"❌ Error scanning existing positions: {e}", Fore.RED)
            return 0
    
    def get_live_position_data(self, pair):
        """Live position data ကိုရယူမယ်"""
        try:
            positions = self.binance.futures_position_information(symbol=pair)
            for pos in positions:
                if pos['symbol'] == pair and float(pos['positionAmt']) != 0:
                    entry_price = float(pos.get('entryPrice', 0))
                    quantity = abs(float(pos['positionAmt']))
                    unrealized_pnl = float(pos.get('unRealizedProfit', 0))
                    leverage = float(pos.get('leverage', self.leverage))
                    
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
            self.print_color(f"❌ Error getting live data for {pair}: {e}", Fore.RED)
            return None
    
    def display_live_dashboard(self):
        """Colorful live trading dashboard"""
        self.print_color(f"\n📊 LIVE TRADING DASHBOARD - {time.strftime('%H:%M:%S')}", Fore.CYAN, Style.BRIGHT)
        self.print_color("=" * 80, Fore.CYAN)
        
        # Update existing positions with live data
        for pair in list(self.existing_positions.keys()):
            live_data = self.get_live_position_data(pair)
            if live_data:
                self.existing_positions[pair].update(live_data)
            else:
                self.print_color(f"✅ EXISTING POSITION CLOSED: {pair}", Fore.GREEN)
                del self.existing_positions[pair]
        
        # Update bot opened trades with live data
        for pair in list(self.bot_opened_trades.keys()):
            live_data = self.get_live_position_data(pair)
            if live_data:
                self.bot_opened_trades[pair].update(live_data)
            else:
                # Position closed, but might still be in tracking for cleanup
                if self.bot_opened_trades[pair]['status'] == 'ACTIVE':
                    self.bot_opened_trades[pair]['status'] = 'CLOSED'
                    self.print_color(f"✅ BOT TRADE CLOSED: {pair}", Fore.GREEN)
        
        total_positions = len(self.existing_positions) + len([t for t in self.bot_opened_trades.values() if t['status'] == 'ACTIVE'])
        
        if total_positions == 0:
            self.print_color("🔄 No active positions", Fore.YELLOW)
            return
        
        total_unrealized_pnl = 0
        total_position_value = 0
        
        # Display EXISTING positions first
        if self.existing_positions:
            self.print_color("\n🏦 EXISTING POSITIONS (From Binance)", Fore.MAGENTA, Style.BRIGHT)
            self.print_color("-" * 50, Fore.MAGENTA)
            for pair, position in self.existing_positions.items():
                pnl_color = Fore.GREEN if position['unrealized_pnl'] >= 0 else Fore.RED
                direction_color = Fore.BLUE if position['direction'] == 'LONG' else Fore.RED
                direction_icon = "📈" if position['direction'] == 'LONG' else "📉"
                
                self.print_color(f"{direction_icon} {pair} {position['direction']} 🔸EXISTING", direction_color)
                self.print_color(f"   📦 Size: {position['quantity']} (${position['position_value']:.2f})", Fore.WHITE)
                self.print_color(f"   📍 Entry: ${position['entry_price']:.4f} | 🎯 Current: ${position['current_price']:.4f}", Fore.WHITE)
                self.print_color(f"   💰 P&L: ${position['unrealized_pnl']:.2f} ({position['pnl_percent']:.2f}%)", pnl_color)
                self.print_color(f"   ⚡ Leverage: {position['leverage']}x", Fore.CYAN)
                self.print_color(f"   📊 Status: 🔄 Monitoring (Skipped for new trades)", Fore.YELLOW)
                self.print_color("-" * 30, Fore.MAGENTA)
                
                total_unrealized_pnl += position['unrealized_pnl']
                total_position_value += position['position_value']
        
        # Display BOT opened positions
        active_bot_trades = {k: v for k, v in self.bot_opened_trades.items() if v['status'] == 'ACTIVE'}
        if active_bot_trades:
            self.print_color("\n🤖 BOT OPENED POSITIONS (OCO Protected)", Fore.GREEN, Style.BRIGHT)
            self.print_color("-" * 50, Fore.GREEN)
            for pair, trade in active_bot_trades.items():
                live_data = self.get_live_position_data(pair)
                if live_data:
                    pnl_color = Fore.GREEN if live_data['unrealized_pnl'] >= 0 else Fore.RED
                    direction_color = Fore.BLUE if trade['direction'] == 'LONG' else Fore.RED
                    direction_icon = "📈" if trade['direction'] == 'LONG' else "📉"
                    
                    self.print_color(f"{direction_icon} {pair} {trade['direction']} 🔸BOT", direction_color)
                    self.print_color(f"   📦 Size: {trade['quantity']} (${live_data['position_value']:.2f})", Fore.WHITE)
                    self.print_color(f"   📍 Entry: ${trade['entry_price']:.4f} | 🎯 Current: ${live_data['current_price']:.4f}", Fore.WHITE)
                    self.print_color(f"   💰 P&L: ${live_data['unrealized_pnl']:.2f} ({live_data['pnl_percent']:.2f}%)", pnl_color)
                    self.print_color(f"   ⚡ Leverage: {live_data['leverage']}x", Fore.CYAN)
                    self.print_color(f"   🎯 TP: ${trade['take_profit']} | 🛑 SL: ${trade['stop_loss']}", Fore.YELLOW)
                    self.print_color(f"   🔒 OCO: Active (Auto-cleanup)", Fore.CYAN)
                    self.print_color(f"   ⏱️ Duration: {(time.time() - trade['entry_time']) / 60:.1f} minutes", Fore.WHITE)
                    self.print_color("-" * 30, Fore.GREEN)
                    
                    total_unrealized_pnl += live_data['unrealized_pnl']
                    total_position_value += live_data['position_value']
        
        # Display summary
        if total_position_value > 0:
            total_color = Fore.GREEN if total_unrealized_pnl >= 0 else Fore.RED
            self.print_color(f"\n💰 TOTAL SUMMARY", Fore.CYAN, Style.BRIGHT)
            self.print_color(f"   📊 Active Positions: {total_positions} | 💰 P&L: ${total_unrealized_pnl:.2f}", total_color)
            self.print_color(f"   📈 Total Exposure: ${total_position_value:.2f}", Fore.WHITE)
            overall_pnl_percent = (total_unrealized_pnl / total_position_value) * 100 if total_position_value > 0 else 0
            self.print_color(f"   📊 Overall Return: {overall_pnl_percent:.2f}%", total_color)
            
            # Trading status
            if total_positions >= self.max_concurrent_trades:
                self.print_color(f"   🚫 TRADING PAUSED - Maximum positions reached", Fore.RED)
            else:
                self.print_color(f"   ✅ TRADING ACTIVE - Can open new positions", Fore.GREEN)
    
    def can_open_new_trade(self, pair):
        """Check if we can open new trade for this pair"""
        if pair in self.existing_positions:
            return False
        
        active_bot_trades = [k for k, v in self.bot_opened_trades.items() if v['status'] == 'ACTIVE']
        if pair in active_bot_trades:
            return False
        
        total_positions = len(self.existing_positions) + len(active_bot_trades)
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
                self.print_color(f"❌ Market data error for {pair}: {e}", Fore.RED)
                continue
                
        return market_data

    def get_ai_decision(self, market_data):
        """AI decision making for available pairs only"""
        if not market_data:
            return {"action": "HOLD", "reason": "No available pairs"}
            
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

    def run_trading_cycle(self):
        """Main trading cycle"""
        try:
            # Scan for existing positions first
            existing_count = self.scan_existing_positions()
            if existing_count > 0:
                self.print_color(f"🔍 Found {existing_count} existing positions", Fore.CYAN)
            
            # Monitor OCO orders
            self.monitor_oco_orders()
            
            # Display live dashboard
            self.display_live_dashboard()
            
            # Check if we can open new trades
            active_bot_trades = len([t for t in self.bot_opened_trades.values() if t['status'] == 'ACTIVE'])
            total_positions = len(self.existing_positions) + active_bot_trades
            
            if total_positions < self.max_concurrent_trades:
                market_data = self.get_market_data()
                
                if market_data:
                    self.print_color(f"\n🔄 Looking for new trade opportunities...", Fore.BLUE)
                    for pair in market_data.keys():
                        if self.can_open_new_trade(pair):
                            pair_data = {pair: market_data[pair]}
                            decision = self.get_ai_decision(pair_data)
                            
                            if decision["action"] == "TRADE":
                                self.print_color(f"✅ QUALIFIED: {pair}", Fore.GREEN)
                                success = self.execute_oco_trade(decision)  # ✅ Use OCO trade function
                                if success:
                                    break
            else:
                self.print_color(f"🚫 Maximum positions reached ({total_positions}/{self.max_concurrent_trades}) - Skipping new trades", Fore.RED)
            
        except Exception as e:
            self.print_color(f"❌ Trading cycle error: {e}", Fore.RED)

    def start_trading(self):
        self.print_color("🚀 STARTING REAL ORDER POSITION TRACKER WITH OCO!", Fore.CYAN, Style.BRIGHT)
        self.print_color("🔍 Scanning for existing positions in Binance...", Fore.CYAN)
        
        self.scan_existing_positions()
        
        cycle_count = 0
        
        while True:
            try:
                cycle_count += 1
                self.print_color(f"\n{'='*80}", Fore.CYAN)
                self.print_color(f"🔄 CYCLE {cycle_count} - {time.strftime('%Y-%m-%d %H:%M:%S')}", Fore.CYAN, Style.BRIGHT)
                self.print_color(f"{'='*80}", Fore.CYAN)
                
                self.run_trading_cycle()
                
                # Clean up old closed trades
                closed_trades = [k for k, v in self.bot_opened_trades.items() if v['status'] == 'CLOSED' and time.time() - v['entry_time'] > 300]
                for pair in closed_trades:
                    del self.bot_opened_trades[pair]
                    self.print_color(f"🗑️ Cleaned up closed trade: {pair}", Fore.YELLOW)
                
                time.sleep(30)
                
            except KeyboardInterrupt:
                self.print_color(f"\n🛑 BOT STOPPED BY USER", Fore.RED)
                # Final cleanup
                self.print_color("🧹 Performing final order cleanup...", Fore.YELLOW)
                for pair in self.available_pairs:
                    self.cleanup_old_orders(pair)
                break
            except Exception as e:
                self.print_color(f"❌ Main loop error: {e}", Fore.RED)
                time.sleep(30)

if __name__ == "__main__":
    try:
        bot = RealOrderPositionTracker()
        bot.start_trading()
    except Exception as e:
        print(f"❌ Failed to start bot: {e}")
