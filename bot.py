import os
import requests
import json
import time
import re
import math
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
from datetime import datetime
import pytz

# Install required packages first: 
# pip install colorama python-binance python-dotenv numpy pytz

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
        
        # Thailand timezone
        self.thailand_tz = pytz.timezone('Asia/Bangkok')
        
        # SCALPING parameters
        self.trade_size_usd = 50
        self.leverage = 5
        
        # Multi-pair parameters
        self.max_concurrent_trades = 1
        self.available_pairs = ["SOLUSDT", "AVAXUSDT", "XRPUSDT", "LINKUSDT", "DOTUSDT"]
        
        # Track both bot-opened and existing positions
        self.bot_opened_trades = {}
        self.existing_positions = {}
        
        # Trade history
        self.trade_history_file = "trade_history.json"
        self.trade_history = self.load_trade_history()
        
        # Precision settings
        self.quantity_precision = {}
        self.price_precision = {}
        
        # Initialize Binance client
        self.binance = Client(self.binance_api_key, self.binance_secret)
        
        self.print_color(f"🤖 REAL ORDER POSITION TRACKER ACTIVATED!", Fore.CYAN)
        self.print_color(f"💵 Trade Size: ${self.trade_size_usd}", Fore.GREEN)
        self.print_color(f"📈 Max Trades: {self.max_concurrent_trades}", Fore.YELLOW)
        self.print_color(f"🧠 Using DeepSeek AI for Trading Decisions", Fore.MAGENTA)
        self.print_color(f"🇹🇭 Timezone: Thailand (Asia/Bangkok)", Fore.BLUE)
        self.print_color(f"📊 Trade History: {self.trade_history_file}", Fore.CYAN)
        
        self.validate_config()
        self.setup_futures()
        self.load_symbol_precision()
    
    def load_trade_history(self):
        """Load trade history from JSON file"""
        try:
            if os.path.exists(self.trade_history_file):
                with open(self.trade_history_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            self.print_color(f"❌ Error loading trade history: {e}", Fore.RED)
            return []
    
    def save_trade_history(self):
        """Save trade history to JSON file"""
        try:
            with open(self.trade_history_file, 'w') as f:
                json.dump(self.trade_history, f, indent=2)
        except Exception as e:
            self.print_color(f"❌ Error saving trade history: {e}", Fore.RED)
    
    def add_trade_to_history(self, trade_data):
        """Add completed trade to history"""
        try:
            trade_data['close_time'] = self.get_thailand_time()
            trade_data['close_timestamp'] = time.time()
            self.trade_history.append(trade_data)
            
            # Keep only last 100 trades
            if len(self.trade_history) > 100:
                self.trade_history = self.trade_history[-100:]
            
            self.save_trade_history()
            self.print_color(f"📝 Trade saved to history: {trade_data['pair']} {trade_data['direction']}", Fore.CYAN)
        except Exception as e:
            self.print_color(f"❌ Error adding trade to history: {e}", Fore.RED)
    
    def show_trade_history(self, limit=10):
        """Display recent trade history"""
        if not self.trade_history:
            self.print_color("📊 No trade history found", Fore.YELLOW)
            return
        
        self.print_color(f"\n📊 TRADE HISTORY (Last {min(limit, len(self.trade_history))} trades)", Fore.CYAN)
        self.print_color("=" * 80, Fore.CYAN)
        
        for i, trade in enumerate(reversed(self.trade_history[-limit:])):
            pnl = trade.get('pnl', 0)
            pnl_color = Fore.GREEN if pnl > 0 else Fore.RED if pnl < 0 else Fore.YELLOW
            status_color = Fore.GREEN if trade.get('status') == 'CLOSED' else Fore.RED
            
            self.print_color(f"{i+1}. {trade['pair']} {trade['direction']} | "
                           f"Entry: ${trade.get('entry_price', 0):.4f} | "
                           f"Exit: ${trade.get('exit_price', 0):.4f} | "
                           f"P&L: ${pnl:.2f}", pnl_color)
            self.print_color(f"   TP: ${trade.get('take_profit', 0):.4f} | "
                           f"SL: ${trade.get('stop_loss', 0):.4f} | "
                           f"Status: {trade.get('status', 'UNKNOWN')} | "
                           f"Time: {trade.get('close_time', 'N/A')}", status_color)
    
    def get_thailand_time(self):
        """Get current Thailand time"""
        now_utc = datetime.now(pytz.utc)
        thailand_time = now_utc.astimezone(self.thailand_tz)
        return thailand_time.strftime('%Y-%m-%d %H:%M:%S')
    
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
        precision = self.price_precision.get(pair, 3)
        return round(price, precision)
    
    def get_quantity(self, pair, price):
        """Calculate proper quantity for $50 position"""
        try:
            # Simple fixed quantity approach based on pair
            fixed_quantities = {
                "SOLUSDT": 0.3,    # ~$50 at current price
                "AVAXUSDT": 2.0,   # ~$50
                "XRPUSDT": 80.0,   # ~$50  
                "LINKUSDT": 3.0,   # ~$50
                "DOTUSDT": 4.0     # ~$50
            }
            
            quantity = fixed_quantities.get(pair)
            
            if quantity is None or quantity <= 0:
                # Fallback calculation
                quantity = round(self.trade_size_usd / price, 2)
                quantity = max(quantity, 0.01)  # Minimum safety
                
            # Apply precision
            precision = self.quantity_precision.get(pair, 3)
            quantity = round(quantity, precision)
            
            # Final validation
            if quantity <= 0:
                self.print_color(f"❌ Invalid quantity: {quantity} for {pair}", Fore.RED)
                return None
                
            actual_value = quantity * price
            
            self.print_color(f"📦 Quantity for {pair}: {quantity} = ${actual_value:.2f}", Fore.CYAN)
            
            return quantity
            
        except Exception as e:
            self.print_color(f"❌ Quantity calculation failed: {e}", Fore.RED)
            return None

    def parse_ai_response(self, text):
        """Parse AI response and extract trading decision"""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                decision_data = json.loads(json_str)
                
                direction = decision_data.get('direction', 'HOLD').upper()
                confidence = decision_data.get('confidence', 50)
                reason = decision_data.get('reason', 'AI Analysis')
                
                # Validate direction
                if direction not in ['LONG', 'SHORT', 'HOLD']:
                    direction = 'HOLD'
                
                # Validate confidence
                try:
                    confidence = float(confidence)
                    if confidence < 0 or confidence > 100:
                        confidence = 50
                except:
                    confidence = 50
                
                return direction, confidence, reason
            
            return 'HOLD', 50, 'No valid JSON found'
            
        except Exception as e:
            self.print_color(f"❌ AI response parsing failed: {e}", Fore.RED)
            return 'HOLD', 50, 'Parsing failed'

    def get_deepseek_analysis(self, pair, market_data):
        """Get AI analysis from DeepSeek API"""
        try:
            if not self.deepseek_key:
                self.print_color("❌ DeepSeek API key not found", Fore.RED)
                return "HOLD", 0, "No API key"
            
            current_price = market_data['current_price']
            
            prompt = f"""
            Analyze {pair} at ${current_price:.4f} for scalping.
            Respond with JSON: {{"direction": "LONG|SHORT|HOLD", "confidence": 65, "reason": "brief explanation"}}
            """
            
            headers = {
                "Authorization": f"Bearer {self.deepseek_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a crypto trading analyst. Respond with valid JSON only."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 300
            }
            
            self.print_color(f"🧠 Consulting DeepSeek AI for {pair}...", Fore.MAGENTA)
            
            response = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content'].strip()
                
                # Parse the AI response
                direction, confidence, reason = self.parse_ai_response(ai_response)
                
                self.print_color(f"✅ AI Analysis: {direction} ({confidence}%)", Fore.MAGENTA)
                
                return direction, confidence, reason
                
            else:
                self.print_color(f"❌ DeepSeek API error: {response.status_code}", Fore.RED)
                return "HOLD", 0, f"API Error"
                
        except Exception as e:
            self.print_color(f"❌ DeepSeek analysis failed: {e}", Fore.RED)
            return "HOLD", 0, f"Error"

    def get_price_history(self, pair, limit=10):
        """Get recent price history for analysis"""
        try:
            klines = self.binance.futures_klines(
                symbol=pair,
                interval=Client.KLINE_INTERVAL_1MINUTE,
                limit=limit
            )
            
            prices = [float(k[4]) for k in klines]  # Closing prices
            return {
                'prices': prices,
                'current_price': prices[-1] if prices else 0
            }
        except Exception as e:
            # Return current price only as fallback
            try:
                ticker = self.binance.futures_symbol_ticker(symbol=pair)
                current_price = float(ticker['price'])
                return {
                    'prices': [current_price] * 5,
                    'current_price': current_price
                }
            except:
                return {'prices': [], 'current_price': 0}

    def get_ai_decision(self, pair_data):
        """Get AI-powered trading decision"""
        try:
            pair = list(pair_data.keys())[0]
            current_price = pair_data[pair]['price']
            
            self.print_color(f"🔍 Analyzing {pair}...", Fore.BLUE)
            
            # Get price history for analysis
            market_data = self.get_price_history(pair)
            market_data['current_price'] = current_price
            
            # Get AI analysis
            direction, confidence, reason = self.get_deepseek_analysis(pair, market_data)
            
            if direction == "HOLD" or confidence < 60:
                self.print_color(f"🎯 AI Decision: HOLD ({confidence}%)", Fore.YELLOW)
                return {
                    "action": "HOLD", 
                    "pair": pair,
                    "direction": direction,
                    "confidence": confidence,
                    "reason": reason
                }
            else:
                self.print_color(f"🎯 AI Decision: {direction} ({confidence}%)", Fore.GREEN)
                return {
                    "action": "TRADE",
                    "pair": pair,
                    "direction": direction,
                    "confidence": confidence,
                    "reason": reason
                }
                
        except Exception as e:
            self.print_color(f"❌ AI decision failed: {e}", Fore.RED)
            return {
                "action": "HOLD",
                "pair": list(pair_data.keys())[0],
                "direction": "HOLD",
                "confidence": 0,
                "reason": f"Error: {str(e)}"
            }

    def execute_trade(self, decision):
        """Execute trade with proper error handling"""
        try:
            pair = decision["pair"]
            
            if not self.can_open_new_trade(pair):
                self.print_color(f"🚫 Cannot open {pair} - position exists", Fore.RED)
                return False
            
            direction = decision["direction"]
            confidence = decision["confidence"]
            reason = decision["reason"]
            
            # Get current price
            ticker = self.binance.futures_symbol_ticker(symbol=pair)
            current_price = float(ticker['price'])
            
            # Calculate quantity
            quantity = self.get_quantity(pair, current_price)
            if quantity is None:
                return False
            
            # Calculate TP/SL
            if direction == "LONG":
                stop_loss = current_price * 0.995
                take_profit = current_price * 1.008
                stop_side = 'SELL'
                entry_side = 'BUY'
            else:  # SHORT
                stop_loss = current_price * 1.005
                take_profit = current_price * 0.992
                stop_side = 'BUY'
                entry_side = 'SELL'
            
            stop_loss = self.format_price(pair, stop_loss)
            take_profit = self.format_price(pair, take_profit)
            
            direction_color = Fore.BLUE if direction == 'LONG' else Fore.RED
            self.print_color(f"🎯 EXECUTING: {pair} {direction}", direction_color)
            self.print_color(f"   Size: {quantity} | Entry: ${current_price:.4f}", Fore.WHITE)
            self.print_color(f"   TP: ${take_profit:.4f} | SL: ${stop_loss:.4f}", Fore.YELLOW)
            
            # Step 1: Open position
            try:
                self.print_color(f"📤 Placing {entry_side} order...", Fore.YELLOW)
                
                order = self.binance.futures_create_order(
                    symbol=pair,
                    side=entry_side,
                    type='MARKET',
                    quantity=quantity
                )
                
                self.print_color(f"✅ {direction} ORDER EXECUTED", Fore.GREEN)
                
                # Wait for position
                time.sleep(2)
                
                # Step 2: Place TP/SL orders
                self.print_color(f"📤 Placing TP/SL orders...", Fore.YELLOW)
                
                # Stop Loss
                sl_order = self.binance.futures_create_order(
                    symbol=pair,
                    side=stop_side,
                    type='STOP_MARKET',
                    quantity=quantity,
                    stopPrice=stop_loss,
                    reduceOnly=True,
                    timeInForce='GTC'
                )
                
                # Take Profit
                tp_order = self.binance.futures_create_order(
                    symbol=pair,
                    side=stop_side,
                    type='TAKE_PROFIT_MARKET',
                    quantity=quantity,
                    stopPrice=take_profit,
                    reduceOnly=True,
                    timeInForce='GTC'
                )
                
                self.print_color(f"✅ TP/SL PLACED", Fore.GREEN)
                
                # Store trade info
                self.bot_opened_trades[pair] = {
                    "pair": pair,
                    "direction": direction,
                    "entry_price": current_price,
                    "quantity": quantity,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "entry_time": time.time(),
                    "status": 'ACTIVE',
                    'ai_confidence': confidence,
                    'entry_time_th': self.get_thailand_time()
                }
                
                self.print_color(f"🚀 TRADE ACTIVATED: {pair} {direction}", Fore.GREEN)
                return True
                
            except BinanceAPIException as e:
                self.print_color(f"❌ Binance Error: {e}", Fore.RED)
                return False
            except Exception as e:
                self.print_color(f"❌ Execution Error: {e}", Fore.RED)
                return False
            
        except Exception as e:
            self.print_color(f"❌ Trade failed: {e}", Fore.RED)
            return False

    def monitor_positions(self):
        """Monitor active positions and track closed trades"""
        try:
            for pair, trade in list(self.bot_opened_trades.items()):
                if trade['status'] == 'ACTIVE':
                    # Check if position still exists
                    position_info = self.get_live_position_data(pair)
                    
                    if position_info is None:
                        # Position closed - calculate P&L
                        self.print_color(f"✅ Position closed: {pair}", Fore.GREEN)
                        
                        # Get current price for exit price
                        current_price = self.get_current_price(pair)
                        if current_price:
                            # Calculate P&L
                            if trade['direction'] == 'LONG':
                                pnl = (current_price - trade['entry_price']) * trade['quantity']
                            else:  # SHORT
                                pnl = (trade['entry_price'] - current_price) * trade['quantity']
                            
                            # Add to trade history
                            closed_trade = trade.copy()
                            closed_trade['exit_price'] = current_price
                            closed_trade['pnl'] = pnl
                            closed_trade['status'] = 'CLOSED'
                            self.add_trade_to_history(closed_trade)
                        
                        trade['status'] = 'CLOSED'
                    
        except Exception as e:
            self.print_color(f"❌ Monitoring error: {e}", Fore.RED)

    def get_current_price(self, pair):
        """Get current price for a pair"""
        try:
            ticker = self.binance.futures_symbol_ticker(symbol=pair)
            return float(ticker['price'])
        except:
            return None

    def scan_existing_positions(self):
        """Scan existing positions in Binance"""
        try:
            positions = self.binance.futures_position_information()
            self.existing_positions = {}
            
            for pos in positions:
                pair = pos['symbol']
                position_amt = float(pos['positionAmt'])
                
                if position_amt != 0 and pair in self.available_pairs:
                    entry_price = float(pos.get('entryPrice', 0))
                    unrealized_pnl = float(pos.get('unRealizedProfit', 0))
                    
                    # Get current price
                    try:
                        ticker = self.binance.futures_symbol_ticker(symbol=pair)
                        current_price = float(ticker['price'])
                    except:
                        current_price = entry_price
                    
                    # Calculate direction
                    if position_amt < 0:
                        direction = "SHORT"
                    else:
                        direction = "LONG"
                    
                    self.existing_positions[pair] = {
                        'direction': direction,
                        'entry_price': entry_price,
                        'quantity': abs(position_amt),
                        'current_price': current_price,
                        'unrealized_pnl': unrealized_pnl,
                        'status': 'ACTIVE'
                    }
            
            return len(self.existing_positions)
            
        except Exception as e:
            self.print_color(f"❌ Error scanning positions: {e}", Fore.RED)
            return 0
    
    def get_live_position_data(self, pair):
        """Get live position data"""
        try:
            positions = self.binance.futures_position_information(symbol=pair)
            for pos in positions:
                if pos['symbol'] == pair and float(pos['positionAmt']) != 0:
                    entry_price = float(pos.get('entryPrice', 0))
                    quantity = abs(float(pos['positionAmt']))
                    unrealized_pnl = float(pos.get('unRealizedProfit', 0))
                    
                    # Get current price
                    ticker = self.binance.futures_symbol_ticker(symbol=pair)
                    current_price = float(ticker['price'])
                    
                    # Calculate direction
                    if pos['positionAmt'].startswith('-'):
                        direction = "SHORT"
                    else:
                        direction = "LONG"
                    
                    return {
                        'direction': direction,
                        'entry_price': entry_price,
                        'quantity': quantity,
                        'current_price': current_price,
                        'unrealized_pnl': unrealized_pnl,
                        'status': 'ACTIVE'
                    }
            return None
        except Exception as e:
            self.print_color(f"❌ Error getting live data: {e}", Fore.RED)
            return None
    
    def display_dashboard(self):
        """Display trading dashboard with TP/SL information"""
        self.print_color(f"\n📊 DASHBOARD - {self.get_thailand_time()}", Fore.CYAN)
        self.print_color("=" * 80, Fore.CYAN)
        
        # Update positions
        for pair in list(self.existing_positions.keys()):
            live_data = self.get_live_position_data(pair)
            if live_data:
                self.existing_positions[pair].update(live_data)
            else:
                del self.existing_positions[pair]
        
        for pair in list(self.bot_opened_trades.keys()):
            live_data = self.get_live_position_data(pair)
            if not live_data and self.bot_opened_trades[pair]['status'] == 'ACTIVE':
                self.bot_opened_trades[pair]['status'] = 'CLOSED'
        
        total_positions = len(self.existing_positions) + len([t for t in self.bot_opened_trades.values() if t['status'] == 'ACTIVE'])
        
        if total_positions == 0:
            self.print_color("🔄 No active positions", Fore.YELLOW)
            return
        
        # Display existing positions (manual trades)
        for pair, position in self.existing_positions.items():
            pnl_color = Fore.GREEN if position['unrealized_pnl'] >= 0 else Fore.RED
            direction_color = Fore.BLUE if position['direction'] == 'LONG' else Fore.RED
            
            self.print_color(f"{position['direction']} {pair}", direction_color)
            self.print_color(f"   Size: {position['quantity']} | Entry: ${position['entry_price']:.4f}", Fore.WHITE)
            self.print_color(f"   Current: ${position['current_price']:.4f} | P&L: ${position['unrealized_pnl']:.2f}", pnl_color)
        
        # Display bot-opened positions
        for pair, trade in self.bot_opened_trades.items():
            if trade['status'] == 'ACTIVE':
                live_data = self.get_live_position_data(pair)
                if live_data:
                    pnl_color = Fore.GREEN if live_data['unrealized_pnl'] >= 0 else Fore.RED
                    direction_color = Fore.BLUE if trade['direction'] == 'LONG' else Fore.RED
                    
                    self.print_color(f"{trade['direction']} {pair} 🤖", direction_color)
                    self.print_color(f"   Size: {trade['quantity']} | Entry: ${trade['entry_price']:.4f}", Fore.WHITE)
                    self.print_color(f"   Current: ${live_data['current_price']:.4f} | P&L: ${live_data['unrealized_pnl']:.2f}", pnl_color)
                    self.print_color(f"   TP: ${trade['take_profit']:.4f} | SL: ${trade['stop_loss']:.4f}", Fore.YELLOW)
    
    def can_open_new_trade(self, pair):
        """Check if we can open new trade"""
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
        """Get market data for available pairs"""
        market_data = {}
        
        for pair in self.available_pairs:
            if not self.can_open_new_trade(pair):
                continue
                
            try:
                ticker = self.binance.futures_symbol_ticker(symbol=pair)
                if 'price' not in ticker:
                    continue
                    
                price = float(ticker['price'])
                market_data[pair] = {'price': price}
                
            except Exception as e:
                continue
                
        return market_data

    def run_trading_cycle(self):
        """Main trading cycle"""
        try:
            # Scan existing positions
            self.scan_existing_positions()
            
            # Monitor positions
            self.monitor_positions()
            
            # Display dashboard
            self.display_dashboard()
            
            # Show recent trade history every 5 cycles
            if hasattr(self, 'cycle_count') and self.cycle_count % 5 == 0:
                self.show_trade_history(5)
            
            # Check if we can open new trades
            active_bot_trades = len([t for t in self.bot_opened_trades.values() if t['status'] == 'ACTIVE'])
            total_positions = len(self.existing_positions) + active_bot_trades
            
            if total_positions < self.max_concurrent_trades:
                market_data = self.get_market_data()
                
                if market_data:
                    self.print_color(f"\n🔄 Looking for opportunities...", Fore.BLUE)
                    for pair in market_data.keys():
                        if self.can_open_new_trade(pair):
                            pair_data = {pair: market_data[pair]}
                            decision = self.get_ai_decision(pair_data)
                            
                            if decision["action"] == "TRADE":
                                self.print_color(f"✅ QUALIFIED: {pair}", Fore.GREEN)
                                success = self.execute_trade(decision)
                                if success:
                                    self.print_color(f"🎯 Trade executed!", Fore.GREEN)
                                    break
                                else:
                                    self.print_color(f"❌ Trade failed", Fore.RED)
                            else:
                                self.print_color(f"⏸️  HOLD: {pair}", Fore.YELLOW)
            
        except Exception as e:
            self.print_color(f"❌ Trading cycle error: {e}", Fore.RED)

    def start_trading(self):
        """Main trading loop"""
        self.print_color("🚀 STARTING TRADING BOT!", Fore.CYAN)
        
        self.cycle_count = 0
        
        while True:
            try:
                self.cycle_count += 1
                self.print_color(f"\n🔄 CYCLE {self.cycle_count}", Fore.CYAN)
                self.print_color("=" * 50, Fore.CYAN)
                
                self.run_trading_cycle()
                
                self.print_color(f"⏳ Waiting 30 seconds...", Fore.BLUE)
                time.sleep(30)
                
            except KeyboardInterrupt:
                self.print_color(f"\n🛑 BOT STOPPED", Fore.RED)
                self.show_trade_history(10)  # Show history when stopping
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
