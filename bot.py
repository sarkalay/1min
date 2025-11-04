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
from datetime import datetime, timedelta
import pytz

# Colorama setup
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    print("Warning: Colorama not installed. Run: pip install colorama")

# Load environment variables
load_dotenv()

# Global color variables for fallback
if not COLORAMA_AVAILABLE:
    class DummyColors:
        def __getattr__(self, name):
            return ''
    
    Fore = DummyColors()
    Back = DummyColors() 
    Style = DummyColors()

class OneMinScalpingBot:
    def __init__(self):
        # Load config from .env file
        self.binance_api_key = os.getenv('BINANCE_API_KEY')
        self.binance_secret = os.getenv('BINANCE_SECRET_KEY')
        self.deepseek_key = os.getenv('DEEPSEEK_API_KEY')
        
        # Store colorama references
        self.Fore = Fore
        self.Back = Back
        self.Style = Style
        self.COLORAMA_AVAILABLE = COLORAMA_AVAILABLE
        
        # Thailand timezone
        self.thailand_tz = pytz.timezone('Asia/Bangkok')
        
        # 1MIN SCALPING PARAMETERS
        self.trade_size_usd = 50
        self.leverage = 5
        self.tp_percent = 0.008   # +0.8%
        self.sl_percent = 0.005   # -0.5%
        
        # Multi-pair parameters
        self.max_concurrent_trades = 5
        self.available_pairs = ["SOLUSDT", "AVAXUSDT", "XRPUSDT", "LINKUSDT", "DOTUSDT"]
        
        # Track bot-opened trades only
        self.bot_opened_trades = {}
        
        # Trade history
        self.trade_history_file = "1min_scalping_history.json"
        self.trade_history = self.load_trade_history()
        
        # Precision settings
        self.quantity_precision = {}
        self.price_precision = {}
        
        # Initialize Binance client
        try:
            self.binance = Client(self.binance_api_key, self.binance_secret)
            self.print_color(f"🎯 1MIN SCALPING BOT ACTIVATED!", self.Fore.CYAN + self.Style.BRIGHT)
            self.print_color(f"TP: +0.8% | SL: -0.5% | R:R = 1.6", self.Fore.GREEN)
            self.print_color(f"Trade Size: ${self.trade_size_usd} | Leverage: {self.leverage}x", self.Fore.YELLOW)
            self.print_color(f"Chart: 1MIN | Max Trades: {self.max_concurrent_trades}", self.Fore.MAGENTA)
        except Exception as e:
            self.print_color(f"Binance initialization failed: {e}", self.Fore.RED)
            # Create dummy client for paper trading
            self.binance = None
        
        self.validate_config()
        if self.binance:
            self.setup_futures()
            self.load_symbol_precision()
    
    def load_trade_history(self):
        try:
            if os.path.exists(self.trade_history_file):
                with open(self.trade_history_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            self.print_color(f"Error loading trade history: {e}", self.Fore.RED)
            return []
    
    def save_trade_history(self):
        try:
            with open(self.trade_history_file, 'w') as f:
                json.dump(self.trade_history, f, indent=2)
        except Exception as e:
            self.print_color(f"Error saving trade history: {e}", self.Fore.RED)
    
    def add_trade_to_history(self, trade_data):
        try:
            trade_data['close_time'] = self.get_thailand_time()
            trade_data['close_timestamp'] = time.time()
            self.trade_history.append(trade_data)
            if len(self.trade_history) > 100:
                self.trade_history = self.trade_history[-100:]
            self.save_trade_history()
            self.print_color(f"Trade saved: {trade_data['pair']} {trade_data['direction']}", self.Fore.CYAN)
        except Exception as e:
            self.print_color(f"Error adding trade to history: {e}", self.Fore.RED)
    
    def show_trade_history(self, limit=10):
        if not self.trade_history:
            self.print_color("No trade history found", self.Fore.YELLOW)
            return
        self.print_color(f"\n📋 1MIN SCALPING HISTORY (Last {min(limit, len(self.trade_history))} trades)", self.Fore.CYAN)
        self.print_color("=" * 90, self.Fore.CYAN)
        for i, trade in enumerate(reversed(self.trade_history[-limit:])):
            pnl = trade.get('pnl', 0)
            pnl_color = self.Fore.GREEN if pnl > 0 else self.Fore.RED if pnl < 0 else self.Fore.YELLOW
            direction_icon = "📈" if trade['direction'] == 'LONG' else "📉"
            self.print_color(f"{i+1}. {direction_icon} {trade['pair']} {trade['direction']} | Entry: ${trade.get('entry_price', 0):.4f} | Exit: ${trade.get('exit_price', 0):.4f} | P&L: ${pnl:.2f}", pnl_color)
            self.print_color(f"   TP: ${trade.get('take_profit', 0):.4f} | SL: ${trade.get('stop_loss', 0):.4f} | Time: {trade.get('close_time', 'N/A')}", self.Fore.YELLOW)
    
    def get_thailand_time(self):
        now_utc = datetime.now(pytz.utc)
        thailand_time = now_utc.astimezone(self.thailand_tz)
        return thailand_time.strftime('%Y-%m-%d %H:%M:%S')
    
    def print_color(self, text, color="", style=""):
        if self.COLORAMA_AVAILABLE:
            print(f"{style}{color}{text}")
        else:
            print(text)
    
    def validate_config(self):
        if not all([self.binance_api_key, self.binance_secret, self.deepseek_key]):
            self.print_color("Missing API keys!", self.Fore.RED)
            return False
        try:
            if self.binance:
                self.binance.futures_exchange_info()
                self.print_color("Binance connection successful!", self.Fore.GREEN)
            else:
                self.print_color("Binance client not available - Paper Trading only", self.Fore.YELLOW)
                return True
        except Exception as e:
            self.print_color(f"Binance connection failed: {e}", self.Fore.RED)
            return False
        return True

    def setup_futures(self):
        if not self.binance:
            return
            
        try:
            for pair in self.available_pairs:
                try:
                    self.binance.futures_change_leverage(symbol=pair, leverage=self.leverage)
                    self.print_color(f"Leverage set for {pair}", self.Fore.GREEN)
                except Exception as e:
                    self.print_color(f"Leverage setup failed for {pair}: {e}", self.Fore.YELLOW)
            self.print_color("Futures setup completed!", self.Fore.GREEN)
        except Exception as e:
            self.print_color(f"Futures setup failed: {e}", self.Fore.RED)
    
    def load_symbol_precision(self):
        if not self.binance:
            # Set default precision for paper trading
            for pair in self.available_pairs:
                self.quantity_precision[pair] = 3
                self.price_precision[pair] = 4
            self.print_color("Default precision set for paper trading", self.Fore.GREEN)
            return
            
        try:
            exchange_info = self.binance.futures_exchange_info()
            for symbol in exchange_info['symbols']:
                pair = symbol['symbol']
                if pair not in self.available_pairs:
                    continue
                for f in symbol['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        step_size = f['stepSize']
                        qty_precision = len(step_size.split('.')[1].rstrip('0')) if '.' in step_size else 0
                        self.quantity_precision[pair] = qty_precision
                    elif f['filterType'] == 'PRICE_FILTER':
                        tick_size = f['tickSize']
                        price_precision = len(tick_size.split('.')[1].rstrip('0')) if '.' in tick_size else 0
                        self.price_precision[pair] = price_precision
            self.print_color("Symbol precision loaded", self.Fore.GREEN)
        except Exception as e:
            self.print_color(f"Error loading symbol precision: {e}", self.Fore.RED)
    
    def format_price(self, pair, price):
        if price <= 0:
            return 0.0
        precision = self.price_precision.get(pair, 4)
        return round(price, precision)
    
    def get_quantity(self, pair, price):
        try:
            if not price or price <= 0:
                self.print_color(f"Invalid price: {price} for {pair}", self.Fore.RED)
                return None

            fixed_quantities = {
                "SOLUSDT": 0.3, "AVAXUSDT": 3.0, "XRPUSDT": 20.0, "LINKUSDT": 3.2, "DOTUSDT": 18.0
            }
            quantity = fixed_quantities.get(pair)
            
            if not quantity or quantity <= 0:
                quantity = round(self.trade_size_usd / price, 4)
                quantity = max(quantity, 0.001)

            precision = self.quantity_precision.get(pair, 3)
            quantity = round(quantity, precision)
            
            if quantity <= 0:
                self.print_color(f"Invalid quantity: {quantity} for {pair}", self.Fore.RED)
                return None
                
            actual_value = quantity * price
            self.print_color(f"Quantity for {pair}: {quantity} = ${actual_value:.2f}", self.Fore.CYAN)
            return quantity
            
        except Exception as e:
            self.print_color(f"Quantity calculation failed: {e}", self.Fore.RED)
            return None

    def parse_ai_response(self, text):
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                decision_data = json.loads(json_str)
                direction = decision_data.get('direction', 'HOLD').upper()
                confidence = float(decision_data.get('confidence', 50))
                reason = decision_data.get('reason', 'AI Analysis')
                
                if direction not in ['LONG', 'SHORT', 'HOLD']:
                    direction = 'HOLD'
                if confidence < 0 or confidence > 100:
                    confidence = 50
                return direction, confidence, reason
            return 'HOLD', 50, 'No valid JSON found'
        except Exception as e:
            self.print_color(f"AI response parsing failed: {e}", self.Fore.RED)
            return 'HOLD', 50, 'Parsing failed'

    def get_deepseek_analysis(self, pair, market_data):
        try:
            if not self.deepseek_key:
                self.print_color("DeepSeek API key not found", self.Fore.RED)
                return "HOLD", 0, "No API key"
            
            current_price = market_data['current_price']
            
            # 1-minute scalping focused prompt
            prompt = f"""
            Analyze {pair} for 1-minute scalping trading. Current price: ${current_price:.4f}
            
            Timeframe: 1-minute charts
            Strategy: Quick scalping
            Holding time: 1-5 minutes
            Target: +0.8% profit
            Stop Loss: -0.5%
            
            Provide your trading recommendation based on your analysis of the 1-minute chart.
            
            Respond with this JSON format only:
            {{
                "direction": "LONG|SHORT|HOLD",
                "confidence": 0-100,
                "reason": "Your analysis reason for 1-minute timeframe"
            }}
            """
            
            headers = {"Authorization": f"Bearer {self.deepseek_key}", "Content-Type": "application/json"}
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are a crypto scalping specialist. Analyze 1-minute charts for quick trading opportunities. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 300
            }
            
            self.print_color(f"🤖 AI Analyzing {pair} on 1MIN chart...", self.Fore.MAGENTA)
            response = requests.post("https://api.deepseek.com/chat/completions", headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content'].strip()
                direction, confidence, reason = self.parse_ai_response(ai_response)
                
                # Log the AI decision with icons
                direction_icon = "📈" if direction == "LONG" else "📉" if direction == "SHORT" else "⏸️"
                color = self.Fore.BLUE if direction == "LONG" else self.Fore.RED if direction == "SHORT" else self.Fore.YELLOW
                self.print_color(f"{direction_icon} 1MIN AI: {direction} ({confidence}%) - {reason}", color)
                return direction, confidence, reason
            else:
                self.print_color(f"DeepSeek API error: {response.status_code}", self.Fore.RED)
                return "HOLD", 0, f"API Error"
                
        except Exception as e:
            self.print_color(f"DeepSeek analysis failed: {e}", self.Fore.RED)
            return "HOLD", 0, f"Error"

    def get_price_history(self, pair, limit=20):
        try:
            if self.binance:
                # Use 1-minute interval
                klines = self.binance.futures_klines(symbol=pair, interval=Client.KLINE_INTERVAL_1MINUTE, limit=limit)
                prices = [float(k[4]) for k in klines]  # Closing prices
                highs = [float(k[2]) for k in klines]   # High prices
                lows = [float(k[3]) for k in klines]    # Low prices
                return {
                    'prices': prices, 
                    'highs': highs,
                    'lows': lows,
                    'current_price': prices[-1] if prices else 0
                }
            else:
                # For paper trading, get real price data
                try:
                    ticker = self.binance.futures_symbol_ticker(symbol=pair) if self.binance else None
                    if ticker:
                        current_price = float(ticker['price'])
                        return {
                            'prices': [current_price] * 10, 
                            'highs': [current_price * 1.01] * 10,
                            'lows': [current_price * 0.99] * 10,
                            'current_price': current_price
                        }
                except:
                    pass
                # Fallback to realistic prices
                base_prices = {
                    "SOLUSDT": 180.50, "AVAXUSDT": 35.20, "XRPUSDT": 0.62, 
                    "LINKUSDT": 18.75, "DOTUSDT": 8.90
                }
                current_price = base_prices.get(pair, 100)
                return {
                    'prices': [current_price] * 10,
                    'highs': [current_price * 1.01] * 10,
                    'lows': [current_price * 0.99] * 10,
                    'current_price': current_price
                }
        except Exception as e:
            try:
                if self.binance:
                    ticker = self.binance.futures_symbol_ticker(symbol=pair)
                    current_price = float(ticker['price'])
                    return {
                        'prices': [current_price] * 10,
                        'highs': [current_price * 1.01] * 10,
                        'lows': [current_price * 0.99] * 10,
                        'current_price': current_price
                    }
                else:
                    return {
                        'prices': [100] * 10,
                        'highs': [101] * 10,
                        'lows': [99] * 10,
                        'current_price': 100
                    }
            except:
                return {
                    'prices': [],
                    'highs': [],
                    'lows': [],
                    'current_price': 0
                }

    def get_ai_decision(self, pair_data):
        try:
            pair = list(pair_data.keys())[0]
            current_price = pair_data[pair]['price']
            if current_price <= 0:
                return {"action": "HOLD", "pair": pair, "direction": "HOLD", "confidence": 0, "reason": "Invalid price"}
            
            self.print_color(f"🔍 Analyzing {pair} at ${current_price:.4f} (1MIN)...", self.Fore.BLUE)
            market_data = self.get_price_history(pair)
            market_data['current_price'] = current_price
            direction, confidence, reason = self.get_deepseek_analysis(pair, market_data)
            
            if direction == "HOLD" or confidence < 70:
                self.print_color(f"⏸️ 1MIN AI Decision: HOLD ({confidence}%)", self.Fore.YELLOW)
                return {"action": "HOLD", "pair": pair, "direction": direction, "confidence": confidence, "reason": reason}
            else:
                direction_icon = "📈" if direction == "LONG" else "📉"
                color = self.Fore.BLUE if direction == "LONG" else self.Fore.RED
                self.print_color(f"🎯 1MIN AI Decision: {direction} {direction_icon} ({confidence}%)", color + self.Style.BRIGHT)
                return {"action": "TRADE", "pair": pair, "direction": direction, "confidence": confidence, "reason": reason}
                
        except Exception as e:
            self.print_color(f"AI decision failed: {e}", self.Fore.RED)
            return {"action": "HOLD", "pair": list(pair_data.keys())[0], "direction": "HOLD", "confidence": 0, "reason": f"Error: {str(e)}"}

    def get_current_price(self, pair):
        try:
            if self.binance:
                ticker = self.binance.futures_symbol_ticker(symbol=pair)
                return float(ticker['price'])
            else:
                # For paper trading, get real price from public API
                try:
                    import requests
                    url = f"https://api.binance.com/api/v3/ticker/price"
                    params = {'symbol': pair}
                    response = requests.get(url, params=params, timeout=5)
                    data = response.json()
                    return float(data['price'])
                except:
                    # Fallback to realistic prices
                    base_prices = {
                        "SOLUSDT": 180.50, "AVAXUSDT": 35.20, "XRPUSDT": 0.62,
                        "LINKUSDT": 18.75, "DOTUSDT": 8.90
                    }
                    base_price = base_prices.get(pair, 100)
                    return base_price
        except:
            return None

    def get_market_data(self):
        market_data = {}
        for pair in self.available_pairs:
            try:
                price = self.get_current_price(pair)
                if price and price > 0:
                    market_data[pair] = {'price': price}
            except Exception as e:
                continue
        return market_data

    def can_open_new_trade(self, pair):
        if pair in self.bot_opened_trades and self.bot_opened_trades[pair]['status'] == 'ACTIVE':
            return False
        return len(self.bot_opened_trades) < self.max_concurrent_trades

    def execute_trade(self, decision):
        """LIVE TRADING - Clear display of all trade details"""
        try:
            pair = decision["pair"]
            if not self.can_open_new_trade(pair):
                self.print_color(f"❌ Cannot open {pair} - position exists", self.Fore.RED)
                return False
            
            direction = decision["direction"]
            confidence = decision["confidence"]
            reason = decision["reason"]
            
            # Get current price
            ticker = self.binance.futures_symbol_ticker(symbol=pair)
            current_price = float(ticker['price'])
            if current_price <= 0:
                self.print_color(f"❌ Invalid price for {pair}", self.Fore.RED)
                return False
            
            # Calculate quantity
            quantity = self.get_quantity(pair, current_price)
            if quantity is None:
                return False
            
            # Calculate TP/SL
            if direction == "LONG":
                take_profit = current_price * (1 + self.tp_percent)   # +0.8%
                stop_loss = current_price * (1 - self.sl_percent)     # -0.5%
            else:  # SHORT
                take_profit = current_price * (1 - self.tp_percent)   # -0.8%
                stop_loss = current_price * (1 + self.sl_percent)     # +0.5%
            
            take_profit = self.format_price(pair, take_profit)
            stop_loss = self.format_price(pair, stop_loss)
            
            # Display trade details CLEARLY
            direction_color = self.Fore.BLUE if direction == 'LONG' else self.Fore.RED
            direction_icon = "📈" if direction == 'LONG' else "📉"
            
            self.print_color(f"\n🎯 LIVE TRADE EXECUTION DETAILS", self.Fore.CYAN + self.Style.BRIGHT)
            self.print_color("=" * 60, self.Fore.CYAN)
            self.print_color(f"{direction_icon} DIRECTION: {direction}", direction_color + self.Style.BRIGHT)
            self.print_color(f"💰 PAIR: {pair}", self.Fore.WHITE)
            self.print_color(f"💵 ENTRY PRICE: ${current_price:.4f}", self.Fore.GREEN)
            self.print_color(f"📊 QUANTITY: {quantity}", self.Fore.WHITE)
            self.print_color(f"🎯 TAKE PROFIT: ${take_profit:.4f} (+{self.tp_percent*100:.1f}%)", self.Fore.GREEN)
            self.print_color(f"🛑 STOP LOSS: ${stop_loss:.4f} (-{self.sl_percent*100:.1f}%)", self.Fore.RED)
            self.print_color(f"🤖 AI CONFIDENCE: {confidence}%", self.Fore.MAGENTA)
            self.print_color(f"📝 REASON: {reason}", self.Fore.YELLOW)
            self.print_color("=" * 60, self.Fore.CYAN)
            
            # Execute the trade
            entry_side = 'BUY' if direction == 'LONG' else 'SELL'
            try:
                # Market entry order
                order = self.binance.futures_create_order(
                    symbol=pair,
                    side=entry_side,
                    type='MARKET',
                    quantity=quantity
                )
                self.print_color(f"✅ {direction} ORDER EXECUTED SUCCESSFULLY!", self.Fore.GREEN + self.Style.BRIGHT)
                time.sleep(2)
                
                # Place TP/SL orders
                stop_side = 'SELL' if direction == 'LONG' else 'BUY'
                
                # Stop Loss order
                self.binance.futures_create_order(
                    symbol=pair,
                    side=stop_side,
                    type='STOP_MARKET',
                    quantity=quantity,
                    stopPrice=stop_loss,
                    reduceOnly=True,
                    timeInForce='GTC'
                )
                
                # Take Profit order
                self.binance.futures_create_order(
                    symbol=pair,
                    side=stop_side,
                    type='TAKE_PROFIT_MARKET',
                    quantity=quantity,
                    stopPrice=take_profit,
                    reduceOnly=True,
                    timeInForce='GTC'
                )
                
                self.print_color(f"✅ TP/SL ORDERS PLACED SUCCESSFULLY!", self.Fore.GREEN)
                
                # Record the trade
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
                    'ai_reason': reason,
                    'entry_time_th': self.get_thailand_time()
                }
                
                self.print_color(f"🚀 LIVE TRADE ACTIVATED: {pair} {direction} {direction_icon}", self.Fore.GREEN + self.Style.BRIGHT)
                return True
                
            except BinanceAPIException as e:
                self.print_color(f"❌ Binance Error: {e}", self.Fore.RED)
                return False
            except Exception as e:
                self.print_color(f"❌ Execution Error: {e}", self.Fore.RED)
                return False
            
        except Exception as e:
            self.print_color(f"❌ Trade failed: {e}", self.Fore.RED)
            return False

    def get_live_position_data(self, pair):
        """Get real-time position data from Binance"""
        try:
            positions = self.binance.futures_position_information(symbol=pair)
            for pos in positions:
                if pos['symbol'] == pair and float(pos['positionAmt']) != 0:
                    entry_price = float(pos.get('entryPrice', 0))
                    quantity = abs(float(pos['positionAmt']))
                    unrealized_pnl = float(pos.get('unRealizedProfit', 0))
                    ticker = self.binance.futures_symbol_ticker(symbol=pair)
                    current_price = float(ticker['price'])
                    direction = "SHORT" if pos['positionAmt'].startswith('-') else "LONG"
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
            self.print_color(f"Error getting live data: {e}", self.Fore.RED)
            return None

    def monitor_positions(self):
        """Monitor and display active positions"""
        try:
            for pair, trade in list(self.bot_opened_trades.items()):
                if trade['status'] != 'ACTIVE':
                    continue
                
                live_data = self.get_live_position_data(pair)
                if not live_data:
                    self.close_trade_with_cleanup(pair, trade)
                    continue
                    
                # Display position status
                direction_icon = "📈" if trade['direction'] == 'LONG' else "📉"
                pnl_color = self.Fore.GREEN if live_data['unrealized_pnl'] >= 0 else self.Fore.RED
                
                self.print_color(f"\n📊 LIVE POSITION: {pair} {direction_icon}", self.Fore.CYAN)
                self.print_color(f"   Direction: {trade['direction']} | Entry: ${trade['entry_price']:.4f}", self.Fore.WHITE)
                self.print_color(f"   Current: ${live_data['current_price']:.4f} | P&L: ${live_data['unrealized_pnl']:.2f}", pnl_color)
                self.print_color(f"   TP: ${trade['take_profit']:.4f} | SL: ${trade['stop_loss']:.4f}", self.Fore.YELLOW)
                    
        except Exception as e:
            self.print_color(f"Monitoring error: {e}", self.Fore.RED)

    def close_trade_with_cleanup(self, pair, trade):
        """Close trade and cleanup orders"""
        try:
            # Cancel any open orders
            open_orders = self.binance.futures_get_open_orders(symbol=pair)
            canceled = 0
            for order in open_orders:
                if order['reduceOnly'] and order['symbol'] == pair:
                    try:
                        self.binance.futures_cancel_order(symbol=pair, orderId=order['orderId'])
                        canceled += 1
                    except Exception as e:
                        self.print_color(f"Failed to cancel order {order['orderId']}: {e}", self.Fore.RED)
            
            # Get final P&L
            final_pnl = self.get_final_pnl(pair, trade)
            
            # Record trade completion
            trade['status'] = 'CLOSED'
            trade['exit_time_th'] = self.get_thailand_time()
            trade['exit_price'] = self.get_current_price(pair)
            trade['pnl'] = final_pnl
            
            closed_trade = trade.copy()
            self.add_trade_to_history(closed_trade)
            
            # Display closure
            pnl_color = self.Fore.GREEN if final_pnl > 0 else self.Fore.RED
            direction_icon = "📈" if trade['direction'] == 'LONG' else "📉"
            self.print_color(f"\n🔚 TRADE CLOSED: {pair} {direction_icon} {trade['direction']}", pnl_color)
            self.print_color(f"   Final P&L: ${final_pnl:.2f}", pnl_color)
            if canceled > 0:
                self.print_color(f"   Cleaned up {canceled} order(s)", self.Fore.CYAN)
                
        except Exception as e:
            self.print_color(f"Cleanup failed for {pair}: {e}", self.Fore.RED)

    def get_final_pnl(self, pair, trade):
        """Calculate final P&L for closed trade"""
        try:
            live = self.get_live_position_data(pair)
            if live and 'unrealized_pnl' in live:
                return live['unrealized_pnl']
            current = self.get_current_price(pair)
            if not current:
                return 0
            if trade['direction'] == 'LONG':
                return (current - trade['entry_price']) * trade['quantity']
            else:
                return (trade['entry_price'] - current) * trade['quantity']
        except:
            return 0

    def display_dashboard(self):
        """Display clear dashboard with all active positions"""
        self.print_color(f"\n📊 LIVE TRADING DASHBOARD - {self.get_thailand_time()}", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color("=" * 80, self.Fore.CYAN)
        
        active_count = 0
        for pair, trade in self.bot_opened_trades.items():
            if trade['status'] == 'ACTIVE':
                active_count += 1
                live_data = self.get_live_position_data(pair)
                if live_data:
                    direction_icon = "📈" if trade['direction'] == 'LONG' else "📉"
                    pnl_color = self.Fore.GREEN if live_data['unrealized_pnl'] >= 0 else self.Fore.RED
                    
                    self.print_color(f"{direction_icon} {pair} {trade['direction']}", self.Fore.WHITE + self.Style.BRIGHT)
                    self.print_color(f"   Entry: ${trade['entry_price']:.4f} | Current: ${live_data['current_price']:.4f}", self.Fore.WHITE)
                    self.print_color(f"   P&L: ${live_data['unrealized_pnl']:.2f}", pnl_color)
                    self.print_color(f"   TP: ${trade['take_profit']:.4f} | SL: ${trade['stop_loss']:.4f}", self.Fore.YELLOW)
                    self.print_color("   " + "-" * 50, self.Fore.CYAN)
        
        if active_count == 0:
            self.print_color("No active positions", self.Fore.YELLOW)
        else:
            self.print_color(f"Total Active Positions: {active_count}", self.Fore.CYAN)

    def run_trading_cycle(self):
        """Main trading cycle with clear AI decisions"""
        try:
            self.monitor_positions()
            self.display_dashboard()
            
            if hasattr(self, 'cycle_count') and self.cycle_count % 5 == 0:
                self.show_trade_history(5)
            
            # AI SCANS ALL PAIRS
            market_data = self.get_market_data()
            if market_data:
                self.print_color(f"\n🤖 1MIN AI SCANNING {len(market_data)} PAIRS...", self.Fore.BLUE + self.Style.BRIGHT)
                
                for pair in market_data.keys():
                    if self.can_open_new_trade(pair):
                        pair_data = {pair: market_data[pair]}
                        decision = self.get_ai_decision(pair_data)
                        
                        if decision["action"] == "TRADE":
                            direction_icon = "📈" if decision['direction'] == "LONG" else "📉"
                            self.print_color(f"🚀 1MIN QUALIFIED: {pair} {decision['direction']} {direction_icon} ({decision['confidence']}%)", self.Fore.GREEN + self.Style.BRIGHT)
                            success = self.execute_trade(decision)
                        else:
                            self.print_color(f"⏸️ 1MIN HOLD: {pair} ({decision['confidence']}%)", self.Fore.YELLOW)
                    else:
                        self.print_color(f"↗️ 1MIN SKIPPED: {pair} (position limit reached)", self.Fore.MAGENTA)
            else:
                self.print_color("No market data available", self.Fore.YELLOW)
                
        except Exception as e:
            self.print_color(f"Trading cycle error: {e}", self.Fore.RED)

    def start_trading(self):
        """Start live trading with clear output"""
        self.print_color("🚀 STARTING 1MIN LIVE TRADING BOT!", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color("⚠️  REAL MONEY TRADING - BE CAREFUL!", self.Fore.RED + self.Style.BRIGHT)
        self.cycle_count = 0
        
        while True:
            try:
                self.cycle_count += 1
                self.print_color(f"\n🔄 LIVE TRADING CYCLE {self.cycle_count}", self.Fore.CYAN)
                self.print_color("=" * 50, self.Fore.CYAN)
                self.run_trading_cycle()
                self.print_color(f"⏰ Waiting 30 seconds for next 1MIN analysis...", self.Fore.BLUE)
                time.sleep(30)
                
            except KeyboardInterrupt:
                self.print_color(f"\n🛑 LIVE TRADING STOPPED", self.Fore.RED + self.Style.BRIGHT)
                self.show_trade_history(10)
                break
            except Exception as e:
                self.print_color(f"Main loop error: {e}", self.Fore.RED)
                time.sleep(30)


class OneMinPaperTradingBot:
    def __init__(self, real_bot):
        self.real_bot = real_bot
        self.paper_balance = 1000
        self.paper_positions = {}
        self.paper_history = []
        
        self.real_bot.print_color("🤖 1MIN PAPER TRADING BOT INITIALIZED!", self.real_bot.Fore.GREEN + self.real_bot.Style.BRIGHT)
        self.real_bot.print_color(f"💰 Starting Paper Balance: ${self.paper_balance}", self.real_bot.Fore.CYAN)
        self.real_bot.print_color(f"🎯 Strategy: 1MIN Scalping | TP: +0.8% | SL: -0.5%", self.real_bot.Fore.MAGENTA)
        
    def paper_execute_trade(self, decision):
        """Paper trading with clear display"""
        try:
            pair = decision["pair"]
            direction = decision["direction"]
            confidence = decision["confidence"]
            reason = decision["reason"]
            
            current_price = self.real_bot.get_current_price(pair)
            if current_price <= 0:
                return False
            
            quantity = self.real_bot.get_quantity(pair, current_price)
            if quantity is None:
                return False
            
            # Calculate TP/SL
            if direction == "LONG":
                take_profit = current_price * (1 + self.real_bot.tp_percent)
                stop_loss = current_price * (1 - self.real_bot.sl_percent)
            else:  # SHORT
                take_profit = current_price * (1 - self.real_bot.tp_percent)
                stop_loss = current_price * (1 + self.real_bot.sl_percent)
            
            take_profit = self.real_bot.format_price(pair, take_profit)
            stop_loss = self.real_bot.format_price(pair, stop_loss)
            
            # Display trade details clearly
            direction_color = self.real_bot.Fore.BLUE if direction == 'LONG' else self.real_bot.Fore.RED
            direction_icon = "📈" if direction == 'LONG' else "📉"
            
            self.real_bot.print_color(f"\n🎯 PAPER TRADE EXECUTION", self.real_bot.Fore.CYAN + self.real_bot.Style.BRIGHT)
            self.real_bot.print_color("=" * 60, self.real_bot.Fore.CYAN)
            self.real_bot.print_color(f"{direction_icon} DIRECTION: {direction}", direction_color)
            self.real_bot.print_color(f"💰 PAIR: {pair}", self.real_bot.Fore.WHITE)
            self.real_bot.print_color(f"💵 ENTRY: ${current_price:.4f}", self.real_bot.Fore.GREEN)
            self.real_bot.print_color(f"🎯 TP: ${take_profit:.4f}", self.real_bot.Fore.GREEN)
            self.real_bot.print_color(f"🛑 SL: ${stop_loss:.4f}", self.real_bot.Fore.RED)
            self.real_bot.print_color(f"🤖 CONFIDENCE: {confidence}%", self.real_bot.Fore.MAGENTA)
            self.real_bot.print_color("=" * 60, self.real_bot.Fore.CYAN)
            
            # Record paper position
            self.paper_positions[pair] = {
                "pair": pair, "direction": direction, "entry_price": current_price,
                "quantity": quantity, "stop_loss": stop_loss, "take_profit": take_profit,
                "entry_time": time.time(), "status": 'ACTIVE', 'ai_confidence': confidence,
                'entry_time_th': self.real_bot.get_thailand_time()
            }
            
            return True
            
        except Exception as e:
            self.real_bot.print_color(f"Paper trade failed: {e}", self.real_bot.Fore.RED)
            return False

    def monitor_paper_positions(self):
        """Monitor paper positions with clear display"""
        try:
            for pair, trade in list(self.paper_positions.items()):
                if trade['status'] != 'ACTIVE':
                    continue
                
                current_price = self.real_bot.get_current_price(pair)
                if not current_price:
                    continue
                
                # Check TP/SL
                should_close = False
                close_reason = ""
                pnl = 0
                
                if trade['direction'] == 'LONG':
                    if current_price >= trade['take_profit']:
                        should_close = True
                        close_reason = "TP HIT"
                        pnl = (current_price - trade['entry_price']) * trade['quantity']
                    elif current_price <= trade['stop_loss']:
                        should_close = True
                        close_reason = "SL HIT" 
                        pnl = (current_price - trade['entry_price']) * trade['quantity']
                else:  # SHORT
                    if current_price <= trade['take_profit']:
                        should_close = True
                        close_reason = "TP HIT"
                        pnl = (trade['entry_price'] - current_price) * trade['quantity']
                    elif current_price >= trade['stop_loss']:
                        should_close = True
                        close_reason = "SL HIT"
                        pnl = (trade['entry_price'] - current_price) * trade['quantity']
                
                if should_close:
                    # Close position
                    trade['status'] = 'CLOSED'
                    trade['exit_price'] = current_price
                    trade['pnl'] = pnl
                    trade['close_reason'] = close_reason
                    trade['close_time'] = self.real_bot.get_thailand_time()
                    
                    self.paper_balance += pnl
                    self.paper_history.append(trade.copy())
                    
                    # Display closure clearly
                    pnl_color = self.real_bot.Fore.GREEN if pnl > 0 else self.real_bot.Fore.RED
                    direction_icon = "📈" if trade['direction'] == 'LONG' else "📉"
                    self.real_bot.print_color(f"\n🔚 PAPER TRADE CLOSED: {pair} {direction_icon}", pnl_color)
                    self.real_bot.print_color(f"   P&L: ${pnl:.2f} | Reason: {close_reason}", pnl_color)
                    
                    del self.paper_positions[pair]
                    
        except Exception as e:
            self.real_bot.print_color(f"Paper monitoring error: {e}", self.real_bot.Fore.RED)

    def get_paper_portfolio_status(self):
        """Display portfolio status clearly"""
        total_trades = len(self.paper_history)
        winning_trades = len([t for t in self.paper_history if t.get('pnl', 0) > 0])
        total_pnl = sum(trade.get('pnl', 0) for trade in self.paper_history)
        
        self.real_bot.print_color(f"\n💼 PAPER TRADING PORTFOLIO", self.real_bot.Fore.CYAN + self.real_bot.Style.BRIGHT)
        self.real_bot.print_color("=" * 60, self.real_bot.Fore.CYAN)
        self.real_bot.print_color(f"Active Positions: {len(self.paper_positions)}", self.real_bot.Fore.WHITE)
        self.real_bot.print_color(f"Balance: ${self.paper_balance:.2f}", self.real_bot.Fore.WHITE)
        self.real_bot.print_color(f"Total Trades: {total_trades}", self.real_bot.Fore.WHITE)
        
        if total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100
            self.real_bot.print_color(f"Win Rate: {win_rate:.1f}%", self.real_bot.Fore.GREEN if win_rate > 50 else self.real_bot.Fore.YELLOW)
            self.real_bot.print_color(f"Total P&L: ${total_pnl:.2f}", self.real_bot.Fore.GREEN if total_pnl > 0 else self.real_bot.Fore.RED)

    def run_paper_trading_cycle(self):
        """Paper trading cycle with clear output"""
        try:
            self.monitor_paper_positions()
            
            market_data = self.real_bot.get_market_data()
            if market_data:
                self.real_bot.print_color(f"\n🤖 1MIN AI SCANNING FOR PAPER TRADES...", self.real_bot.Fore.BLUE + self.real_bot.Style.BRIGHT)
                
                for pair in market_data.keys():
                    if pair not in self.paper_positions and len(self.paper_positions) < self.real_bot.max_concurrent_trades:
                        pair_data = {pair: market_data[pair]}
                        decision = self.real_bot.get_ai_decision(pair_data)
                        
                        if decision["action"] == "TRADE":
                            direction_icon = "📈" if decision['direction'] == "LONG" else "📉"
                            self.real_bot.print_color(f"🎯 1MIN AI SIGNAL: {pair} {decision['direction']} {direction_icon}", self.real_bot.Fore.GREEN + self.real_bot.Style.BRIGHT)
                            self.paper_execute_trade(decision)
            
            self.get_paper_portfolio_status()
            
        except Exception as e:
            self.real_bot.print_color(f"Paper trading error: {e}", self.real_bot.Fore.RED)

    def start_paper_trading(self):
        """Start paper trading"""
        self.real_bot.print_color("🚀 STARTING 1MIN PAPER TRADING!", self.real_bot.Fore.GREEN + self.real_bot.Style.BRIGHT)
        self.real_bot.print_color("💰 NO REAL MONEY AT RISK", self.real_bot.Fore.GREEN)
        
        cycle_count = 0
        while True:
            try:
                cycle_count += 1
                self.real_bot.print_color(f"\n🔄 PAPER CYCLE {cycle_count}", self.real_bot.Fore.CYAN)
                self.real_bot.print_color("=" * 50, self.real_bot.Fore.CYAN)
                self.run_paper_trading_cycle()
                self.real_bot.print_color(f"⏰ Waiting 30 seconds...", self.real_bot.Fore.BLUE)
                time.sleep(30)
                
            except KeyboardInterrupt:
                self.real_bot.print_color(f"\n🛑 PAPER TRADING STOPPED", self.real_bot.Fore.RED + self.real_bot.Style.BRIGHT)
                break
            except Exception as e:
                self.real_bot.print_color(f"Paper trading error: {e}", self.real_bot.Fore.RED)
                time.sleep(30)


if __name__ == "__main__":
    try:
        real_bot = OneMinScalpingBot()
        
        print("\n" + "="*60)
        print("🤖 1MIN AI SCALPING BOT")
        print("="*60)
        print("SELECT TRADING MODE:")
        print("1. 🔴 Live Trading (Real Money)")
        print("2. 🟢 Paper Trading (No Risk)")
        
        choice = input("Enter choice (1-2): ").strip()
        
        if choice == "1":
            print("⚠️  WARNING: REAL MONEY TRADING!")
            confirm = input("Type 'YES' to confirm: ").strip()
            if confirm.upper() == 'YES':
                real_bot.start_trading()
            else:
                print("Using Paper Trading mode...")
                paper_bot = OneMinPaperTradingBot(real_bot)
                paper_bot.start_paper_trading()
        else:
            paper_bot = OneMinPaperTradingBot(real_bot)
            paper_bot.start_paper_trading()
            
    except Exception as e:
        print(f"Failed to start bot: {e}")
