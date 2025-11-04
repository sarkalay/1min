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

class ThreeMinScalpingBot:
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
        
        # 3MIN SCALPING PARAMETERS
        self.trade_size_usd = 50
        self.leverage = 5
        self.tp_percent = 0.008   # +0.8%
        self.sl_percent = 0.005   # -0.5%
        
        # Multi-pair parameters
        self.max_concurrent_trades = 2
        self.available_pairs = ["SOLUSDT", "AVAXUSDT", "XRPUSDT", "LINKUSDT", "DOTUSDT"]
        
        # Track bot-opened trades only
        self.bot_opened_trades = {}
        
        # Trade history
        self.trade_history_file = "3min_scalping_history.json"
        self.trade_history = self.load_trade_history()
        
        # Precision settings
        self.quantity_precision = {}
        self.price_precision = {}
        
        # Initialize Binance client
        try:
            self.binance = Client(self.binance_api_key, self.binance_secret)
            self.print_color(f"🎯 3MIN SCALPING BOT ACTIVATED!", self.Fore.CYAN + self.Style.BRIGHT)
            self.print_color(f"TP: +0.8% | SL: -0.5% | R:R = 1.6", self.Fore.GREEN)
            self.print_color(f"Trade Size: ${self.trade_size_usd} | Leverage: {self.leverage}x", self.Fore.YELLOW)
            self.print_color(f"Chart: 3MIN | Max Trades: {self.max_concurrent_trades}", self.Fore.MAGENTA)
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
        self.print_color(f"\n📋 3MIN SCALPING HISTORY (Last {min(limit, len(self.trade_history))} trades)", self.Fore.CYAN)
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
                "SOLUSDT": 0.3, "AVAXUSDT": 2.0, "XRPUSDT": 20.0, "LINKUSDT": 3.2, "DOTUSDT": 18.0
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
            
            # 3-minute scalping focused prompt
            prompt = f"""
            Analyze {pair} for 3-minute scalping trading. Current price: ${current_price:.4f}
            
            Timeframe: 3-minute charts
            Strategy: Quick scalping
            Holding time: 3-10 minutes
            Target: +0.8% profit
            Stop Loss: -0.5%
            
            Provide your trading recommendation based on your analysis of the 3-minute chart.
            
            Respond with this JSON format only:
            {{
                "direction": "LONG|SHORT|HOLD",
                "confidence": 0-100,
                "reason": "Your analysis reason for 3-minute timeframe"
            }}
            """
            
            headers = {"Authorization": f"Bearer {self.deepseek_key}", "Content-Type": "application/json"}
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are a crypto scalping specialist. Analyze 3-minute charts for quick trading opportunities. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 300
            }
            
            self.print_color(f"🤖 AI Analyzing {pair} on 3MIN chart...", self.Fore.MAGENTA)
            response = requests.post("https://api.deepseek.com/chat/completions", headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content'].strip()
                direction, confidence, reason = self.parse_ai_response(ai_response)
                
                # Log the AI decision with icons
                direction_icon = "📈" if direction == "LONG" else "📉" if direction == "SHORT" else "⏸️"
                color = self.Fore.BLUE if direction == "LONG" else self.Fore.RED if direction == "SHORT" else self.Fore.YELLOW
                self.print_color(f"{direction_icon} 3MIN AI: {direction} ({confidence}%) - {reason}", color)
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
                # Use 3-minute interval
                klines = self.binance.futures_klines(symbol=pair, interval=Client.KLINE_INTERVAL_3MINUTE, limit=limit)
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
            
            self.print_color(f"🔍 Analyzing {pair} at ${current_price:.4f} (3MIN)...", self.Fore.BLUE)
            market_data = self.get_price_history(pair)
            market_data['current_price'] = current_price
            direction, confidence, reason = self.get_deepseek_analysis(pair, market_data)
            
            if direction == "HOLD" or confidence < 70:
                self.print_color(f"⏸️ 3MIN AI Decision: HOLD ({confidence}%)", self.Fore.YELLOW)
                return {"action": "HOLD", "pair": pair, "direction": direction, "confidence": confidence, "reason": reason}
            else:
                direction_icon = "📈" if direction == "LONG" else "📉"
                color = self.Fore.BLUE if direction == "LONG" else self.Fore.RED
                self.print_color(f"🎯 3MIN AI Decision: {direction} {direction_icon} ({confidence}%)", color + self.Style.BRIGHT)
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
        # This is just a placeholder for paper trading
        self.print_color("💡 LIVE TRADING DISABLED - Using Paper Trading", self.Fore.YELLOW)
        return False

    def display_dashboard(self):
        self.print_color(f"\n📊 3MIN SCALPING DASHBOARD - {self.get_thailand_time()}", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color("=" * 80, self.Fore.CYAN)
        self.print_color("💡 LIVE TRADING DISABLED - PAPER TRADING MODE", self.Fore.YELLOW)

    def start_trading(self):
        self.print_color("🚀 STARTING 3MIN SCALPING BOT!", self.Fore.CYAN + self.Style.BRIGHT)
        self.cycle_count = 0
        while True:
            try:
                self.cycle_count += 1
                self.print_color(f"\n🔄 3MIN CYCLE {self.cycle_count}", self.Fore.CYAN)
                self.print_color("=" * 50, self.Fore.CYAN)
                self.run_trading_cycle()
                self.print_color(f"⏰ Waiting 60 seconds for next 3MIN analysis...", self.Fore.BLUE)
                time.sleep(60)  # Wait 60 seconds for 3-minute timeframe
            except KeyboardInterrupt:
                self.print_color(f"\n🛑 BOT STOPPED", self.Fore.RED + self.Style.BRIGHT)
                self.show_trade_history(10)
                break
            except Exception as e:
                self.print_color(f"Main loop error: {e}", self.Fore.RED)
                time.sleep(60)

    def run_trading_cycle(self):
        try:
            self.display_dashboard()
            
            if hasattr(self, 'cycle_count') and self.cycle_count % 5 == 0:
                self.show_trade_history(5)
            
            # AI SCANS ALL PAIRS
            market_data = self.get_market_data()
            if market_data:
                self.print_color(f"\n🤖 3MIN AI SCANNING {len(market_data)} PAIRS...", self.Fore.BLUE + self.Style.BRIGHT)
                
                for pair in market_data.keys():
                    if self.can_open_new_trade(pair):
                        pair_data = {pair: market_data[pair]}
                        decision = self.get_ai_decision(pair_data)
                        
                        if decision["action"] == "TRADE":
                            direction_icon = "📈" if decision['direction'] == "LONG" else "📉"
                            self.print_color(f"🚀 3MIN QUALIFIED: {pair} {decision['direction']} {direction_icon} ({decision['confidence']}%)", self.Fore.GREEN + self.Style.BRIGHT)
                            success = self.execute_trade(decision)
                            if success:
                                pass
                        else:
                            self.print_color(f"⏸️ 3MIN HOLD: {pair} ({decision['confidence']}%)", self.Fore.YELLOW)
                    else:
                        self.print_color(f"↗️ 3MIN SKIPPED: {pair} (already active)", self.Fore.MAGENTA)
            else:
                self.print_color("No market data available", self.Fore.YELLOW)
                
        except Exception as e:
            self.print_color(f"Trading cycle error: {e}", self.Fore.RED)


class ThreeMinPaperTradingBot:
    def __init__(self, real_bot):
        self.real_bot = real_bot
        self.paper_balance = 1000
        self.paper_positions = {}
        self.paper_history = []
        self.is_paper_trading = True
        
        self.real_bot.print_color("🤖 3MIN PAPER TRADING BOT INITIALIZED!", self.real_bot.Fore.GREEN + self.real_bot.Style.BRIGHT)
        self.real_bot.print_color(f"💰 Starting Paper Balance: ${self.paper_balance}", self.real_bot.Fore.CYAN)
        self.real_bot.print_color(f"🎯 Strategy: 3MIN Scalping | TP: +0.8% | SL: -0.5%", self.real_bot.Fore.MAGENTA)
        
    def paper_execute_trade(self, decision):
        """AI decision ကို paper trade အဖြစ်အစားထိုးလုပ်မယ်"""
        try:
            pair = decision["pair"]
            direction = decision["direction"]
            confidence = decision["reason"]
            
            # Real market price ကိုယူမယ်
            current_price = self.real_bot.get_current_price(pair)
            
            if current_price <= 0:
                return False
            
            # Real bot ရဲ့ quantity calculation ကိုသုံးမယ်
            quantity = self.real_bot.get_quantity(pair, current_price)
            if quantity is None:
                return False
            
            # 3MIN SCALPING TP/SL
            if direction == "LONG":
                take_profit = current_price * (1 + self.real_bot.tp_percent)   # +0.8%
                stop_loss = current_price * (1 - self.real_bot.sl_percent)     # -0.5%
                tp_sl_ratio = f"TP: +{self.real_bot.tp_percent*100:.1f}% | SL: -{self.real_bot.sl_percent*100:.1f}%"
            else:  # SHORT
                take_profit = current_price * (1 - self.real_bot.tp_percent)   # -0.8%
                stop_loss = current_price * (1 + self.real_bot.sl_percent)     # +0.5%
                tp_sl_ratio = f"TP: -{self.real_bot.tp_percent*100:.1f}% | SL: +{self.real_bot.sl_percent*100:.1f}%"
            
            take_profit = self.real_bot.format_price(pair, take_profit)
            stop_loss = self.real_bot.format_price(pair, stop_loss)
            
            # Paper position ကို record လုပ်မယ်
            self.paper_positions[pair] = {
                "pair": pair,
                "direction": direction,
                "entry_price": current_price,
                "quantity": quantity,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "entry_time": time.time(),
                "status": 'ACTIVE',
                'ai_confidence': confidence,
                'entry_time_th': self.real_bot.get_thailand_time()
            }
            
            # Real bot ရဲ့ display ကိုသုံးမယ်
            direction_color = self.real_bot.Fore.BLUE if direction == 'LONG' else self.real_bot.Fore.RED
            direction_icon = "📈" if direction == 'LONG' else "📉"
            
            self.real_bot.print_color(f"📝 {direction_icon} 3MIN PAPER TRADE EXECUTED: {pair} {direction}", direction_color)
            self.real_bot.print_color(f"   Size: {quantity} | Entry: ${current_price:.4f}", self.real_bot.Fore.WHITE)
            self.real_bot.print_color(f"   TP: ${take_profit:.4f} | SL: ${stop_loss:.4f}", self.real_bot.Fore.YELLOW)
            self.real_bot.print_color(f"   {tp_sl_ratio}", self.real_bot.Fore.CYAN)
            self.real_bot.print_color(f"   AI Confidence: {confidence}%", self.real_bot.Fore.MAGENTA)
            
            return True
            
        except Exception as e:
            self.real_bot.print_color(f"Paper trade failed: {e}", self.real_bot.Fore.RED)
            return False

    def monitor_paper_positions(self):
        """Paper positions တွေကို monitor လုပ်မယ်"""
        try:
            for pair, trade in list(self.paper_positions.items()):
                if trade['status'] != 'ACTIVE':
                    continue
                
                # Real market price ကိုယူမယ်
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
                    # Close the paper position
                    trade['status'] = 'CLOSED'
                    trade['exit_price'] = current_price
                    trade['pnl'] = pnl
                    trade['close_reason'] = close_reason
                    trade['close_time'] = self.real_bot.get_thailand_time()
                    
                    # Update paper balance
                    self.paper_balance += pnl
                    
                    # Add to history
                    self.paper_history.append(trade.copy())
                    
                    # Display result
                    pnl_color = self.real_bot.Fore.GREEN if pnl > 0 else self.real_bot.Fore.RED
                    direction_icon = "📈" if trade['direction'] == 'LONG' else "📉"
                    self.real_bot.print_color(f"📊 {direction_icon} 3MIN PAPER TRADE CLOSED: {pair} {trade['direction']}", pnl_color)
                    self.real_bot.print_color(f"   Exit: ${current_price:.4f} | P&L: ${pnl:.2f}", pnl_color)
                    self.real_bot.print_color(f"   Reason: {close_reason}", self.real_bot.Fore.YELLOW)
                    
                    # Remove from active positions
                    del self.paper_positions[pair]
                else:
                    # Show unrealized P&L
                    if trade['direction'] == 'LONG':
                        unrealized_pnl = (current_price - trade['entry_price']) * trade['quantity']
                    else:
                        unrealized_pnl = (trade['entry_price'] - current_price) * trade['quantity']
                    
                    pnl_color = self.real_bot.Fore.GREEN if unrealized_pnl > 0 else self.real_bot.Fore.RED
                    self.real_bot.print_color(f"   {pair} Unrealized P&L: ${unrealized_pnl:.2f}", pnl_color)
                    
        except Exception as e:
            self.real_bot.print_color(f"Paper monitoring error: {e}", self.real_bot.Fore.RED)

    def get_paper_portfolio_status(self):
        """Paper trading portfolio status ပြမယ်"""
        total_trades = len(self.paper_history)
        winning_trades = len([t for t in self.paper_history if t.get('pnl', 0) > 0])
        losing_trades = len([t for t in self.paper_history if t.get('pnl', 0) < 0])
        total_pnl = sum(trade.get('pnl', 0) for trade in self.paper_history)
        
        # Calculate unrealized P&L
        unrealized_total = 0
        for pair, position in self.paper_positions.items():
            current_price = self.real_bot.get_current_price(pair)
            if current_price:
                if position['direction'] == 'LONG':
                    unrealized = (current_price - position['entry_price']) * position['quantity']
                else:
                    unrealized = (position['entry_price'] - current_price) * position['quantity']
                unrealized_total += unrealized
        
        total_value = self.paper_balance + unrealized_total
        
        self.real_bot.print_color(f"\n💼 3MIN PAPER TRADING PORTFOLIO", self.real_bot.Fore.CYAN + self.real_bot.Style.BRIGHT)
        self.real_bot.print_color("=" * 60, self.real_bot.Fore.CYAN)
        self.real_bot.print_color(f"Active Positions: {len(self.paper_positions)}", self.real_bot.Fore.WHITE)
        self.real_bot.print_color(f"Paper Balance: ${self.paper_balance:.2f}", self.real_bot.Fore.WHITE)
        self.real_bot.print_color(f"Unrealized P&L: ${unrealized_total:.2f}", 
                                self.real_bot.Fore.GREEN if unrealized_total > 0 else self.real_bot.Fore.RED)
        self.real_bot.print_color(f"Total Value: ${total_value:.2f}", self.real_bot.Fore.CYAN)
        self.real_bot.print_color(f"Total Trades: {total_trades}", self.real_bot.Fore.WHITE)
        
        if total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100
            self.real_bot.print_color(f"Win Rate: {win_rate:.1f}%", 
                                    self.real_bot.Fore.GREEN if win_rate > 50 else self.real_bot.Fore.YELLOW)
            self.real_bot.print_color(f"Total P&L: ${total_pnl:.2f}", 
                                    self.real_bot.Fore.GREEN if total_pnl > 0 else self.real_bot.Fore.RED)
            avg_trade_pnl = total_pnl / total_trades
            self.real_bot.print_color(f"Avg Trade P&L: ${avg_trade_pnl:.2f}", 
                                    self.real_bot.Fore.GREEN if avg_trade_pnl > 0 else self.real_bot.Fore.RED)

    def show_paper_trade_history(self, limit=10):
        """Paper trade history ပြမယ်"""
        if not self.paper_history:
            self.real_bot.print_color("No paper trade history found", self.real_bot.Fore.YELLOW)
            return
            
        self.real_bot.print_color(f"\n📋 3MIN PAPER TRADE HISTORY (Last {min(limit, len(self.paper_history))} trades)", self.real_bot.Fore.CYAN)
        self.real_bot.print_color("=" * 90, self.real_bot.Fore.CYAN)
        
        for i, trade in enumerate(reversed(self.paper_history[-limit:])):
            pnl = trade.get('pnl', 0)
            pnl_color = self.real_bot.Fore.GREEN if pnl > 0 else self.real_bot.Fore.RED if pnl < 0 else self.real_bot.Fore.YELLOW
            direction_icon = "📈" if trade['direction'] == 'LONG' else "📉"
            
            self.real_bot.print_color(f"{i+1}. {direction_icon} {trade['pair']} {trade['direction']} | Entry: ${trade.get('entry_price', 0):.4f} | Exit: ${trade.get('exit_price', 0):.4f} | P&L: ${pnl:.2f}", pnl_color)
            self.real_bot.print_color(f"   TP: ${trade.get('take_profit', 0):.4f} | SL: ${trade.get('stop_loss', 0):.4f} | Reason: {trade.get('close_reason', 'N/A')}", self.real_bot.Fore.YELLOW)

    def run_paper_trading_cycle(self):
        """3MIN AI paper trading cycle"""
        try:
            # Monitor existing paper positions
            self.monitor_paper_positions()
            
            # Get AI decisions for all pairs
            market_data = self.real_bot.get_market_data()
            if market_data:
                self.real_bot.print_color(f"\n🤖 3MIN AI SCANNING {len(market_data)} PAIRS...", self.real_bot.Fore.BLUE + self.real_bot.Style.BRIGHT)
                
                for pair in market_data.keys():
                    # Check if we can open new paper position
                    if pair not in self.paper_positions and len(self.paper_positions) < self.real_bot.max_concurrent_trades:
                        pair_data = {pair: market_data[pair]}
                        decision = self.real_bot.get_ai_decision(pair_data)
                        
                        if decision["action"] == "TRADE":
                            direction_icon = "📈" if decision['direction'] == "LONG" else "📉"
                            self.real_bot.print_color(f"🎯 3MIN AI PAPER SIGNAL: {pair} {decision['direction']} {direction_icon} ({decision['confidence']}%)", self.real_bot.Fore.GREEN + self.real_bot.Style.BRIGHT)
                            self.paper_execute_trade(decision)
                        else:
                            self.real_bot.print_color(f"⏸️ 3MIN AI HOLD: {pair} ({decision['confidence']}%)", self.real_bot.Fore.YELLOW)
                    else:
                        if pair in self.paper_positions:
                            direction_icon = "📈" if self.paper_positions[pair]['direction'] == 'LONG' else "📉"
                            self.real_bot.print_color(f"↗️ 3MIN ACTIVE PAPER POSITION: {pair} {direction_icon}", self.real_bot.Fore.MAGENTA)
            
            # Show portfolio status
            self.get_paper_portfolio_status()
            
        except Exception as e:
            self.real_bot.print_color(f"3MIN Paper trading cycle error: {e}", self.real_bot.Fore.RED)

    def start_paper_trading(self):
        """Start 3MIN AI paper trading"""
        self.real_bot.print_color("🚀 STARTING 3MIN AI PAPER TRADING BOT!", self.real_bot.Fore.GREEN + self.real_bot.Style.BRIGHT)
        self.real_bot.print_color("📝 3MIN CHART ANALYSIS + PAPER EXECUTION", self.real_bot.Fore.CYAN)
        self.real_bot.print_color("💰 NO REAL MONEY AT RISK", self.real_bot.Fore.GREEN)
        self.real_bot.print_color(f"🎯 TP: +{self.real_bot.tp_percent*100:.1f}% | SL: -{self.real_bot.sl_percent*100:.1f}%", self.real_bot.Fore.MAGENTA)
        
        cycle_count = 0
        while True:
            try:
                cycle_count += 1
                self.real_bot.print_color(f"\n🔄 3MIN PAPER TRADING CYCLE {cycle_count}", self.real_bot.Fore.CYAN)
                self.real_bot.print_color("=" * 50, self.real_bot.Fore.CYAN)
                
                self.run_paper_trading_cycle()
                
                # Show history every 10 cycles
                if cycle_count % 10 == 0:
                    self.show_paper_trade_history(5)
                
                self.real_bot.print_color(f"⏰ Waiting 60 seconds for next 3MIN analysis...", self.real_bot.Fore.BLUE)
                time.sleep(60)  # Wait 60 seconds for 3-minute timeframe
                
            except KeyboardInterrupt:
                self.real_bot.print_color(f"\n🛑 3MIN PAPER TRADING STOPPED", self.real_bot.Fore.RED + self.real_bot.Style.BRIGHT)
                self.show_paper_trade_history(10)
                break
            except Exception as e:
                self.real_bot.print_color(f"3MIN Paper trading error: {e}", self.real_bot.Fore.RED)
                time.sleep(60)


if __name__ == "__main__":
    try:
        # Real bot ကိုတည်ဆောက်မယ်
        real_bot = ThreeMinScalpingBot()
        
        print("\n" + "="*60)
        print("🤖 3MIN AI SCALPING BOT")
        print("="*60)
        print("SELECT TRADING MODE:")
        print("1. 🔴 Live Trading (Real Money - RISKY)")
        print("2. 🟢 3MIN Paper Trading (RECOMMENDED - NO RISK)")
        
        choice = input("Enter choice (1-2): ").strip()
        
        if choice == "1":
            print("⚠️  WARNING: You are using REAL MONEY!")
            confirm = input("Type 'YES' to confirm: ").strip()
            if confirm.upper() == 'YES':
                real_bot.start_trading()
            else:
                print("Using 3MIN Paper Trading mode instead...")
                paper_bot = ThreeMinPaperTradingBot(real_bot)
                paper_bot.start_paper_trading()
        else:
            # Default to 3MIN Paper Trading
            paper_bot = ThreeMinPaperTradingBot(real_bot)
            paper_bot.start_paper_trading()
            
    except Exception as e:
        print(f"Failed to start bot: {e}")
