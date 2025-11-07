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

class AggressiveOneMinScalpingBot:
    def __init__(self):
        # Load config from .env file
        self.binance_api_key = os.getenv('BINANCE_API_KEY')
        self.binance_secret = os.getenv('BINANCE_SECRET_KEY')
        self.openrouter_key = os.getenv('OPENROUTER_API_KEY')
        
        # Store colorama references
        self.Fore = Fore
        self.Back = Back
        self.Style = Style
        self.COLORAMA_AVAILABLE = COLORAMA_AVAILABLE
        
        # Thailand timezone
        self.thailand_tz = pytz.timezone('Asia/Bangkok')
        
        # AGGRESSIVE 1MIN SCALPING PARAMETERS
        self.trade_size_usd = 100  # Increased size for aggressive trading
        self.leverage = 10  # Higher leverage
        self.tp_percent = 0.012   # +1.2% - More aggressive TP
        self.sl_percent = 0.008   # -0.8% - Tighter SL
        
        # Multi-pair parameters - More pairs for more opportunities
        self.max_concurrent_trades = 8  # Increased concurrent trades
        self.available_pairs = ["SOLUSDT", "AVAXUSDT", "XRPUSDT", "LINKUSDT", "DOTUSDT", 
                               "ADAUSDT", "MATICUSDT", "DOGEUSDT", "ATOMUSDT", "NEARUSDT"]
        
        # Track bot-opened trades only
        self.bot_opened_trades = {}
        
        # REAL TRADE HISTORY - For live trading only
        self.real_trade_history_file = "aggressive_1min_scalping_real_history.json"
        self.real_trade_history = self.load_real_trade_history()
        
        # Trading statistics for real trading
        self.real_total_trades = 0
        self.real_winning_trades = 0
        self.real_total_pnl = 0.0
        
        # Precision settings
        self.quantity_precision = {}
        self.price_precision = {}
        
        # Initialize Binance client
        try:
            self.binance = Client(self.binance_api_key, self.binance_secret)
            self.print_color(f"🔥 AGGRESSIVE 1MIN SCALPING BOT ACTIVATED! 🔥", self.Fore.RED + self.Style.BRIGHT)
            self.print_color(f"🎯 TP: +1.2% | SL: -0.8% | R:R = 1.5", self.Fore.GREEN + self.Style.BRIGHT)
            self.print_color(f"💰 Trade Size: ${self.trade_size_usd} | Leverage: {self.leverage}x", self.Fore.YELLOW + self.Style.BRIGHT)
            self.print_color(f"⏰ Chart: 1MIN | Max Trades: {self.max_concurrent_trades}", self.Fore.MAGENTA + self.Style.BRIGHT)
            self.print_color(f"🎲 Pairs: {len(self.available_pairs)}", self.Fore.CYAN + self.Style.BRIGHT)
            self.print_color(f"🤖 AI Model: Qwen3 Max (via OpenRouter)", self.Fore.BLUE + self.Style.BRIGHT)
        except Exception as e:
            self.print_color(f"Binance initialization failed: {e}", self.Fore.RED)
            # Create dummy client for paper trading
            self.binance = None
        
        self.validate_config()
        if self.binance:
            self.setup_futures()
            self.load_symbol_precision()
    
    def load_real_trade_history(self):
        """Load only REAL trading history"""
        try:
            if os.path.exists(self.real_trade_history_file):
                with open(self.real_trade_history_file, 'r') as f:
                    history = json.load(f)
                    # Calculate statistics from loaded history
                    self.real_total_trades = len(history)
                    self.real_winning_trades = len([t for t in history if t.get('pnl', 0) > 0])
                    self.real_total_pnl = sum(t.get('pnl', 0) for t in history)
                    return history
            return []
        except Exception as e:
            self.print_color(f"Error loading real trade history: {e}", self.Fore.RED)
            return []
    
    def save_real_trade_history(self):
        """Save only REAL trading history"""
        try:
            with open(self.real_trade_history_file, 'w') as f:
                json.dump(self.real_trade_history, f, indent=2)
        except Exception as e:
            self.print_color(f"Error saving real trade history: {e}", self.Fore.RED)
    
    def add_real_trade_to_history(self, trade_data):
        """Add trade to REAL trading history only"""
        try:
            trade_data['close_time'] = self.get_thailand_time()
            trade_data['close_timestamp'] = time.time()
            trade_data['trade_type'] = 'REAL'  # Mark as real trade
            self.real_trade_history.append(trade_data)
            
            # Update REAL statistics
            self.real_total_trades += 1
            pnl = trade_data.get('pnl', 0)
            self.real_total_pnl += pnl
            if pnl > 0:
                self.real_winning_trades += 1
                
            if len(self.real_trade_history) > 200:  # Keep more history
                self.real_trade_history = self.real_trade_history[-200:]
            self.save_real_trade_history()
            self.print_color(f"📝 REAL Trade saved: {trade_data['pair']} {trade_data['direction']} P&L: ${pnl:.2f}", self.Fore.CYAN)
        except Exception as e:
            self.print_color(f"Error adding real trade to history: {e}", self.Fore.RED)
    
    def show_real_trade_history(self, limit=15):
        """Show only REAL trading history"""
        if not self.real_trade_history:
            self.print_color("No REAL trade history found", self.Fore.YELLOW)
            return
        
        self.print_color(f"\n🔥 REAL TRADING HISTORY (Last {min(limit, len(self.real_trade_history))} trades)", self.Fore.RED + self.Style.BRIGHT)
        self.print_color("=" * 100, self.Fore.RED)
        
        recent_trades = self.real_trade_history[-limit:]
        for i, trade in enumerate(reversed(recent_trades)):
            pnl = trade.get('pnl', 0)
            pnl_color = self.Fore.GREEN + self.Style.BRIGHT if pnl > 0 else self.Fore.RED + self.Style.BRIGHT if pnl < 0 else self.Fore.YELLOW
            direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
            close_reason = trade.get('close_reason', 'MANUAL')
            
            self.print_color(f"{i+1:2d}. {direction_icon} {trade['pair']} | Entry: ${trade.get('entry_price', 0):.4f} | Exit: ${trade.get('exit_price', 0):.4f} | P&L: ${pnl:.2f}", pnl_color)
            self.print_color(f"     TP: ${trade.get('take_profit', 0):.4f} | SL: ${trade.get('stop_loss', 0):.4f} | {close_reason} | Time: {trade.get('close_time', 'N/A')}", self.Fore.YELLOW)
    
    def show_real_trading_stats(self):
        """Show only REAL trading statistics"""
        if self.real_total_trades == 0:
            return
            
        win_rate = (self.real_winning_trades / self.real_total_trades) * 100
        avg_trade = self.real_total_pnl / self.real_total_trades
        
        self.print_color(f"\n📊 REAL TRADING STATISTICS", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color("=" * 60, self.Fore.CYAN)
        self.print_color(f"Total REAL Trades: {self.real_total_trades} | Winning Trades: {self.real_winning_trades}", self.Fore.WHITE)
        self.print_color(f"REAL Win Rate: {win_rate:.1f}%", self.Fore.GREEN + self.Style.BRIGHT if win_rate > 50 else self.Fore.YELLOW)
        self.print_color(f"Total REAL P&L: ${self.real_total_pnl:.2f}", self.Fore.GREEN + self.Style.BRIGHT if self.real_total_pnl > 0 else self.Fore.RED + self.Style.BRIGHT)
        self.print_color(f"Average REAL P&L per Trade: ${avg_trade:.2f}", self.Fore.WHITE)
    
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
        if not all([self.binance_api_key, self.binance_secret, self.openrouter_key]):
            self.print_color("Missing API keys!", self.Fore.RED)
            return False
        try:
            if self.binance:
                self.binance.futures_exchange_info()
                self.print_color("✅ Binance connection successful!", self.Fore.GREEN + self.Style.BRIGHT)
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
                    self.binance.futures_change_margin_type(symbol=pair, marginType='ISOLATED')
                    self.print_color(f"✅ Leverage set for {pair}", self.Fore.GREEN)
                except Exception as e:
                    self.print_color(f"Leverage setup failed for {pair}: {e}", self.Fore.YELLOW)
            self.print_color("✅ Futures setup completed!", self.Fore.GREEN + self.Style.BRIGHT)
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
            self.print_color("✅ Symbol precision loaded", self.Fore.GREEN + self.Style.BRIGHT)
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

            # Aggressive quantity calculation
            base_quantities = {
                "SOLUSDT": 0.5, "AVAXUSDT": 5.0, "XRPUSDT": 30.0, 
                "LINKUSDT": 5.0, "DOTUSDT": 25.0, "ADAUSDT": 80.0,
                "MATICUSDT": 60.0, "DOGEUSDT": 200.0, "ATOMUSDT": 8.0,
                "NEARUSDT": 12.0
            }
            
            quantity = base_quantities.get(pair)
            if not quantity or quantity <= 0:
                quantity = round(self.trade_size_usd / price, 4)
                quantity = max(quantity, 0.001)

            precision = self.quantity_precision.get(pair, 3)
            quantity = round(quantity, precision)
            
            if quantity <= 0:
                self.print_color(f"Invalid quantity: {quantity} for {pair}", self.Fore.RED)
                return None
                
            actual_value = quantity * price * self.leverage
            self.print_color(f"📊 Quantity for {pair}: {quantity} = ${actual_value:.2f} (with leverage)", self.Fore.CYAN)
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
                entry_price = float(decision_data.get('entry_price', 0))
                confidence = float(decision_data.get('confidence', 50))
                reason = decision_data.get('reason', 'AI Analysis')
                take_profit = decision_data.get('take_profit')
                stop_loss = decision_data.get('stop_loss')
                
                if direction not in ['LONG', 'SHORT', 'HOLD']:
                    direction = 'HOLD'
                if confidence < 0 or confidence > 100:
                    confidence = 50
                if entry_price <= 0:
                    entry_price = None
                    
                # Convert TP/SL to floats if provided
                if take_profit:
                    take_profit = float(take_profit)
                if stop_loss:
                    stop_loss = float(stop_loss)
                    
                return direction, entry_price, confidence, reason, take_profit, stop_loss
            return 'HOLD', None, 50, 'No valid JSON found', None, None
        except Exception as e:
            self.print_color(f"AI response parsing failed: {e}", self.Fore.RED)
            return 'HOLD', None, 50, 'Parsing failed', None, None

    def get_qwen3_max_analysis(self, pair, market_data):
        try:
            if not self.openrouter_key:
                self.print_color("OpenRouter API key not found", self.Fore.RED)
                return "HOLD", None, 0, "No API key", None, None
            
            current_price = market_data['current_price']
            price_change = market_data.get('price_change', 0)
            volume_change = market_data.get('volume_change', 0)
            
            # AGGRESSIVE TRADING PROMPT - AI controls everything
            prompt = f"""
            AGGRESSIVE 1-MINUTE SCALPING ANALYSIS for {pair}
            
            CURRENT MARKET DATA:
            - Current Price: ${current_price:.6f}
            - Price Change (5min): {price_change:.2f}%
            - Volume Change: {volume_change:.2f}%
            - Recent Prices: {market_data.get('prices', [])[-8:]} (latest on right)
            - Highs: {market_data.get('highs', [])[-5:]}
            - Lows: {market_data.get('lows', [])[-5:]}
            
            AGGRESSIVE TRADING STRATEGY:
            - Look for strong momentum signals
            - High conviction entries only
            - Aggressive position sizing
            - Quick scalps (1-3 minutes)
            
            YOU CONTROL EVERYTHING:
            - Direction (LONG/SHORT/HOLD)
            - Entry Price (exact price)
            - Take Profit (aggressive target)
            - Stop Loss (tight protection)
            - Confidence level
            
            Return VALID JSON only:
            {{
                "direction": "LONG" | "SHORT" | "HOLD",
                "entry_price": number,
                "take_profit": number,
                "stop_loss": number,
                "confidence": 0-100,
                "reason": "brief aggressive reason"
            }}
            
            Be aggressive but smart. Look for clear signals.
            """

            headers = {
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com",
                "X-Title": "Aggressive Scalping Bot"
            }
            
            data = {
                "model": "qwen/qwen3-max",
                "messages": [
                    {"role": "system", "content": "You are an AGGRESSIVE 1-minute scalper with deep thinking capabilities. Analyze market data thoroughly and take calculated risks. Return perfect JSON only with TP/SL. Think step by step before making decisions."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.4,  # Slightly higher temperature for aggressive decisions
                "max_tokens": 500
            }
            
            self.print_color(f"🤖 Qwen3 Max Analyzing {pair} for AGGRESSIVE entries...", self.Fore.MAGENTA + self.Style.BRIGHT)
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=45)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content'].strip()
                direction, entry_price, confidence, reason, take_profit, stop_loss = self.parse_ai_response(ai_response)
                
                # Log the AI decision
                direction_icon = "🟢 LONG" if direction == "LONG" else "🔴 SHORT" if direction == "SHORT" else "🟡 HOLD"
                color = self.Fore.GREEN + self.Style.BRIGHT if direction == "LONG" else self.Fore.RED + self.Style.BRIGHT if direction == "SHORT" else self.Fore.YELLOW
                
                self.print_color(f"{direction_icon} {pair} | Entry: ${entry_price} | Confidence: {confidence}%", color)
                if take_profit and stop_loss:
                    self.print_color(f"   AI TP: ${take_profit:.4f} | AI SL: ${stop_loss:.4f}", self.Fore.CYAN)
                self.print_color(f"   Reason: {reason}", self.Fore.YELLOW)
                return direction, entry_price, confidence, reason, take_profit, stop_loss
            else:
                self.print_color(f"Qwen3 Max API error: {response.status_code} - {response.text}", self.Fore.RED)
                return "HOLD", None, 0, f"API Error", None, None
                
        except Exception as e:
            self.print_color(f"Qwen3 Max analysis failed: {e}", self.Fore.RED)
            return "HOLD", None, 0, f"Error", None, None

    def get_price_history(self, pair, limit=15):
        try:
            if self.binance:
                klines = self.binance.futures_klines(symbol=pair, interval=Client.KLINE_INTERVAL_1MINUTE, limit=limit)
                prices = [float(k[4]) for k in klines]
                highs = [float(k[2]) for k in klines]
                lows = [float(k[3]) for k in klines]
                volumes = [float(k[5]) for k in klines]
                
                current_price = prices[-1] if prices else 0
                price_change = ((current_price - prices[-6]) / prices[-6] * 100) if len(prices) >= 6 else 0
                volume_change = ((volumes[-1] - volumes[-6]) / volumes[-6] * 100) if len(volumes) >= 6 else 0
                
                return {
                    'prices': prices, 
                    'highs': highs,
                    'lows': lows,
                    'volumes': volumes,
                    'current_price': current_price,
                    'price_change': price_change,
                    'volume_change': volume_change
                }
            else:
                current_price = self.get_current_price(pair)
                return {
                    'prices': [current_price] * 10, 
                    'highs': [current_price * 1.01] * 10,
                    'lows': [current_price * 0.99] * 10,
                    'volumes': [100000] * 10,
                    'current_price': current_price,
                    'price_change': 0.5,
                    'volume_change': 10.2
                }
        except Exception as e:
            current_price = self.get_current_price(pair)
            return {
                'prices': [current_price] * 10,
                'highs': [current_price * 1.01] * 10,
                'lows': [current_price * 0.99] * 10,
                'volumes': [100000] * 10,
                'current_price': current_price,
                'price_change': 0.5,
                'volume_change': 10.2
            }

    def get_ai_decision(self, pair_data):
        try:
            pair = list(pair_data.keys())[0]
            current_price = pair_data[pair]['price']
            if current_price <= 0:
                return {"action": "HOLD", "pair": pair, "direction": "HOLD", "confidence": 0, "reason": "Invalid price"}
            
            self.print_color(f"🔍 Analyzing {pair} at ${current_price:.4f} for AGGRESSIVE entry...", self.Fore.BLUE + self.Style.BRIGHT)
            market_data = self.get_price_history(pair)
            market_data['current_price'] = current_price
            
            direction, entry_price, confidence, reason, take_profit, stop_loss = self.get_qwen3_max_analysis(pair, market_data)
            
            if direction == "HOLD" or confidence < 75:  # Higher confidence threshold
                self.print_color(f"🟡 HOLD {pair} ({confidence}% confidence)", self.Fore.YELLOW)
                return {"action": "HOLD", "pair": pair, "direction": direction, "confidence": confidence, "reason": reason}
            else:
                direction_icon = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
                color = self.Fore.GREEN + self.Style.BRIGHT if direction == "LONG" else self.Fore.RED + self.Style.BRIGHT
                self.print_color(f"🎯 AGGRESSIVE SIGNAL: {direction_icon} {pair} @ ${entry_price} ({confidence}%)", color)
                
                return {
                    "action": "TRADE",
                    "pair": pair,
                    "direction": direction,
                    "entry_price": entry_price,
                    "take_profit": take_profit,
                    "stop_loss": stop_loss,
                    "confidence": confidence,
                    "reason": reason
                }
                
        except Exception as e:
            self.print_color(f"AI decision failed: {e}", self.Fore.RED)
            return {"action": "HOLD", "pair": list(pair_data.keys())[0], "direction": "HOLD", "confidence": 0, "reason": f"Error: {str(e)}"}

    def get_current_price(self, pair):
        try:
            if self.binance:
                ticker = self.binance.futures_symbol_ticker(symbol=pair)
                return float(ticker['price'])
            else:
                try:
                    import requests
                    url = f"https://api.binance.com/api/v3/ticker/price"
                    params = {'symbol': pair}
                    response = requests.get(url, params=params, timeout=5)
                    data = response.json()
                    return float(data['price'])
                except:
                    base_prices = {
                        "SOLUSDT": 180.50, "AVAXUSDT": 35.20, "XRPUSDT": 0.62,
                        "LINKUSDT": 18.75, "DOTUSDT": 8.90, "ADAUSDT": 0.48,
                        "MATICUSDT": 0.78, "DOGEUSDT": 0.12, "ATOMUSDT": 10.25,
                        "NEARUSDT": 7.80
                    }
                    return base_prices.get(pair, 100)
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
        """AGGRESSIVE LIVE TRADING - AI controls everything"""
        try:
            pair = decision["pair"]
            if not self.can_open_new_trade(pair):
                self.print_color(f"🚫 Cannot open {pair} - position exists or max trades reached", self.Fore.RED)
                return False
            
            direction = decision["direction"]
            entry_price = decision["entry_price"]
            confidence = decision["confidence"]
            reason = decision["reason"]
            ai_take_profit = decision.get("take_profit")
            ai_stop_loss = decision.get("stop_loss")
            
            # Use AI's entry price
            if entry_price is None or entry_price <= 0:
                self.print_color(f"Invalid AI entry price", self.Fore.RED)
                return False

            # Calculate quantity
            quantity = self.get_quantity(pair, entry_price)
            if quantity is None:
                return False
            
            # Use AI's TP/SL if provided, otherwise use default aggressive values
            if ai_take_profit and ai_stop_loss:
                take_profit = ai_take_profit
                stop_loss = ai_stop_loss
                tp_sl_source = "AI"
            else:
                if direction == "LONG":
                    take_profit = entry_price * (1 + self.tp_percent)
                    stop_loss = entry_price * (1 - self.sl_percent)
                else:
                    take_profit = entry_price * (1 - self.tp_percent)
                    stop_loss = entry_price * (1 + self.sl_percent)
                tp_sl_source = "DEFAULT"
            
            take_profit = self.format_price(pair, take_profit)
            stop_loss = self.format_price(pair, stop_loss)
            
            # Display aggressive trade details
            direction_color = self.Fore.GREEN + self.Style.BRIGHT if direction == 'LONG' else self.Fore.RED + self.Style.BRIGHT
            direction_icon = "🟢 LONG" if direction == 'LONG' else "🔴 SHORT"
            
            self.print_color(f"\n🎯 AGGRESSIVE LIVE TRADE EXECUTION", self.Fore.CYAN + self.Style.BRIGHT)
            self.print_color("=" * 70, self.Fore.CYAN)
            self.print_color(f"{direction_icon} {pair}", direction_color)
            self.print_color(f"ENTRY PRICE: ${entry_price:.4f}", self.Fore.GREEN + self.Style.BRIGHT)
            self.print_color(f"QUANTITY: {quantity} (Leverage: {self.leverage}x)", self.Fore.WHITE)
            self.print_color(f"TAKE PROFIT: ${take_profit:.4f} ({tp_sl_source})", self.Fore.GREEN)
            self.print_color(f"STOP LOSS: ${stop_loss:.4f} ({tp_sl_source})", self.Fore.RED)
            self.print_color(f"AI CONFIDENCE: {confidence}%", self.Fore.MAGENTA + self.Style.BRIGHT)
            self.print_color(f"REASON: {reason}", self.Fore.YELLOW)
            self.print_color("=" * 70, self.Fore.CYAN)
            
            # Execute live trade
            entry_side = 'BUY' if direction == 'LONG' else 'SELL'
            try:
                order = self.binance.futures_create_order(
                    symbol=pair,
                    side=entry_side,
                    type='MARKET',
                    quantity=quantity
                )
                self.print_color(f"✅ {direction} ORDER EXECUTED!", self.Fore.GREEN + self.Style.BRIGHT)
                time.sleep(1)
                
                # Set stop loss and take profit
                stop_side = 'SELL' if direction == 'LONG' else 'BUY'
                self.binance.futures_create_order(
                    symbol=pair, side=stop_side, type='STOP_MARKET',
                    quantity=quantity, stopPrice=stop_loss, reduceOnly=True
                )
                self.binance.futures_create_order(
                    symbol=pair, side=stop_side, type='TAKE_PROFIT_MARKET',
                    quantity=quantity, stopPrice=take_profit, reduceOnly=True
                )
                
                self.bot_opened_trades[pair] = {
                    "pair": pair, "direction": direction, "entry_price": entry_price,
                    "quantity": quantity, "stop_loss": stop_loss, "take_profit": take_profit,
                    "entry_time": time.time(), "status": 'ACTIVE', 'ai_confidence': confidence,
                    'ai_reason': reason, 'entry_time_th': self.get_thailand_time(),
                    'tp_sl_source': tp_sl_source
                }
                
                self.print_color(f"🔥 LIVE TRADE ACTIVATED: {pair} {direction}", self.Fore.GREEN + self.Style.BRIGHT)
                return True
                
            except Exception as e:
                self.print_color(f"❌ Execution Error: {e}", self.Fore.RED)
                return False
            
        except Exception as e:
            self.print_color(f"❌ Trade failed: {e}", self.Fore.RED)
            return False

    def get_live_position_data(self, pair):
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
        try:
            closed_trades = []
            for pair, trade in list(self.bot_opened_trades.items()):
                if trade['status'] != 'ACTIVE':
                    continue
                
                live_data = self.get_live_position_data(pair)
                if not live_data:
                    # Position closed
                    self.close_real_trade_with_cleanup(pair, trade, "AUTO CLOSE")
                    closed_trades.append(pair)
                    continue
                    
                direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
                pnl_color = self.Fore.GREEN + self.Style.BRIGHT if live_data['unrealized_pnl'] >= 0 else self.Fore.RED + self.Style.BRIGHT
                
                self.print_color(f"\n📊 LIVE: {pair} {direction_icon} | P&L: ${live_data['unrealized_pnl']:.2f}", pnl_color)
                self.print_color(f"   Entry: ${trade['entry_price']:.4f} | Current: ${live_data['current_price']:.4f}", self.Fore.WHITE)
                self.print_color(f"   TP: ${trade['take_profit']:.4f} | SL: ${trade['stop_loss']:.4f}", self.Fore.YELLOW)
                    
            return closed_trades
        except Exception as e:
            self.print_color(f"Monitoring error: {e}", self.Fore.RED)
            return []

    def close_real_trade_with_cleanup(self, pair, trade, close_reason="MANUAL"):
        try:
            # Cancel existing orders
            open_orders = self.binance.futures_get_open_orders(symbol=pair)
            canceled = 0
            for order in open_orders:
                if order['reduceOnly'] and order['symbol'] == pair:
                    try:
                        self.binance.futures_cancel_order(symbol=pair, orderId=order['orderId'])
                        canceled += 1
                    except: pass
            
            final_pnl = self.get_final_pnl(pair, trade)
            trade['status'] = 'CLOSED'
            trade['exit_time_th'] = self.get_thailand_time()
            trade['exit_price'] = self.get_current_price(pair)
            trade['pnl'] = final_pnl
            trade['close_reason'] = close_reason
            
            closed_trade = trade.copy()
            self.add_real_trade_to_history(closed_trade)  # Save to REAL history only
            
            pnl_color = self.Fore.GREEN + self.Style.BRIGHT if final_pnl > 0 else self.Fore.RED + self.Style.BRIGHT
            direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
            self.print_color(f"\n🔚 REAL TRADE CLOSED: {pair} {direction_icon}", pnl_color)
            self.print_color(f"   Final P&L: ${final_pnl:.2f} | Reason: {close_reason}", pnl_color)
            if canceled > 0:
                self.print_color(f"   Cleaned up {canceled} order(s)", self.Fore.CYAN)
                
            del self.bot_opened_trades[pair]
            
        except Exception as e:
            self.print_color(f"Cleanup failed for {pair}: {e}", self.Fore.RED)

    def get_final_pnl(self, pair, trade):
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
        self.print_color(f"\n🔥 AGGRESSIVE LIVE TRADING DASHBOARD - {self.get_thailand_time()}", self.Fore.RED + self.Style.BRIGHT)
        self.print_color("=" * 90, self.Fore.RED)
        
        active_count = 0
        total_unrealized = 0
        
        for pair, trade in self.bot_opened_trades.items():
            if trade['status'] == 'ACTIVE':
                active_count += 1
                live_data = self.get_live_position_data(pair)
                if live_data:
                    direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
                    pnl_color = self.Fore.GREEN + self.Style.BRIGHT if live_data['unrealized_pnl'] >= 0 else self.Fore.RED + self.Style.BRIGHT
                    total_unrealized += live_data['unrealized_pnl']
                    
                    self.print_color(f"{direction_icon} {pair}", self.Fore.WHITE + self.Style.BRIGHT)
                    self.print_color(f"   Entry: ${trade['entry_price']:.4f} | Current: ${live_data['current_price']:.4f}", self.Fore.WHITE)
                    self.print_color(f"   P&L: ${live_data['unrealized_pnl']:.2f}", pnl_color)
                    self.print_color(f"   TP: ${trade['take_profit']:.4f} | SL: ${trade['stop_loss']:.4f}", self.Fore.YELLOW)
                    self.print_color(f"   AI Confidence: {trade.get('ai_confidence', 0)}%", self.Fore.MAGENTA)
                    self.print_color("   " + "-" * 60, self.Fore.CYAN)
        
        if active_count == 0:
            self.print_color("No active positions", self.Fore.YELLOW)
        else:
            total_color = self.Fore.GREEN + self.Style.BRIGHT if total_unrealized >= 0 else self.Fore.RED + self.Style.BRIGHT
            self.print_color(f"📊 Active Positions: {active_count} | Total Unrealized P&L: ${total_unrealized:.2f}", total_color)

    def run_trading_cycle(self):
        try:
            closed_trades = self.monitor_positions()
            self.display_dashboard()
            
            # Show REAL stats every 5 cycles
            if hasattr(self, 'cycle_count') and self.cycle_count % 5 == 0:
                self.show_real_trade_history(8)
                self.show_real_trading_stats()
            
            market_data = self.get_market_data()
            if market_data:
                self.print_color(f"\n🔍 AGGRESSIVE AI SCANNING {len(market_data)} PAIRS...", self.Fore.BLUE + self.Style.BRIGHT)
                
                qualified_signals = 0
                for pair in market_data.keys():
                    if self.can_open_new_trade(pair):
                        pair_data = {pair: market_data[pair]}
                        decision = self.get_ai_decision(pair_data)
                        
                        if decision["action"] == "TRADE":
                            qualified_signals += 1
                            direction_icon = "🟢 LONG" if decision['direction'] == "LONG" else "🔴 SHORT"
                            self.print_color(f"🎯 QUALIFIED: {pair} {direction_icon}", self.Fore.GREEN + self.Style.BRIGHT)
                            success = self.execute_trade(decision)
                            if success:
                                time.sleep(1)  # Small delay between executions
                        else:
                            if decision['confidence'] >= 70:  # Show high confidence holds
                                self.print_color(f"🟡 HIGH CONFIDENCE HOLD: {pair} ({decision['confidence']}%)", self.Fore.YELLOW)
                    else:
                        if pair not in self.bot_opened_trades:
                            self.print_color(f"⏸️  SKIPPED: {pair} (max trades)", self.Fore.MAGENTA)
                
                if qualified_signals == 0:
                    self.print_color("No qualified signals this cycle", self.Fore.YELLOW)
                else:
                    self.print_color(f"🎯 {qualified_signals} qualified signals found", self.Fore.GREEN + self.Style.BRIGHT)
            else:
                self.print_color("No market data available", self.Fore.RED)
                
        except Exception as e:
            self.print_color(f"Cycle error: {e}", self.Fore.RED)

    def start_trading(self):
        self.print_color("🔥 STARTING AGGRESSIVE 1MIN LIVE TRADING BOT!", self.Fore.RED + self.Style.BRIGHT)
        self.print_color("⚠️  REAL MONEY TRADING - HIGH RISK! ⚠️", self.Fore.RED + self.Style.BRIGHT)
        self.print_color("🤖 Qwen3 Max AI FULLY CONTROLS: Entry, TP, SL, Direction", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color("💾 REAL trades saved to: aggressive_1min_scalping_real_history.json", self.Fore.GREEN)
        self.cycle_count = 0
        
        while True:
            try:
                self.cycle_count += 1
                self.print_color(f"\n🎯 AGGRESSIVE CYCLE {self.cycle_count}", self.Fore.RED + self.Style.BRIGHT)
                self.print_color("=" * 60, self.Fore.RED)
                self.run_trading_cycle()
                self.print_color(f"⏳ Waiting 25 seconds for next cycle...", self.Fore.BLUE)
                time.sleep(25)  # Slightly faster cycles for aggressive trading
                
            except KeyboardInterrupt:
                self.print_color(f"\n🛑 AGGRESSIVE TRADING STOPPED", self.Fore.RED + self.Style.BRIGHT)
                self.show_real_trade_history(10)
                self.show_real_trading_stats()
                break
            except Exception as e:
                self.print_color(f"Main loop error: {e}", self.Fore.RED)
                time.sleep(25)


class AggressiveOneMinPaperTradingBot:
    def __init__(self, real_bot):
        self.real_bot = real_bot
        # Copy colorama attributes from real_bot
        self.Fore = real_bot.Fore
        self.Back = real_bot.Back
        self.Style = real_bot.Style
        self.COLORAMA_AVAILABLE = real_bot.COLORAMA_AVAILABLE
        
        self.paper_balance = 5000  # Higher paper balance for aggressive trading
        self.paper_positions = {}
        self.paper_history_file = "aggressive_1min_scalping_paper_history.json"
        self.paper_history = self.load_paper_history()
        
        self.real_bot.print_color("🔥 AGGRESSIVE 1MIN PAPER TRADING BOT INITIALIZED!", self.Fore.GREEN + self.Style.BRIGHT)
        self.real_bot.print_color(f"💰 Starting Paper Balance: ${self.paper_balance}", self.Fore.CYAN + self.Style.BRIGHT)
        self.real_bot.print_color(f"🎯 Strategy: AGGRESSIVE 1MIN Scalping | TP: +1.2% | SL: -0.8%", self.Fore.MAGENTA + self.Style.BRIGHT)
        self.real_bot.print_color(f"🤖 Qwen3 Max AI Full Control: Entry, TP, SL, Direction", self.Fore.CYAN + self.Style.BRIGHT)
        self.real_bot.print_color(f"💾 Paper trades saved to: {self.paper_history_file}", self.Fore.GREEN)
        
    def load_paper_history(self):
        """Load PAPER trading history only"""
        try:
            if os.path.exists(self.paper_history_file):
                with open(self.paper_history_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            self.real_bot.print_color(f"Error loading paper trade history: {e}", self.Fore.RED)
            return []
    
    def save_paper_history(self):
        """Save PAPER trading history only"""
        try:
            with open(self.paper_history_file, 'w') as f:
                json.dump(self.paper_history, f, indent=2)
        except Exception as e:
            self.real_bot.print_color(f"Error saving paper trade history: {e}", self.Fore.RED)
    
    def add_paper_trade_to_history(self, trade_data):
        """Add trade to PAPER trading history only"""
        try:
            trade_data['close_time'] = self.real_bot.get_thailand_time()
            trade_data['close_timestamp'] = time.time()
            trade_data['trade_type'] = 'PAPER'  # Mark as paper trade
            self.paper_history.append(trade_data)
            
            if len(self.paper_history) > 200:  # Keep more history
                self.paper_history = self.paper_history[-200:]
            self.save_paper_history()
            self.real_bot.print_color(f"📝 PAPER Trade saved: {trade_data['pair']} {trade_data['direction']} P&L: ${trade_data.get('pnl', 0):.2f}", self.Fore.CYAN)
        except Exception as e:
            self.real_bot.print_color(f"Error adding paper trade to history: {e}", self.Fore.RED)
    
    def show_paper_trade_history(self, limit=15):
        """Show only PAPER trading history"""
        if not self.paper_history:
            self.real_bot.print_color("No PAPER trade history found", self.Fore.YELLOW)
            return
        
        self.real_bot.print_color(f"\n📝 PAPER TRADING HISTORY (Last {min(limit, len(self.paper_history))} trades)", self.Fore.GREEN + self.Style.BRIGHT)
        self.real_bot.print_color("=" * 100, self.Fore.GREEN)
        
        recent_trades = self.paper_history[-limit:]
        for i, trade in enumerate(reversed(recent_trades)):
            pnl = trade.get('pnl', 0)
            pnl_color = self.Fore.GREEN + self.Style.BRIGHT if pnl > 0 else self.Fore.RED + self.Style.BRIGHT if pnl < 0 else self.Fore.YELLOW
            direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
            close_reason = trade.get('close_reason', 'MANUAL')
            
            self.real_bot.print_color(f"{i+1:2d}. {direction_icon} {trade['pair']} | Entry: ${trade.get('entry_price', 0):.4f} | Exit: ${trade.get('exit_price', 0):.4f} | P&L: ${pnl:.2f}", pnl_color)
            self.real_bot.print_color(f"     TP: ${trade.get('take_profit', 0):.4f} | SL: ${trade.get('stop_loss', 0):.4f} | {close_reason} | Time: {trade.get('close_time', 'N/A')}", self.Fore.YELLOW)
    
    def paper_execute_trade(self, decision):
        try:
            pair = decision["pair"]
            direction = decision["direction"]
            entry_price = decision["entry_price"]
            confidence = decision["confidence"]
            reason = decision["reason"]
            ai_take_profit = decision.get("take_profit")
            ai_stop_loss = decision.get("stop_loss")
            
            if entry_price is None or entry_price <= 0:
                return False
            
            quantity = self.real_bot.get_quantity(pair, entry_price)
            if quantity is None:
                return False
            
            # Use AI's TP/SL if provided
            if ai_take_profit and ai_stop_loss:
                take_profit = ai_take_profit
                stop_loss = ai_stop_loss
                tp_sl_source = "AI"
            else:
                if direction == "LONG":
                    take_profit = entry_price * (1 + self.real_bot.tp_percent)
                    stop_loss = entry_price * (1 - self.real_bot.sl_percent)
                else:
                    take_profit = entry_price * (1 - self.real_bot.tp_percent)
                    stop_loss = entry_price * (1 + self.real_bot.sl_percent)
                tp_sl_source = "DEFAULT"
            
            take_profit = self.real_bot.format_price(pair, take_profit)
            stop_loss = self.real_bot.format_price(pair, stop_loss)
            
            direction_color = self.Fore.GREEN + self.Style.BRIGHT if direction == 'LONG' else self.Fore.RED + self.Style.BRIGHT
            direction_icon = "🟢 LONG" if direction == 'LONG' else "🔴 SHORT"
            
            self.real_bot.print_color(f"\n🎯 PAPER TRADE EXECUTION", self.Fore.CYAN + self.Style.BRIGHT)
            self.real_bot.print_color("=" * 70, self.Fore.CYAN)
            self.real_bot.print_color(f"{direction_icon} {pair}", direction_color)
            self.real_bot.print_color(f"ENTRY (AI): ${entry_price:.4f}", self.Fore.GREEN + self.Style.BRIGHT)
            self.real_bot.print_color(f"TP: ${take_profit:.4f} ({tp_sl_source})", self.Fore.GREEN)
            self.real_bot.print_color(f"SL: ${stop_loss:.4f} ({tp_sl_source})", self.Fore.RED)
            self.real_bot.print_color(f"CONFIDENCE: {confidence}%", self.Fore.MAGENTA + self.Style.BRIGHT)
            self.real_bot.print_color("=" * 70, self.Fore.CYAN)
            
            self.paper_positions[pair] = {
                "pair": pair, "direction": direction, "entry_price": entry_price,
                "quantity": quantity, "stop_loss": stop_loss, "take_profit": take_profit,
                "entry_time": time.time(), "status": 'ACTIVE', 'ai_confidence': confidence,
                'entry_time_th': self.real_bot.get_thailand_time(), 'tp_sl_source': tp_sl_source
            }
            
            return True
            
        except Exception as e:
            self.real_bot.print_color(f"❌ Paper trade failed: {e}", self.Fore.RED)
            return False

    def monitor_paper_positions(self):
        try:
            closed_positions = []
            for pair, trade in list(self.paper_positions.items()):
                if trade['status'] != 'ACTIVE':
                    continue
                
                current_price = self.real_bot.get_current_price(pair)
                if not current_price:
                    continue
                
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
                else:
                    if current_price <= trade['take_profit']:
                        should_close = True
                        close_reason = "TP HIT"
                        pnl = (trade['entry_price'] - current_price) * trade['quantity']
                    elif current_price >= trade['stop_loss']:
                        should_close = True
                        close_reason = "SL HIT"
                        pnl = (trade['entry_price'] - current_price) * trade['quantity']
                
                if should_close:
                    trade['status'] = 'CLOSED'
                    trade['exit_price'] = current_price
                    trade['pnl'] = pnl
                    trade['close_reason'] = close_reason
                    trade['close_time'] = self.real_bot.get_thailand_time()
                    
                    self.paper_balance += pnl
                    self.add_paper_trade_to_history(trade.copy())  # Save to PAPER history only
                    closed_positions.append(pair)
                    
                    pnl_color = self.Fore.GREEN + self.Style.BRIGHT if pnl > 0 else self.Fore.RED + self.Style.BRIGHT
                    direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
                    self.real_bot.print_color(f"\n🔚 PAPER TRADE CLOSED: {pair} {direction_icon}", pnl_color)
                    self.real_bot.print_color(f"   P&L: ${pnl:.2f} | Reason: {close_reason}", pnl_color)
                    self.real_bot.print_color(f"   New Balance: ${self.paper_balance:.2f}", self.Fore.CYAN)
                    
                    del self.paper_positions[pair]
                    
            return closed_positions
                    
        except Exception as e:
            self.real_bot.print_color(f"Paper monitoring error: {e}", self.Fore.RED)
            return []

    def get_paper_portfolio_status(self):
        total_trades = len(self.paper_history)
        winning_trades = len([t for t in self.paper_history if t.get('pnl', 0) > 0])
        total_pnl = sum(trade.get('pnl', 0) for trade in self.paper_history)
        
        self.real_bot.print_color(f"\n📊 PAPER TRADING PORTFOLIO", self.Fore.CYAN + self.Style.BRIGHT)
        self.real_bot.print_color("=" * 70, self.Fore.CYAN)
        self.real_bot.print_color(f"Active Positions: {len(self.paper_positions)}", self.Fore.WHITE)
        self.real_bot.print_color(f"Balance: ${self.paper_balance:.2f}", self.Fore.WHITE + self.Style.BRIGHT)
        self.real_bot.print_color(f"Total PAPER Trades: {total_trades}", self.Fore.WHITE)
        
        if total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100
            self.real_bot.print_color(f"PAPER Win Rate: {win_rate:.1f}%", self.Fore.GREEN + self.Style.BRIGHT if win_rate > 50 else self.Fore.YELLOW)
            self.real_bot.print_color(f"Total PAPER P&L: ${total_pnl:.2f}", self.Fore.GREEN + self.Style.BRIGHT if total_pnl > 0 else self.Fore.RED + self.Style.BRIGHT)
            avg_trade = total_pnl / total_trades
            self.real_bot.print_color(f"Average PAPER P&L: ${avg_trade:.2f}", self.Fore.WHITE)

    def run_paper_trading_cycle(self):
        try:
            closed_positions = self.monitor_paper_positions()
            
            market_data = self.real_bot.get_market_data()
            if market_data:
                self.real_bot.print_color(f"\n🔍 AGGRESSIVE AI SCANNING FOR PAPER TRADES...", self.Fore.BLUE + self.Style.BRIGHT)
                
                qualified_signals = 0
                for pair in market_data.keys():
                    if pair not in self.paper_positions and len(self.paper_positions) < self.real_bot.max_concurrent_trades:
                        pair_data = {pair: market_data[pair]}
                        decision = self.real_bot.get_ai_decision(pair_data)
                        
                        if decision["action"] == "TRADE":
                            qualified_signals += 1
                            direction_icon = "🟢 LONG" if decision['direction'] == "LONG" else "🔴 SHORT"
                            self.real_bot.print_color(f"🎯 AI SIGNAL: {pair} {direction_icon}", self.Fore.GREEN + self.Style.BRIGHT)
                            self.paper_execute_trade(decision)
                            time.sleep(0.5)  # Small delay between paper executions
                
                if qualified_signals > 0:
                    self.real_bot.print_color(f"🎯 {qualified_signals} qualified paper signals executed", self.Fore.GREEN + self.Style.BRIGHT)
            
            # Show paper history every 8 cycles
            if hasattr(self, 'paper_cycle_count') and self.paper_cycle_count % 8 == 0:
                self.show_paper_trade_history(8)
            
            self.get_paper_portfolio_status()
            
        except Exception as e:
            self.real_bot.print_color(f"Paper trading error: {e}", self.Fore.RED)

    def start_paper_trading(self):
        self.real_bot.print_color("🔥 STARTING AGGRESSIVE 1MIN PAPER TRADING!", self.Fore.GREEN + self.Style.BRIGHT)
        self.real_bot.print_color("💸 NO REAL MONEY AT RISK", self.Fore.GREEN)
        self.real_bot.print_color("🤖 Qwen3 Max AI Full Control: Entry, TP, SL, Direction", self.Fore.CYAN)
        
        self.paper_cycle_count = 0
        while True:
            try:
                self.paper_cycle_count += 1
                self.real_bot.print_color(f"\n🎯 PAPER CYCLE {self.paper_cycle_count}", self.Fore.CYAN)
                self.real_bot.print_color("=" * 60, self.Fore.CYAN)
                self.run_paper_trading_cycle()
                self.real_bot.print_color(f"⏳ Waiting 25 seconds...", self.Fore.BLUE)
                time.sleep(25)
                
            except KeyboardInterrupt:
                self.real_bot.print_color(f"\n🛑 PAPER TRADING STOPPED", self.Fore.RED + self.Style.BRIGHT)
                
                # Show final paper trading results
                total_trades = len(self.paper_history)
                if total_trades > 0:
                    winning_trades = len([t for t in self.paper_history if t.get('pnl', 0) > 0])
                    total_pnl = sum(trade.get('pnl', 0) for trade in self.paper_history)
                    win_rate = (winning_trades / total_trades) * 100
                    
                    self.real_bot.print_color(f"\n📊 FINAL PAPER TRADING RESULTS", self.Fore.CYAN + self.Style.BRIGHT)
                    self.real_bot.print_color("=" * 50, self.Fore.CYAN)
                    self.real_bot.print_color(f"Total PAPER Trades: {total_trades}", self.Fore.WHITE)
                    self.real_bot.print_color(f"PAPER Win Rate: {win_rate:.1f}%", self.Fore.GREEN)
                    self.real_bot.print_color(f"Total PAPER P&L: ${total_pnl:.2f}", self.Fore.GREEN if total_pnl > 0 else self.Fore.RED)
                    self.real_bot.print_color(f"Final PAPER Balance: ${self.paper_balance:.2f}", self.Fore.CYAN + self.Style.BRIGHT)
                
                break
            except Exception as e:
                self.real_bot.print_color(f"Paper trading error: {e}", self.Fore.RED)
                time.sleep(25)

if __name__ == "__main__":
    try:
        real_bot = AggressiveOneMinScalpingBot()
        
        print("\n" + "="*80)
        print("🔥 AGGRESSIVE 1MIN AI SCALPING BOT")
        print("="*80)
        print("SELECT TRADING MODE:")
        print("1. 🔥 Live Trading (Real Money - HIGH RISK)")
        print("2. 💸 Paper Trading (No Risk)")
        
        choice = input("Enter choice (1-2): ").strip()
        
        if choice == "1":
            print("⚠️  WARNING: REAL MONEY TRADING! HIGH RISK! ⚠️")
            print("🤖 Qwen3 Max AI FULLY CONTROLS: Entry, TP, SL, Direction")
            print("💾 REAL trades saved to: aggressive_1min_scalping_real_history.json")
            confirm = input("Type 'AGGRESSIVE' to confirm: ").strip()
            if confirm.upper() == 'AGGRESSIVE':
                real_bot.start_trading()
            else:
                print("Using Paper Trading mode...")
                paper_bot = AggressiveOneMinPaperTradingBot(real_bot)
                paper_bot.start_paper_trading()
        else:
            paper_bot = AggressiveOneMinPaperTradingBot(real_bot)
            paper_bot.start_paper_trading()
            
    except Exception as e:
        print(f"Failed to start bot: {e}")
