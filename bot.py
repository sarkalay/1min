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

class FullyAutonomous1HourAITrader:
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
        
        # 🎯 FULLY AUTONOMOUS 1HOUR AI TRADING PARAMETERS
        self.total_budget = 5000  # $5000 budget for AI to manage
        self.available_budget = 5000  # Current available budget
        self.max_position_size_percent = 25  # Max 25% of budget per trade for 1hr
        self.max_concurrent_trades = 6  # Maximum concurrent positions - CHANGED TO 6
        
        # AI can trade selected 6 major pairs only
        self.available_pairs = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "DOGEUSDT", "AVAXUSDT"
        ]
        
        # Track AI-opened trades
        self.ai_opened_trades = {}
        
        # REAL TRADE HISTORY
        self.real_trade_history_file = "fully_autonomous_1hour_ai_trading_history.json"
        self.real_trade_history = self.load_real_trade_history()
        
        # Trading statistics
        self.real_total_trades = 0
        self.real_winning_trades = 0
        self.real_total_pnl = 0.0
        
        # Precision settings
        self.quantity_precision = {}
        self.price_precision = {}
        
        # Initialize Binance client
        try:
            self.binance = Client(self.binance_api_key, self.binance_secret)
            self.print_color(f"🤖 FULLY AUTONOMOUS 1HOUR AI TRADER ACTIVATED! 🤖", self.Fore.CYAN + self.Style.BRIGHT)
            self.print_color(f"💰 TOTAL BUDGET: ${self.total_budget}", self.Fore.GREEN + self.Style.BRIGHT)
            self.print_color(f"🎯 AI FULL CONTROL: Analysis, Entry, Size, TP, SL", self.Fore.MAGENTA + self.Style.BRIGHT)
            self.print_color(f"⏰ Timeframe: 1HOUR | Max Positions: {self.max_concurrent_trades}", self.Fore.YELLOW + self.Style.BRIGHT)
            self.print_color(f"📈 Pairs: {len(self.available_pairs)} selected (BTC, ETH, BNB, SOL, DOGE, AVAX)", self.Fore.BLUE + self.Style.BRIGHT)
            self.print_color(f"⚡ Leverage Range: 5x to 20x", self.Fore.RED + self.Style.BRIGHT)
            self.print_color(f"🧠 AI Model: DeepSeek Chat V3.1 (via OpenRouter)", self.Fore.CYAN + self.Style.BRIGHT)  # CHANGED MODEL
        except Exception as e:
            self.print_color(f"Binance initialization failed: {e}", self.Fore.RED)
            self.binance = None
        
        self.validate_config()
        if self.binance:
            self.setup_futures()
            self.load_symbol_precision()
    
    def load_real_trade_history(self):
        """Load trading history"""
        try:
            if os.path.exists(self.real_trade_history_file):
                with open(self.real_trade_history_file, 'r') as f:
                    history = json.load(f)
                    self.real_total_trades = len(history)
                    self.real_winning_trades = len([t for t in history if t.get('pnl', 0) > 0])
                    self.real_total_pnl = sum(t.get('pnl', 0) for t in history)
                    return history
            return []
        except Exception as e:
            self.print_color(f"Error loading trade history: {e}", self.Fore.RED)
            return []
    
    def save_real_trade_history(self):
        """Save trading history"""
        try:
            with open(self.real_trade_history_file, 'w') as f:
                json.dump(self.real_trade_history, f, indent=2)
        except Exception as e:
            self.print_color(f"Error saving trade history: {e}", self.Fore.RED)
    
    def add_trade_to_history(self, trade_data):
        """Add trade to history"""
        try:
            trade_data['close_time'] = self.get_thailand_time()
            trade_data['close_timestamp'] = time.time()
            trade_data['trade_type'] = 'REAL'
            self.real_trade_history.append(trade_data)
            
            # Update statistics
            self.real_total_trades += 1
            pnl = trade_data.get('pnl', 0)
            self.real_total_pnl += pnl
            if pnl > 0:
                self.real_winning_trades += 1
                
            if len(self.real_trade_history) > 200:
                self.real_trade_history = self.real_trade_history[-200:]
            self.save_real_trade_history()
            self.print_color(f"📝 Trade saved: {trade_data['pair']} {trade_data['direction']} P&L: ${pnl:.2f}", self.Fore.CYAN)
        except Exception as e:
            self.print_color(f"Error adding trade to history: {e}", self.Fore.RED)
    
    def show_trade_history(self, limit=15):
        """Show trading history"""
        if not self.real_trade_history:
            self.print_color("No trade history found", self.Fore.YELLOW)
            return
        
        self.print_color(f"\n📊 1HOUR TRADING HISTORY (Last {min(limit, len(self.real_trade_history))} trades)", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color("=" * 120, self.Fore.CYAN)
        
        recent_trades = self.real_trade_history[-limit:]
        for i, trade in enumerate(reversed(recent_trades)):
            pnl = trade.get('pnl', 0)
            pnl_color = self.Fore.GREEN + self.Style.BRIGHT if pnl > 0 else self.Fore.RED + self.Style.BRIGHT if pnl < 0 else self.Fore.YELLOW
            direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
            position_size = trade.get('position_size_usd', 0)
            leverage = trade.get('leverage', 1)
            
            self.print_color(f"{i+1:2d}. {direction_icon} {trade['pair']} | Size: ${position_size:.2f} | Leverage: {leverage}x | P&L: ${pnl:.2f}", pnl_color)
            self.print_color(f"     Entry: ${trade.get('entry_price', 0):.4f} | Exit: ${trade.get('exit_price', 0):.4f} | {trade.get('close_reason', 'N/A')}", self.Fore.YELLOW)
    
    def show_trading_stats(self):
        """Show trading statistics"""
        if self.real_total_trades == 0:
            return
            
        win_rate = (self.real_winning_trades / self.real_total_trades) * 100
        avg_trade = self.real_total_pnl / self.real_total_trades
        
        self.print_color(f"\n📈 1HOUR TRADING STATISTICS", self.Fore.GREEN + self.Style.BRIGHT)
        self.print_color("=" * 60, self.Fore.GREEN)
        self.print_color(f"Total Trades: {self.real_total_trades} | Winning Trades: {self.real_winning_trades}", self.Fore.WHITE)
        self.print_color(f"Win Rate: {win_rate:.1f}%", self.Fore.GREEN + self.Style.BRIGHT if win_rate > 50 else self.Fore.YELLOW)
        self.print_color(f"Total P&L: ${self.real_total_pnl:.2f}", self.Fore.GREEN + self.Style.BRIGHT if self.real_total_pnl > 0 else self.Fore.RED + self.Style.BRIGHT)
        self.print_color(f"Average P&L per Trade: ${avg_trade:.2f}", self.Fore.WHITE)
        self.print_color(f"Available Budget: ${self.available_budget:.2f}", self.Fore.CYAN + self.Style.BRIGHT)
    
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
                    # Set initial leverage to 10x (AI can change later)
                    self.binance.futures_change_leverage(symbol=pair, leverage=10)
                    self.binance.futures_change_margin_type(symbol=pair, marginType='ISOLATED')
                    self.print_color(f"✅ Leverage set for {pair}", self.Fore.GREEN)
                except Exception as e:
                    self.print_color(f"Leverage setup failed for {pair}: {e}", self.Fore.YELLOW)
            self.print_color("✅ Futures setup completed!", self.Fore.GREEN + self.Style.BRIGHT)
        except Exception as e:
            self.print_color(f"Futures setup failed: {e}", self.Fore.RED)
    
    def load_symbol_precision(self):
        if not self.binance:
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
    
    def get_market_news_sentiment(self):
        """Get recent cryptocurrency news sentiment"""
        try:
            news_sources = [
                "CoinDesk", "Cointelegraph", "CryptoSlate", "Decrypt", "Binance Blog"
            ]
            return f"Monitoring: {', '.join(news_sources)}"
        except:
            return "General crypto market news monitoring"
    
    def get_ai_trading_decision(self, pair, market_data):
        """AI makes COMPLETE trading decisions with budget management for 1HOUR timeframe"""
        try:
            if not self.openrouter_key:
                return self.get_fallback_decision(pair, market_data)
            
            current_price = market_data['current_price']
            news_sentiment = self.get_market_news_sentiment()
            
            # 🧠 COMPREHENSIVE AI TRADING PROMPT FOR 1HOUR TIMEFRAME
            prompt = f"""
            YOU ARE A FULLY AUTONOMOUS AI TRADER with ${self.available_budget:.2f} budget.

            MARKET ANALYSIS FOR {pair} (1HOUR TIMEFRAME):
            - Current Price: ${current_price:.6f}
            - 1Hour Price Change: {market_data.get('price_change', 0):.2f}%
            - Volume Change: {market_data.get('volume_change', 0):.2f}%
            - Recent Prices (last 6 hours): {market_data.get('prices', [])[-6:]}
            - Support/Resistance Levels: {market_data.get('support_levels', [])} / {market_data.get('resistance_levels', [])}
            - Market News: {news_sentiment}

            YOUR TRADING PARAMETERS:
            - Total Budget: ${self.total_budget}
            - Available Budget: ${self.available_budget:.2f}
            - Maximum Position Size: ${self.total_budget * self.max_position_size_percent/100:.2f} ({self.max_position_size_percent}% of budget)
            - Timeframe: 1HOUR (Higher timeframe = bigger moves)
            - Current Active Positions: {len(self.ai_opened_trades)}
            - ALLOWED LEVERAGE RANGE: 5x to 20x ONLY - YOU MUST USE LEVERAGE BETWEEN 5-20x
            - MAX CONCURRENT TRADES: {self.max_concurrent_trades}

            YOU HAVE COMPLETE CONTROL OVER:
            ✅ Trade Decision (LONG/SHORT/HOLD)
            ✅ Position Size ($ amount to risk)
            ✅ Entry Price (exact price)
            ✅ Take Profit (realistic target for 1hr)
            ✅ Stop Loss (risk management for 1hr)
            ✅ Leverage (5-20x ONLY - MUST BE IN THIS RANGE)
            ✅ Reasoning based on 1hr technicals + fundamentals

            RISK MANAGEMENT RULES FOR 1HOUR TRADING:
            - Never risk more than {self.max_position_size_percent}% of total budget per trade
            - Use leverage between 5x to 20x ONLY
            - Maintain proper risk-reward ratios (minimum 1:2 for 1hr trades)
            - Consider overall portfolio exposure
            - 1hr timeframe requires wider stops and bigger targets

            Return VALID JSON only:
            {{
                "decision": "LONG" | "SHORT" | "HOLD",
                "position_size_usd": number (max {self.total_budget * self.max_position_size_percent/100:.0f}),
                "entry_price": number,
                "take_profit": number,
                "stop_loss": number,
                "leverage": number (5-20 ONLY - MUST BE IN THIS RANGE),
                "confidence": 0-100,
                "reasoning": "detailed 1hr timeframe analysis including technicals, market sentiment, and risk management rationale"
            }}

            Think step by step for 1HOUR TIMEFRAME: 
            - Analyze the 1hr chart structure
            - Check momentum and volume for 1hr moves
            - Consider higher timeframe context
            - Calculate optimal position size for 1hr holds
            - Set realistic TP/SL based on 1hr volatility
            - MUST USE LEVERAGE BETWEEN 5x to 20x
            - 1hr trades need bigger targets and stops than lower timeframes
            """

            headers = {
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com",
                "X-Title": "Fully Autonomous 1Hour AI Trader"
            }
            
            data = {
                "model": "deepseek/deepseek-chat-v3.1",  # CHANGED TO DEEPSEEK
                "messages": [
                    {"role": "system", "content": "You are a fully autonomous AI trader managing a $5000 portfolio on 1HOUR timeframe. You MUST use leverage between 5x to 20x. Make calculated trading decisions considering 1hr technical analysis, market sentiment, and strict risk management. Always return valid JSON with complete trading parameters including leverage between 5-20x."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 800
            }
            
            self.print_color(f"🧠 DeepSeek V3.1 Analyzing {pair} on 1HOUR with ${self.available_budget:.2f} available...", self.Fore.MAGENTA + self.Style.BRIGHT)
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content'].strip()
                return self.parse_ai_trading_decision(ai_response, pair, current_price)
            else:
                self.print_color(f"DeepSeek API error: {response.status_code}", self.Fore.RED)
                return self.get_fallback_decision(pair, market_data)
                
        except Exception as e:
            self.print_color(f"DeepSeek analysis failed: {e}", self.Fore.RED)
            return self.get_fallback_decision(pair, market_data)

    def parse_ai_trading_decision(self, ai_response, pair, current_price):
        """Parse AI's complete trading decision"""
        try:
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                decision_data = json.loads(json_str)
                
                decision = decision_data.get('decision', 'HOLD').upper()
                position_size_usd = float(decision_data.get('position_size_usd', 0))
                entry_price = float(decision_data.get('entry_price', 0))
                take_profit = float(decision_data.get('take_profit', 0))
                stop_loss = float(decision_data.get('stop_loss', 0))
                leverage = int(decision_data.get('leverage', 10))
                confidence = float(decision_data.get('confidence', 50))
                reasoning = decision_data.get('reasoning', 'AI Analysis')
                
                # Validate inputs with leverage constraint
                if decision not in ['LONG', 'SHORT', 'HOLD']:
                    decision = 'HOLD'
                if position_size_usd > self.total_budget * self.max_position_size_percent/100:
                    position_size_usd = self.total_budget * self.max_position_size_percent/100
                
                # ENFORCE LEVERAGE RANGE: 5x to 20x ONLY
                if leverage < 5:
                    leverage = 5
                    self.print_color(f"⚠️  Leverage adjusted to minimum 5x", self.Fore.YELLOW)
                elif leverage > 20:
                    leverage = 20
                    self.print_color(f"⚠️  Leverage adjusted to maximum 20x", self.Fore.YELLOW)
                    
                if entry_price <= 0:
                    entry_price = current_price
                    
                return {
                    "decision": decision,
                    "position_size_usd": position_size_usd,
                    "entry_price": entry_price,
                    "take_profit": take_profit,
                    "stop_loss": stop_loss,
                    "leverage": leverage,
                    "confidence": confidence,
                    "reasoning": reasoning
                }
            return self.get_fallback_decision(pair, {'current_price': current_price})
        except Exception as e:
            self.print_color(f"DeepSeek response parsing failed: {e}", self.Fore.RED)
            return self.get_fallback_decision(pair, {'current_price': current_price})

    def get_fallback_decision(self, pair, market_data):
        """Fallback decision if AI fails"""
        return {
            "decision": "HOLD",
            "position_size_usd": 0,
            "entry_price": market_data['current_price'],
            "take_profit": 0,
            "stop_loss": 0,
            "leverage": 10,
            "confidence": 0,
            "reasoning": "Fallback: AI analysis unavailable"
        }

    def get_price_history(self, pair, limit=12):
        """Get 1hour price history with technical levels"""
        try:
            if self.binance:
                klines = self.binance.futures_klines(symbol=pair, interval=Client.KLINE_INTERVAL_1HOUR, limit=limit)
                prices = [float(k[4]) for k in klines]
                highs = [float(k[2]) for k in klines]
                lows = [float(k[3]) for k in klines]
                volumes = [float(k[5]) for k in klines]
                
                current_price = prices[-1] if prices else 0
                # Calculate 4-hour price change for 1hr context
                price_change = ((current_price - prices[-4]) / prices[-4] * 100) if len(prices) >= 4 else 0
                volume_change = ((volumes[-1] - volumes[-4]) / volumes[-4] * 100) if len(volumes) >= 4 else 0
                
                # Calculate support/resistance levels for 1hr
                support_levels = [min(lows[-6:]), min(lows[-12:])]
                resistance_levels = [max(highs[-6:]), max(highs[-12:])]
                
                return {
                    'prices': prices,
                    'highs': highs,
                    'lows': lows,
                    'volumes': volumes,
                    'current_price': current_price,
                    'price_change': price_change,
                    'volume_change': volume_change,
                    'support_levels': [round(l, 4) for l in support_levels],
                    'resistance_levels': [round(l, 4) for l in resistance_levels]
                }
            else:
                current_price = self.get_current_price(pair)
                return {
                    'prices': [current_price] * 12,
                    'highs': [current_price * 1.03] * 12,
                    'lows': [current_price * 0.97] * 12,
                    'volumes': [100000] * 12,
                    'current_price': current_price,
                    'price_change': 1.2,
                    'volume_change': 15.5,
                    'support_levels': [current_price * 0.97, current_price * 0.95],
                    'resistance_levels': [current_price * 1.03, current_price * 1.05]
                }
        except Exception as e:
            current_price = self.get_current_price(pair)
            return {
                'current_price': current_price,
                'price_change': 0,
                'volume_change': 0,
                'support_levels': [],
                'resistance_levels': []
            }

    def get_current_price(self, pair):
        try:
            if self.binance:
                ticker = self.binance.futures_symbol_ticker(symbol=pair)
                return float(ticker['price'])
            else:
                # Mock prices for paper trading
                mock_prices = {
                    "BTCUSDT": 45000, "ETHUSDT": 2500, "BNBUSDT": 300,
                    "SOLUSDT": 180, "DOGEUSDT": 0.12, "AVAXUSDT": 35
                }
                return mock_prices.get(pair, 100)
        except:
            return 100

    def calculate_quantity(self, pair, entry_price, position_size_usd, leverage):
        """Calculate quantity based on position size and leverage"""
        try:
            if entry_price <= 0:
                return None
                
            # Calculate notional value
            notional_value = position_size_usd * leverage
            
            # Calculate quantity
            quantity = notional_value / entry_price
            
            # Apply precision
            precision = self.quantity_precision.get(pair, 3)
            quantity = round(quantity, precision)
            
            if quantity <= 0:
                return None
                
            self.print_color(f"📊 Position: ${position_size_usd} | Leverage: {leverage}x | Notional: ${notional_value:.2f} | Quantity: {quantity}", self.Fore.CYAN)
            return quantity
            
        except Exception as e:
            self.print_color(f"Quantity calculation failed: {e}", self.Fore.RED)
            return None

    def can_open_new_position(self, pair, position_size_usd):
        """Check if new position can be opened"""
        if pair in self.ai_opened_trades:
            return False, "Position already exists"
        
        if len(self.ai_opened_trades) >= self.max_concurrent_trades:
            return False, f"Max concurrent trades reached ({self.max_concurrent_trades})"
            
        if position_size_usd > self.available_budget:
            return False, f"Insufficient budget: ${position_size_usd:.2f} > ${self.available_budget:.2f}"
            
        max_allowed = self.total_budget * self.max_position_size_percent / 100
        if position_size_usd > max_allowed:
            return False, f"Position size too large: ${position_size_usd:.2f} > ${max_allowed:.2f}"
            
        return True, "OK"

    def execute_ai_trade(self, pair, ai_decision):
        """Execute trade based on AI's complete decision"""
        try:
            decision = ai_decision["decision"]
            position_size_usd = ai_decision["position_size_usd"]
            entry_price = ai_decision["entry_price"]
            take_profit = ai_decision["take_profit"]
            stop_loss = ai_decision["stop_loss"]
            leverage = ai_decision["leverage"]
            confidence = ai_decision["confidence"]
            reasoning = ai_decision["reasoning"]
            
            if decision == "HOLD" or position_size_usd <= 0:
                self.print_color(f"🟡 DeepSeek decides to HOLD {pair} (Confidence: {confidence}%)", self.Fore.YELLOW)
                return False
            
            # Check if we can open position
            can_open, reason = self.can_open_new_position(pair, position_size_usd)
            if not can_open:
                self.print_color(f"🚫 Cannot open {pair}: {reason}", self.Fore.RED)
                return False
            
            # Calculate quantity
            quantity = self.calculate_quantity(pair, entry_price, position_size_usd, leverage)
            if quantity is None:
                return False
            
            # Format prices
            take_profit = self.format_price(pair, take_profit)
            stop_loss = self.format_price(pair, stop_loss)
            
            # Display AI trade decision
            direction_color = self.Fore.GREEN + self.Style.BRIGHT if decision == 'LONG' else self.Fore.RED + self.Style.BRIGHT
            direction_icon = "🟢 LONG" if decision == 'LONG' else "🔴 SHORT"
            
            self.print_color(f"\n🤖 1HOUR DEEPSEEK TRADE EXECUTION", self.Fore.CYAN + self.Style.BRIGHT)
            self.print_color("=" * 80, self.Fore.CYAN)
            self.print_color(f"{direction_icon} {pair}", direction_color)
            self.print_color(f"POSITION SIZE: ${position_size_usd:.2f}", self.Fore.GREEN + self.Style.BRIGHT)
            self.print_color(f"LEVERAGE: {leverage}x ⚡", self.Fore.RED + self.Style.BRIGHT)
            self.print_color(f"ENTRY PRICE: ${entry_price:.4f}", self.Fore.WHITE)
            self.print_color(f"QUANTITY: {quantity}", self.Fore.CYAN)
            self.print_color(f"TAKE PROFIT: ${take_profit:.4f}", self.Fore.GREEN)
            self.print_color(f"STOP LOSS: ${stop_loss:.4f}", self.Fore.RED)
            self.print_color(f"CONFIDENCE: {confidence}%", self.Fore.YELLOW + self.Style.BRIGHT)
            self.print_color(f"1HOUR REASONING: {reasoning}", self.Fore.WHITE)
            self.print_color("=" * 80, self.Fore.CYAN)
            
            # Execute live trade
            if self.binance:
                entry_side = 'BUY' if decision == 'LONG' else 'SELL'
                
                # Set leverage (5-20x)
                try:
                    self.binance.futures_change_leverage(symbol=pair, leverage=leverage)
                    self.print_color(f"✅ Leverage set to {leverage}x for {pair}", self.Fore.GREEN)
                except Exception as e:
                    self.print_color(f"Leverage change failed: {e}", self.Fore.YELLOW)
                
                # Execute order
                order = self.binance.futures_create_order(
                    symbol=pair,
                    side=entry_side,
                    type='MARKET',
                    quantity=quantity
                )
                
                # Set stop loss and take profit
                stop_side = 'SELL' if decision == 'LONG' else 'BUY'
                self.binance.futures_create_order(
                    symbol=pair, side=stop_side, type='STOP_MARKET',
                    quantity=quantity, stopPrice=stop_loss, reduceOnly=True
                )
                self.binance.futures_create_order(
                    symbol=pair, side=stop_side, type='TAKE_PROFIT_MARKET',
                    quantity=quantity, stopPrice=take_profit, reduceOnly=True
                )
            
            # Update budget and track trade
            self.available_budget -= position_size_usd
            
            self.ai_opened_trades[pair] = {
                "pair": pair,
                "direction": decision,
                "entry_price": entry_price,
                "quantity": quantity,
                "position_size_usd": position_size_usd,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "leverage": leverage,
                "entry_time": time.time(),
                "status": 'ACTIVE',
                'ai_confidence': confidence,
                'ai_reasoning': reasoning,
                'entry_time_th': self.get_thailand_time()
            }
            
            self.print_color(f"✅ 1HOUR DEEPSEEK TRADE EXECUTED: {pair} {decision} | Leverage: {leverage}x | Budget Used: ${position_size_usd:.2f}", self.Fore.GREEN + self.Style.BRIGHT)
            return True
            
        except Exception as e:
            self.print_color(f"❌ DeepSeek trade execution failed: {e}", self.Fore.RED)
            return False

    def get_live_position_data(self, pair):
        """Get live position data from Binance"""
        try:
            if not self.binance:
                return None
                
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
        """Monitor and update open positions"""
        try:
            closed_trades = []
            for pair, trade in list(self.ai_opened_trades.items()):
                if trade['status'] != 'ACTIVE':
                    continue
                
                if self.binance:
                    live_data = self.get_live_position_data(pair)
                    if not live_data:
                        # Position closed
                        self.close_trade_with_cleanup(pair, trade, "AUTO CLOSE")
                        closed_trades.append(pair)
                        continue
                else:
                    # Paper trading - check TP/SL
                    current_price = self.get_current_price(pair)
                    if self.check_paper_tp_sl(pair, trade, current_price):
                        closed_trades.append(pair)
                    
            return closed_trades
        except Exception as e:
            self.print_color(f"Monitoring error: {e}", self.Fore.RED)
            return []

    def check_paper_tp_sl(self, pair, trade, current_price):
        """Check if paper trade hit TP/SL"""
        try:
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
                self.close_paper_trade(pair, trade, close_reason, current_price, pnl)
                return True
            return False
                    
        except Exception as e:
            self.print_color(f"Paper TP/SL check failed: {e}", self.Fore.RED)
            return False

    def close_paper_trade(self, pair, trade, close_reason, current_price, pnl):
        """Close paper trade"""
        try:
            trade['status'] = 'CLOSED'
            trade['exit_price'] = current_price
            trade['pnl'] = pnl
            trade['close_reason'] = close_reason
            trade['close_time'] = self.get_thailand_time()
            
            # Return used budget plus P&L
            self.available_budget += trade['position_size_usd'] + pnl
            
            self.add_trade_to_history(trade.copy())
            
            pnl_color = self.Fore.GREEN + self.Style.BRIGHT if pnl > 0 else self.Fore.RED + self.Style.BRIGHT
            direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
            self.print_color(f"\n🔚 1HOUR PAPER TRADE CLOSED: {pair} {direction_icon}", pnl_color)
            self.print_color(f"   Leverage: {trade['leverage']}x | P&L: ${pnl:.2f} | Reason: {close_reason}", pnl_color)
            self.print_color(f"   New Available Budget: ${self.available_budget:.2f}", self.Fore.CYAN)
            
            del self.ai_opened_trades[pair]
            
        except Exception as e:
            self.print_color(f"Paper trade close failed: {e}", self.Fore.RED)

    def close_trade_with_cleanup(self, pair, trade, close_reason="MANUAL"):
        """Close real trade with cleanup"""
        try:
            if self.binance:
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
            
            # Return used budget plus P&L
            self.available_budget += trade['position_size_usd'] + final_pnl
            
            self.add_trade_to_history(trade.copy())
            
            pnl_color = self.Fore.GREEN + self.Style.BRIGHT if final_pnl > 0 else self.Fore.RED + self.Style.BRIGHT
            direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
            self.print_color(f"\n🔚 1HOUR TRADE CLOSED: {pair} {direction_icon}", pnl_color)
            self.print_color(f"   Leverage: {trade['leverage']}x | Final P&L: ${final_pnl:.2f} | Reason: {close_reason}", pnl_color)
            self.print_color(f"   Available Budget: ${self.available_budget:.2f}", self.Fore.CYAN)
                
            del self.ai_opened_trades[pair]
            
        except Exception as e:
            self.print_color(f"Cleanup failed for {pair}: {e}", self.Fore.RED)

    def get_final_pnl(self, pair, trade):
        """Calculate final P&L for trade"""
        try:
            if self.binance:
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
        """Display trading dashboard"""
        self.print_color(f"\n🤖 1HOUR DEEPSEEK AI TRADING DASHBOARD - {self.get_thailand_time()}", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color("=" * 90, self.Fore.CYAN)
        
        active_count = 0
        total_unrealized = 0
        
        for pair, trade in self.ai_opened_trades.items():
            if trade['status'] == 'ACTIVE':
                active_count += 1
                current_price = self.get_current_price(pair)
                
                direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
                
                if trade['direction'] == 'LONG':
                    unrealized_pnl = (current_price - trade['entry_price']) * trade['quantity']
                else:
                    unrealized_pnl = (trade['entry_price'] - current_price) * trade['quantity']
                    
                total_unrealized += unrealized_pnl
                pnl_color = self.Fore.GREEN + self.Style.BRIGHT if unrealized_pnl >= 0 else self.Fore.RED + self.Style.BRIGHT
                
                self.print_color(f"{direction_icon} {pair}", self.Fore.WHITE + self.Style.BRIGHT)
                self.print_color(f"   Size: ${trade['position_size_usd']:.2f} | Leverage: {trade['leverage']}x ⚡", self.Fore.WHITE)
                self.print_color(f"   Entry: ${trade['entry_price']:.4f} | Current: ${current_price:.4f}", self.Fore.WHITE)
                self.print_color(f"   P&L: ${unrealized_pnl:.2f}", pnl_color)
                self.print_color(f"   TP: ${trade['take_profit']:.4f} | SL: ${trade['stop_loss']:.4f}", self.Fore.YELLOW)
                self.print_color("   " + "-" * 60, self.Fore.CYAN)
        
        if active_count == 0:
            self.print_color("No active 1hour positions", self.Fore.YELLOW)
        else:
            total_color = self.Fore.GREEN + self.Style.BRIGHT if total_unrealized >= 0 else self.Fore.RED + self.Style.BRIGHT
            self.print_color(f"📊 Active 1Hour Positions: {active_count}/{self.max_concurrent_trades} | Total Unrealized P&L: ${total_unrealized:.2f}", total_color)

    def run_trading_cycle(self):
        """Run one complete trading cycle"""
        try:
            self.monitor_positions()
            self.display_dashboard()
            
            # Show stats every 2 cycles (since 1hr timeframe is longer)
            if hasattr(self, 'cycle_count') and self.cycle_count % 2 == 0:
                self.show_trade_history(8)
                self.show_trading_stats()
            
            self.print_color(f"\n🔍 DEEPSEEK SCANNING {len(self.available_pairs)} 1HOUR PAIRS WITH ${self.available_budget:.2f} AVAILABLE...", self.Fore.BLUE + self.Style.BRIGHT)
            
            qualified_signals = 0
            for pair in self.available_pairs:
                if self.available_budget > 100:  # Minimum $100 to consider trading
                    market_data = self.get_price_history(pair)
                    ai_decision = self.get_ai_trading_decision(pair, market_data)
                    
                    if ai_decision["decision"] in ["LONG", "SHORT"] and ai_decision["position_size_usd"] > 0:
                        qualified_signals += 1
                        direction_icon = "🟢 LONG" if ai_decision['decision'] == "LONG" else "🔴 SHORT"
                        leverage_info = f"Leverage: {ai_decision['leverage']}x"
                        self.print_color(f"🎯 1HOUR DEEPSEEK SIGNAL: {pair} {direction_icon} | Size: ${ai_decision['position_size_usd']:.2f} | {leverage_info}", self.Fore.GREEN + self.Style.BRIGHT)
                        success = self.execute_ai_trade(pair, ai_decision)
                        if success:
                            time.sleep(3)  # Longer delay between executions for 1hr
                
            if qualified_signals == 0:
                self.print_color("No qualified 1hour DeepSeek signals this cycle", self.Fore.YELLOW)
                
        except Exception as e:
            self.print_color(f"1Hour trading cycle error: {e}", self.Fore.RED)

    def start_trading(self):
        """Start the fully autonomous 1hour AI trading"""
        self.print_color("🚀 STARTING FULLY AUTONOMOUS 1HOUR DEEPSEEK AI TRADER!", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color("💰 AI MANAGING $5000 PORTFOLIO", self.Fore.GREEN + self.Style.BRIGHT)
        self.print_color("🤖 DEEPSEEK V3.1 FULL CONTROL: Analysis, Sizing, Entry, TP, SL", self.Fore.MAGENTA + self.Style.BRIGHT)
        self.print_color("⏰ 1HOUR TIMEFRAME | LEVERAGE: 5x to 20x ⚡", self.Fore.RED + self.Style.BRIGHT)
        self.print_color(f"🎯 MAX POSITIONS: {self.max_concurrent_trades} | SELECTED PAIRS: BTC, ETH, BNB, SOL, DOGE, AVAX", self.Fore.YELLOW + self.Style.BRIGHT)
        
        self.cycle_count = 0
        while True:
            try:
                self.cycle_count += 1
                self.print_color(f"\n🔄 1HOUR DEEPSEEK TRADING CYCLE {self.cycle_count}", self.Fore.CYAN + self.Style.BRIGHT)
                self.print_color("=" * 60, self.Fore.CYAN)
                self.run_trading_cycle()
                self.print_color(f"⏳ DeepSeek analyzing next 1hour opportunities in 5 minutes...", self.Fore.BLUE)
                time.sleep(300)  # 5 minutes between cycles for 1hr timeframe
                
            except KeyboardInterrupt:
                self.print_color(f"\n🛑 1HOUR DEEPSEEK TRADING STOPPED", self.Fore.RED + self.Style.BRIGHT)
                self.show_trade_history(15)
                self.show_trading_stats()
                break
            except Exception as e:
                self.print_color(f"1Hour main loop error: {e}", self.Fore.RED)
                time.sleep(300)


if __name__ == "__main__":
    try:
        ai_trader = FullyAutonomous1HourAITrader()
        
        print("\n" + "="*80)
        print("🤖 FULLY AUTONOMOUS 1HOUR DEEPSEEK AI TRADER")
        print("="*80)
        print("SELECT MODE:")
        print("1. 🚀 Live Trading (DeepSeek Manages Real $5000)")
        print("2. 💸 Paper Trading (Virtual $5000)")
        
        choice = input("Enter choice (1-2): ").strip()
        
        if choice == "1":
            print("⚠️  WARNING: REAL MONEY TRADING! ⚠️")
            print("🤖 DEEPSEEK V3.1 HAS COMPLETE CONTROL OVER $5000")
            print("⚡ LEVERAGE: 5x to 20x ONLY")
            print("⏰ TIMEFRAME: 1HOUR")
            print(f"🎯 MAX POSITIONS: {ai_trader.max_concurrent_trades} | PAIRS: BTC, ETH, BNB, SOL, DOGE, AVAX")
            confirm = input("Type 'DEEPSEEK' to confirm: ").strip()
            if confirm.upper() == 'DEEPSEEK':
                ai_trader.start_trading()
            else:
                print("Exiting...")
        else:
            print("Paper trading mode coming soon...")
            # Paper trading implementation would be similar to previous version
            
    except Exception as e:
        print(f"Failed to start 1Hour DeepSeek AI trader: {e}")
