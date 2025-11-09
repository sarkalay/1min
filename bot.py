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

class FullyAutonomous15MinAITrader:
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
        
        # 🎯 FULLY AUTONOMOUS AI TRADING PARAMETERS
        self.total_budget = 5000  # $5000 budget for AI to manage
        self.available_budget = 5000  # Current available budget
        self.max_position_size_percent = 20  # Max 20% of budget per trade
        self.max_concurrent_trades = 5  # Maximum concurrent positions
        
        # AI can trade all major pairs
        self.available_pairs = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "XRPUSDT", 
            "BNBUSDT", "DOGEUSDT"
        ]
        
        # Track AI-opened trades
        self.ai_opened_trades = {}
        
        # REAL TRADE HISTORY
        self.real_trade_history_file = "fully_autonomous_ai_trading_history.json"
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
            self.print_color(f"🤖 FULLY AUTONOMOUS AI TRADER ACTIVATED! 🤖", self.Fore.CYAN + self.Style.BRIGHT)
            self.print_color(f"💰 TOTAL BUDGET: ${self.total_budget}", self.Fore.GREEN + self.Style.BRIGHT)
            self.print_color(f"🎯 AI FULL CONTROL: Analysis, Entry, Size, TP, SL", self.Fore.MAGENTA + self.Style.BRIGHT)
            self.print_color(f"⏰ Timeframe: 15MIN | Max Positions: {self.max_concurrent_trades}", self.Fore.YELLOW + self.Style.BRIGHT)
            self.print_color(f"📈 Pairs: {len(self.available_pairs)} major cryptocurrencies", self.Fore.BLUE + self.Style.BRIGHT)
            self.print_color(f"🧠 AI Model: Qwen3 Max (via OpenRouter)", self.Fore.CYAN + self.Style.BRIGHT)
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
        
        self.print_color(f"\n📊 TRADING HISTORY (Last {min(limit, len(self.real_trade_history))} trades)", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color("=" * 120, self.Fore.CYAN)
        
        recent_trades = self.real_trade_history[-limit:]
        for i, trade in enumerate(reversed(recent_trades)):
            pnl = trade.get('pnl', 0)
            pnl_color = self.Fore.GREEN + self.Style.BRIGHT if pnl > 0 else self.Fore.RED + self.Style.BRIGHT if pnl < 0 else self.Fore.YELLOW
            direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
            position_size = trade.get('position_size_usd', 0)
            
            self.print_color(f"{i+1:2d}. {direction_icon} {trade['pair']} | Size: ${position_size:.2f} | P&L: ${pnl:.2f}", pnl_color)
            self.print_color(f"     Entry: ${trade.get('entry_price', 0):.4f} | Exit: ${trade.get('exit_price', 0):.4f} | {trade.get('close_reason', 'N/A')}", self.Fore.YELLOW)
    
    def show_trading_stats(self):
        """Show trading statistics"""
        if self.real_total_trades == 0:
            return
            
        win_rate = (self.real_winning_trades / self.real_total_trades) * 100
        avg_trade = self.real_total_pnl / self.real_total_trades
        
        self.print_color(f"\n📈 TRADING STATISTICS", self.Fore.GREEN + self.Style.BRIGHT)
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
                    # Set moderate leverage for safety
                    self.binance.futures_change_leverage(symbol=pair, leverage=5)
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
            # This is a simplified version - in reality you'd use news API
            news_sources = [
                "CoinDesk", "Cointelegraph", "CryptoSlate", "Decrypt"
            ]
            return f"Monitoring: {', '.join(news_sources)}"
        except:
            return "General crypto market news monitoring"
    
    def get_ai_trading_decision(self, pair, market_data):
        """AI makes COMPLETE trading decisions with budget management"""
        try:
            if not self.openrouter_key:
                return self.get_fallback_decision(pair, market_data)
            
            current_price = market_data['current_price']
            news_sentiment = self.get_market_news_sentiment()
            
            # 🧠 COMPREHENSIVE AI TRADING PROMPT WITH BUDGET MANAGEMENT
            prompt = f"""
            YOU ARE A FULLY AUTONOMOUS AI TRADER with ${self.available_budget:.2f} budget.

            MARKET ANALYSIS FOR {pair}:
            - Current Price: ${current_price:.6f}
            - 15min Price Change: {market_data.get('price_change', 0):.2f}%
            - Volume Change: {market_data.get('volume_change', 0):.2f}%
            - Recent Prices: {market_data.get('prices', [])[-6:]}
            - Support/Resistance Levels: {market_data.get('support_levels', [])} / {market_data.get('resistance_levels', [])}
            - Market News: {news_sentiment}

            YOUR TRADING PARAMETERS:
            - Total Budget: ${self.total_budget}
            - Available Budget: ${self.available_budget:.2f}
            - Maximum Position Size: ${self.total_budget * self.max_position_size_percent/100:.2f} ({self.max_position_size_percent}% of budget)
            - Timeframe: 15MIN
            - Current Active Positions: {len(self.ai_opened_trades)}

            YOU HAVE COMPLETE CONTROL OVER:
            ✅ Trade Decision (LONG/SHORT/HOLD)
            ✅ Position Size ($ amount to risk)
            ✅ Entry Price (exact price)
            ✅ Take Profit (realistic target)
            ✅ Stop Loss (risk management)
            ✅ Leverage (1-5x, be careful!)
            ✅ Reasoning based on technicals + fundamentals

            RISK MANAGEMENT RULES:
            - Never risk more than {self.max_position_size_percent}% of total budget per trade
            - Maintain proper risk-reward ratios (minimum 1:1.5)
            - Consider overall portfolio exposure
            - Be aware of market volatility

            Return VALID JSON only:
            {{
                "decision": "LONG" | "SHORT" | "HOLD",
                "position_size_usd": number (max {self.total_budget * self.max_position_size_percent/100:.0f}),
                "entry_price": number,
                "take_profit": number,
                "stop_loss": number,
                "leverage": number (1-5),
                "confidence": 0-100,
                "reasoning": "detailed analysis including technicals, market sentiment, and risk management rationale"
            }}

            Think step by step: Analyze the 15min chart, check momentum, consider market news, 
            calculate optimal position size, set realistic TP/SL based on volatility.
            """

            headers = {
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com",
                "X-Title": "Fully Autonomous AI Trader"
            }
            
            data = {
                "model": "qwen/qwen3-max",
                "messages": [
                    {"role": "system", "content": "You are a fully autonomous AI trader managing a $5000 portfolio. Make calculated trading decisions considering technical analysis, market sentiment, and strict risk management. Always return valid JSON with complete trading parameters."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 800
            }
            
            self.print_color(f"🧠 AI Analyzing {pair} with ${self.available_budget:.2f} available...", self.Fore.MAGENTA + self.Style.BRIGHT)
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content'].strip()
                return self.parse_ai_trading_decision(ai_response, pair, current_price)
            else:
                self.print_color(f"AI API error: {response.status_code}", self.Fore.RED)
                return self.get_fallback_decision(pair, market_data)
                
        except Exception as e:
            self.print_color(f"AI analysis failed: {e}", self.Fore.RED)
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
                leverage = int(decision_data.get('leverage', 3))
                confidence = float(decision_data.get('confidence', 50))
                reasoning = decision_data.get('reasoning', 'AI Analysis')
                
                # Validate inputs
                if decision not in ['LONG', 'SHORT', 'HOLD']:
                    decision = 'HOLD'
                if position_size_usd > self.total_budget * self.max_position_size_percent/100:
                    position_size_usd = self.total_budget * self.max_position_size_percent/100
                if leverage < 1 or leverage > 5:
                    leverage = 3
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
            self.print_color(f"AI response parsing failed: {e}", self.Fore.RED)
            return self.get_fallback_decision(pair, {'current_price': current_price})

    def get_fallback_decision(self, pair, market_data):
        """Fallback decision if AI fails"""
        return {
            "decision": "HOLD",
            "position_size_usd": 0,
            "entry_price": market_data['current_price'],
            "take_profit": 0,
            "stop_loss": 0,
            "leverage": 3,
            "confidence": 0,
            "reasoning": "Fallback: AI analysis unavailable"
        }

    def get_price_history(self, pair, limit=20):
        """Get 15min price history with technical levels"""
        try:
            if self.binance:
                klines = self.binance.futures_klines(symbol=pair, interval=Client.KLINE_INTERVAL_15MINUTE, limit=limit)
                prices = [float(k[4]) for k in klines]
                highs = [float(k[2]) for k in klines]
                lows = [float(k[3]) for k in klines]
                volumes = [float(k[5]) for k in klines]
                
                current_price = prices[-1] if prices else 0
                price_change = ((current_price - prices[-4]) / prices[-4] * 100) if len(prices) >= 4 else 0
                volume_change = ((volumes[-1] - volumes[-4]) / volumes[-4] * 100) if len(volumes) >= 4 else 0
                
                # Calculate simple support/resistance levels
                support_levels = [min(lows[-5:]), min(lows[-10:])]
                resistance_levels = [max(highs[-5:]), max(highs[-10:])]
                
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
                    'prices': [current_price] * 10,
                    'highs': [current_price * 1.02] * 10,
                    'lows': [current_price * 0.98] * 10,
                    'volumes': [100000] * 10,
                    'current_price': current_price,
                    'price_change': 0.5,
                    'volume_change': 10.2,
                    'support_levels': [current_price * 0.98, current_price * 0.96],
                    'resistance_levels': [current_price * 1.02, current_price * 1.04]
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
                    "BTCUSDT": 45000, "ETHUSDT": 2500, "SOLUSDT": 180,
                    "AVAXUSDT": 35, "XRPUSDT": 0.6, "LINKUSDT": 18,
                    "DOTUSDT": 8, "ADAUSDT": 0.45, "MATICUSDT": 0.75,
                    "DOGEUSDT": 0.12, "ATOMUSDT": 10, "NEARUSDT": 7.5,
                    "BNBUSDT": 300, "LTCUSDT": 70, "BCHUSDT": 400
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
                
            self.print_color(f"📊 Position: ${position_size_usd} | Leverage: {leverage}x | Quantity: {quantity}", self.Fore.CYAN)
            return quantity
            
        except Exception as e:
            self.print_color(f"Quantity calculation failed: {e}", self.Fore.RED)
            return None

    def can_open_new_position(self, pair, position_size_usd):
        """Check if new position can be opened"""
        if pair in self.ai_opened_trades:
            return False, "Position already exists"
        
        if len(self.ai_opened_trades) >= self.max_concurrent_trades:
            return False, "Max concurrent trades reached"
            
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
                self.print_color(f"🟡 AI decides to HOLD {pair} (Confidence: {confidence}%)", self.Fore.YELLOW)
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
            
            self.print_color(f"\n🤖 AI TRADE EXECUTION", self.Fore.CYAN + self.Style.BRIGHT)
            self.print_color("=" * 80, self.Fore.CYAN)
            self.print_color(f"{direction_icon} {pair}", direction_color)
            self.print_color(f"POSITION SIZE: ${position_size_usd:.2f}", self.Fore.GREEN + self.Style.BRIGHT)
            self.print_color(f"ENTRY PRICE: ${entry_price:.4f}", self.Fore.WHITE)
            self.print_color(f"LEVERAGE: {leverage}x", self.Fore.MAGENTA)
            self.print_color(f"QUANTITY: {quantity}", self.Fore.CYAN)
            self.print_color(f"TAKE PROFIT: ${take_profit:.4f}", self.Fore.GREEN)
            self.print_color(f"STOP LOSS: ${stop_loss:.4f}", self.Fore.RED)
            self.print_color(f"CONFIDENCE: {confidence}%", self.Fore.YELLOW + self.Style.BRIGHT)
            self.print_color(f"REASONING: {reasoning}", self.Fore.WHITE)
            self.print_color("=" * 80, self.Fore.CYAN)
            
            # Execute live trade
            if self.binance:
                entry_side = 'BUY' if decision == 'LONG' else 'SELL'
                
                # Set leverage
                try:
                    self.binance.futures_change_leverage(symbol=pair, leverage=leverage)
                except:
                    pass
                
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
            
            self.print_color(f"✅ AI TRADE EXECUTED: {pair} {decision} | Budget Used: ${position_size_usd:.2f}", self.Fore.GREEN + self.Style.BRIGHT)
            return True
            
        except Exception as e:
            self.print_color(f"❌ AI trade execution failed: {e}", self.Fore.RED)
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
            self.print_color(f"\n🔚 PAPER TRADE CLOSED: {pair} {direction_icon}", pnl_color)
            self.print_color(f"   P&L: ${pnl:.2f} | Reason: {close_reason}", pnl_color)
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
            self.print_color(f"\n🔚 TRADE CLOSED: {pair} {direction_icon}", pnl_color)
            self.print_color(f"   Final P&L: ${final_pnl:.2f} | Reason: {close_reason}", pnl_color)
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
        self.print_color(f"\n🤖 AI TRADING DASHBOARD - {self.get_thailand_time()}", self.Fore.CYAN + self.Style.BRIGHT)
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
                self.print_color(f"   Size: ${trade['position_size_usd']:.2f} | Leverage: {trade['leverage']}x", self.Fore.WHITE)
                self.print_color(f"   Entry: ${trade['entry_price']:.4f} | Current: ${current_price:.4f}", self.Fore.WHITE)
                self.print_color(f"   P&L: ${unrealized_pnl:.2f}", pnl_color)
                self.print_color(f"   TP: ${trade['take_profit']:.4f} | SL: ${trade['stop_loss']:.4f}", self.Fore.YELLOW)
                self.print_color("   " + "-" * 60, self.Fore.CYAN)
        
        if active_count == 0:
            self.print_color("No active positions", self.Fore.YELLOW)
        else:
            total_color = self.Fore.GREEN + self.Style.BRIGHT if total_unrealized >= 0 else self.Fore.RED + self.Style.BRIGHT
            self.print_color(f"📊 Active Positions: {active_count} | Total Unrealized P&L: ${total_unrealized:.2f}", total_color)

    def run_trading_cycle(self):
        """Run one complete trading cycle"""
        try:
            self.monitor_positions()
            self.display_dashboard()
            
            # Show stats every 3 cycles
            if hasattr(self, 'cycle_count') and self.cycle_count % 3 == 0:
                self.show_trade_history(8)
                self.show_trading_stats()
            
            self.print_color(f"\n🔍 AI SCANNING {len(self.available_pairs)} PAIRS WITH ${self.available_budget:.2f} AVAILABLE...", self.Fore.BLUE + self.Style.BRIGHT)
            
            qualified_signals = 0
            for pair in self.available_pairs:
                if self.available_budget > 100:  # Minimum $100 to consider trading
                    market_data = self.get_price_history(pair)
                    ai_decision = self.get_ai_trading_decision(pair, market_data)
                    
                    if ai_decision["decision"] in ["LONG", "SHORT"] and ai_decision["position_size_usd"] > 0:
                        qualified_signals += 1
                        direction_icon = "🟢 LONG" if ai_decision['decision'] == "LONG" else "🔴 SHORT"
                        self.print_color(f"🎯 AI SIGNAL: {pair} {direction_icon} | Size: ${ai_decision['position_size_usd']:.2f}", self.Fore.GREEN + self.Style.BRIGHT)
                        success = self.execute_ai_trade(pair, ai_decision)
                        if success:
                            time.sleep(2)  # Delay between executions
                
            if qualified_signals == 0:
                self.print_color("No qualified AI signals this cycle", self.Fore.YELLOW)
                
        except Exception as e:
            self.print_color(f"Trading cycle error: {e}", self.Fore.RED)

    def start_trading(self):
        """Start the fully autonomous AI trading"""
        self.print_color("🚀 STARTING FULLY AUTONOMOUS AI TRADER!", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color("💰 AI MANAGING $5000 PORTFOLIO", self.Fore.GREEN + self.Style.BRIGHT)
        self.print_color("🤖 COMPLETE AI CONTROL: Analysis, Sizing, Entry, TP, SL", self.Fore.MAGENTA + self.Style.BRIGHT)
        self.print_color("⏰ 15MIN TIMEFRAME | RISK MANAGEMENT: 20% max per trade", self.Fore.YELLOW + self.Style.BRIGHT)
        
        self.cycle_count = 0
        while True:
            try:
                self.cycle_count += 1
                self.print_color(f"\n🔄 AI TRADING CYCLE {self.cycle_count}", self.Fore.CYAN + self.Style.BRIGHT)
                self.print_color("=" * 60, self.Fore.CYAN)
                self.run_trading_cycle()
                self.print_color(f"⏳ AI analyzing next opportunities in 2 minutes...", self.Fore.BLUE)
                time.sleep(120)  # 2 minutes between cycles
                
            except KeyboardInterrupt:
                self.print_color(f"\n🛑 AI TRADING STOPPED", self.Fore.RED + self.Style.BRIGHT)
                self.show_trade_history(15)
                self.show_trading_stats()
                break
            except Exception as e:
                self.print_color(f"Main loop error: {e}", self.Fore.RED)
                time.sleep(120)


class FullyAutonomousPaperTrader:
    def __init__(self, real_bot):
        self.real_bot = real_bot
        # Copy colorama attributes from real_bot
        self.Fore = real_bot.Fore
        self.Back = real_bot.Back
        self.Style = real_bot.Style
        self.COLORAMA_AVAILABLE = real_bot.COLORAMA_AVAILABLE
        
        self.paper_balance = 5000  # Virtual $5000 budget
        self.available_budget = 5000
        self.paper_positions = {}
        self.paper_history_file = "fully_autonomous_paper_trading_history.json"
        self.paper_history = self.load_paper_history()
        
        self.real_bot.print_color("🤖 FULLY AUTONOMOUS PAPER TRADER INITIALIZED!", self.Fore.GREEN + self.Style.BRIGHT)
        self.real_bot.print_color(f"💰 Virtual Budget: ${self.paper_balance}", self.Fore.CYAN + self.Style.BRIGHT)
        self.real_bot.print_color(f"🎯 AI Full Control: Analysis, Sizing, Entry, TP, SL", self.Fore.MAGENTA + self.Style.BRIGHT)
        self.real_bot.print_color(f"⏰ 15MIN Timeframe | Risk: 20% max per trade", self.Fore.YELLOW + self.Style.BRIGHT)
        self.real_bot.print_color(f"💾 Paper trades saved to: {self.paper_history_file}", self.Fore.GREEN)
        
    def load_paper_history(self):
        """Load PAPER trading history"""
        try:
            if os.path.exists(self.paper_history_file):
                with open(self.paper_history_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            self.real_bot.print_color(f"Error loading paper trade history: {e}", self.Fore.RED)
            return []
    
    def save_paper_history(self):
        """Save PAPER trading history"""
        try:
            with open(self.paper_history_file, 'w') as f:
                json.dump(self.paper_history, f, indent=2)
        except Exception as e:
            self.real_bot.print_color(f"Error saving paper trade history: {e}", self.Fore.RED)
    
    def add_paper_trade_to_history(self, trade_data):
        """Add trade to PAPER trading history"""
        try:
            trade_data['close_time'] = self.real_bot.get_thailand_time()
            trade_data['close_timestamp'] = time.time()
            trade_data['trade_type'] = 'PAPER'
            self.paper_history.append(trade_data)
            
            if len(self.paper_history) > 200:
                self.paper_history = self.paper_history[-200:]
            self.save_paper_history()
            self.real_bot.print_color(f"📝 PAPER Trade saved: {trade_data['pair']} {trade_data['direction']} P&L: ${trade_data.get('pnl', 0):.2f}", self.Fore.CYAN)
        except Exception as e:
            self.real_bot.print_color(f"Error adding paper trade to history: {e}", self.Fore.RED)
    
    def show_paper_trade_history(self, limit=15):
        """Show PAPER trading history"""
        if not self.paper_history:
            self.real_bot.print_color("No PAPER trade history found", self.Fore.YELLOW)
            return
        
        self.real_bot.print_color(f"\n📝 PAPER TRADING HISTORY (Last {min(limit, len(self.paper_history))} trades)", self.Fore.GREEN + self.Style.BRIGHT)
        self.real_bot.print_color("=" * 120, self.Fore.GREEN)
        
        recent_trades = self.paper_history[-limit:]
        for i, trade in enumerate(reversed(recent_trades)):
            pnl = trade.get('pnl', 0)
            pnl_color = self.Fore.GREEN + self.Style.BRIGHT if pnl > 0 else self.Fore.RED + self.Style.BRIGHT if pnl < 0 else self.Fore.YELLOW
            direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
            position_size = trade.get('position_size_usd', 0)
            
            self.real_bot.print_color(f"{i+1:2d}. {direction_icon} {trade['pair']} | Size: ${position_size:.2f} | P&L: ${pnl:.2f}", pnl_color)
            self.real_bot.print_color(f"     Entry: ${trade.get('entry_price', 0):.4f} | Exit: ${trade.get('exit_price', 0):.4f} | {trade.get('close_reason', 'N/A')}", self.Fore.YELLOW)
    
    def paper_execute_trade(self, pair, ai_decision):
        """Execute paper trade based on AI's decision"""
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
                self.real_bot.print_color(f"🟡 AI decides to HOLD {pair} (Confidence: {confidence}%)", self.Fore.YELLOW)
                return False
            
            # Check if we can open position
            if pair in self.paper_positions:
                self.real_bot.print_color(f"🚫 Cannot open {pair}: Position already exists", self.Fore.RED)
                return False
            
            if len(self.paper_positions) >= 5:
                self.real_bot.print_color(f"🚫 Cannot open {pair}: Max concurrent trades reached", self.Fore.RED)
                return False
                
            if position_size_usd > self.available_budget:
                self.real_bot.print_color(f"🚫 Cannot open {pair}: Insufficient budget", self.Fore.RED)
                return False
            
            max_allowed = 5000 * 0.2  # 20% of total budget
            if position_size_usd > max_allowed:
                position_size_usd = max_allowed
                self.real_bot.print_color(f"📝 Adjusted position size to max allowed: ${position_size_usd:.2f}", self.Fore.YELLOW)
            
            # Calculate quantity
            notional_value = position_size_usd * leverage
            quantity = notional_value / entry_price
            quantity = round(quantity, 3)  # Simple precision
            
            # Format prices
            take_profit = round(take_profit, 4)
            stop_loss = round(stop_loss, 4)
            
            # Display AI trade decision
            direction_color = self.Fore.GREEN + self.Style.BRIGHT if decision == 'LONG' else self.Fore.RED + self.Style.BRIGHT
            direction_icon = "🟢 LONG" if decision == 'LONG' else "🔴 SHORT"
            
            self.real_bot.print_color(f"\n🤖 PAPER TRADE EXECUTION", self.Fore.CYAN + self.Style.BRIGHT)
            self.real_bot.print_color("=" * 80, self.Fore.CYAN)
            self.real_bot.print_color(f"{direction_icon} {pair}", direction_color)
            self.real_bot.print_color(f"POSITION SIZE: ${position_size_usd:.2f}", self.Fore.GREEN + self.Style.BRIGHT)
            self.real_bot.print_color(f"ENTRY PRICE: ${entry_price:.4f}", self.Fore.WHITE)
            self.real_bot.print_color(f"LEVERAGE: {leverage}x", self.Fore.MAGENTA)
            self.real_bot.print_color(f"QUANTITY: {quantity}", self.Fore.CYAN)
            self.real_bot.print_color(f"TAKE PROFIT: ${take_profit:.4f}", self.Fore.GREEN)
            self.real_bot.print_color(f"STOP LOSS: ${stop_loss:.4f}", self.Fore.RED)
            self.real_bot.print_color(f"CONFIDENCE: {confidence}%", self.Fore.YELLOW + self.Style.BRIGHT)
            self.real_bot.print_color(f"REASONING: {reasoning}", self.Fore.WHITE)
            self.real_bot.print_color("=" * 80, self.Fore.CYAN)
            
            # Update budget and track trade
            self.available_budget -= position_size_usd
            
            self.paper_positions[pair] = {
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
                'entry_time_th': self.real_bot.get_thailand_time()
            }
            
            self.real_bot.print_color(f"✅ PAPER TRADE EXECUTED: {pair} {decision} | Budget Used: ${position_size_usd:.2f}", self.Fore.GREEN + self.Style.BRIGHT)
            return True
            
        except Exception as e:
            self.real_bot.print_color(f"❌ Paper trade execution failed: {e}", self.Fore.RED)
            return False

    def monitor_paper_positions(self):
        """Monitor paper positions for TP/SL"""
        try:
            closed_positions = []
            for pair, trade in list(self.paper_positions.items()):
                if trade['status'] != 'ACTIVE':
                    continue
                
                current_price = self.real_bot.get_current_price(pair)
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
                    
                    # Return used budget plus P&L
                    self.available_budget += trade['position_size_usd'] + pnl
                    self.paper_balance = self.available_budget  # Update total balance
                    
                    self.add_paper_trade_to_history(trade.copy())
                    closed_positions.append(pair)
                    
                    pnl_color = self.Fore.GREEN + self.Style.BRIGHT if pnl > 0 else self.Fore.RED + self.Style.BRIGHT
                    direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
                    self.real_bot.print_color(f"\n🔚 PAPER TRADE CLOSED: {pair} {direction_icon}", pnl_color)
                    self.real_bot.print_color(f"   P&L: ${pnl:.2f} | Reason: {close_reason}", pnl_color)
                    self.real_bot.print_color(f"   New Available Budget: ${self.available_budget:.2f}", self.Fore.CYAN)
                    
                    del self.paper_positions[pair]
                    
            return closed_positions
                    
        except Exception as e:
            self.real_bot.print_color(f"Paper monitoring error: {e}", self.Fore.RED)
            return []

    def get_paper_portfolio_status(self):
        """Show paper trading portfolio status"""
        total_trades = len(self.paper_history)
        winning_trades = len([t for t in self.paper_history if t.get('pnl', 0) > 0])
        total_pnl = sum(trade.get('pnl', 0) for trade in self.paper_history)
        
        self.real_bot.print_color(f"\n📊 PAPER TRADING PORTFOLIO", self.Fore.CYAN + self.Style.BRIGHT)
        self.real_bot.print_color("=" * 70, self.Fore.CYAN)
        self.real_bot.print_color(f"Active Positions: {len(self.paper_positions)}", self.Fore.WHITE)
        self.real_bot.print_color(f"Available Budget: ${self.available_budget:.2f}", self.Fore.WHITE + self.Style.BRIGHT)
        self.real_bot.print_color(f"Total Paper Trades: {total_trades}", self.Fore.WHITE)
        
        if total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100
            self.real_bot.print_color(f"Paper Win Rate: {win_rate:.1f}%", self.Fore.GREEN + self.Style.BRIGHT if win_rate > 50 else self.Fore.YELLOW)
            self.real_bot.print_color(f"Total Paper P&L: ${total_pnl:.2f}", self.Fore.GREEN + self.Style.BRIGHT if total_pnl > 0 else self.Fore.RED + self.Style.BRIGHT)
            avg_trade = total_pnl / total_trades
            self.real_bot.print_color(f"Average Paper P&L: ${avg_trade:.2f}", self.Fore.WHITE)

    def display_paper_dashboard(self):
        """Display paper trading dashboard"""
        self.real_bot.print_color(f"\n🤖 PAPER TRADING DASHBOARD - {self.real_bot.get_thailand_time()}", self.Fore.GREEN + self.Style.BRIGHT)
        self.real_bot.print_color("=" * 90, self.Fore.GREEN)
        
        active_count = 0
        total_unrealized = 0
        
        for pair, trade in self.paper_positions.items():
            if trade['status'] == 'ACTIVE':
                active_count += 1
                current_price = self.real_bot.get_current_price(pair)
                
                direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
                
                if trade['direction'] == 'LONG':
                    unrealized_pnl = (current_price - trade['entry_price']) * trade['quantity']
                else:
                    unrealized_pnl = (trade['entry_price'] - current_price) * trade['quantity']
                    
                total_unrealized += unrealized_pnl
                pnl_color = self.Fore.GREEN + self.Style.BRIGHT if unrealized_pnl >= 0 else self.Fore.RED + self.Style.BRIGHT
                
                self.real_bot.print_color(f"{direction_icon} {pair}", self.Fore.WHITE + self.Style.BRIGHT)
                self.real_bot.print_color(f"   Size: ${trade['position_size_usd']:.2f} | Leverage: {trade['leverage']}x", self.Fore.WHITE)
                self.real_bot.print_color(f"   Entry: ${trade['entry_price']:.4f} | Current: ${current_price:.4f}", self.Fore.WHITE)
                self.real_bot.print_color(f"   P&L: ${unrealized_pnl:.2f}", pnl_color)
                self.real_bot.print_color(f"   TP: ${trade['take_profit']:.4f} | SL: ${trade['stop_loss']:.4f}", self.Fore.YELLOW)
                self.real_bot.print_color("   " + "-" * 60, self.Fore.GREEN)
        
        if active_count == 0:
            self.real_bot.print_color("No active paper positions", self.Fore.YELLOW)
        else:
            total_color = self.Fore.GREEN + self.Style.BRIGHT if total_unrealized >= 0 else self.Fore.RED + self.Style.BRIGHT
            self.real_bot.print_color(f"📊 Active Paper Positions: {active_count} | Total Unrealized P&L: ${total_unrealized:.2f}", total_color)

    def run_paper_trading_cycle(self):
        """Run one complete paper trading cycle"""
        try:
            self.monitor_paper_positions()
            self.display_paper_dashboard()
            
            # Show paper history every 5 cycles
            if hasattr(self, 'paper_cycle_count') and self.paper_cycle_count % 5 == 0:
                self.show_paper_trade_history(8)
            
            self.get_paper_portfolio_status()
            
            self.real_bot.print_color(f"\n🔍 AI SCANNING FOR PAPER TRADES WITH ${self.available_budget:.2f} AVAILABLE...", self.Fore.BLUE + self.Style.BRIGHT)
            
            qualified_signals = 0
            for pair in self.real_bot.available_pairs:
                if self.available_budget > 100:  # Minimum $100 to consider trading
                    market_data = self.real_bot.get_price_history(pair)
                    ai_decision = self.real_bot.get_ai_trading_decision(pair, market_data)
                    
                    if ai_decision["decision"] in ["LONG", "SHORT"] and ai_decision["position_size_usd"] > 0:
                        qualified_signals += 1
                        direction_icon = "🟢 LONG" if ai_decision['decision'] == "LONG" else "🔴 SHORT"
                        self.real_bot.print_color(f"🎯 PAPER AI SIGNAL: {pair} {direction_icon} | Size: ${ai_decision['position_size_usd']:.2f}", self.Fore.GREEN + self.Style.BRIGHT)
                        self.paper_execute_trade(pair, ai_decision)
                        time.sleep(1)  # Small delay between paper executions
                
            if qualified_signals > 0:
                self.real_bot.print_color(f"🎯 {qualified_signals} qualified paper signals executed", self.Fore.GREEN + self.Style.BRIGHT)
            else:
                self.real_bot.print_color("No qualified paper signals this cycle", self.Fore.YELLOW)
            
        except Exception as e:
            self.real_bot.print_color(f"Paper trading cycle error: {e}", self.Fore.RED)

    def start_paper_trading(self):
        """Start paper trading"""
        self.real_bot.print_color("🚀 STARTING FULLY AUTONOMOUS PAPER TRADING!", self.Fore.GREEN + self.Style.BRIGHT)
        self.real_bot.print_color("💰 VIRTUAL $5000 BUDGET - NO REAL MONEY", self.Fore.CYAN + self.Style.BRIGHT)
        self.real_bot.print_color("🤖 AI Full Control: Analysis, Sizing, Entry, TP, SL", self.Fore.MAGENTA + self.Style.BRIGHT)
        self.real_bot.print_color("⏰ 15MIN Timeframe | Risk Management: 20% max per trade", self.Fore.YELLOW + self.Style.BRIGHT)
        
        self.paper_cycle_count = 0
        while True:
            try:
                self.paper_cycle_count += 1
                self.real_bot.print_color(f"\n🔄 PAPER TRADING CYCLE {self.paper_cycle_count}", self.Fore.GREEN)
                self.real_bot.print_color("=" * 60, self.Fore.GREEN)
                self.run_paper_trading_cycle()
                self.real_bot.print_color(f"⏳ AI analyzing next paper opportunities in 2 minutes...", self.Fore.BLUE)
                time.sleep(120)  # 2 minutes between cycles
                
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
                    self.real_bot.print_color(f"Total Paper Trades: {total_trades}", self.Fore.WHITE)
                    self.real_bot.print_color(f"Paper Win Rate: {win_rate:.1f}%", self.Fore.GREEN)
                    self.real_bot.print_color(f"Total Paper P&L: ${total_pnl:.2f}", self.Fore.GREEN if total_pnl > 0 else self.Fore.RED)
                    self.real_bot.print_color(f"Final Paper Balance: ${self.paper_balance:.2f}", self.Fore.CYAN + self.Style.BRIGHT)
                
                break
            except Exception as e:
                self.real_bot.print_color(f"Paper trading error: {e}", self.Fore.RED)
                time.sleep(120)

if __name__ == "__main__":
    try:
        ai_trader = FullyAutonomous15MinAITrader()
        
        print("\n" + "="*80)
        print("🤖 FULLY AUTONOMOUS AI TRADER")
        print("="*80)
        print("SELECT MODE:")
        print("1. 🚀 Live Trading (AI Manages Real $5000)")
        print("2. 💸 Paper Trading (Virtual $5000)")
        
        choice = input("Enter choice (1-2): ").strip()
        
        if choice == "1":
            print("⚠️  WARNING: REAL MONEY TRADING! ⚠️")
            print("🤖 AI HAS COMPLETE CONTROL OVER $5000")
            print("📊 AI DECIDES: Position Sizing, Entry, TP, SL, Leverage")
            confirm = input("Type 'AUTONOMOUS' to confirm: ").strip()
            if confirm.upper() == 'AUTONOMOUS':
                ai_trader.start_trading()
            else:
                print("Using Paper Trading mode instead...")
                paper_bot = FullyAutonomousPaperTrader(ai_trader)
                paper_bot.start_paper_trading()
        else:
            paper_bot = FullyAutonomousPaperTrader(ai_trader)
            paper_bot.start_paper_trading()
            
    except Exception as e:
        print(f"Failed to start AI trader: {e}")
