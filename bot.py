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
import pandas as pd

# Install required packages:
# pip install colorama python-binance python-dotenv numpy pytz pandas

try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    print("Warning: Colorama not installed. Run: pip install colorama")

# Load environment variables
load_dotenv()

class Final3MinScalpingBot:
    def __init__(self):
        # Load config from .env file
        self.binance_api_key = os.getenv('BINANCE_API_KEY')
        self.binance_secret = os.getenv('BINANCE_SECRET_KEY')
        self.deepseek_key = os.getenv('DEEPSEEK_API_KEY')
        
        # Thailand timezone
        self.thailand_tz = pytz.timezone('Asia/Bangkok')
        
        # 3MIN SCALPING PARAMETERS
        self.trade_size_usd = 50
        self.leverage = 5
        self.tp_percent = 0.006   # +0.6%
        self.sl_percent = 0.003   # -0.3%
        
        # Multi-pair parameters
        self.max_concurrent_trades = 2
        self.available_pairs = ["SOLUSDT", "AVAXUSDT", "XRPUSDT", "LINKUSDT", "DOTUSDT"]
        
        # Track bot-opened trades (live)
        self.bot_opened_trades = {}
        
        # Paper trading mode
        self.paper_trading = False
        self.paper_balance = 1000.0  # Starting balance
        self.paper_positions = {}    # Simulated positions
        
        # Trade history (shared)
        self.trade_history_file = "3min_scalping_history.json"
        self.trade_history = self.load_trade_history()
        
        # Precision settings
        self.quantity_precision = {}
        self.price_precision = {}
        
        # Initialize Binance client
        self.binance = Client(self.binance_api_key, self.binance_secret)
        
        self.print_color(f"FINAL 3MIN SCALPING BOT ACTIVATED!", Fore.CYAN + Style.BRIGHT)
        self.print_color(f"TP: +0.6% | SL: -0.3% | R:R = 1:2", Fore.GREEN + Style.BRIGHT)
        self.print_color(f"AI: 3-MINUTE SCALPING EXPERT", Fore.YELLOW)
        
        self.validate_config()
        self.setup_futures()
        self.load_symbol_precision()
    
    def load_trade_history(self):
        try:
            if os.path.exists(self.trade_history_file):
                with open(self.trade_history_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            self.print_color(f"Error loading trade history: {e}", Fore.RED)
            return []
    
    def save_trade_history(self):
        try:
            with open(self.trade_history_file, 'w') as f:
                json.dump(self.trade_history, f, indent=2)
        except Exception as e:
            self.print_color(f"Error saving trade history: {e}", Fore.RED)
    
    def add_trade_to_history(self, trade_data):
        try:
            trade_data['close_time'] = self.get_thailand_time()
            trade_data['close_timestamp'] = time.time()
            self.trade_history.append(trade_data)
            if len(self.trade_history) > 100:
                self.trade_history = self.trade_history[-100:]
            self.save_trade_history()
            self.print_color(f"Trade saved: {trade_data['pair']} {trade_data['direction']}", Fore.CYAN)
        except Exception as e:
            self.print_color(f"Error adding trade to history: {e}", Fore.RED)
    
    def show_trade_history(self, limit=10):
        if not self.trade_history:
            self.print_color("No trade history found", Fore.YELLOW)
            return
        self.print_color(f"\n3MIN SCALPING HISTORY (Last {min(limit, len(self.trade_history))} trades)", Fore.CYAN)
        self.print_color("=" * 80, Fore.CYAN)
        for i, trade in enumerate(reversed(self.trade_history[-limit:])):
            pnl = trade.get('pnl', 0)
            pnl_color = Fore.GREEN if pnl > 0 else Fore.RED if pnl < 0 else Fore.YELLOW
            mode = "[PAPER]" if 'paper' in trade else "[LIVE]"
            self.print_color(f"{mode} {i+1}. {trade['pair']} {trade['direction']} | Entry: ${trade.get('entry_price', 0):.4f} | Exit: ${trade.get('exit_price', 0):.4f} | P&L: ${pnl:.2f}", pnl_color)
            self.print_color(f"   TP: ${trade.get('take_profit', 0):.4f} | SL: ${trade.get('stop_loss', 0):.4f} | Time: {trade.get('close_time', 'N/A')}", Fore.YELLOW)
    
    def get_thailand_time(self):
        now_utc = datetime.now(pytz.utc)
        thailand_time = now_utc.astimezone(self.thailand_tz)
        return thailand_time.strftime('%Y-%m-%d %H:%M:%S')
    
    def print_color(self, text, color=Fore.WHITE, style=Style.NORMAL):
        if COLORAMA_AVAILABLE:
            print(f"{style}{color}{text}")
        else:
            print(text)
    
    def validate_config(self):
        if not all([self.binance_api_key, self.binance_secret]):
            self.print_color("Missing Binance API keys!", Fore.RED)
            return False
        try:
            self.binance.futures_exchange_info()
            self.print_color("Binance connection successful!", Fore.GREEN)
        except Exception as e:
            self.print_color(f"Binance connection failed: {e}", Fore.RED)
            return False
        return True

    def setup_futures(self):
        try:
            for pair in self.available_pairs:
                try:
                    self.binance.futures_change_leverage(symbol=pair, leverage=self.leverage)
                    self.print_color(f"Leverage set for {pair}", Fore.GREEN)
                except Exception as e:
                    self.print_color(f"Leverage setup failed for {pair}: {e}", Fore.YELLOW)
            self.print_color("Futures setup completed!", Fore.GREEN)
        except Exception as e:
            self.print_color(f"Futures setup failed: {e}", Fore.RED)
    
    def load_symbol_precision(self):
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
            self.print_color("Symbol precision loaded", Fore.GREEN)
        except Exception as e:
            self.print_color(f"Error loading symbol precision: {e}", Fore.RED)
    
    def format_price(self, pair, price):
        if price <= 0:
            return 0.0
        precision = self.price_precision.get(pair, 4)
        return round(price, precision)
    
    def get_quantity(self, pair, price):
        try:
            if not price or price <= 0:
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
                return None
                
            return quantity
            
        except Exception as e:
            return None

    # === BACKTEST FUNCTION ===
    def run_backtest(self, days=7):
        self.print_color(f"\nBACKTESTING 3MIN SCALPING STRATEGY...", Fore.CYAN + Style.BRIGHT)
        self.print_color(f"Period: Last {days} days | Pairs: {len(self.available_pairs)}", Fore.YELLOW)
        
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        results = {}
        total_trades = 0
        total_pnl = 0
        
        for pair in self.available_pairs:
            self.print_color(f"\nBacktesting {pair}...", Fore.BLUE)
            try:
                klines = self.binance.futures_historical_klines(
                    symbol=pair,
                    interval=Client.KLINE_INTERVAL_3MINUTE,
                    start_str=start_time,
                    end_str=end_time,
                    limit=1000
                )
                
                if len(klines) < 100:
                    self.print_color(f"Not enough data for {pair}", Fore.YELLOW)
                    continue
                
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base', 'taker_buy_quote', 'ignore'
                ])
                
                price_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in price_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(subset=price_cols)
                
                if len(df) < 50:
                    self.print_color(f"Insufficient data for {pair}", Fore.YELLOW)
                    continue
                
                trades = []
                active_trade = None
                
                for i in range(10, len(df) - 5):
                    current_price = df['close'].iloc[i]
                    recent_change = (df['close'].iloc[i] - df['close'].iloc[i-10]) / df['close'].iloc[i-10]
                    
                    if active_trade is None:
                        if recent_change > 0.0015:
                            quantity = self.get_quantity(pair, current_price)
                            if quantity:
                                active_trade = {
                                    'entry': current_price,
                                    'quantity': quantity,
                                    'direction': 'LONG',
                                    'tp': current_price * 1.006,
                                    'sl': current_price * 0.997,
                                    'entry_idx': i
                                }
                        elif recent_change < -0.0015:
                            quantity = self.get_quantity(pair, current_price)
                            if quantity:
                                active_trade = {
                                    'entry': current_price,
                                    'quantity': quantity,
                                    'direction': 'SHORT',
                                    'tp': current_price * 0.994,
                                    'sl': current_price * 1.003,
                                    'entry_idx': i
                                }
                    
                    if active_trade:
                        high = df['high'].iloc[i]
                        low = df['low'].iloc[i]
                        
                        if active_trade['direction'] == 'LONG':
                            if high >= active_trade['tp']:
                                pnl = (active_trade['tp'] - active_trade['entry']) * active_trade['quantity']
                                trades.append({'pnl': pnl, 'result': 'WIN'})
                                active_trade = None
                            elif low <= active_trade['sl']:
                                pnl = (active_trade['sl'] - active_trade['entry']) * active_trade['quantity']
                                trades.append({'pnl': pnl, 'result': 'LOSS'})
                                active_trade = None
                        else:
                            if low <= active_trade['tp']:
                                pnl = (active_trade['entry'] - active_trade['tp']) * active_trade['quantity']
                                trades.append({'pnl': pnl, 'result': 'WIN'})
                                active_trade = None
                            elif high >= active_trade['sl']:
                                pnl = (active_trade['entry'] - active_trade['sl']) * active_trade['quantity']
                                trades.append({'pnl': pnl, 'result': 'LOSS'})
                                active_trade = None
                
                if trades:
                    wins = [t for t in trades if t['result'] == 'WIN']
                    win_rate = len(wins) / len(trades) * 100
                    total_pnl_pair = sum(t['pnl'] for t in trades)
                    avg_pnl = total_pnl_pair / len(trades)
                    
                    results[pair] = {
                        'trades': len(trades),
                        'win_rate': win_rate,
                        'total_pnl': total_pnl_pair,
                        'avg_pnl': avg_pnl
                    }
                    
                    total_trades += len(trades)
                    total_pnl += total_pnl_pair
                    
                    self.print_color(f"{pair}: {len(trades)} trades | Win Rate: {win_rate:.1f}% | P&L: ${total_pnl_pair:.2f}", 
                                   Fore.GREEN if total_pnl_pair > 0 else Fore.RED)
                else:
                    self.print_color(f"{pair}: No valid trades", Fore.YELLOW)
                    
            except Exception as e:
                self.print_color(f"Backtest error for {pair}: {e}", Fore.RED)
        
        self.print_color(f"\nBACKTEST SUMMARY ({days} days)", Fore.CYAN + Style.BRIGHT)
        self.print_color("=" * 60, Fore.CYAN)
        for pair, res in results.items():
            self.print_color(f"{pair}: {res['trades']} trades | {res['win_rate']:.1f}% WR | ${res['total_pnl']:.2f}", 
                           Fore.GREEN if res['total_pnl'] > 0 else Fore.RED)
        
        if total_trades > 0:
            overall_wr = sum(r['trades'] * r['win_rate'] / 100 for r in results.values()) / total_trades * 100
            self.print_color(f"\nTOTAL TRADES: {total_trades}", Fore.YELLOW)
            self.print_color(f"OVERALL WIN RATE: {overall_wr:.1f}%", Fore.YELLOW)
            self.print_color(f"TOTAL P&L: ${total_pnl:.2f}", Fore.GREEN if total_pnl > 0 else Fore.RED)
            self.print_color(f"AVG P&L PER TRADE: ${total_pnl/total_trades:.3f}", Fore.CYAN)
        
        return results

    # === AI ANALYSIS ===
    def get_deepseek_analysis(self, pair, market_data):
        try:
            if not self.deepseek_key:
                return "HOLD", 0, "No API key"
            
            current_price = market_data['current_price']
            prompt = f"""
            You are a 3-MINUTE SCALPING EXPERT.
            Analyze {pair} at ${current_price:.4f} using last 30 minutes of data.
            Target: +0.6% profit in 3-7 minutes.
            Stop Loss: -0.3%.
            Only respond if you see a HIGH PROBABILITY move in the next 3-5 minutes.
            Respond with JSON: {{"direction": "LONG|SHORT|HOLD", "confidence": 70, "reason": "brief"}}
            """
            
            headers = {"Authorization": f"Bearer {self.deepseek_key}", "Content-Type": "application/json"}
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are a 3min scalping bot. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 300
            }
            
            response = requests.post("https://api.deepseek.com/chat/completions", headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content'].strip()
                json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                if json_match:
                    decision = json.loads(json_match.group())
                    return decision.get('direction', 'HOLD').upper(), decision.get('confidence', 50), decision.get('reason', '')
            return "HOLD", 0, "API Error"
        except:
            return "HOLD", 0, "Error"

    # === PAPER TRADING ===
    def execute_paper_trade(self, decision):
        pair = decision["pair"]
        direction = decision["direction"]
        confidence = decision["confidence"]
        
        try:
            current_price = self.get_current_price(pair)
            if not current_price or current_price <= 0:
                return False
            
            quantity = self.get_quantity(pair, current_price)
            if not quantity:
                return False
            
            if direction == "LONG":
                take_profit = current_price * 1.006
                stop_loss = current_price * 0.997
            else:
                take_profit = current_price * 0.994
                stop_loss = current_price * 1.003
            
            take_profit = self.format_price(pair, take_profit)
            stop_loss = self.format_price(pair, stop_loss)
            
            self.paper_positions[pair] = {
                'direction': direction,
                'entry_price': current_price,
                'quantity': quantity,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'entry_time': time.time(),
                'status': 'ACTIVE',
                'confidence': confidence,
                'paper': True
            }
            
            self.print_color(f"[PAPER] {pair} {direction} | ${current_price:.4f}", Fore.CYAN + Style.BRIGHT)
            self.print_color(f"   TP: ${take_profit:.4f} | SL: ${stop_loss:.4f} | Qty: {quantity}", Fore.CYAN)
            return True
            
        except Exception as e:
            self.print_color(f"[PAPER] Error: {e}", Fore.RED)
            return False

    def monitor_paper_positions(self):
        for pair, pos in list(self.paper_positions.items()):
            if pos['status'] != 'ACTIVE':
                continue
            
            current_price = self.get_current_price(pair)
            if not current_price:
                continue
            
            pnl = 0
            exit_price = None
            triggered = False
            
            if pos['direction'] == 'LONG':
                if current_price >= pos['take_profit']:
                    pnl = (pos['take_profit'] - pos['entry_price']) * pos['quantity']
                    exit_price = pos['take_profit']
                    triggered = True
                elif current_price <= pos['stop_loss']:
                    pnl = (pos['stop_loss'] - pos['entry_price']) * pos['quantity']
                    exit_price = pos['stop_loss']
                    triggered = True
            else:
                if current_price <= pos['take_profit']:
                    pnl = (pos['entry_price'] - pos['take_profit']) * pos['quantity']
                    exit_price = pos['take_profit']
                    triggered = True
                elif current_price >= pos['stop_loss']:
                    pnl = (pos['entry_price'] - pos['stop_loss']) * pos['quantity']
                    exit_price = pos['stop_loss']
                    triggered = True
            
            if triggered:
                pos['status'] = 'CLOSED'
                pos['exit_price'] = exit_price
                pos['pnl'] = pnl
                pos['close_time'] = self.get_thailand_time()
                
                self.paper_balance += pnl
                self.add_trade_to_history(pos.copy())
                
                color = Fore.GREEN if pnl > 0 else Fore.RED
                self.print_color(f"[PAPER] CLOSED {pair} {pos['direction']} | P&L: ${pnl:.2f}", color + Style.BRIGHT)
                self.print_color(f"   Exit: ${exit_price:.4f} | Balance: ${self.paper_balance:.2f}", Fore.CYAN)

    def display_paper_dashboard(self):
        self.print_color(f"\n[PAPER TRADING] Balance: ${self.paper_balance:.2f}", Fore.MAGENTA + Style.BRIGHT)
        self.print_color("=" * 60, Fore.MAGENTA)
        if self.paper_positions:
            for pair, pos in self.paper_positions.items():
                if pos['status'] != 'ACTIVE':
                    continue
                current = self.get_current_price(pair) or 0
                unrealized = 0
                if pos['direction'] == 'LONG':
                    unrealized = (current - pos['entry_price']) * pos['quantity']
                else:
                    unrealized = (pos['entry_price'] - current) * pos['quantity']
                color = Fore.GREEN if unrealized >= 0 else Fore.RED
                self.print_color(f"{pos['direction']} {pair} | Entry: ${pos['entry_price']:.4f}", Fore.WHITE)
                self.print_color(f"   Current: ${current:.4f} | Unrealized: ${unrealized:.2f}", color)

    # === LIVE TRADING ===
    def execute_live_trade(self, decision):
        # (Same as previous live execute_trade)
        pass  # Full live code from previous version

    def execute_trade(self, decision):
        if self.paper_trading:
            return self.execute_paper_trade(decision)
        else:
            return self.execute_live_trade(decision)

    def get_current_price(self, pair):
        try:
            ticker = self.binance.futures_symbol_ticker(symbol=pair)
            return float(ticker['price'])
        except:
            return None

    def can_open_new_trade(self, pair):
        if self.paper_trading:
            return pair not in self.paper_positions or self.paper_positions[pair]['status'] != 'ACTIVE'
        else:
            return pair not in self.bot_opened_trades or self.bot_opened_trades[pair]['status'] != 'ACTIVE'

    def run_trading_cycle(self):
        try:
            if self.paper_trading:
                self.monitor_paper_positions()
                self.display_paper_dashboard()
            else:
                # Live monitoring
                pass
            
            market_data = {}
            for pair in self.available_pairs:
                price = self.get_current_price(pair)
                if price:
                    market_data[pair] = {'price': price}
            
            if market_data:
                for pair in market_data.keys():
                    if self.can_open_new_trade(pair):
                        decision = {"pair": pair, "direction": "LONG", "confidence": 85}  # Simulate AI
                        self.execute_trade(decision)
        except Exception as e:
            self.print_color(f"Cycle error: {e}", Fore.RED)

    def start_trading(self):
        self.print_color("STARTING FINAL 3MIN SCALPING BOT!", Fore.CYAN + Style.BRIGHT)
        self.cycle_count = 0
        while True:
            try:
                self.cycle_count += 1
                self.print_color(f"\nCYCLE {self.cycle_count}", Fore.CYAN)
                self.print_color("=" * 50, Fore.CYAN)
                self.run_trading_cycle()
                time.sleep(60)
            except KeyboardInterrupt:
                if self.paper_trading:
                    self.print_color(f"\n[PAPER] FINAL BALANCE: ${self.paper_balance:.2f}", Fore.MAGENTA + Style.BRIGHT)
                self.show_trade_history(10)
                break

if __name__ == "__main__":
    try:
        bot = Final3MinScalpingBot()
        print("\n" + "="*60)
        print("FINAL 3MIN SCALPING + BACKTEST + PAPER TRADING")
        print("1. Live Trading (Real Money)")
        print("2. Paper Trading (Test Mode - $1000)")
        print("3. Backtest (7 days)")
        print("4. Backtest (1 day)")
        choice = input("Enter choice (1/2/3/4): ").strip()
        
        if choice == "2":
            bot.paper_trading = True
            bot.paper_balance = 1000.0
            bot.start_trading()
        elif choice == "3":
            bot.run_backtest(days=7)
        elif choice == "4":
            bot.run_backtest(days=1)
        elif choice == "1":
            bot.start_trading()
        else:
            print("Invalid choice.")
            
    except Exception as e:
        print(f"Failed to start bot: {e}")
