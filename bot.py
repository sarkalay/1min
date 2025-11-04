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

# Install required packages first: 
# pip install colorama python-binance python-dotenv numpy pytz

try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    print("Warning: Colorama not installed. Run: pip install colorama")

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
        self.max_concurrent_trades = 3  # ← အခု ၃ ခု ဖွင့်လို့ ရတယ်
        self.available_pairs = ["SOLUSDT", "AVAXUSDT", "XRPUSDT", "LINKUSDT", "DOTUSDT"]
        
        # Track bot-opened trades only
        self.bot_opened_trades = {}
        
        # Trade history
        self.trade_history_file = "trade_history.json"
        self.trade_history = self.load_trade_history()
        
        # Precision settings
        self.quantity_precision = {}
        self.price_precision = {}
        
        # Initialize Binance client
        self.binance = Client(self.binance_api_key, self.binance_secret)
        
        self.print_color(f"FINAL AI ALWAYS SCAN BOT ACTIVATED!", Fore.CYAN + Style.BRIGHT)
        self.print_color(f"Trade Size: ${self.trade_size_usd} | Leverage: {self.leverage}x", Fore.GREEN)
        self.print_color(f"Max Trades: {self.max_concurrent_trades} | AI: ALWAYS ON", Fore.YELLOW)
        self.print_color(f"TP/SL Auto-Cleanup | No Dangling Orders", Fore.MAGENTA)
        
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
        self.print_color(f"\nTRADE HISTORY (Last {min(limit, len(self.trade_history))} trades)", Fore.CYAN)
        self.print_color("=" * 80, Fore.CYAN)
        for i, trade in enumerate(reversed(self.trade_history[-limit:])):
            pnl = trade.get('pnl', 0)
            pnl_color = Fore.GREEN if pnl > 0 else Fore.RED if pnl < 0 else Fore.YELLOW
            self.print_color(f"{i+1}. {trade['pair']} {trade['direction']} | Entry: ${trade.get('entry_price', 0):.4f} | Exit: ${trade.get('exit_price', 0):.4f} | P&L: ${pnl:.2f}", pnl_color)
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
        if not all([self.binance_api_key, self.binance_secret, self.deepseek_key]):
            self.print_color("Missing API keys!", Fore.RED)
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
                self.print_color(f"Invalid price: {price} for {pair}", Fore.RED)
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
                self.print_color(f"Invalid quantity: {quantity} for {pair}", Fore.RED)
                return None
                
            actual_value = quantity * price
            self.print_color(f"Quantity for {pair}: {quantity} = ${actual_value:.2f}", Fore.CYAN)
            return quantity
            
        except Exception as e:
            self.print_color(f"Quantity calculation failed: {e}", Fore.RED)
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
            self.print_color(f"AI response parsing failed: {e}", Fore.RED)
            return 'HOLD', 50, 'Parsing failed'

    def get_deepseek_analysis(self, pair, market_data):
        try:
            if not self.deepseek_key:
                self.print_color("DeepSeek API key not found", Fore.RED)
                return "HOLD", 0, "No API key"
            
            current_price = market_data['current_price']
            prompt = f"Analyze {pair} at ${current_price:.4f} for scalping. Respond with JSON: {{\"direction\": \"LONG|SHORT|HOLD\", \"confidence\": 70, \"reason\": \"brief explanation\"}}"
            
            headers = {"Authorization": f"Bearer {self.deepseek_key}", "Content-Type": "application/json"}
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are a crypto trading analyst. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 300
            }
            
            self.print_color(f"Consulting DeepSeek AI for {pair}...", Fore.MAGENTA)
            response = requests.post("https://api.deepseek.com/chat/completions", headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content'].strip()
                direction, confidence, reason = self.parse_ai_response(ai_response)
                self.print_color(f"AI Analysis: {direction} ({confidence}%)", Fore.MAGENTA)
                return direction, confidence, reason
            else:
                self.print_color(f"DeepSeek API error: {response.status_code}", Fore.RED)
                return "HOLD", 0, f"API Error"
                
        except Exception as e:
            self.print_color(f"DeepSeek analysis failed: {e}", Fore.RED)
            return "HOLD", 0, f"Error"

    def get_price_history(self, pair, limit=10):
        try:
            klines = self.binance.futures_klines(symbol=pair, interval=Client.KLINE_INTERVAL_1MINUTE, limit=limit)
            prices = [float(k[4]) for k in klines]
            return {'prices': prices, 'current_price': prices[-1] if prices else 0}
        except Exception as e:
            try:
                ticker = self.binance.futures_symbol_ticker(symbol=pair)
                current_price = float(ticker['price'])
                return {'prices': [current_price] * 5, 'current_price': current_price}
            except:
                return {'prices': [], 'current_price': 0}

    def get_ai_decision(self, pair_data):
        try:
            pair = list(pair_data.keys())[0]
            current_price = pair_data[pair]['price']
            if current_price <= 0:
                return {"action": "HOLD", "pair": pair, "direction": "HOLD", "confidence": 0, "reason": "Invalid price"}
            
            self.print_color(f"Analyzing {pair}...", Fore.BLUE)
            market_data = self.get_price_history(pair)
            market_data['current_price'] = current_price
            direction, confidence, reason = self.get_deepseek_analysis(pair, market_data)
            
            if direction == "HOLD" or confidence < 70:
                self.print_color(f"AI Decision: HOLD ({confidence}%)", Fore.YELLOW)
                return {"action": "HOLD", "pair": pair, "direction": direction, "confidence": confidence, "reason": reason}
            else:
                self.print_color(f"AI Decision: {direction} ({confidence}%)", Fore.GREEN)
                return {"action": "TRADE", "pair": pair, "direction": direction, "confidence": confidence, "reason": reason}
                
        except Exception as e:
            self.print_color(f"AI decision failed: {e}", Fore.RED)
            return {"action": "HOLD", "pair": list(pair_data.keys())[0], "direction": "HOLD", "confidence": 0, "reason": f"Error: {str(e)}"}

    def execute_trade(self, decision):
        try:
            pair = decision["pair"]
            if not self.can_open_new_trade(pair):
                self.print_color(f"Cannot open {pair} - position exists", Fore.RED)
                return False
            
            direction = decision["direction"]
            confidence = decision["confidence"]
            
            ticker = self.binance.futures_symbol_ticker(symbol=pair)
            current_price = float(ticker['price'])
            if current_price <= 0:
                self.print_color(f"Invalid price for {pair}", Fore.RED)
                return False
            
            quantity = self.get_quantity(pair, current_price)
            if quantity is None:
                return False
            
            if direction == "LONG":
                tp_raw = current_price * 1.008
                sl_raw = current_price * 0.995
            else:
                tp_raw = current_price * 0.992
                sl_raw = current_price * 1.005
            
            take_profit = self.format_price(pair, tp_raw)
            stop_loss = self.format_price(pair, sl_raw)
            
            if direction == "LONG":
                if take_profit <= current_price or stop_loss >= current_price:
                    self.print_color(f"Invalid TP/SL for LONG", Fore.RED)
                    return False
            else:
                if take_profit >= current_price or stop_loss <= current_price:
                    self.print_color(f"Invalid TP/SL for SHORT", Fore.RED)
                    return False
            
            direction_color = Fore.BLUE if direction == 'LONG' else Fore.RED
            self.print_color(f"EXECUTING: {pair} {direction}", direction_color)
            self.print_color(f"   Size: {quantity} | Entry: ${current_price:.4f}", Fore.WHITE)
            self.print_color(f"   TP: ${take_profit:.4f} | SL: ${stop_loss:.4f}", Fore.YELLOW)
            
            entry_side = 'BUY' if direction == 'LONG' else 'SELL'
            try:
                order = self.binance.futures_create_order(
                    symbol=pair,
                    side=entry_side,
                    type='MARKET',
                    quantity=quantity
                )
                self.print_color(f"{direction} ORDER EXECUTED", Fore.GREEN)
                time.sleep(2)
                
                stop_side = 'SELL' if direction == 'LONG' else 'BUY'
                
                self.binance.futures_create_order(
                    symbol=pair,
                    side=stop_side,
                    type='STOP_MARKET',
                    quantity=quantity,
                    stopPrice=stop_loss,
                    reduceOnly=True,
                    timeInForce='GTC'
                )
                
                self.binance.futures_create_order(
                    symbol=pair,
                    side=stop_side,
                    type='TAKE_PROFIT_MARKET',
                    quantity=quantity,
                    stopPrice=take_profit,
                    reduceOnly=True,
                    timeInForce='GTC'
                )
                
                self.print_color(f"TP/SL PLACED", Fore.GREEN)
                
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
                
                self.print_color(f"TRADE ACTIVATED: {pair} {direction}", Fore.GREEN + Style.BRIGHT)
                return True
                
            except BinanceAPIException as e:
                self.print_color(f"Binance Error: {e}", Fore.RED)
                return False
            except Exception as e:
                self.print_color(f"Execution Error: {e}", Fore.RED)
                return False
            
        except Exception as e:
            self.print_color(f"Trade failed: {e}", Fore.RED)
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
            self.print_color(f"Error getting live data: {e}", Fore.RED)
            return None

    def get_current_price(self, pair):
        try:
            ticker = self.binance.futures_symbol_ticker(symbol=pair)
            return float(ticker['price'])
        except:
            return None

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

    def close_trade_with_cleanup(self, pair, trade):
        try:
            open_orders = self.binance.futures_get_open_orders(symbol=pair)
            canceled = 0
            for order in open_orders:
                if order['reduceOnly'] and order['symbol'] == pair:
                    try:
                        self.binance.futures_cancel_order(symbol=pair, orderId=order['orderId'])
                        self.print_color(f"Canceled dangling {order['type']} order: {order['orderId']}", Fore.YELLOW)
                        canceled += 1
                    except Exception as e:
                        self.print_color(f"Failed to cancel order {order['orderId']}: {e}", Fore.RED)
            
            trade['status'] = 'CLOSED'
            trade['exit_time_th'] = self.get_thailand_time()
            final_pnl = self.get_final_pnl(pair, trade)
            closed_trade = trade.copy()
            closed_trade['pnl'] = final_pnl
            closed_trade['exit_price'] = self.get_current_price(pair)
            self.add_trade_to_history(closed_trade)
            
            self.print_color(f"TRADE CLOSED: {pair} {trade['direction']} | P&L: ${final_pnl:.2f}", 
                            Fore.GREEN if final_pnl > 0 else Fore.RED)
            if canceled > 0:
                self.print_color(f"Cleaned up {canceled} dangling order(s)", Fore.CYAN)
                
        except Exception as e:
            self.print_color(f"Cleanup failed for {pair}: {e}", Fore.RED)

    def monitor_positions(self):
        try:
            for pair, trade in list(self.bot_opened_trades.items()):
                if trade['status'] != 'ACTIVE':
                    continue
                
                live_data = self.get_live_position_data(pair)
                if not live_data:
                    self.close_trade_with_cleanup(pair, trade)
                    continue
        except Exception as e:
            self.print_color(f"Monitoring error: {e}", Fore.RED)

    def scan_existing_positions(self):
        try:
            positions = self.binance.futures_position_information()
            self.existing_positions = {}
            
            for pos in positions:
                pair = pos['symbol']
                position_amt = float(pos['positionAmt'])
                
                if position_amt == 0 or pair not in self.available_pairs:
                    continue
                
                if pair in self.bot_opened_trades and self.bot_opened_trades[pair]['status'] == 'ACTIVE':
                    continue
                
                entry_price = float(pos.get('entryPrice', 0))
                unrealized_pnl = float(pos.get('unRealizedProfit', 0))
                try:
                    current_price = float(self.binance.futures_symbol_ticker(symbol=pair)['price'])
                except:
                    current_price = entry_price
                
                direction = "SHORT" if position_amt < 0 else "LONG"
                
                self.existing_positions[pair] = {
                    'direction': direction,
                    'entry_price': entry_price,
                    'quantity': abs(position_amt),
                    'current_price': current_price,
                    'unrealized_pnl': unrealized_pnl,
                    'status': 'ACTIVE',
                    'source': 'MANUAL'
                }
            
            return len(self.existing_positions)
            
        except Exception as e:
            self.print_color(f"Error scanning positions: {e}", Fore.RED)
            return 0

    def display_dashboard(self):
        self.print_color(f"\nDASHBOARD - {self.get_thailand_time()}", Fore.CYAN + Style.BRIGHT)
        self.print_color("=" * 80, Fore.CYAN)
        
        displayed = set()
        
        for pair, trade in self.bot_opened_trades.items():
            if trade['status'] != 'ACTIVE':
                continue
            live = self.get_live_position_data(pair)
            if not live:
                continue
            
            displayed.add(pair)
            direction_color = Fore.BLUE if trade['direction'] == 'LONG' else Fore.RED
            pnl_color = Fore.GREEN if live['unrealized_pnl'] >= 0 else Fore.RED
            
            self.print_color(f"{trade['direction']} {pair} [AI]", direction_color + Style.BRIGHT)
            self.print_color(f"   Size: {trade['quantity']} | Entry: ${trade['entry_price']:.4f}", Fore.WHITE)
            self.print_color(f"   Current: ${live['current_price']:.4f} | P&L: ${live['unrealized_pnl']:.2f}", pnl_color)
            
            cp = live['current_price']
            if trade['direction'] == 'LONG':
                tp_dist = ((trade['take_profit'] - cp) / cp) * 100
                sl_dist = ((cp - trade['stop_loss']) / cp) * 100
            else:
                tp_dist = ((cp - trade['take_profit']) / cp) * 100
                sl_dist = ((trade['stop_loss'] - cp) / cp) * 100
            self.print_color(f"   TP: ${trade['take_profit']:.4f} | SL: ${trade['stop_loss']:.4f}", Fore.YELLOW)
            self.print_color(f"   TP: +{tp_dist:.2f}% | SL: -{sl_dist:.2f}%", Fore.CYAN)
        
        for pair, pos in self.existing_positions.items():
            if pair in displayed:
                continue
            
            pnl_color = Fore.GREEN if pos['unrealized_pnl'] >= 0 else Fore.RED
            direction_color = Fore.BLUE if pos['direction'] == 'LONG' else Fore.RED
            
            self.print_color(f"{pos['direction']} {pair}", direction_color)
            self.print_color(f"   Size: {pos['quantity']} | Entry: ${pos['entry_price']:.4f}", Fore.WHITE)
            self.print_color(f"   Current: ${pos['current_price']:.4f} | P&L: ${pos['unrealized_pnl']:.2f}", pnl_color)
            self.print_color(f"   TP/SL: Manual (Not set by bot)", Fore.YELLOW)
        
        if not displayed and not self.existing_positions:
            self.print_color("No active positions", Fore.YELLOW)

    def can_open_new_trade(self, pair):
        if pair in self.bot_opened_trades and self.bot_opened_trades[pair]['status'] == 'ACTIVE':
            return False
        if pair in self.existing_positions:
            return False
        total_active = len(self.bot_opened_trades) + len(self.existing_positions)
        return total_active < self.max_concurrent_trades
    
    def get_market_data(self):
        market_data = {}
        for pair in self.available_pairs:
            try:
                ticker = self.binance.futures_symbol_ticker(symbol=pair)
                price = float(ticker['price'])
                if price > 0:
                    market_data[pair] = {'price': price}
            except Exception as e:
                continue
        return market_data

    def run_trading_cycle(self):
        try:
            self.scan_existing_positions()
            self.monitor_positions()
            self.display_dashboard()
            
            if hasattr(self, 'cycle_count') and self.cycle_count % 5 == 0:
                self.show_trade_history(5)
            
            # AI ALWAYS SCANS ALL PAIRS
            market_data = self.get_market_data()
            if market_data:
                self.print_color(f"\nAI ALWAYS SCANNING {len(market_data)} PAIRS...", Fore.BLUE + Style.BRIGHT)
                for pair in market_data.keys():
                    if self.can_open_new_trade(pair):
                        pair_data = {pair: market_data[pair]}
                        decision = self.get_ai_decision(pair_data)
                        if decision["action"] == "TRADE":
                            self.print_color(f"QUALIFIED: {pair} {decision['direction']} ({decision['confidence']}%)", Fore.GREEN + Style.BRIGHT)
                            success = self.execute_trade(decision)
                            if success:
                                # Continue scanning, but don't open more than max_concurrent
                                pass
                        else:
                            self.print_color(f"HOLD: {pair} ({decision['confidence']}%)", Fore.YELLOW)
                    else:
                        self.print_color(f"SKIPPED: {pair} (already active)", Fore.MAGENTA)
            else:
                self.print_color("No market data available", Fore.YELLOW)
                
        except Exception as e:
            self.print_color(f"Trading cycle error: {e}", Fore.RED)

    def start_trading(self):
        self.print_color("STARTING FINAL AI ALWAYS SCAN BOT!", Fore.CYAN + Style.BRIGHT)
        self.cycle_count = 0
        while True:
            try:
                self.cycle_count += 1
                self.print_color(f"\nCYCLE {self.cycle_count}", Fore.CYAN)
                self.print_color("=" * 50, Fore.CYAN)
                self.run_trading_cycle()
                self.print_color(f"Waiting 30 seconds...", Fore.BLUE)
                time.sleep(30)
            except KeyboardInterrupt:
                self.print_color(f"\nBOT STOPPED", Fore.RED + Style.BRIGHT)
                self.show_trade_history(10)
                break
            except Exception as e:
                self.print_color(f"Main loop error: {e}", Fore.RED)
                time.sleep(30)

if __name__ == "__main__":
    try:
        bot = RealOrderPositionTracker()
        print("\n" + "="*50)
        print("FINAL AI ALWAYS SCAN BOT READY")
        print("1. Live Trading")
        choice = input("Enter choice (1): ").strip()
        if choice == "1":
            bot.start_trading()
        else:
            print("Backtest not included.")
    except Exception as e:
        print(f"Failed to start bot: {e}")
