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

try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

load_dotenv()

class Final3MinScalpingBot:
    def __init__(self):
        self.binance_api_key = os.getenv('BINANCE_API_KEY')
        self.binance_secret = os.getenv('BINANCE_SECRET_KEY')
        self.deepseek_key = os.getenv('DEEPSEEK_API_KEY')
        
        self.thailand_tz = pytz.timezone('Asia/Bangkok')
        
        self.trade_size_usd = 50
        self.leverage = 5
        self.tp_percent = 0.006
        self.sl_percent = 0.003
        
        self.max_concurrent_trades = 2
        self.available_pairs = ["SOLUSDT", "AVAXUSDT", "XRPUSDT", "LINKUSDT", "DOTUSDT"]
        
        self.paper_trading = False
        self.paper_balance = 1000.0
        self.paper_positions = {}
        
        self.trade_history_file = "3min_scalping_history.json"
        self.trade_history = self.load_trade_history()
        
        self.quantity_precision = {}
        self.price_precision = {}
        
        self.binance = Client(self.binance_api_key, self.binance_secret)
        
        self.print_color(f"FINAL 3MIN SCALPING BOT ACTIVATED!", Fore.CYAN + Style.BRIGHT)
        self.validate_config()
        self.setup_futures()
        self.load_symbol_precision()
    
    # ... (load_trade_history, save, add, show, get_thailand_time, print_color, validate, setup, precision, format_price, get_quantity - same)

    def get_price_history(self, pair, limit=20):
        try:
            klines = self.binance.futures_klines(symbol=pair, interval=Client.KLINE_INTERVAL_3MINUTE, limit=limit)
            prices = [float(k[4]) for k in klines]
            return {'prices': prices, 'current_price': prices[-1] if prices else 0}
        except:
            return {'prices': [], 'current_price': 0}

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
                    {"role": "system", "content": "Respond with valid JSON only."},
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
        except Exception as e:
            return "HOLD", 0, f"Error: {e}"

    def get_ai_decision(self, pair, price):
        try:
            market_data = self.get_price_history(pair)
            market_data['current_price'] = price
            direction, confidence, reason = self.get_deepseek_analysis(pair, market_data)
            if direction in ['LONG', 'SHORT'] and confidence >= 70:
                return {"action": "TRADE", "pair": pair, "direction": direction, "confidence": confidence, "reason": reason}
            else:
                return {"action": "HOLD", "pair": pair, "direction": direction, "confidence": confidence, "reason": reason}
        except Exception as e:
            return {"action": "HOLD", "pair": pair, "direction": "HOLD", "confidence": 0, "reason": str(e)}

    def execute_paper_trade(self, decision):
        # ... (same as before)
        pass  # Use previous paper execute code

    def monitor_paper_positions(self):
        # ... (same)
        pass

    def display_paper_dashboard(self):
        # ... (same)
        pass

    def get_current_price(self, pair):
        try:
            ticker = self.binance.futures_symbol_ticker(symbol=pair)
            return float(ticker['price'])
        except:
            return None

    def can_open_new_trade(self, pair):
        if self.paper_trading:
            return pair not in self.paper_positions or self.paper_positions[pair]['status'] != 'ACTIVE'
        return True

    def run_trading_cycle(self):
        try:
            if self.paper_trading:
                self.monitor_paper_positions()
                self.display_paper_dashboard()
            
            active_count = sum(1 for p in self.paper_positions.values() if p['status'] == 'ACTIVE') if self.paper_trading else 0
            if active_count >= self.max_concurrent_trades:
                self.print_color(f"Max concurrent trades reached ({self.max_concurrent_trades})", Fore.YELLOW)
                return
            
            self.print_color(f"\nAI SCANNING {len(self.available_pairs)} PAIRS...", Fore.BLUE + Style.BRIGHT)
            
            for pair in self.available_pairs:
                if active_count >= self.max_concurrent_trades:
                    break
                if not self.can_open_new_trade(pair):
                    continue
                price = self.get_current_price(pair)
                if not price:
                    continue
                decision = self.get_ai_decision(pair, price)
                if decision["action"] == "TRADE":
                    self.print_color(f"AI SIGNAL: {pair} {decision['direction']} ({decision['confidence']}%) → {decision['reason']}", Fore.GREEN + Style.BRIGHT)
                    if self.execute_paper_trade(decision):
                        active_count += 1
                else:
                    self.print_color(f"AI: {pair} HOLD ({decision['confidence']}%)", Fore.MAGENTA)
                    
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
    bot = Final3MinScalpingBot()
    print("\n" + "="*60)
    print("FINAL 3MIN SCALPING + BACKTEST + PAPER TRADING")
    print("1. Live Trading")
    print("2. Paper Trading (AI-Powered)")
    print("3. Backtest (7 days)")
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "2":
        bot.paper_trading = True
        bot.start_trading()
    elif choice == "3":
        bot.run_backtest(days=7)
    elif choice == "1":
        bot.start_trading()
