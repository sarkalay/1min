import os
import requests
import json
import time
import re
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
from datetime import datetime
import pytz

# Colorama
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORAMA = True
except ImportError:
    COLORAMA = False
    print("pip install colorama")

load_dotenv()

if not COLORAMA:
    class Dummy:
        def __getattr__(self, name): return ''
    Fore = Style = Dummy()

class PureAIBot:
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret = os.getenv('BINANCE_SECRET_KEY')
        self.deepseek_key = os.getenv('DEEPSEEK_API_KEY')
        
        self.th_tz = pytz.timezone('Asia/Bangkok')
        self.pairs = ["SOLUSDT", "AVAXUSDT", "XRPUSDT", "LINKUSDT", "DOTUSDT"]
        
        # FIXED SIZE
        self.fixed_size_usd = 50
        
        # PAPER STATE
        self.paper_balance = 1000.0
        self.paper_positions = {}
        self.paper_history = []
        
        # LIVE STATE
        self.active_trades = {}
        self.trade_history = []
        
        self.printc("PURE AI SCALPING BOT", Fore.CYAN + Style.BRIGHT)
        self.printc(f"FIXED SIZE: ${self.fixed_size_usd}", Fore.YELLOW + Style.BRIGHT)
        self.printc("AI DECIDES: Direction, TP, SL, Entry, Confidence", Fore.MAGENTA)

    def printc(self, text, color=Fore.WHITE):
        print(f"{color}{text}{Style.RESET_ALL if COLORAMA else ''}")

    def get_th_time(self):
        return datetime.now(self.th_tz).strftime('%Y-%m-%d %H:%M:%S')

    def get_client(self):
        return Client(self.api_key, self.secret) if self.api_key and self.secret else None

    def get_klines(self, symbol, limit=50):
        try:
            client = self.get_client()
            if client:
                klines = client.futures_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=limit)
                return [float(k[4]) for k in klines]
            else:
                return [100 + i*0.1 for i in range(limit)]
        except:
            return [100] * limit

    def get_price(self, symbol):
        try:
            client = self.get_client()
            if client:
                return float(client.futures_symbol_ticker(symbol=symbol)['price'])
            else:
                import requests
                resp = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}", timeout=5)
                return float(resp.json()['price'])
        except:
            return 100.0

    def ask_ai_full_analysis(self, symbol):
        prices = self.get_klines(symbol, 50)
        current = prices[-1]
        
        prompt = f"""
        Analyze {symbol} on 1-minute chart.
        Current price: ${current:.6f}
        Last 10 closes: {prices[-10:]} (latest on right)

        Decide: LONG, SHORT, or HOLD.
        If trade: give entry, TP, SL, confidence, reason.
        Trade size is FIXED $50 — DO NOT include it.

        Return VALID JSON only:
        {{
            "action": "LONG" | "SHORT" | "HOLD",
            "confidence": 0-100,
            "entry_price": float,
            "take_profit": float,
            "stop_loss": float,
            "reason": "short reason"
        }}
        """

        try:
            headers = {"Authorization": f"Bearer {self.deepseek_key}", "Content-Type": "application/json"}
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are a 1-min scalper. Return perfect JSON only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 400
            }
            self.printc(f"AI ANALYZING {symbol}...", Fore.MAGENTA)
            resp = requests.post("https://api.deepseek.com/chat/completions", headers=headers, json=data, timeout=40)
            
            if resp.status_code != 200:
                return None
                
            text = resp.json()['choices'][0]['message']['content'].strip()
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if not json_match:
                return None
                
            decision = json.loads(json_match.group())
            required = ['action', 'confidence', 'take_profit', 'stop_loss', 'reason']
            if not all(k in decision for k in required):
                return None
            if decision['action'] not in ['LONG', 'SHORT', 'HOLD']:
                return None
            return decision
            
        except Exception as e:
            self.printc(f"AI Error: {e}", Fore.RED)
            return None

    def can_open_trade(self, is_paper):
        if is_paper:
            return len(self.paper_positions) < 3
        else:
            return len(self.active_trades) < 3

    def execute_trade(self, decision, symbol, is_paper):
        price = self.get_price(symbol)
        quantity = round(self.fixed_size_usd / price, 6)
        if quantity <= 0:
            return False

        trade = {
            'symbol': symbol,
            'direction': decision['action'],
            'entry': price,
            'quantity': quantity,
            'tp': decision['take_profit'],
            'sl': decision['stop_loss'],
            'confidence': decision['confidence'],
            'reason': decision['reason'],
            'time': self.get_th_time()
        }

        if is_paper:
            # PAPER
            self.paper_positions[symbol] = trade
            self.paper_balance -= self.fixed_size_usd
            self.printc(f"PAPER TRADE: {symbol} {trade['direction']}", Fore.GREEN)
            self.printc(f"   ${self.fixed_size_usd} → {quantity} | TP: ${trade['tp']:.4f} | SL: ${trade['sl']:.4f}", Fore.CYAN)
            return True
        else:
            # LIVE
            client = self.get_client()
            if not client:
                self.printc("NO BINANCE KEYS → CANNOT TRADE LIVE", Fore.RED)
                return False
            try:
                side = 'BUY' if decision['action'] == 'LONG' else 'SELL'
                client.futures_create_order(symbol=symbol, side=side, type='MARKET', quantity=quantity)
                self.printc("LIVE ORDER EXECUTED", Fore.GREEN + Style.BRIGHT)

                time.sleep(2)
                close_side = 'SELL' if decision['action'] == 'LONG' else 'BUY'
                client.futures_create_order(symbol=symbol, side=close_side, type='STOP_MARKET', quantity=quantity, stopPrice=trade['sl'], reduceOnly=True)
                client.futures_create_order(symbol=symbol, side=close_side, type='TAKE_PROFIT_MARKET', quantity=quantity, stopPrice=trade['tp'], reduceOnly=True)
                self.printc("TP & SL PLACED", Fore.GREEN)

                self.active_trades[symbol] = trade
                return True
            except Exception as e:
                self.printc(f"LIVE FAILED: {e}", Fore.RED)
                return False

    def monitor_paper(self):
        for symbol, trade in list(self.paper_positions.items()):
            current = self.get_price(symbol)
            if (trade['direction'] == 'LONG' and current >= trade['tp']) or (trade['direction'] == 'SHORT' and current <= trade['tp']):
                self.close_paper_trade(symbol, trade, current, "TP")
            elif (trade['direction'] == 'LONG' and current <= trade['sl']) or (trade['direction'] == 'SHORT' and current >= trade['sl']):
                self.close_paper_trade(symbol, trade, current, "SL")

    def close_paper_trade(self, symbol, trade, exit_price, reason):
        pnl = (exit_price - trade['entry']) * trade['quantity'] if trade['direction'] == 'LONG' else (trade['entry'] - exit_price) * trade['quantity']
        self.paper_balance += pnl
        self.paper_history.append({**trade, 'exit_price': exit_price, 'pnl': pnl, 'close_reason': reason})
        del self.paper_positions[symbol]
        color = Fore.GREEN if pnl > 0 else Fore.RED
        self.printc(f"{reason}: {symbol} | P&L: ${pnl:.2f} | Balance: ${self.paper_balance:.2f}", color)

    def monitor_live(self):
        client = self.get_client()
        if not client:
            return
        for symbol, trade in list(self.active_trades.items()):
            try:
                pos = client.futures_position_information(symbol=symbol)
                for p in pos:
                    if p['symbol'] == symbol and float(p['positionAmt']) != 0:
                        current = float(p['markPrice'])
                        if (trade['direction'] == 'LONG' and current >= trade['tp']) or (trade['direction'] == 'SHORT' and current <= trade['tp']):
                            self.close_live_trade(symbol, trade, current, "TP")
                        elif (trade['direction'] == 'LONG' and current <= trade['sl']) or (trade['direction'] == 'SHORT' and current >= trade['sl']):
                            self.close_live_trade(symbol, trade, current, "SL")
                        break
            except:
                pass

    def close_live_trade(self, symbol, trade, exit_price, reason):
        client = self.get_client()
        if not client:
            return
        try:
            pos = client.futures_position_information(symbol=symbol)
            for p in pos:
                if p['symbol'] == symbol and float(p['positionAmt']) != 0:
                    pnl = float(p['unRealizedProfit'])
                    client.futures_create_order(symbol=symbol, side='SELL' if trade['direction']=='LONG' else 'BUY', type='MARKET', quantity=abs(float(p['positionAmt'])), reduceOnly=True)
                    self.trade_history.append({**trade, 'exit_price': exit_price, 'pnl': pnl, 'close_reason': reason})
                    del self.active_trades[symbol]
                    color = Fore.GREEN if pnl > 0 else Fore.RED
                    self.printc(f"{reason}: {symbol} | P&L: ${pnl:.2f}", color)
                    break
        except Exception as e:
            self.printc(f"Close failed: {e}", Fore.RED)

    def show_status(self, is_paper):
        if is_paper:
            self.printc(f"\nPAPER | Balance: ${self.paper_balance:.2f} | Active: {len(self.paper_positions)}", Fore.CYAN + Style.BRIGHT)
            if self.paper_history:
                win_rate = len([t for t in self.paper_history if t.get('pnl',0)>0]) / len(self.paper_history) * 100
                self.printc(f"Win Rate: {win_rate:.1f}% | Trades: {len(self.paper_history)}", Fore.YELLOW)
        else:
            self.printc(f"\nLIVE | Active: {len(self.active_trades)}", Fore.CYAN + Style.BRIGHT)

    def run_paper(self):
        self.printc("STARTING PAPER TRADING MODE", Fore.GREEN + Style.BRIGHT)
        self.printc("NO RISK - TESTING AI STRATEGY", Fore.YELLOW)
        cycle = 0
        while True:
            try:
                cycle += 1
                self.printc(f"\nPAPER CYCLE {cycle}", Fore.CYAN)
                self.monitor_paper()
                self.show_status(is_paper=True)

                for symbol in self.pairs:
                    if symbol in self.paper_positions:
                        continue
                    if not self.can_open_trade(is_paper=True):
                        continue
                    decision = self.ask_ai_full_analysis(symbol)
                    if decision and decision['action'] in ['LONG', 'SHORT']:
                        self.execute_trade(decision, symbol, is_paper=True)

                time.sleep(25)
            except KeyboardInterrupt:
                self.printc("\nPAPER BOT STOPPED", Fore.RED)
                break

    def run_live(self):
        if not self.get_client():
            self.printc("NO BINANCE KEYS → CANNOT RUN LIVE", Fore.RED)
            return
        confirm = input("TYPE 'YES' TO START REAL TRADING: ").strip().upper()
        if confirm != 'YES':
            self.printc("LIVE TRADING CANCELLED", Fore.YELLOW)
            return

        self.printc("STARTING LIVE TRADING", Fore.RED + Style.BRIGHT)
        self.printc("REAL MONEY AT RISK!", Fore.RED + Style.BRIGHT)
        cycle = 0
        while True:
            try:
                cycle += 1
                self.printc(f"\nLIVE CYCLE {cycle}", Fore.CYAN)
                self.monitor_live()
                self.show_status(is_paper=False)

                for symbol in self.pairs:
                    if symbol in self.active_trades:
                        continue
                    if not self.can_open_trade(is_paper=False):
                        continue
                    decision = self.ask_ai_full_analysis(symbol)
                    if decision and decision['action'] in ['LONG', 'SHORT']:
                        self.execute_trade(decision, symbol, is_paper=False)

                time.sleep(25)
            except KeyboardInterrupt:
                self.printc("\nLIVE BOT STOPPED", Fore.RED)
                break


if __name__ == "__main__":
    bot = PureAIBot()
    
    print("\n" + "="*60)
    print("PURE AI SCALPING BOT")
    print("="*60)
    print("1. REAL TRADING (Real Money)")
    print("2. PAPER TRADING (No Risk)")
    print("="*60)
    
    choice = input("Choose (1 or 2): ").strip()
    
    if choice == "1":
        bot.run_live()
    elif choice == "2":
        bot.run_paper()
    else:
        print("Invalid choice. Run again.")
