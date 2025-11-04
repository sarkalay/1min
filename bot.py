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
        self.client = Client(self.api_key, self.secret) if self.api_key and self.secret else None
        
        self.th_tz = pytz.timezone('Asia/Bangkok')
        self.pairs = ["SOLUSDT", "AVAXUSDT", "XRPUSDT", "LINKUSDT", "DOTUSDT"]
        
        # PAPER TRADING STATE
        self.paper_mode = not bool(self.client)  # Auto-detect: no key = paper
        self.paper_balance = 1000.0
        self.paper_positions = {}
        self.paper_history = []
        
        # FIXED SIZE
        self.fixed_size_usd = 50
        
        # REAL TRADES
        self.active_trades = {}
        self.trade_history = []
        
        self.printc("PURE AI BOT INITIALIZED", Fore.CYAN + Style.BRIGHT)
        if self.paper_mode:
            self.printc("PAPER TRADING MODE (NO RISK)", Fore.GREEN + Style.BRIGHT)
            self.printc(f"STARTING BALANCE: ${self.paper_balance}", Fore.YELLOW)
        else:
            self.printc("LIVE TRADING MODE (REAL MONEY)", Fore.RED + Style.BRIGHT)
        self.printc(f"FIXED TRADE SIZE: ${self.fixed_size_usd}", Fore.MAGENTA + Style.BRIGHT)
        self.printc("AI DECIDES: Direction, TP, SL, Entry, Confidence", Fore.CYAN)

    def printc(self, text, color=Fore.WHITE):
        print(f"{color}{text}{Style.RESET_ALL if COLORAMA else ''}")

    def get_th_time(self):
        return datetime.now(self.th_tz).strftime('%Y-%m-%d %H:%M:%S')

    def get_klines(self, symbol, limit=50):
        try:
            if self.client:
                klines = self.client.futures_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=limit)
                return [float(k[4]) for k in klines]
            else:
                return [100 + i*0.1 for i in range(limit)]
        except:
            return [100] * limit

    def get_price(self, symbol):
        try:
            if self.client:
                return float(self.client.futures_symbol_ticker(symbol=symbol)['price'])
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
        Analyze {symbol} on 1-minute chart for scalping.
        Current price: ${current:.6f}
        Last 10 closes: {prices[-10:]} (latest on right)

        You decide everything:
        - LONG, SHORT, or HOLD
        - Entry price
        - Take Profit
        - Stop Loss
        - Confidence (0-100)
        - 1-sentence reason

        Trade size is FIXED at $50 — DO NOT include it.

        Return VALID JSON only:
        {{
            "action": "LONG" | "SHORT" | "HOLD",
            "confidence": 0-100,
            "entry_price": float,
            "take_profit": float,
            "stop_loss": float,
            "reason": "short clear reason"
        }}
        """

        try:
            headers = {"Authorization": f"Bearer {self.deepseek_key}", "Content-Type": "application/json"}
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are a 1-minute scalping AI. Be decisive. Return perfect JSON only."},
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

    def can_open_trade(self, symbol):
        if self.paper_mode:
            return symbol not in self.paper_positions and len(self.paper_positions) < 3
        else:
            return symbol not in self.active_trades and len(self.active_trades) < 3

    def execute_trade(self, decision, symbol):
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

        if self.paper_mode:
            # PAPER EXECUTION
            self.paper_positions[symbol] = trade
            self.paper_balance -= self.fixed_size_usd  # simulate entry cost
            self.printc(f"PAPER TRADE OPENED: {symbol} {trade['direction']}", Fore.GREEN)
            self.printc(f"   Size: ${self.fixed_size_usd} | Qty: {quantity}", Fore.YELLOW)
            self.printc(f"   Entry: ${price:.4f} | TP: ${trade['tp']:.4f} | SL: ${trade['sl']:.4f}", Fore.CYAN)
            self.printc(f"   Conf: {trade['confidence']}% | {trade['reason']}", Fore.MAGENTA)
            return True
        else:
            # LIVE EXECUTION
            try:
                side = 'BUY' if decision['action'] == 'LONG' else 'SELL'
                order = self.client.futures_create_order(
                    symbol=symbol, side=side, type='MARKET', quantity=quantity
                )
                self.printc("LIVE MARKET ORDER EXECUTED", Fore.GREEN + Style.BRIGHT)

                time.sleep(2)
                close_side = 'SELL' if decision['action'] == 'LONG' else 'BUY'
                self.client.futures_create_order(symbol=symbol, side=close_side, type='STOP_MARKET', quantity=quantity, stopPrice=trade['sl'], reduceOnly=True)
                self.client.futures_create_order(symbol=symbol, side=close_side, type='TAKE_PROFIT_MARKET', quantity=quantity, stopPrice=trade['tp'], reduceOnly=True)
                self.printc("TP & SL PLACED", Fore.GREEN)

                self.active_trades[symbol] = trade
                return True
            except Exception as e:
                self.printc(f"LIVE EXECUTION FAILED: {e}", Fore.RED)
                return False

    def monitor_paper(self):
        for symbol, trade in list(self.paper_positions.items()):
            current = self.get_price(symbol)
            if (trade['direction'] == 'LONG' and current >= trade['tp']) or (trade['direction'] == 'SHORT' and current <= trade['tp']):
                self.close_paper_trade(symbol, trade, current, "TP HIT")
            elif (trade['direction'] == 'LONG' and current <= trade['sl']) or (trade['direction'] == 'SHORT' and current >= trade['sl']):
                self.close_paper_trade(symbol, trade, current, "SL HIT")

    def close_paper_trade(self, symbol, trade, exit_price, reason):
        pnl = (exit_price - trade['entry']) * trade['quantity'] if trade['direction'] == 'LONG' else (trade['entry'] - exit_price) * trade['quantity']
        self.paper_balance += pnl
        self.paper_history.append({**trade, 'exit_price': exit_price, 'pnl': pnl, 'close_reason': reason, 'close_time': self.get_th_time()})
        del self.paper_positions[symbol]
        
        color = Fore.GREEN if pnl > 0 else Fore.RED
        self.printc(f"{reason}: {symbol} | P&L: ${pnl:.2f} | Balance: ${self.paper_balance:.2f}", color)

    def monitor_live(self):
        for symbol, trade in list(self.active_trades.items()):
            try:
                pos = self.client.futures_position_information(symbol=symbol)
                for p in pos:
                    if p['symbol'] == symbol and float(p['positionAmt']) != 0:
                        current = float(p['markPrice'])
                        unrealized = float(p['unRealizedProfit'])
                        if (trade['direction'] == 'LONG' and current >= trade['tp']) or (trade['direction'] == 'SHORT' and current <= trade['tp']):
                            self.close_live_trade(symbol, trade, current, "TP HIT")
                        elif (trade['direction'] == 'LONG' and current <= trade['sl']) or (trade['direction'] == 'SHORT' and current >= trade['sl']):
                            self.close_live_trade(symbol, trade, current, "SL HIT")
                        break
            except:
                pass

    def close_live_trade(self, symbol, trade, exit_price, reason):
        try:
            pos = self.client.futures_position_information(symbol=symbol)
            for p in pos:
                if p['symbol'] == symbol and float(p['positionAmt']) != 0:
                    pnl = float(p['unRealizedProfit'])
                    self.trade_history.append({**trade, 'exit_price': exit_price, 'pnl': pnl, 'close_reason': reason})
                    self.client.futures_create_order(symbol=symbol, side='SELL' if trade['direction']=='LONG' else 'BUY', type='MARKET', quantity=abs(float(p['positionAmt'])), reduceOnly=True)
                    del self.active_trades[symbol]
                    color = Fore.GREEN if pnl > 0 else Fore.RED
                    self.printc(f"{reason}: {symbol} | P&L: ${pnl:.2f}", color)
                    break
        except Exception as e:
            self.printc(f"Close failed: {e}", Fore.RED)

    def show_portfolio(self):
        if self.paper_mode:
            self.printc(f"\nPAPER PORTFOLIO | Balance: ${self.paper_balance:.2f} | Trades: {len(self.paper_history)}", Fore.CYAN + Style.BRIGHT)
            win_rate = (len([t for t in self.paper_history if t.get('pnl',0)>0]) / len(self.paper_history) * 100) if self.paper_history else 0
            self.printc(f"Win Rate: {win_rate:.1f}% | Active: {len(self.paper_positions)}", Fore.YELLOW)
        else:
            self.printc(f"\nLIVE DASHBOARD | Active: {len(self.active_trades)}", Fore.CYAN + Style.BRIGHT)

    def run(self):
        self.printc("STARTING AI SCALPING BOT...", Fore.CYAN + Style.BRIGHT)
        cycle = 0
        while True:
            try:
                cycle += 1
                self.printc(f"\nCYCLE {cycle} | {self.get_th_time()}", Fore.CYAN)

                if self.paper_mode:
                    self.monitor_paper()
                else:
                    self.monitor_live()

                self.show_portfolio()

                for symbol in self.pairs:
                    if not self.can_open_trade(symbol):
                        continue
                    decision = self.ask_ai_full_analysis(symbol)
                    if decision and decision['action'] in ['LONG', 'SHORT']:
                        self.execute_trade(decision, symbol)

                time.sleep(25)
            except KeyboardInterrupt:
                self.printc("\nBOT STOPPED", Fore.RED + Style.BRIGHT)
                break
            except Exception as e:
                self.printc(f"ERROR: {e}", Fore.RED)
                time.sleep(10)


if __name__ == "__main__":
    bot = PureAIBot()
    bot.run()
