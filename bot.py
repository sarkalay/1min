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

# Install colorama first: pip install colorama
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    print("⚠️  Colorama not installed. Run: pip install colorama")

# Load environment variables
load_dotenv()

class RealOrderPositionTracker:
    def __init__(self):
        # Load config from .env file
        self.binance_api_key = os.getenv('BINANCE_API_KEY')
        self.binance_secret = os.getenv('BINANCE_SECRET_KEY')
        self.deepseek_key = os.getenv('DEEPSEEK_API_KEY')
        
        # SCALPING parameters
        self.trade_size_usd = 50
        self.leverage = 5
        
        # Multi-pair parameters
        self.max_concurrent_trades = 1
        self.available_pairs = ["SOLUSDT", "AVAXUSDT", "XRPUSDT", "LINKUSDT", "DOTUSDT"]
        
        # Track both bot-opened and existing positions
        self.bot_opened_trades = {}
        self.existing_positions = {}
        
        # Precision settings
        self.quantity_precision = {}
        self.price_precision = {}
        
        # Initialize Binance client
        self.binance = Client(self.binance_api_key, self.binance_secret)
        
        self.print_color(f"🤖 REAL ORDER POSITION TRACKER WITH OCO ACTIVATED!", Fore.CYAN)
        self.print_color(f"💵 Trade Size: ${self.trade_size_usd}", Fore.GREEN)
        self.print_color(f"📈 Max Trades: {self.max_concurrent_trades}", Fore.YELLOW)
        self.print_color(f"🎯 Using OCO Orders for TP/SL", Fore.CYAN)
        self.print_color(f"🧠 Using DeepSeek AI ONLY for Trading Decisions", Fore.MAGENTA)
        
        self.validate_config()
        self.setup_futures()
        self.load_symbol_precision()
    
    def print_color(self, text, color=Fore.WHITE, style=Style.NORMAL):
        """Colorful print function"""
        if COLORAMA_AVAILABLE:
            print(f"{style}{color}{text}")
        else:
            print(text)
    
    def validate_config(self):
        if not all([self.binance_api_key, self.binance_secret, self.deepseek_key]):
            self.print_color("❌ Missing API keys!", Fore.RED)
            return False
        try:
            self.binance.futures_exchange_info()
            self.print_color("✅ Binance connection successful!", Fore.GREEN)
        except Exception as e:
            self.print_color(f"❌ Binance connection failed: {e}", Fore.RED)
            return False
        return True

    def setup_futures(self):
        try:
            for pair in self.available_pairs:
                try:
                    self.binance.futures_change_leverage(symbol=pair, leverage=self.leverage)
                    self.print_color(f"✅ Leverage set for {pair}", Fore.GREEN)
                except Exception as e:
                    self.print_color(f"⚠️ Leverage setup failed for {pair}: {e}", Fore.YELLOW)
            self.print_color("✅ Futures setup completed!", Fore.GREEN)
        except Exception as e:
            self.print_color(f"❌ Futures setup failed: {e}", Fore.RED)
    
    def load_symbol_precision(self):
        try:
            exchange_info = self.binance.futures_exchange_info()
            for symbol in exchange_info['symbols']:
                pair = symbol['symbol']
                for f in symbol['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        step_size = f['stepSize']
                        if '1' in step_size:
                            qty_precision = 0
                        else:
                            qty_precision = len(step_size.split('.')[1].rstrip('0'))
                        self.quantity_precision[pair] = qty_precision
                    elif f['filterType'] == 'PRICE_FILTER':
                        tick_size = f['tickSize']
                        if '1' in tick_size:
                            price_precision = 0
                        else:
                            price_precision = len(tick_size.split('.')[1].rstrip('0'))
                        self.price_precision[pair] = price_precision
            self.print_color("✅ Symbol precision loaded", Fore.GREEN)
        except Exception as e:
            self.print_color(f"❌ Error loading symbol precision: {e}", Fore.RED)
    
    def format_price(self, pair, price):
        precision = self.price_precision.get(pair, 4)
        return round(price, precision)
    
    def get_quantity(self, pair, price):
        """✅ CORRECTED: Calculate proper quantity for $50 position"""
        try:
            # Binance FUTURES minimum quantities
            min_quantities = {
                "SOLUSDT": 0.01,    # 0.01 SOL minimum
                "AVAXUSDT": 0.1,    # 0.1 AVAX minimum  
                "XRPUSDT": 1.0,     # 1 XRP minimum
                "LINKUSDT": 0.1,    # 0.1 LINK minimum
                "DOTUSDT": 0.1      # 0.1 DOT minimum
            }
            
            min_qty = min_quantities.get(pair, 0.1)
            precision = self.quantity_precision.get(pair, 3)
            
            # Calculate required quantity for $50
            required_quantity = self.trade_size_usd / price
            
            # Use the larger of required quantity or minimum
            quantity = max(required_quantity, min_qty)
            quantity = round(quantity, precision)
            
            # For low-priced tokens, ensure we get close to $50
            actual_value = quantity * price
            if actual_value < self.trade_size_usd * 0.8:  # If less than $40
                # Add minimum units until we're close to $50
                while actual_value < self.trade_size_usd * 0.9 and quantity < min_qty * 100:  # Safety limit
                    quantity += min_qty
                    quantity = round(quantity, precision)
                    actual_value = quantity * price
            
            actual_value = quantity * price
            
            # Display calculation details
            status_color = Fore.GREEN if abs(actual_value - self.trade_size_usd) < 10 else Fore.YELLOW
            status_icon = "✅" if abs(actual_value - self.trade_size_usd) < 10 else "⚠️"
            
            self.print_color(f"{status_icon} QUANTITY CALCULATION for {pair}:", Fore.CYAN)
            self.print_color(f"   🎯 Target: ${self.trade_size_usd}", Fore.WHITE)
            self.print_color(f"   📈 Price: ${price:.4f}", Fore.WHITE)
            self.print_color(f"   📦 Quantity: {quantity} {pair.replace('USDT', '')}", Fore.GREEN)
            self.print_color(f"   💰 Actual Position: ${actual_value:.2f}", status_color)
            self.print_color(f"   📏 Minimum: {min_qty}", Fore.WHITE)
            
            return quantity
            
        except Exception as e:
            self.print_color(f"❌ Quantity calculation failed: {e}", Fore.RED)
            return None

    def parse_ai_response(self, text):
        """Parse AI response and extract trading decision"""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                decision_data = json.loads(json_str)
                
                direction = decision_data.get('direction', 'HOLD').upper()
                confidence = decision_data.get('confidence', 50)
                reason = decision_data.get('reason', 'AI Analysis')
                
                # Validate direction
                if direction not in ['LONG', 'SHORT', 'HOLD']:
                    direction = 'HOLD'
                
                # Validate confidence
                try:
                    confidence = float(confidence)
                    if confidence < 0 or confidence > 100:
                        confidence = 50
                except:
                    confidence = 50
                
                return direction, confidence, reason
            
            # If no JSON found, try to extract information from text
            direction = 'HOLD'
            confidence = 50
            reason = 'AI Analysis'
            
            # Look for direction keywords
            if 'LONG' in text.upper():
                direction = 'LONG'
            elif 'SHORT' in text.upper():
                direction = 'SHORT'
            
            # Look for confidence percentage
            confidence_match = re.search(r'(\d+)%', text)
            if confidence_match:
                confidence = int(confidence_match.group(1))
            else:
                # Look for confidence number
                confidence_match = re.search(r'confidence.*?(\d+)', text, re.IGNORECASE)
                if confidence_match:
                    confidence = int(confidence_match.group(1))
            
            return direction, confidence, reason
            
        except Exception as e:
            self.print_color(f"❌ AI response parsing failed: {e}", Fore.RED)
            return 'HOLD', 50, 'Parsing failed'

    def get_deepseek_analysis(self, pair, market_data):
        """Get AI analysis from DeepSeek API - AI ONLY"""
        try:
            if not self.deepseek_key:
                self.print_color("❌ DeepSeek API key not found - CANNOT TRADE", Fore.RED)
                return "HOLD", 0, "No API key"
            
            current_price = market_data['current_price']
            price_history = market_data.get('prices', [])
            
            # Create a more structured prompt for better responses
            prompt = f"""
            CRYPTOCURRENCY TRADING ANALYSIS REQUEST

            TRADING PAIR: {pair}
            CURRENT PRICE: ${current_price:.4f}
            
            RECENT PRICE MOVEMENT:
            {[f'${p:.4f}' for p in price_history[-5:]] if price_history else 'No history available'}
            
            TRADING PARAMETERS:
            - Position Size: ${self.trade_size_usd}
            - Leverage: {self.leverage}x
            - Take Profit: 0.8% 
            - Stop Loss: 0.5%
            - Trading Style: Scalping (short-term)
            
            REQUIRED RESPONSE FORMAT (STRICT JSON):
            {{
                "direction": "LONG|SHORT|HOLD",
                "confidence": 65,
                "reason": "Brief technical explanation"
            }}
            
            ANALYSIS CRITERIA:
            - LONG: Expecting price increase in next 1-2 hours
            - SHORT: Expecting price decrease in next 1-2 hours  
            - HOLD: No clear direction or low confidence
            - Confidence: 0-100% based on signal strength
            
            Provide ONLY the JSON response, no other text.
            """
            
            headers = {
                "Authorization": f"Bearer {self.deepseek_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional crypto trading analyst. Analyze the market data and provide trading decisions in EXACT JSON format. Respond ONLY with valid JSON, no other text."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.1,  # Lower temperature for more consistent responses
                "max_tokens": 300
            }
            
            self.print_color(f"🧠 Consulting DeepSeek AI for {pair}...", Fore.MAGENTA)
            
            response = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers=headers,
                json=data,
                timeout=45  # Increased timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content'].strip()
                
                self.print_color(f"📄 Raw AI Response: {ai_response}", Fore.BLUE)
                
                # Parse the AI response
                direction, confidence, reason = self.parse_ai_response(ai_response)
                
                self.print_color(f"✅ DeepSeek Analysis Result:", Fore.MAGENTA)
                self.print_color(f"   📊 Direction: {direction}", Fore.CYAN)
                self.print_color(f"   🎯 Confidence: {confidence}%", Fore.GREEN)
                self.print_color(f"   💡 Reason: {reason}", Fore.WHITE)
                
                return direction, confidence, reason
                
            else:
                self.print_color(f"❌ DeepSeek API error: {response.status_code} - {response.text}", Fore.RED)
                return "HOLD", 0, f"API Error: {response.status_code}"
                
        except requests.exceptions.Timeout:
            self.print_color("❌ DeepSeek API timeout", Fore.RED)
            return "HOLD", 0, "API Timeout"
        except Exception as e:
            self.print_color(f"❌ DeepSeek analysis failed: {e}", Fore.RED)
            return "HOLD", 0, f"Error: {str(e)}"

    def get_price_history(self, pair, limit=10):
        """Get recent price history for analysis"""
        try:
            klines = self.binance.futures_klines(
                symbol=pair,
                interval=Client.KLINE_INTERVAL_1MINUTE,
                limit=limit
            )
            
            prices = [float(k[4]) for k in klines]  # Closing prices
            return {
                'prices': prices,
                'current_price': prices[-1] if prices else 0
            }
        except Exception as e:
            self.print_color(f"❌ Error getting price history for {pair}: {e}", Fore.RED)
            # Return current price only as fallback
            try:
                ticker = self.binance.futures_symbol_ticker(symbol=pair)
                current_price = float(ticker['price'])
                return {
                    'prices': [current_price] * 5,  # Fake some history
                    'current_price': current_price
                }
            except:
                return {'prices': [], 'current_price': 0}

    def get_ai_decision(self, pair_data):
        """Get AI-powered trading decision - AI ONLY"""
        try:
            pair = list(pair_data.keys())[0]
            current_price = pair_data[pair]['price']
            
            self.print_color(f"🔍 Analyzing {pair} with DeepSeek AI...", Fore.BLUE)
            
            # Get price history for analysis
            market_data = self.get_price_history(pair)
            market_data['current_price'] = current_price
            
            # Get AI analysis - NO FALLBACK TO TECHNICAL ANALYSIS
            direction, confidence, reason = self.get_deepseek_analysis(pair, market_data)
            
            if direction == "HOLD" or confidence < 60:
                self.print_color(f"🎯 AI Decision: HOLD (Confidence: {confidence}%)", Fore.YELLOW)
                self.print_color(f"   💡 Reason: {reason}", Fore.WHITE)
                return {
                    "action": "HOLD", 
                    "pair": pair,
                    "direction": direction,
                    "confidence": confidence,
                    "reason": reason
                }
            else:
                self.print_color(f"🎯 AI Decision: {direction} (Confidence: {confidence}%)", Fore.GREEN)
                self.print_color(f"   💡 Reason: {reason}", Fore.WHITE)
                return {
                    "action": "TRADE",
                    "pair": pair,
                    "direction": direction,
                    "confidence": confidence,
                    "reason": reason
                }
                
        except Exception as e:
            self.print_color(f"❌ AI decision process failed: {e}", Fore.RED)
            # NO FALLBACK - AI ONLY
            return {
                "action": "HOLD",
                "pair": list(pair_data.keys())[0],
                "direction": "HOLD",
                "confidence": 0,
                "reason": f"AI System Error: {str(e)}"
            }

    def cleanup_old_orders(self, pair):
        """Clean up any existing orders for a pair"""
        try:
            self.binance.futures_cancel_all_open_orders(symbol=pair)
            self.print_color(f"🧹 Cleaned up old orders for {pair}", Fore.YELLOW)
            return True
        except Exception as e:
            self.print_color(f"⚠️ No orders to clean for {pair}: {e}", Fore.YELLOW)
            return True
    
    def execute_oco_trade(self, decision):
        """✅ OCO Order System - TP/SL တစ်ခုထိရင် အခြားတစ်ခု အလိုလို cancel"""
        try:
            pair = decision["pair"]
            
            if not self.can_open_new_trade(pair):
                self.print_color(f"🚫 Cannot open {pair} - position exists or limit reached", Fore.RED)
                return False
            
            direction = decision["direction"]
            confidence = decision["confidence"]
            reason = decision["reason"]
            
            # Clean up any existing orders first
            self.cleanup_old_orders(pair)
            
            # Get current price
            ticker = self.binance.futures_symbol_ticker(symbol=pair)
            current_price = float(ticker['price'])
            
            # Calculate quantity - ✅ CORRECTED VERSION
            quantity = self.get_quantity(pair, current_price)
            if quantity is None:
                return False
            
            # Calculate TP/SL with better risk management
            if direction == "LONG":
                stop_loss = current_price * 0.995   # 0.5% SL
                take_profit = current_price * 1.008 # 0.8% TP
                stop_side = 'SELL'
            else:  # SHORT
                stop_loss = current_price * 1.005   # 0.5% SL
                take_profit = current_price * 0.992 # 0.8% TP
                stop_side = 'BUY'
            
            stop_loss = self.format_price(pair, stop_loss)
            take_profit = self.format_price(pair, take_profit)
            
            direction_color = Fore.BLUE if direction == 'LONG' else Fore.RED
            self.print_color(f"🎯 EXECUTING OCO TRADE: {pair} {direction}", direction_color)
            self.print_color(f"   📦 Size: {quantity} | 🎯 TP: ${take_profit} | 🛑 SL: ${stop_loss}", Fore.WHITE)
            self.print_color(f"   🧠 AI Confidence: {confidence}%", Fore.MAGENTA)
            self.print_color(f"   💡 AI Reason: {reason}", Fore.WHITE)
            
            # ✅ Step 1: Open position with MARKET order
            try:
                if direction == "LONG":
                    order = self.binance.futures_create_order(
                        symbol=pair,
                        side='BUY',
                        type='MARKET',
                        quantity=quantity
                    )
                    self.print_color(f"✅ REAL LONG ORDER EXECUTED: {quantity} {pair} @ ${current_price}", Fore.GREEN)
                else:
                    order = self.binance.futures_create_order(
                        symbol=pair,
                        side='SELL',
                        type='MARKET',
                        quantity=quantity
                    )
                    self.print_color(f"✅ REAL SHORT ORDER EXECUTED: {quantity} {pair} @ ${current_price}", Fore.GREEN)
                
                # ✅ Step 2: Wait for position to open
                time.sleep(2)
                
                # ✅ Step 3: Place OCO ORDER using STOP_MARKET and TAKE_PROFIT_MARKET
                
                # Place STOP LOSS order
                sl_order = self.binance.futures_create_order(
                    symbol=pair,
                    side=stop_side,
                    type='STOP_MARKET',
                    quantity=quantity,
                    stopPrice=stop_loss,
                    reduceOnly=True,
                    timeInForce='GTC'
                )
                
                # Place TAKE PROFIT order  
                tp_order = self.binance.futures_create_order(
                    symbol=pair,
                    side=stop_side,
                    type='TAKE_PROFIT_MARKET',
                    quantity=quantity,
                    stopPrice=take_profit,
                    reduceOnly=True,
                    timeInForce='GTC'
                )
                
                self.print_color(f"✅ OCO ORDERS PLACED - TP/SL activated!", Fore.GREEN)
                self.print_color(f"   🛑 SL Order ID: {sl_order['orderId']}", Fore.YELLOW)
                self.print_color(f"   🎯 TP Order ID: {tp_order['orderId']}", Fore.YELLOW)
                
                # Store trade info with both order IDs
                self.bot_opened_trades[pair] = {
                    "pair": pair,
                    "direction": direction,
                    "entry_price": current_price,
                    "quantity": quantity,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "sl_order_id": sl_order['orderId'],
                    "tp_order_id": tp_order['orderId'],
                    "entry_time": time.time(),
                    "source": "BOT",
                    'status': 'ACTIVE',
                    'ai_confidence': confidence,
                    'ai_reason': reason
                }
                
                self.print_color(f"🚀 OCO TRADE ACTIVATED: {pair} {direction}", Fore.GREEN)
                return True
                
            except BinanceAPIException as e:
                self.print_color(f"❌ Binance API Error: {e}", Fore.RED)
                # Clean up on error
                self.cleanup_old_orders(pair)
                return False
            except Exception as e:
                self.print_color(f"❌ Order execution failed: {e}", Fore.RED)
                self.cleanup_old_orders(pair)
                return False
            
        except Exception as e:
            self.print_color(f"❌ Trade execution failed: {e}", Fore.RED)
            return False

    # ... (ကျန်တဲ့ function တွေ အတူတူပဲ ထားလိုက်ပါ)

    def run_trading_cycle(self):
        """Main trading cycle"""
        try:
            # Scan for existing positions first
            existing_count = self.scan_existing_positions()
            if existing_count > 0:
                self.print_color(f"🔍 Found {existing_count} existing positions", Fore.CYAN)
            
            # Monitor OCO orders
            self.monitor_oco_orders()
            
            # Display live dashboard
            self.display_live_dashboard()
            
            # Check if we can open new trades
            active_bot_trades = len([t for t in self.bot_opened_trades.values() if t['status'] == 'ACTIVE'])
            total_positions = len(self.existing_positions) + active_bot_trades
            
            if total_positions < self.max_concurrent_trades:
                market_data = self.get_market_data()
                
                if market_data:
                    self.print_color(f"\n🔄 Looking for new trade opportunities...", Fore.BLUE)
                    for pair in market_data.keys():
                        if self.can_open_new_trade(pair):
                            pair_data = {pair: market_data[pair]}
                            decision = self.get_ai_decision(pair_data)
                            
                            if decision["action"] == "TRADE":
                                self.print_color(f"✅ QUALIFIED: {pair}", Fore.GREEN)
                                success = self.execute_oco_trade(decision)
                                if success:
                                    self.print_color(f"🎯 Trade executed successfully!", Fore.GREEN)
                                    break
                            else:
                                self.print_color(f"⏸️  AI suggests HOLD for {pair} (Confidence: {decision['confidence']}%)", Fore.YELLOW)
                                self.print_color(f"   💡 Reason: {decision['reason']}", Fore.WHITE)
                else:
                    self.print_color(f"📊 No trading pairs available", Fore.BLUE)
            else:
                self.print_color(f"🚫 Maximum positions reached ({total_positions}/{self.max_concurrent_trades}) - Skipping new trades", Fore.RED)
            
        except Exception as e:
            self.print_color(f"❌ Trading cycle error: {e}", Fore.RED)

    # ... (ကျန်တဲ့ function တွေ အတူတူပဲ)

if __name__ == "__main__":
    try:
        bot = RealOrderPositionTracker()
        bot.start_trading()
    except Exception as e:
        print(f"❌ Failed to start bot: {e}")
