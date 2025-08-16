import configparser
from pathlib import Path
from oandapyV20 import API
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.accounts as accounts
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class BrokerConnector:
    """
    Handles all communication and order execution with the OANDA API.
    """
    def __init__(self):
        """Initializes the API connection using credentials from config.ini."""
        print("Initializing Broker Connector...")
        try:
            PROJECT_ROOT = Path(__file__).resolve().parents[1]
            config_path = PROJECT_ROOT / 'src' / 'config.ini'
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found at {config_path}")
            config = configparser.ConfigParser()
            config.read(config_path)
            if 'oanda' not in config:
                raise KeyError("Section [oanda] missing in config.ini")
            self.account_id = config['oanda']['account_id']
            self.access_token = config['oanda']['access_token']
            self.environment = config['oanda']['environment']
            if not all([self.account_id, self.access_token, self.environment]):
                raise ValueError("Missing required config keys: account_id, access_token, or environment")
            self.client = API(access_token=self.access_token, environment=self.environment)
            # Test connection
            r = accounts.AccountSummary(accountID=self.account_id)
            self.client.request(r)
            print("Broker connection successful.")
        except Exception as e:
            print(f"FATAL: Could not initialize broker connection. Error: {str(e)}")
            self.client = None
            self.account_id = None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: print(f"  -> Retrying account balance fetch (attempt {retry_state.attempt_number})...")
    )
    def get_account_balance(self):
        """Fetches the current balance of the OANDA account."""
        if not self.client:
            return None
        r = accounts.AccountSummary(accountID=self.account_id)
        response = self.client.request(r)
        balance = float(response['account']['balance'])
        return balance

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: print(f"  -> Retrying price fetch (attempt {retry_state.attempt_number})...")
    )
    def get_current_price(self, instrument="GBP_JPY"):
        """Fetches the latest bid/ask price for an instrument."""
        if not self.client:
            return None
        params = {"instruments": instrument}
        r = pricing.PricingInfo(accountID=self.account_id, params=params)
        response = self.client.request(r)
        ask = float(response['prices'][0]['asks'][0]['price'])
        bid = float(response['prices'][0]['bids'][0]['price'])
        return {'ask': ask, 'bid': bid}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: print(f"  -> Retrying order creation (attempt {retry_state.attempt_number})...")
    )
    def create_market_order(self, instrument, units, sl_price, tp_price):
        """
        Creates a market order with a specified stop loss and take profit.
        """
        if not self.client:
            return None
        order_data = {
            "order": {
                "instrument": instrument,
                "units": str(units),
                "type": "MARKET",
                "positionFill": "DEFAULT",
                "takeProfitOnFill": {
                    "price": str(tp_price)
                },
                "stopLossOnFill": {
                    "price": str(sl_price)
                }
            }
        }
        r = orders.OrderCreate(accountID=self.account_id, data=order_data)
        response = self.client.request(r)
        print("  -> Order creation request sent. Response:")
        print(response)
        return response

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: print(f"  -> Retrying open trade check (attempt {retry_state.attempt_number})...")
    )
    def get_open_trade(self, instrument="GBP_JPY"):
        """Checks for an open trade for a specific instrument."""
        if not self.client:
            return None
        r = trades.OpenTrades(accountID=self.account_id)
        response = self.client.request(r)
        for trade in response.get('trades', []):
            if trade['instrument'] == instrument:
                return trade
        return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: print(f"  -> Retrying trade close (attempt {retry_state.attempt_number})...")
    )
    def close_trade(self, trade_id):
        """Closes a specific trade by its ID."""
        if not self.client:
            return None
        r = trades.TradeClose(accountID=self.account_id, tradeID=trade_id)
        response = self.client.request(r)
        print(f"  -> Trade {trade_id} close request sent. Response:")
        print(response)
        return response