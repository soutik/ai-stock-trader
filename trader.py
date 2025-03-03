import openai
import requests
import logging
import random
import datetime
import time
import json
import os
import yfinance as yf  # Ensure yfinance is installed: pip install yfinance
import pandas as pd

# Set up logging for production-level traceability.
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# Portfolio class to manage cash, holdings, and trade transactions.
class Portfolio:
    def __init__(self, initial_cash=100000):
        self.cash = initial_cash
        self.holdings = {}  # mapping symbol -> number of shares
        self.transactions = []  # record of all transactions

    def buy(self, symbol, price, shares, trade_date):
        cost = price * shares
        if self.cash < cost:
            logging.warning("[%s] Insufficient cash to buy %d shares of %s at %.2f",
                            trade_date.strftime("%Y-%m-%d"), shares, symbol, price)
            return False
        self.cash -= cost
        self.holdings[symbol] = self.holdings.get(symbol, 0) + shares
        self.transactions.append({
            "date": trade_date.isoformat(),
            "symbol": symbol,
            "action": "BUY",
            "price": price,
            "shares": shares
        })
        logging.info("[%s] Bought %d shares of %s at %.2f", 
                     trade_date.strftime("%Y-%m-%d"), shares, symbol, price)
        return True

    def sell(self, symbol, price, shares, trade_date):
        if self.holdings.get(symbol, 0) < shares:
            logging.warning("[%s] Insufficient shares to sell %d shares of %s",
                            trade_date.strftime("%Y-%m-%d"), shares, symbol)
            return False
        self.holdings[symbol] -= shares
        self.cash += price * shares
        self.transactions.append({
            "date": trade_date.isoformat(),
            "symbol": symbol,
            "action": "SELL",
            "price": price,
            "shares": shares
        })
        logging.info("[%s] Sold %d shares of %s at %.2f", 
                     trade_date.strftime("%Y-%m-%d"), shares, symbol, price)
        return True

    def get_value(self, current_prices):
        total_value = self.cash
        for symbol, shares in self.holdings.items():
            total_value += current_prices.get(symbol, 0) * shares
        return total_value

# NewsFetcher fetches historical news using a news API.
class NewsFetcher:
    def __init__(self, api_key=None):
        self.api_key = api_key

    def fetch_news(self, symbol, simulation_date):
        date_str = simulation_date.strftime("%Y-%m-%d")
        if self.api_key:
            url = (f"https://newsapi.org/v2/everything?q={symbol}&from={date_str}&to={date_str}"
                   f"&sortBy=publishedAt&apiKey={self.api_key}")
            try:
                response = requests.get(url)
                data = response.json()
                if data.get("status") == "ok":
                    articles = data.get("articles", [])
                    logging.info("[%s] Fetched %d news articles for %s", 
                                 date_str, len(articles), symbol)
                    return articles[:5]  # use top 5 articles for analysis
                else:
                    logging.error("[%s] News API error for %s: %s", date_str, symbol, data)
            except Exception as e:
                logging.error("[%s] Exception fetching news for %s: %s", date_str, symbol, e)
        # Fallback dummy news if API key is not provided or an error occurs
        logging.info("[%s] Using dummy news for %s", date_str, symbol)
        return [{
            "title": f"Market update for {symbol} on {date_str}",
            "description": "No API key provided or error occurred in fetching live news."
        }]

# LLMAnalyzer uses OpenAI's API to provide trading recommendations.
class LLMAnalyzer:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key

    def analyze(self, symbol, news_articles, current_price, simulation_date):
        # Build a news summary.
        news_summary = "\n".join(
            f"{article.get('title', 'No title')} - {article.get('description', 'No description')}"
            for article in news_articles
        )
        date_str = simulation_date.strftime("%Y-%m-%d")
        prompt = f"""
You are an expert stock analyst. Given the following news and the current price for {symbol} on {date_str}:

News:
{news_summary}

Current Price: {current_price}

Based on this information, provide a trading recommendation optimized for profit.
"""

        # Define function specification for structured output.
        functions = [
            {
                "name": "trade_recommendation",
                "description": "Return a trading recommendation with the following keys: symbol, buy_limit, sell_limit, and action.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "The stock symbol."
                        },
                        "buy_limit": {
                            "type": "number",
                            "description": "The price below which the stock should be bought."
                        },
                        "sell_limit": {
                            "type": "number",
                            "description": "The price above which the stock should be sold."
                        },
                        "action": {
                            "type": "string",
                            "enum": ["BUY", "SELL", "HOLD"],
                            "description": "The recommended action."
                        }
                    },
                    "required": ["symbol", "buy_limit", "sell_limit", "action"]
                }
            }
        ]

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",  # Ensure this or another function-calling supported model is used.
                messages=[{"role": "user", "content": prompt}],
                functions=functions,
                function_call="auto",  # Automatically call the function if applicable.
                temperature=0.5,
                max_tokens=150
            )
            message = response.choices[0].message
            print(message.function_call)
            # Check if the model returned a function call.
            if message.function_call:
                function_args = message.function_call.arguments
                try:
                    recommendation = json.loads(function_args)
                except Exception as e:
                    logging.error("[%s] Error parsing structured output for %s: %s", date_str, symbol, e)
                    recommendation = None
            else:
                # Fallback: try parsing the message content as JSON.
                try:
                    recommendation = json.loads(message.get("content", ""))
                except Exception as e:
                    logging.error("[%s] Error parsing output for %s: %s", date_str, symbol, e)
                    recommendation = None

            if recommendation is None:
                raise ValueError("No valid recommendation returned.")
            logging.info("[%s] LLM recommendation for %s: %s", date_str, symbol, recommendation)
            return recommendation
        except Exception as e:
            logging.error("[%s] LLM analysis failed for %s: %s", date_str, symbol, e)
            # Fallback recommendation.
            return {
                "symbol": symbol,
                "buy_limit": current_price * 0.95,
                "sell_limit": current_price * 1.05,
                "action": "HOLD"
            }

# StockMarketSimulator now uses real historical data via yfinance.
class StockMarketSimulator:
    def __init__(self, symbols, start_date, end_date):
        self.symbols = symbols
        self.historical_data = {}
        # Download historical data for each symbol.
        for symbol in symbols:
            logging.info("Downloading historical data for %s from %s to %s", 
                         symbol, start_date, end_date)
            data = yf.download(symbol, start=start_date, end=end_date)
            if data.empty:
                logging.error("No data fetched for %s. Check symbol or date range.", symbol)
            else:
                # Ensure the index is a DatetimeIndex
                data.index = pd.to_datetime(data.index)
                self.historical_data[symbol] = data

    def get_price(self, symbol, simulation_date):
        data = self.historical_data.get(symbol)
        if data is None or data.empty:
            raise ValueError(f"No historical data for {symbol}")
        # Find the most recent available trading day at or before simulation_date.
        # data.index is sorted. Use asof to find the nearest date.
        nearest_date = data.index.asof(simulation_date)
        if pd.isna(nearest_date):
            logging.error("No trading data available for %s at or before %s", 
                          symbol, simulation_date)
            return None
        price = data.loc[nearest_date]["Close"]
        # If multiple values are returned (unlikely), take the first.
        if isinstance(price, pd.Series):
            price = price.iloc[0]
        return price

    def update_prices(self, simulation_date):
        current_prices = {}
        for symbol in self.symbols:
            price = self.get_price(symbol, simulation_date)
            if price is None:
                logging.warning("Price for %s on %s not available.", 
                                symbol, simulation_date.strftime("%Y-%m-%d"))
                continue
            current_prices[symbol] = price
        logging.info("Historical prices for %s: %s", 
                     simulation_date.strftime("%Y-%m-%d"), current_prices)
        return current_prices

# The main trading logic that ties all components together.
def main():
    # Configuration and environment setup.
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    simulation_days = 1
    initial_cash = 100000

    # Replace with your actual API keys or set them as environment variables.
    news_api_key = os.getenv("NEWS_API_KEY")  # e.g., from newsapi.org
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Set simulation start and end dates.
    simulation_interval = 7
    simulation_end_date = datetime.datetime.today()
    simulation_start_date = simulation_end_date - datetime.timedelta(days=simulation_days)
    # Format dates for yfinance (end date is exclusive; add one day to include the final day).
    yf_start = simulation_start_date.strftime("%Y-%m-%d")
    yf_end = (simulation_end_date + datetime.timedelta(days=simulation_interval)).strftime("%Y-%m-%d")

    # Instantiate system components.
    portfolio = Portfolio(initial_cash)
    news_fetcher = NewsFetcher(api_key=news_api_key)
    llm_analyzer = LLMAnalyzer(openai_api_key=openai_api_key)
    market_simulator = StockMarketSimulator(symbols, yf_start, yf_end)

    # Simulation loop: each iteration represents a historical trading day.
    current_date = simulation_start_date
    days_run = 0
    while days_run < simulation_days:
        # Only process if we have data (i.e. skip non-trading days)
        current_prices = market_simulator.update_prices(current_date)
        if not current_prices:
            logging.info("[%s] No trading data; skipping.", current_date.strftime("%Y-%m-%d"))
            current_date += datetime.timedelta(days=1)
            continue

        date_str = current_date.strftime("%Y-%m-%d")
        logging.info("=== Trading Day: %s ===", date_str)
        for symbol in symbols:
            if symbol not in current_prices:
                logging.info("[%s] Skipping %s due to missing price data.", date_str, symbol)
                continue
            # Fetch historical news for the simulation date.
            articles = news_fetcher.fetch_news(symbol, current_date)
            current_price = current_prices[symbol]
            # Get trading recommendation from LLM based on news and price.
            recommendation = llm_analyzer.analyze(symbol, articles, current_price, current_date)

            # Execute trade if conditions are met.
            action = recommendation.get("action", "HOLD").upper()
            buy_limit = recommendation.get("buy_limit", current_price)
            sell_limit = recommendation.get("sell_limit", current_price)
            if action == "BUY" and current_price <= buy_limit:
                # Determine maximum shares that can be purchased.
                shares_to_buy = int(portfolio.cash // current_price)
                if shares_to_buy > 0:
                    portfolio.buy(symbol, current_price, shares_to_buy, current_date)
            elif action == "SELL" and current_price >= sell_limit:
                shares_to_sell = portfolio.holdings.get(symbol, 0)
                if shares_to_sell > 0:
                    portfolio.sell(symbol, current_price, shares_to_sell, current_date)
            else:
                logging.info("[%s] No trade executed for %s; action: %s", 
                             date_str, symbol, action)
        # Report the portfolio value at the end of the day.
        portfolio_value = portfolio.get_value(current_prices)
        logging.info("End of Day %s portfolio value: %.2f", date_str, portfolio_value)
        days_run += 1
        # Move to the next calendar day.
        current_date += datetime.timedelta(days=simulation_interval)
        # In production, you might wait until the next trading day; here we simulate with a short delay.
        time.sleep(1)

    final_value = portfolio.get_value(market_simulator.update_prices(simulation_end_date))
    logging.info("Trading simulation complete. Final portfolio value: %.2f", final_value)

if __name__ == "__main__":
    main()
