# Stock Trading Simulation with AI and Historical Data

This project is a stock trading simulator that uses historical data to make trading decisions based on news articles fetched from an external API. The system leverages the capabilities of OpenAI's language model to analyze news articles and provide trading recommendations, which are then executed against a simulated portfolio.

## Features
1. Historical Data Fetching: Uses yfinance to fetch historical data for specified stocks over a given date range.
2. News API Integration: Fetches news articles related to each stock symbol on specific dates using the NewsAPI.
3. AI-Driven Trading Recommendations: Utilizes OpenAI's language model to analyze news and current prices, providing buy/sell recommendations.
4. Portfolio Management: Manages a simulated portfolio of stocks, allowing for buying and selling based on AI recommendations.
5. Simulation Engine: Simulates trading over multiple days, updating portfolio values and transactions accordingly.

## Requirements
- Python 3.x
- Libraries: yfinance, requests, openai, pandas
- API Keys for NewsAPI and OpenAI services (set as environment variables or directly in the script).

## Installation
1. Install required Python packages:
```pip install yfinance requests openai pandas```

2. Set up environment variables for your API keys:
`NEWS_API_KEY`: For NewsAPI access.
`OPENAI_API_KEY`: For OpenAI services.

## Usage
1. Run the script using Python:
```python trading_simulation.py```
2. The script will simulate stock trading based on historical data and AI-generated recommendations over a specified number of days.

## Configuration
- Symbols: Define the list of stocks to be simulated in the symbols variable.
- Simulation Parameters: Adjust the simulation parameters such as simulation_days, simulation_interval, etc., in the script.

## Examples
Fetching Historical Data and News Articles:
```
market_simulator = StockMarketSimulator(symbols, yf_start, yf_end)
current_prices = market_simulator.update_prices(current_date)
articles = news_fetcher.fetch_news(symbol, current_date)
```

## AI-Driven Trading Recommendations:

```
recommendation = llm_analyzer.analyze(symbol, articles, current_price, current_date)
if recommendation["action"] == "BUY" and current_price <= buy_limit:
    portfolio.buy(symbol, current_price, shares_to_buy, current_date)
Portfolio Management:
portfolio = Portfolio(initial_cash)
portfolio.buy(symbol, current_price, shares_to_buy, current_date)
portfolio.sell(symbol, current_price, shares_to_sell, current_date)
```

## Conclusion
This project demonstrates a comprehensive approach to stock trading simulation using historical data and AI-driven decision making. It provides a flexible framework for integrating external APIs and utilizing advanced language models to make informed trading decisions.