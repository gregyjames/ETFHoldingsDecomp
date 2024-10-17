import yfinance as yf


def calculate_valuation_difference(price, value):
    """
    Calculate the percentage by which something is overvalued or undervalued.

    :param price: Current market price
    :param value: Intrinsic value or fair value
    :return: Percentage difference and whether it's overvalued or undervalued
    """
    if value == 0:
        return "Intrinsic value is 0, calculation not possible."

    # Calculate percentage difference
    percentage_difference = ((price - value) / value) * 100

    # Determine whether it's overvalued or undervalued
    if percentage_difference > 0:
        return f"Overvalued by {percentage_difference:.2f}%"
    else:
        return f"Undervalued by {-percentage_difference:.2f}%"


def GrahamValuation(symbol, avg_aaa_rate, current_aaa_rate, pe_no_growth=8.5):
    """
    Calculate the intrinsic value of a stock using Benjamin Graham's valuation formula.

    :param symbol: The stock ticker symbol (e.g., 'AAPL').
    :param avg_aaa_rate: The average historical AAA corporate bond yield (as a percentage, e.g., 4.4).
    :param current_aaa_rate: The current AAA corporate bond yield (as a percentage, e.g., 4.68).
    :param pe_no_growth: The P/E ratio for a no-growth stock (default is 8.5, per Graham's recommendation).

    :return: A tuple containing:
        - The estimated intrinsic value of the stock.
        - The full `yfinance.Ticker` object for further use if needed.

    Example usage:
        value, ticker = GrahamValuation('AAPL', 4.4, 4.68)

    Notes:
        - This formula assumes the relationship between the company’s earnings, growth rate, and bond yields.
        - The earnings per share (EPS) is retrieved from Yahoo Finance.
        - The stock's estimated growth rate is multiplied by 2, following Graham's approach for a growing company.
    """
    ticker = yf.Ticker(symbol)
    eps = ticker.info.get("trailingEps", "EPS data not available")
    growth = (ticker.growth_estimates.iloc[5].iloc[0]) * 100
    two_growth = 2 * growth
    value = (eps * (pe_no_growth + two_growth) * avg_aaa_rate) / current_aaa_rate
    return value, ticker


def main():
    stock = "JEPQ"
    spy = yf.Ticker(stock)
    data = spy.funds_data
    holdings = data.top_holdings.loc[:, ["Holding Percent"]].reset_index()
    print(holdings)

    tickers = holdings["Symbol"].to_list()

    over = 0
    under = 0
    avg_aaa = 4.4  # FRED
    current_aaa = 4.68  # FRED

    for symbol in tickers:
        value, ticker = GrahamValuation(symbol, avg_aaa, current_aaa)
        hist = ticker.history(period="1d").reset_index()["Close"].to_list()[0]
        undervalued = value > hist
        if undervalued:
            under = under + 1
        else:
            over = over + 1
        print(
            symbol
            + " "
            + str(round(hist, 2))
            + "/"
            + str(round(value, 2))
            + " "
            + (calculate_valuation_difference(hist, value))
        )
    if under > over:
        print(f"{stock} is undervalued at {under}/{over} for the top holdings")
    elif over > under:
        print(f"{stock} is overvalued at {over}/{under} for the top holdings")
    else:
        print(f"{stock} is fairly valued at {under}/{over} for the top holdings")


if __name__ == "__main__":
    main()
