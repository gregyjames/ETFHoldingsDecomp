import yfinance as yf
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor
import streamlit.web.bootstrap
import time


def calculate_valuation_difference(price, value):
    """
    Calculate the percentage by which something is overvalued or undervalued.

    :param price: Current market price
    :param value: Intrinsic value or fair value
    :return: Percentage difference and whether it's overvalued or undervalued
    """
    if value == 0:
        return "Intrinsic value is 0, calculation not possible."
    percentage_difference = ((price - value) / value) * 100
    return f"{'Overvalued' if percentage_difference > 0 else 'Undervalued'} by {abs(percentage_difference):.2f}%"


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
    eps = ticker.info.get(
        "trailingEps", 0
    )  # Use default value of 0 if EPS not available
    growth = ticker.growth_estimates.iloc[5].iloc[0] * 100
    hist = ticker.info.get("previousClose", 0)
    two_growth = 2 * growth
    value = (eps * (pe_no_growth + two_growth) * avg_aaa_rate) / current_aaa_rate
    return value, hist


def DCFValuation(symbol, required_rate=0.07, perpetual_rate=0.02, cfgrowth=0.03):
    """
    Perform a Discounted Cash Flow (DCF) valuation for the given stock and return the fair value per share.

    :param symbol: Stock ticker symbol as a string (e.g., 'AAPL').
    :param required_rate: Discount rate or required rate of return (default: 7%).
    :param perpetual_rate: Perpetual growth rate for the terminal value (default: 2%).
    :param cfgrowth: Expected annual growth rate of Free Cash Flow (FCF) during the forecast period (default: 3%).
    :return: Fair value per share (float).
    """
    ticker = yf.Ticker(symbol)

    # Attempt to retrieve necessary financial data; return None if unavailable
    try:
        latest_fcf = ticker.cashflow.loc["Free Cash Flow"].iloc[0]
        outstanding = ticker.info["sharesOutstanding"]
    except (KeyError, IndexError, TypeError):
        return None  # Handle missing or unavailable data

    # Precompute constants for discounting
    discount_factors = [(1 + required_rate) ** year for year in range(1, 5)]

    # Calculate the terminal value, discounted to present value
    tv = latest_fcf * (1 + perpetual_rate) / (required_rate - perpetual_rate)
    dtv = tv / discount_factors[-1]  # Discount terminal value by the last year's factor

    # Calculate the present value of projected FCFs for years 1 to 4
    dcf_sum = sum(
        (latest_fcf * (1 + cfgrowth) ** year) / discount_factors[year - 1]
        for year in range(1, 5)
    )

    # Sum the discounted FCF and terminal value
    total_value = dcf_sum + dtv

    # Return the fair value per share
    return total_value / outstanding


def fetch_valuation_data(
    symbol, avg_aaa, current_aaa, required_rate, perpetual_rate, cfgrowth, model
):
    value, hist = GrahamValuation(symbol, avg_aaa, current_aaa)
    dcfvalue = DCFValuation(
        symbol,
        required_rate=required_rate,
        perpetual_rate=perpetual_rate,
        cfgrowth=cfgrowth,
    )
    if model == "dcf":
        undervalued = dcfvalue > hist
    else:
        undervalued = value > hist
    status = calculate_valuation_difference(hist, value)
    return symbol, hist, value, undervalued, status, dcfvalue


def main(stock, required_rate, perpetual_rate, cfgrowth, avgaaa, currentaaa):
    start_time = time.time()
    spy = yf.Ticker(stock)
    data = spy.funds_data
    holdings = data.top_holdings.loc[:, ["Holding Percent"]].reset_index()
    other = 1 - sum(holdings["Holding Percent"])
    holdings_for_chart = holdings.copy()
    holdings_for_chart.loc[len(holdings_for_chart)] = ["Other", other]

    fig = px.pie(
        holdings_for_chart,
        values="Holding Percent",
        names="Symbol",
        title=f"{stock} Holdings",
        color_discrete_sequence=px.colors.sequential.RdBu,
    )

    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig)

    tickers = holdings["Symbol"].to_list()

    model = st.radio(
        label="Model",
        options=["dcf", "graham"],
        horizontal=True,
    )

    with st.spinner("Calculating..."):
        # Parallel processing for faster API calls
        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(
                    lambda symbol: fetch_valuation_data(
                        symbol,
                        avgaaa,
                        currentaaa,
                        required_rate,
                        perpetual_rate,
                        cfgrowth,
                        model,
                    ),
                    tickers,
                )
            )

        prices = []
        values = []
        dcfvalues = []
        over, under = 0, 0

        for result in results:
            symbol, hist, value, undervalued, status, dcfvalue = result
            prices.append(hist)
            values.append(value)
            dcfvalues.append(dcfvalue)
            if undervalued:
                under += 1
            else:
                over += 1
            # st.text(f"{symbol}: {hist:.2f}/{value:.2f} - {status}")

        col1, col2 = st.columns(2)
        with col1:
            st.header("Overvalued")
            st.text(over)
        with col2:
            st.header("Undervalued")
            st.text(under)

        df = pd.DataFrame(
            {"tickers": tickers, "prices": prices, "values": values, "dcf": dcfvalues}
        )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=df["tickers"],
                y=df["prices"],
                name="Historical Prices",
                marker_color="green",
            )
        )
        fig.add_trace(
            go.Bar(
                x=df["tickers"],
                y=df["values"],
                name="Graham Valuations",
                marker_color="red",
            )
        )
        fig.add_trace(
            go.Bar(
                x=df["tickers"],
                y=df["dcf"],
                name="DCF Valuations",
                marker_color="pink",
            )
        )
        fig.update_layout(
            title="Price vs Valuations", barmode="group", xaxis_tickangle=-45
        )
        st.plotly_chart(fig)
    end_time = time.time()
    execution_time = end_time - start_time  # Calculate the difference
    st.toast(f"Execution time: {execution_time:.2f} seconds")


st.title("ETFHoldingsDecomp")
stock = st.text_input("Enter a ETF ticker:", "SCHD")
st.markdown("#### DCF Model Settings:")
col1, col2, col3 = st.columns(3)
with col1:
    required_rate = st.number_input("Required Rate", 0.0, 1.0, step=0.01, value=0.07)
with col2:
    perpetual_rate = st.number_input("Perpetual Rate", 0.0, 1.0, step=0.01, value=0.02)
with col3:
    cfgrowth = st.number_input("CF Growth Rate", 0.0, 1.0, step=0.01, value=0.03)
st.markdown("#### Graham Model Settings:")
col1, col2 = st.columns(2)
with col1:
    avg_aaa = st.number_input("Average AAA Bond Rate", 0.0, 15.0, step=0.01, value=4.4)
with col2:
    current_aaa = st.number_input(
        "Current AAA Bond Rate", 0.0, 15.0, step=0.01, value=4.68
    )
main(
    stock=stock,
    required_rate=required_rate,
    perpetual_rate=perpetual_rate,
    cfgrowth=cfgrowth,
    avgaaa=avg_aaa,
    currentaaa=current_aaa,
)

if __name__ == "__main__":
    if "__streamlitmagic__" not in locals():
        streamlit.web.bootstrap.run(__file__, False, [], {})
