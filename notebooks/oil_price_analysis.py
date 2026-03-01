import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# --- Configuration ---
wti_ticker = "CL=F"
brent_ticker = "BZ=F"

start_date = "2025-06-01"
end_date = "2026-03-01"

geopolitical_events_data = [
    {"date": "2025-06-13", "description": "Israel strikes Iran"},
    {"date": "2026-02-27", "description": "US/Israel strikes on Iran"},
]

window_size = 30


# --- Data Acquisition ---
def download_oil_prices(ticker1, ticker2, start, end):
    """Downloads and merges oil price data from yfinance."""
    try:
        data1 = yf.download(ticker1, start=start, end=end, progress=False)
        data2 = yf.download(ticker2, start=start, end=end, progress=False)

        print(f"Data1 shape: {data1.shape}, Data2 shape: {data2.shape}")

        # Handle potential None or empty data
        if data1 is None or data1.empty or data2 is None or data2.empty:
            print("Error: No data retrieved from yfinance")
            return pd.DataFrame()

        # Extract Close price - flatten if needed
        def get_close(df):
            if "Close" in df.columns:
                col = df["Close"]
            elif "Adj Close" in df.columns:
                col = df["Adj Close"]
            else:
                print(f"Available columns: {df.columns.tolist()}")
                return None
            # Flatten: squeeze removes single-element dimensions
            return col.squeeze() if hasattr(col, "squeeze") else col

        wti_close = get_close(data1)
        brent_close = get_close(data2)

        if wti_close is None or brent_close is None:
            return pd.DataFrame()

        # Ensure 1D Series
        wti_close = pd.Series(
            wti_close.values.flatten() if wti_close.ndim > 1 else wti_close.values,
            index=data1.index,
        )
        brent_close = pd.Series(
            brent_close.values.flatten()
            if brent_close.ndim > 1
            else brent_close.values,
            index=data2.index,
        )

        oil_prices = pd.DataFrame({"WTI": wti_close, "Brent": brent_close})
        oil_prices.dropna(inplace=True)
        print(f"Successfully downloaded oil price data. Shape: {oil_prices.shape}")
        return oil_prices
    except Exception as e:
        print(f"Error downloading oil price data: {e}")
        import traceback

        traceback.print_exc()
        return pd.DataFrame()

        # Extract Close price - handle both column name and MultiIndex cases
        def get_close(df):
            if "Close" in df.columns:
                col = df["Close"]
            elif "Adj Close" in df.columns:
                col = df["Adj Close"]
            else:
                print(f"Available columns: {df.columns.tolist()}")
                return None
            # Flatten if needed
            if hasattr(col, "values"):
                return pd.Series(col.values, index=df.index)
            return col

        wti_close = get_close(data1)
        brent_close = get_close(data2)

        if wti_close is None or brent_close is None:
            return pd.DataFrame()

        oil_prices = pd.DataFrame({"WTI": wti_close, "Brent": brent_close})
        oil_prices.dropna(inplace=True)
        print(f"Successfully downloaded oil price data. Shape: {oil_prices.shape}")
        return oil_prices
    except Exception as e:
        print(f"Error downloading oil price data: {e}")
        import traceback

        traceback.print_exc()
        return pd.DataFrame()


def get_geopolitical_events(events_list):
    """Converts list of events to DataFrame."""
    geopolitical_events = []
    for event in events_list:
        try:
            event_date = datetime.strptime(event["date"], "%Y-%m-%d")
            geopolitical_events.append(
                {"date": event_date, "description": event["description"]}
            )
        except Exception as e:
            print(f"Could not parse event date: {e}")
    geopolitical_events_df = pd.DataFrame(geopolitical_events)
    if not geopolitical_events_df.empty:
        geopolitical_events_df.set_index("date", inplace=True)
    return geopolitical_events_df


# --- Data Analysis ---
def analyze_price_volatility(oil_prices, geopolitical_events_df):
    """Analyzes price volatility and correlation with geopolitical events."""
    if oil_prices.empty:
        print("Oil price data is not available for analysis.")
        return

    # Calculate returns and volatility
    oil_prices["WTI_Returns"] = oil_prices["WTI"].pct_change()
    oil_prices["Brent_Returns"] = oil_prices["Brent"].pct_change()
    oil_prices["WTI_Volatility"] = oil_prices["WTI_Returns"].rolling(
        window=window_size
    ).std() * (252**0.5)
    oil_prices["Brent_Volatility"] = oil_prices["Brent_Returns"].rolling(
        window=window_size
    ).std() * (252**0.5)

    # Moving averages
    oil_prices["WTI_MA20"] = oil_prices["WTI"].rolling(window=20).mean()
    oil_prices["WTI_MA50"] = oil_prices["WTI"].rolling(window=50).mean()
    oil_prices["Brent_MA20"] = oil_prices["Brent"].rolling(window=20).mean()
    oil_prices["Brent_MA50"] = oil_prices["Brent"].rolling(window=50).mean()

    # --- Plot 1: Price with Moving Averages ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Price + MAs
    ax1 = axes[0, 0]
    ax1.plot(oil_prices.index, oil_prices["WTI"], label="WTI", color="blue", alpha=0.7)
    ax1.plot(
        oil_prices.index,
        oil_prices["WTI_MA20"],
        label="MA20",
        color="orange",
        linestyle="--",
    )
    ax1.plot(
        oil_prices.index,
        oil_prices["WTI_MA50"],
        label="MA50",
        color="green",
        linestyle="--",
    )
    for date, row in geopolitical_events_df.iterrows():
        ax1.axvline(date, color="red", linestyle=":", alpha=0.7)
    ax1.set_title("WTI Crude Oil Price with Moving Averages")
    ax1.set_ylabel("Price (USD)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Brent Price + MAs
    ax2 = axes[0, 1]
    ax2.plot(
        oil_prices.index, oil_prices["Brent"], label="Brent", color="red", alpha=0.7
    )
    ax2.plot(
        oil_prices.index,
        oil_prices["Brent_MA20"],
        label="MA20",
        color="orange",
        linestyle="--",
    )
    ax2.plot(
        oil_prices.index,
        oil_prices["Brent_MA50"],
        label="MA50",
        color="green",
        linestyle="--",
    )
    for date, row in geopolitical_events_df.iterrows():
        ax2.axvline(date, color="black", linestyle=":", alpha=0.7)
    ax2.set_title("Brent Crude Oil Price with Moving Averages")
    ax2.set_ylabel("Price (USD)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Volatility
    ax3 = axes[1, 0]
    ax3.plot(
        oil_prices.index, oil_prices["WTI_Volatility"], label="WTI Vol", color="blue"
    )
    ax3.plot(
        oil_prices.index, oil_prices["Brent_Volatility"], label="Brent Vol", color="red"
    )
    ax3.set_title("30-Day Annualized Volatility")
    ax3.set_ylabel("Volatility")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Returns distribution
    ax4 = axes[1, 1]
    oil_prices["WTI_Returns"].dropna().hist(
        ax=ax4, bins=50, alpha=0.5, label="WTI", color="blue"
    )
    oil_prices["Brent_Returns"].dropna().hist(
        ax=ax4, bins=50, alpha=0.5, label="Brent", color="red"
    )
    ax4.set_title("Daily Returns Distribution")
    ax4.set_xlabel("Return")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("notebooks/figures/oil_analysis_1.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot 2: Event Impact Analysis ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Cumulative returns
    oil_prices["WTI_CumReturn"] = (1 + oil_prices["WTI_Returns"]).cumprod() - 1
    oil_prices["Brent_CumReturn"] = (1 + oil_prices["Brent_Returns"]).cumprod() - 1

    ax1 = axes[0]
    ax1.plot(
        oil_prices.index, oil_prices["WTI_CumReturn"] * 100, label="WTI", color="blue"
    )
    ax1.plot(
        oil_prices.index,
        oil_prices["Brent_CumReturn"] * 100,
        label="Brent",
        color="red",
    )
    ax1.axhline(0, color="black", linestyle="-", alpha=0.3)
    for date, row in geopolitical_events_df.iterrows():
        ax1.axvline(date, color="grey", linestyle="--", alpha=0.7)
    ax1.set_title("Cumulative Returns (%)")
    ax1.set_ylabel("Return (%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # WTI vs Brent spread
    ax2 = axes[1]
    spread = oil_prices["WTI"] - oil_prices["Brent"]
    ax2.plot(oil_prices.index, spread, label="WTI - Brent", color="purple")
    ax2.axhline(0, color="black", linestyle="-", alpha=0.3)
    ax2.set_title("WTI-Brent Spread")
    ax2.set_ylabel("Spread (USD)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("notebooks/figures/oil_analysis_2.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Print Event Impact Statistics ---
    print("\n" + "=" * 60)
    print("GEOPOLITICAL EVENT IMPACT ANALYSIS")
    print("=" * 60)

    if not geopolitical_events_df.empty:
        for date, event in geopolitical_events_df.iterrows():
            print(f"\n--- {event['description']} ({date.strftime('%Y-%m-%d')}) ---")
            start_window = date - timedelta(days=3)
            end_window = date + timedelta(days=3)
            window_data = oil_prices.loc[start_window:end_window]

            if not window_data.empty:
                wti_change = (
                    (window_data["WTI"].iloc[-1] - window_data["WTI"].iloc[0])
                    / window_data["WTI"].iloc[0]
                ) * 100
                brent_change = (
                    (window_data["Brent"].iloc[-1] - window_data["Brent"].iloc[0])
                    / window_data["Brent"].iloc[0]
                ) * 100
                print(
                    f"  3-Day Price Change: WTI: {wti_change:+.2f}% | Brent: {brent_change:+.2f}%"
                )

                # Volatility during event
                wti_vol = window_data["WTI_Volatility"].mean()
                brent_vol = window_data["Brent_Volatility"].mean()
                print(f"  Avg Volatility: WTI: {wti_vol:.2%} | Brent: {brent_vol:.2%}")

    # Current market stats
    print("\n" + "=" * 60)
    print("CURRENT MARKET STATS")
    print("=" * 60)
    latest = oil_prices.iloc[-1]
    print(f"  Latest Prices: WTI: ${latest['WTI']:.2f} | Brent: ${latest['Brent']:.2f}")
    print(
        f"  Latest Volatility: WTI: {latest['WTI_Volatility']:.2%} | Brent: {latest['Brent_Volatility']:.2%}"
    )
    print(
        f"  20-Day Return: WTI: {oil_prices['WTI_Returns'].tail(20).sum() * 100:+.2f}% | Brent: {oil_prices['Brent_Returns'].tail(20).sum() * 100:+.2f}%"
    )

    return oil_prices


# --- Strategy Identification ---
def identify_strategies(oil_prices):
    """Provides data-driven trading recommendations."""
    if oil_prices.empty:
        print("No data available for strategy analysis.")
        return

    latest = oil_prices.iloc[-1]
    wti_price = latest["WTI"]
    brent_price = latest["Brent"]
    wti_vol = latest["WTI_Volatility"]
    brent_vol = latest["Brent_Volatility"]

    # Calculate momentum
    wti_ma20 = latest["WTI_MA20"]
    wti_ma50 = latest["WTI_MA50"]
    brent_ma20 = latest["Brent_MA20"]
    brent_ma50 = latest["Brent_MA50"]

    # Trend analysis
    wti_bullish = wti_price > wti_ma20 > wti_ma50
    brent_bullish = brent_price > brent_ma20 > brent_ma50

    print("\n" + "=" * 60)
    print("TRADING RECOMMENDATIONS")
    print("=" * 60)

    print(f"\n📊 MARKET STATUS:")
    print(f"   WTI: ${wti_price:.2f} | Vol: {wti_vol:.1%}")
    print(f"   Brent: ${brent_price:.2f} | Vol: {brent_vol:.1%}")
    print(
        f"   Trend: {'🟢 BULLISH' if wti_bullish and brent_bullish else '🔴 BEARISH' if not wti_bullish and not brent_bullish else '🟡 MIXED'}"
    )

    print(f"\n📈 RECOMMENDED STRATEGIES:")

    # Strategy 1: Trend following
    if wti_bullish and brent_bullish:
        print(
            "   1. TREND FOLLOWING (Long): Prices above MA20 & MA50 - maintain long positions"
        )
        print(
            f"      Entry: ${wti_price:.2f} | Target: ${wti_price * 1.05:.2f} | Stop: ${wti_price * 0.96:.2f}"
        )
    else:
        print(
            "   1. TREND FOLLOWING: Prices below moving averages - consider short or wait"
        )

    # Strategy 2: Volatility based
    avg_hist_vol = oil_prices["WTI_Volatility"].mean()
    if wti_vol > avg_hist_vol * 1.3:
        print("   2. VOLATILITY SPIKE: Current vol 30%+ above average")
        print("      Consider: Straddles/Strangles if expecting big move")
        print("      Or: Profit from mean reversion (vol usually drops)")
    elif wti_vol < avg_hist_vol * 0.7:
        print("   2. LOW VOLATILITY: Vol below average - consider long volatility")
        print("      Options: Buy straddles, wait for catalyst")

    # Strategy 3: Mean reversion
    wti_50d_avg = oil_prices["WTI"].rolling(50).mean().iloc[-1]
    deviation = (wti_price - wti_50d_avg) / wti_50d_avg
    if deviation > 0.05:
        print(
            f"   3. MEAN REVERSION: WTI {deviation * 100:.1f}% above 50-day avg - potential pullback"
        )
    elif deviation < -0.05:
        print(
            f"   3. MEAN REVERSION: WTI {abs(deviation) * 100:.1f}% below 50-day avg - potential bounce"
        )

    # Geopolitical context
    print(f"\n⚠️  GEOPOLITICAL RISK:")
    print("   - US/Israel strikes on Iran have increased supply risk premium")
    print("   - Monitor for: Strait of Hormuz disruptions, Iranian retaliation")
    print("   - Scenarios: Limited strikes = spike fades | Supply hit = $80+")

    print(f"\n💡 BOTTOM LINE:")
    if wti_bullish and wti_vol > 0.25:
        print("   Current setup favors BULLISH directional plays with tight stops.")
        print("   Volatility is elevated - good for options strategies.")
    elif wti_bullish:
        print(
            "   Trend is bullish but vol is normalizing - prefer futures/options over volatility plays."
        )
    else:
        print(
            "   Mixed signals - consider waiting for clearer trend or hedge downside."
        )

    print("\n" + "=" * 60)


# --- Main Execution ---
if __name__ == "__main__":
    oil_prices_data = download_oil_prices(
        wti_ticker, brent_ticker, start_date, end_date
    )
    geopolitical_events = get_geopolitical_events(geopolitical_events_data)

    if oil_prices_data.empty:
        print("Failed to retrieve oil price data. Exiting.")
        exit()

    print("\n--- Oil Prices (Head) ---")
    print(oil_prices_data.head())

    print("\n--- Geopolitical Events ---")
    print(geopolitical_events)

    print("\n--- Generating Plots & Analysis ---")
    analyzed_data = analyze_price_volatility(oil_prices_data, geopolitical_events)

    print("\n--- Strategy Recommendations ---")
    identify_strategies(analyzed_data)
