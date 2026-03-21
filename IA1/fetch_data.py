import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
 
# ── stock/ETF pools ──
SYMBOLS = symbols_list = [
  # US Internet Giants
  "GOOG",  # Alphabet (Google) 
  "META",  # Meta (Facebook)
  "AMZN",  # Amazon
  "AAPL",  # Apple
  "MSFT",  # Microsoft

  # Cloud/Enterprise Software
  "CRM",   # Salesforce
  "ADBE",  # Adobe
  "NOW",   # ServiceNow
  "WDAY",  # Workday
  "ZM",    # Zoom
  "DDOG",  # Datadog
  "SNOW",  # Snowflake
  "ORCL",  # Oracle 

  # Social Media & Digital Platforms
  "UBER",  # Uber
  "LYFT",  # Lyft

  # E-commerce & Fintech
  "SHOP",  # Shopify
  "PYPL",  # PayPal

  # === Chinese Internet (US-listed) ===
  "BABA",   # Alibaba
  "JD",     # JD.com
  "PDD",    # Pinduoduo
  "BIDU",   # Baidu
  "VIPS",   # Vipshop
 
  # === Semiconductors/Electronics ===
  "SOXX",   # iShares Semiconductor ETF
  "SMH",    # VanEck Semiconductor ETF
  "SOXL",   # 3x Leverage Semiconductor Bull ETF
  "TSM",    # Taiwan Semiconductor Manufacturing Company
  "INTC",   # Intel
  "QCOM",   # Qualcomm
  "MU",     # Micron Technology
  "AVGO",   # Broadcom
  "TXN",    # Texas Instruments
 
  # === Telecommunications/Telecom ===
  "XLC",    # Communication Services Sector ETF
  "VOX",    # Vanguard Communication Services ETF
  "FCOM",   # Fidelity Telecom Services ETF
  "T",      # AT&T
  "VZ",     # Verizon
  "TMUS",   # T-Mobile US
  "CMCSA",  # Comcast
  "DIS",    # The Walt Disney Company
 
  # === Agriculture/Agricultural Products ===
  "VEGI",   # iShares MSCI Global Agriculture Index ETF
  "DBA",    # Invesco DB Agriculture Fund
  "MOO",    # VanEck Agriculture ETF
  "ADM",    # Archer-Daniels-Midland Company
  "DE",     # Deere & Company (John Deere)
  "CF",     # CF Industries Holdings
  "MOS",    # The Mosaic Company
  "TSCO",   # Tractor Supply Company
 
  # === Electric Vehicles/New Energy ===
  "TSLA",   # Tesla
  "NIO",    # NIO Inc.
  "XPEV",   # XPeng Inc.
  "LI",     # Li Auto Inc.
  "BYD",    # BYD Company Limited (Hong Kong ADR)
 
  # === Traditional Finance ===
  "JPM",    # JPMorgan Chase & Co.
  "BAC",    # Bank of America
  "WFC",    # Wells Fargo & Company
  "GS",     # The Goldman Sachs Group
  "MS",     # Morgan Stanley
  "V",      # Visa Inc.
  "MA",     # Mastercard Incorporated
 
  # === Healthcare ===
  "JNJ",    # Johnson & Johnson
  "UNH",    # UnitedHealth Group
  "PFE",    # Pfizer Inc.
  "MRNA",   # Moderna Inc.
  "ABBV",   # AbbVie Inc.
 
  # === Consumer Goods ===
  "KO",     # The Coca-Cola Company
  "PEP",    # PepsiCo Inc.
  "PG",     # Procter & Gamble
  "WMT",    # Walmart Inc.
  "COST",   # Costco Wholesale Corporation
  "MCD",    # McDonald's Corporation
  "SBUX",   # Starbucks Corporation
 
  # === Energy ===
  "XOM",    # Exxon Mobil Corporation
  "CVX",    # Chevron Corporation
  "COP",    # ConocoPhillips
 
  # === Industrials ===
  "BA",     # The Boeing Company
  "CAT",    # Caterpillar Inc.
  "GE",     # General Electric
  "LMT",    # Lockheed Martin Corporation
  "BSX",    # Boston Dynamics (note: primarily private; this may not be directly tradable)
 
  # === Real Estate/REITs ===
  "AMT",    # American Tower Corporation
  "PLD",    # Prologis Inc.
  "CCI",    # Crown Castle Inc.
 
  # === Large-cap ETFs ===
  "SPY",    # SPDR S&P 500 ETF Trust
  "QQQ",    # Invesco QQQ Trust (Nasdaq-100)
  "VTI",    # Vanguard Total Stock Market ETF
  "IWM",    # iShares Russell 2000 ETF (Small-cap)
  "DIA",    # SPDR Dow Jones Industrial Average ETF
 
  # === Sector ETFs ===
  "XLK",    # Technology Select Sector SPDR ETF
  "XLF",    # Financial Select Sector SPDR ETF
  "XLV",    # Healthcare Select Sector SPDR ETF
  "XLE",    # Energy Select Sector SPDR ETF
  "XLI",    # Industrials Select Sector SPDR ETF
  "XLY",    # Consumer Discretionary Select Sector SPDR ETF
  "XLP",    # Consumer Staples Select Sector SPDR ETF
  "XLRE",   # Real Estate Select Sector SPDR ETF
  "XLU",    # Utilities Select Sector SPDR ETF
 
  # === Thematic ETFs ===
  "ARKK",   # ARK Innovation ETF
  "ICLN",   # iShares Global Clean Energy ETF
  "FINX",   # Global FinTech ETF
  "HACK",   # ETFMG Prime Cyber Security ETF
  "ROBO",   # ROBO Global Robotics & Automation ETF
 
  # === Gold/Precious Metals ===
  "GLD",    # SPDR Gold Shares ETF
  "IAU",    # iShares Gold Trust
  "GDXJ",   # VanEck Junior Gold Miners ETF
  "SLV",    # iShares Silver Trust
  "PPLT",   # Aberdeen Standard Physical Platinum Shares ETF
 
  # === Commodities ===
  "DBC",    # Invesco DB Commodity Index Tracking Fund
  "USO",    # United States Oil Fund LP
  "UNG",    # United States Natural Gas Fund LP
 
  # === Bonds/Fixed Income ===
  "TLT",    # iShares 20+ Year Treasury Bond ETF
  "IEF",    # iShares 7-10 Year Treasury Bond ETF
  "LQD",    # iShares Investment Grade Corporate Bond ETF
  "HYG",    # iShares High Yield Corporate Bond ETF
  "TIP",    # iShares TIPS Bond ETF (inflation-protected)
 
  # === International Markets ===
  "VEA",    # Vanguard FTSE Developed Markets ETF
  "VWO",    # Vanguard FTSE Emerging Markets ETF
  "EFA",    # iShares MSCI EAFE ETF
  "FXI",    # iShares China Large-Cap ETF
  "EWJ",    # iShares MSCI Japan ETF
 
  # === Cryptocurrency-related ===
  "COIN",   # Coinbase Global Inc.
  "MSTR",   # MicroStrategy Inc. (Bitcoin exposure)
]
 
WINDOW = 30  # 29 windows + 1
 
 
def fetch_returns_table(start: str, end: str) -> pd.DataFrame:
  """
  fetch daily returns of all stocks, creating trading days x stocks table。
  create window buffer
  """
  buffer_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")

  all_returns = {}
  for sym in SYMBOLS:
    try:
      df = yf.Ticker(sym).history(start=buffer_start, end=end, interval="1d", auto_adjust=True)
      if df.empty:
        print(f"  ⏭ {sym}: no data")
        continue
      ret = df["Close"].pct_change()
      ret.index = ret.index.tz_localize(None)
      all_returns[sym] = ret
      print(f"  ✅ {sym} ({len(df)} days)")
    except Exception as e:
        print(f"  ❌ {sym}: {e}")
    time.sleep(0.15)

  
  returns_df = pd.DataFrame(all_returns).dropna(how="all")
  returns_df = returns_df.fillna(0.0)

  print(f"\n return rate table: {returns_df.shape[0]} trading days x {returns_df.shape[1]} stocks")
  return returns_df
 
 
def build_matrix_samples(returns_df: pd.DataFrame, start: str, end: str, output_dir: str):
  """
  creating samples

  each sample = one trading day (num_stocks x WINDOW) matrix
  - day1~day29: The returns of each stock in the previous 29 days
  - day30: The return rates of each stock on the day (labels)

  save:
    samples.npy  — (num_samples, num_stocks, 30)
    dates.csv    — The trading day corresponding to each sample
    symbols.csv  — Stock sequence (row index)
  """
  start_dt = pd.Timestamp(start)
  end_dt = pd.Timestamp(end)

  dates = returns_df.index
  values = returns_df.values  # (total_days, num_stocks)

  samples = []
  sample_dates = []

  for i in range(WINDOW - 1, len(values)):
      d = dates[i]
      if d < start_dt or d > end_dt:
          continue
      # window shape: (30, num_stocks) → transpose (num_stocks, 30)
      window = values[i - WINDOW + 1 : i + 1, :]
      samples.append(window.T)
      sample_dates.append(str(d.date()))

  samples = np.array(samples, dtype=np.float32)  # (num_samples, num_stocks, WINDOW)
  print(f"Sample: {samples.shape}  →  ({len(sample_dates)} days, {returns_df.shape[1]} stocks, {WINDOW} window)")

  os.makedirs(output_dir, exist_ok=True)
  np.save(os.path.join(output_dir, "samples.npy"), samples)
  pd.DataFrame({"date": sample_dates}).to_csv(os.path.join(output_dir, "dates.csv"), index=False)
  pd.DataFrame({"symbol": returns_df.columns.tolist()}).to_csv(os.path.join(output_dir, "symbols.csv"), index=False)
  print(f"✅ save to {output_dir}/")

 
def main():
  train_start = input("training starts (YYYY-MM-DD): ").strip()
  train_end   = input("training ends (YYYY-MM-DD): ").strip()

  te = datetime.strptime(train_end, "%Y-%m-%d")
  val_start  = (te + timedelta(days=1)).strftime("%Y-%m-%d")
  val_end    = (te + timedelta(days=180)).strftime("%Y-%m-%d") # 400 180
  test_start = (te + timedelta(days=181)).strftime("%Y-%m-%d")
  test_end   =  datetime.today().strftime("%Y-%m-%d") # (te + timedelta(days=700)).strftime("%Y-%m-%d") #
 
  splits = [
      ("train", train_start, train_end),
      ("val",   val_start,   val_end),
      ("test",  test_start,  test_end),
  ]

  print(f"\n📅 Train: {train_start} ~ {train_end}")
  print(f"📅 Val:   {val_start} ~ {val_end}")
  print(f"📅 Test:  {test_start} ~ {test_end}")
  
  base = os.path.dirname(os.path.abspath(__file__))
  for name, s, e in splits:
      print(f"\n{'='*40}\n⬇ {name}\n{'='*40}")
      returns_df = fetch_returns_table(s, e)
      build_matrix_samples(returns_df, s, e, os.path.join(base, "data", name))

  print("\n🎉 Done！")
  print("Each data/{split}/ includes:")
  print("  samples.npy  — shape (num_samples, num_stocks, 30)")
  print("  dates.csv    — The date corresponding to each sample")
  print("  symbols.csv  — Stock sequence (row index)")
 
 
if __name__ == "__main__":
  main()