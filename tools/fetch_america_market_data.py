import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time


def get_time_input(prompt="Enter date (format: YYYYMMDD): "):
    while True:
        date_str = input(prompt)
        try:
            # Validate input format
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            # Convert to yfinance-required format (YYYY-MM-DD)
            formatted_date = date_obj.strftime("%Y-%m-%d")
            return formatted_date
        except ValueError:
            print("Invalid date format. Please use YYYYMMDD, e.g., 20200101")


def calculate_date_ranges(train_start, train_end):
    """Calculate date ranges for train, validation, and test sets."""
    
    # Training set range
    train_start_dt = datetime.strptime(train_start, "%Y-%m-%d")
    train_end_dt = datetime.strptime(train_end, "%Y-%m-%d")
    
    # Need extra buffer data for technical indicators
    buffer_start_dt = train_start_dt - timedelta(days=120)
    
    # Validation set: 400 days after training set
    val_start_dt = train_end_dt + timedelta(days=1) 
    val_end_dt = val_start_dt + timedelta(days=400)
    val_buffer_start_dt = val_start_dt - timedelta(days=120)
    
    # Test set: 60 days after validation set
    test_start_dt = val_end_dt + timedelta(days=1)
    test_end_dt = test_start_dt + timedelta(days=60)
    test_buffer_start_dt = test_start_dt - timedelta(days=120)

    return {
        'train': {
            'start': train_start,
            'end': train_end,
            'buffer_start': buffer_start_dt.strftime("%Y-%m-%d"),
            'data_start': train_start,
        },
        'val': {
            'start': val_start_dt.strftime("%Y-%m-%d"),
            'end': val_end_dt.strftime("%Y-%m-%d"),
            'buffer_start': val_buffer_start_dt.strftime("%Y-%m-%d"),
            'data_start': val_start_dt.strftime("%Y-%m-%d"),
        },
        'test': {
            'start': test_start_dt.strftime("%Y-%m-%d"),
            'end': test_end_dt.strftime("%Y-%m-%d"),
            'buffer_start': test_buffer_start_dt.strftime("%Y-%m-%d"),
            'data_start': test_start_dt.strftime("%Y-%m-%d"),
        }
    }


def fetch_raw_data_for_period(period_name, start_date, end_date, buffer_start):
    """Fetch raw data for a specific period."""
    
    print(f"\n=== Fetching Raw Data for {period_name} ===")
    
    if buffer_start is None:
        buffer_start = start_date
    
    print(f"{period_name} period: {start_date} to {end_date}")
    print(f"Fetching data from: {buffer_start} to {end_date}")
    
    # US stock symbols - diversified portfolio
    symbols = {
        # === Internet/Tech ===
        # US Internet Giants
        "GOOG": "Alphabet (Google) Class C",
        "META": "Meta (Facebook)",
        "AMZN": "Amazon",
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        # Cloud/Enterprise Software
        "CRM": "Salesforce",
        "ADBE": "Adobe",
        "NOW": "ServiceNow",
        "WDAY": "Workday",
        "ZM": "Zoom",
        "DDOG": "Datadog",
        "SNOW": "Snowflake",
        "ORCL": "Oracle",
        # Social Media & Digital Platforms
        "UBER": "Uber",
        "LYFT": "Lyft",
        # E-commerce & Fintech
        "SHOP": "Shopify",
        "PYPL": "PayPal",
        # Chinese Internet (US-listed)
        "BABA": "Alibaba",
        "JD": "JD.com",
        "PDD": "Pinduoduo",
        "BIDU": "Baidu",
        "VIPS": "Vipshop",

        # === Semiconductors/Electronics ===
        "SOXX": "iShares Semiconductor ETF",
        "SMH": "VanEck Semiconductor ETF", 
        "SOXL": "3x Long Semiconductor ETF",
        "TSM": "TSMC",
        "INTC": "Intel",
        "QCOM": "Qualcomm",
        "MU": "Micron Technology",
        "AVGO": "Broadcom",
        "TXN": "Texas Instruments",
        
        # === Communications/Telecom ===
        "XLC": "Communication Services ETF",
        "VOX": "Vanguard Communication Services ETF",
        "FCOM": "Fidelity Communication Services ETF",
        "T": "AT&T",
        "VZ": "Verizon",
        "TMUS": "T-Mobile",
        "CMCSA": "Comcast",
        "DIS": "Disney",
        
        # === Agriculture ===
        "VEGI": "iShares Agribusiness ETF",
        "DBA": "Invesco DB Agriculture Fund",
        "MOO": "VanEck Agribusiness ETF",
        "ADM": "Archer Daniels Midland",
        "DE": "John Deere",
        "CF": "CF Industries",
        "MOS": "The Mosaic Company",
        "TSCO": "Tractor Supply Company",
        
        # === EV/New Energy ===
        "TSLA": "Tesla",
        "NIO": "NIO",
        "XPEV": "XPeng",
        "LI": "Li Auto",
        "BYD": "BYD (ADR)",
        
        # === Traditional Financials ===
        "JPM": "JPMorgan Chase",
        "BAC": "Bank of America",
        "WFC": "Wells Fargo",
        "GS": "Goldman Sachs",
        "MS": "Morgan Stanley",
        "V": "Visa",
        "MA": "Mastercard",
        
        # === Healthcare ===
        "JNJ": "Johnson & Johnson",
        "UNH": "UnitedHealth Group",
        "PFE": "Pfizer",
        "MRNA": "Moderna",
        "ABBV": "AbbVie",
        
        # === Consumer Goods ===
        "KO": "Coca-Cola",
        "PEP": "PepsiCo",
        "PG": "Procter & Gamble",
        "WMT": "Walmart",
        "COST": "Costco",
        "MCD": "McDonald's",
        "SBUX": "Starbucks",
        
        # === Energy ===
        "XOM": "Exxon Mobil",
        "CVX": "Chevron",
        "COP": "ConocoPhillips",
        
        # === Industrials ===
        "BA": "Boeing",
        "CAT": "Caterpillar",
        "GE": "General Electric",
        "LMT": "Lockheed Martin",
        "BSX": "Boston Scientific", # Ticker BSX is Boston Scientific
        
        # === Real Estate/REITs ===
        "AMT": "American Tower",
        "PLD": "Prologis",
        "CCI": "Crown Castle",
        
        # === Broad Market ETFs ===
        "SPY": "SPDR S&P 500 ETF",
        "QQQ": "NASDAQ 100 ETF", 
        "VTI": "Vanguard Total Stock Market ETF",
        "IWM": "Russell 2000 ETF",
        "DIA": "Dow Jones Industrial ETF",
        
        # === Sector ETFs ===
        "XLK": "Technology Sector ETF",
        "XLF": "Financial Sector ETF", 
        "XLV": "Health Care Sector ETF",
        "XLE": "Energy Sector ETF",
        "XLI": "Industrial Sector ETF",
        "XLY": "Consumer Discretionary ETF",
        "XLP": "Consumer Staples ETF",
        "XLRE": "Real Estate Sector ETF",
        "XLU": "Utilities Sector ETF",
        
        # === Thematic ETFs ===
        "ARKK": "ARK Innovation ETF",
        "ICLN": "Clean Energy ETF",
        "FINX": "Fintech ETF",
        "HACK": "Cybersecurity ETF",
        "ROBO": "Robotics ETF",
        
        # === Gold/Precious Metals ===
        "GLD": "SPDR Gold ETF",
        "IAU": "iShares Gold ETF", 
        "GDXJ": "Junior Gold Miners ETF",
        "SLV": "Silver ETF",
        "PPLT": "Platinum ETF",
        
        # === Commodities ===
        "DBC": "Invesco DB Commodity ETF",
        "USO": "United States Oil Fund",
        "UNG": "United States Natural Gas Fund",
        
        # === Bonds/Fixed Income ===
        "TLT": "20+ Year Treasury Bond ETF",
        "IEF": "7-10 Year Treasury Bond ETF",
        "LQD": "Investment Grade Corporate Bond ETF",
        "HYG": "High Yield Corporate Bond ETF",
        "TIP": "TIPS Bond ETF",
        
        # === International Markets ===
        "VEA": "Developed Markets ETF",
        "VWO": "Emerging Markets ETF",
        "EFA": "MSCI EAFE ETF",
        "FXI": "China Large-Cap ETF",
        "EWJ": "Japan ETF",
        
        # === Crypto-related ===
        "COIN": "Coinbase",
        "MSTR": "MicroStrategy (Bitcoin proxy)",
    }
    
    # Remove duplicates from the dict (which resulted from copy-paste in original code)
    symbols = dict(list(symbols.items()))
    print(f"📊 Total pool size: {len(symbols)} stocks & ETFs")

    # Create corresponding directory
    os.makedirs(f'data/raw/{period_name}', exist_ok=True)
    
    successful_downloads = 0
    total_symbols = len(symbols)
    
    for symbol, name in symbols.items():
        print(f"\nFetching {symbol} ({name}) raw {period_name} data...")
        
        try:
            # Fetch data using yfinance
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=buffer_start,
                end=end_date,
                interval="1d",
                auto_adjust=True,  # Auto-adjust prices (splits/dividends)
                back_adjust=True
            )
            
            if df.empty:
                print(f"  ❌ {symbol} No data fetched")
                continue
            
            # Reset index, make Date a column
            df.reset_index(inplace=True)
            
            # Standardize column names
            df = standardize_column_names_yfinance(df)
            
            # Calculate daily return
            df['daily_return'] = df['close'].pct_change().fillna(0)
            
            # Ensure correct date format
            df['date'] = pd.to_datetime(df['date'])
            
            print(f"  ✅ Fetched {len(df)} rows")
            print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
            print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            
            # Save raw data
            filename = f"{symbol}_stock_raw.csv"
            df.to_csv(f"data/raw/{period_name}/{filename}", index=False)
            print(f"  Saved to: data/raw/{period_name}/{filename}")
            
            successful_downloads += 1
            
        except Exception as e:
            print(f"  ❌ Failed to fetch {symbol}: {str(e)}")
            continue
        
        # Avoid rapid requests
        time.sleep(0.2)
    
    print(f"\n{period_name} data fetch complete: {successful_downloads}/{total_symbols} successful")
    

def standardize_column_names_yfinance(df):
    """Standardize yfinance column names"""
    column_mapping = {
        'Date': 'date',
        'Open': 'open', 
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
        'Dividends': 'dividends',
        'Stock Splits': 'stock_splits'
    }

    df = df.rename(columns=column_mapping)
    
    # Handle timezone issues
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    return df


def process_data_by_rows(raw_data_folder, processed_data_folder, data_start_date, 
                         rsi_window=14, bb_window=20):
    """Process data by row count( just upgraded for new features)"""
    
    print(f"\n=== Processing Data (Row-based) ===")
    print(f"Data start date: {data_start_date}")
    print(f"Raw data folder: {raw_data_folder}")
    print(f"Processed data folder: {processed_data_folder}")
    
    os.makedirs(processed_data_folder, exist_ok=True)
    
    # Check if folder exists
    if not os.path.exists(raw_data_folder):
        print(f"❌ Raw data folder not found: {raw_data_folder}")
        return
        
    # --- Step 1: Pre-calculate macro and cross-sectional features ---
    macro_features_df = calculate_macro_features(raw_data_folder)
    percentile_df = calculate_cross_sectional_features(raw_data_folder)
    
    max_required_rows = max(rsi_window, bb_window, 60, 50, 200) # 200 is for market_uptrend
    print(f"Max historical rows needed: {max_required_rows}")
    
    csv_files = [f for f in os.listdir(raw_data_folder) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"❌ No CSV files found in {raw_data_folder}")
        return
    
    # --- Step 2: Loop through files and merge new features ---
    for csv_file in csv_files:
        print(f"\nProcessing file: {csv_file}")
        
        try:
            # Read raw data
            df = pd.read_csv(os.path.join(raw_data_folder, csv_file))
            df['date'] = pd.to_datetime(df['date'])
            
            # Extract ticker from filename
            stock_name = csv_file.replace('_stock_raw.csv', '')
            df['ticker'] = stock_name
            
            print(f" Raw data: {len(df)} rows")
            
            # Calculate all technical indicators
            df = calculate_technical_indicators(df, rsi_window, bb_window)
            
            # === Merge new features ===
            df = add_new_features_to_df(df, macro_features_df, percentile_df)

            # Find row index for data_start_date
            data_start_dt = pd.to_datetime(data_start_date)
            
            if data_start_dt in df['date'].values:
                data_start_idx = df[df['date'] == data_start_dt].index[0]
            else:
                # If start date is not a trading day, find the next one
                future_dates = df[df['date'] > data_start_dt]
                if len(future_dates) > 0:
                    data_start_idx = future_dates.index[0]
                    actual_date = df.loc[data_start_idx, 'date'].strftime('%Y-%m-%d')
                    print(f" {data_start_date} is not a trading day. Using next trading day {actual_date}")
                else:
                    print(f" Error: Could not find data start date")
                    continue
            
            if data_start_idx < max_required_rows:
                print(f" Warning: Insufficient historical data! Need {max_required_rows}, have {data_start_idx}")
            
            # Slice data
            processed_df = df.iloc[data_start_idx:].copy().reset_index(drop=True)
            
            print(f" Sliced data: {len(processed_df)} rows")
            
            # Save processed data
            output_file = csv_file.replace('_raw.csv', '_processed.csv')
            processed_df.to_csv(os.path.join(processed_data_folder, output_file), index=False)
            print(f" Saved to: {processed_data_folder}/{output_file}")
            
        except Exception as e:
            print(f" ❌ Failed to process file {csv_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue


def calculate_technical_indicators(df, rsi_window=14, bb_window=20, base_date="2021-01-01"):
    """Calculate technical indicators"""
    
    # Calculate RSI
    df['RSI'] = calculate_rsi(change=df['daily_return'], period=rsi_window)
    
    # Calculate Bollinger Bands
    df['bb_pos'], df['bb_width'] = calculate_bb(prices=df['close'], bb=bb_window)
    
    # Calculate volume change
    df['volume_change'] = df['volume'].pct_change().fillna(0)
    
    # Close price features
    df['ma_50'] = df['close'].rolling(50).mean()
    df['price_percentile_60'] = df['close'].rolling(window=60).rank(pct=True)
    df['price_vs_ma50'] = df['close'] / df['ma_50'] - 1
    df['c_10d_trend'] = calculate_trend(series=df['close'], window=10)
    df['3d_P'] = calculate_3d_momentum_pct(price_series=df['close'])
    
    # Volume change features
    df['vc_5d_mean'] = calculate_5d_mean(df=df['volume_change'], window=5)
    df['vc_5d_trend'] = calculate_trend(series=df['volume_change'], window=5)
    
    # RSI features
    ori_rsi = df['RSI'] * 50 + 50
    df['rsi_50'] = calculate_rsi(change=df['daily_return'], period=50)
    df['rsi_5d_trend'] = calculate_trend(series=ori_rsi, window=5)
    df['overbought'] = calculate_overbought(original_rsi=ori_rsi)
    df['oversold'] = calculate_oversold(original_rsi=ori_rsi)
    
    # Bollinger Bands features
    df['bb_5d_mean'] = calculate_5d_mean(df=df['bb_pos'], window=5)
    df['bb_breakout_up'] = (df['bb_pos'] > 1.0).astype(int)
    df['bb_breakout_down'] = (df['bb_pos'] < 0.0).astype(int)
    
    # Amplitude features
    # Amplitude features (Fixed Future Warning)
    df['amplitude'] = (df['high'] - df['low']) / df['close'].shift(1)
    # Check if DataFrame is empty
    if not df.empty:
        first_day_amplitude = (df['high'].iloc[0] - df['low'].iloc[0]) / df['open'].iloc[0] if df['open'].iloc[0] != 0 else 0
        df['amplitude'] = df['amplitude'].fillna(first_day_amplitude) # <--- Fix 1
    else:
        df['amplitude'] = 0 # or other default
    
    df['amp_5d_mean'] = calculate_5d_mean(df=df['amplitude'], window=5)
    df['amp_percentile'] = df['amplitude'].rolling(window=20).rank(pct=True)
    
    # 6. momentum_5d - 5-day momentum
    df['momentum_5d'] = df['daily_return'].rolling(5).sum().fillna(0.0)
    
    df['standard_price'] = calculate_standard_price(df, base_date)

    # === Added individual stock volatility ===
    df['volatility_22d'] = df['daily_return'].rolling(window=22).std() * np.sqrt(252) # <--- Fix 2

    return df

def calculate_standard_price(df, base_date="2020-01-01"):
    """Calculate standardized price index from a base date"""
    
    # Convert base date to datetime
    base_dt = pd.to_datetime(base_date)
    
    # Find the position of the base date in the data
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['date'])
    
    # Find the first valid date on or after the base date
    valid_dates = df_copy[df_copy['date'] >= base_dt]
    
    if len(valid_dates) == 0:
        # If base date is after all data, calculate from day one
        print(f"⚠️ Base date {base_date} is after data range. Calculating from first day.")
        standard_price = (1 + df_copy['daily_return']).cumprod()
        return standard_price
    
    elif valid_dates.index[0] == 0:
        # Base date is the first day or earlier
        standard_price = (1 + df_copy['daily_return']).cumprod()
        return standard_price
    
    else:
        # Base date is somewhere in the middle of the data
        base_idx = valid_dates.index[0]
        print(f"📍 Standardized price base: {df_copy.loc[base_idx, 'date'].date()} (Row {base_idx+1})")
        
        # Create standard price series
        standard_price = pd.Series(index=df_copy.index, dtype=float)
        
        # Set as NaN before base date or use forward calculation
        if base_idx > 0:
            # Forward calculate price before base date
            returns_before = df_copy['daily_return'].iloc[:base_idx+1]
            # Cumulatively calculate backwards from base point
            cumulative_before = (1 + returns_before[::-1]).cumprod()[::-1]
            # Standardize, making the base date 1.0
            cumulative_before = cumulative_before / cumulative_before.iloc[-1]
            standard_price.iloc[:base_idx+1] = cumulative_before
        else:
            standard_price.iloc[0] = 1.0
        
        # Calculate normally after base date
        if base_idx < len(df_copy) - 1:
            returns_after = df_copy['daily_return'].iloc[base_idx+1:]
            cumulative_after = (1 + returns_after).cumprod()
            standard_price.iloc[base_idx+1:] = cumulative_after
        
        return standard_price

# for technical indicators
def calculate_rsi(change, period=14):
    gains = change.where(change > 0, 0)
    losses = -change.where(change < 0, 0)
    
    avg_gains = gains.rolling(window=period).mean()
    avg_losses = losses.rolling(window=period).mean()
    
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    rsi = rsi.fillna(50)
    rsi_normalized = (rsi - 50) / 50
    return rsi_normalized


def calculate_bb(prices, bb):
    """Calculate Bollinger Bands"""
    mid = prices.rolling(window=bb).mean()
    std = prices.rolling(window=bb).std()
    
    upp = mid + (2 * std)
    low = mid - (2 * std)
    
    width = (upp - low) / mid
    bb_position = (prices - low) / (upp - low)
    bb_position = bb_position.fillna(0.5)
    return bb_position, width


def calculate_5d_mean(df, window):
    return df.rolling(window=window).mean()


def calculate_trend(series, window):
    def slope_func(x):
        if len(x) < 2:
            return 0
        try:
            time_x = np.arange(len(x))
            slope, _, _, _, _ = stats.linregress(time_x, x)
            avg_price = np.mean(x)
            return slope / avg_price if avg_price != 0 else 0
        except:
            return 0
    
    return series.rolling(window=window).apply(slope_func, raw=True)


def calculate_3d_momentum_pct(price_series, window=3):
    return price_series.pct_change(periods=window)


def calculate_overbought(original_rsi, threshold=70, window=20):
    overbought_mask = original_rsi > threshold
    overbought_days = overbought_mask.rolling(window=window).mean()
    return overbought_days.fillna(0)


def calculate_oversold(original_rsi, threshold=30, window=20):
    oversold_mask = original_rsi < threshold
    oversold_days = oversold_mask.rolling(window=window).mean()
    return oversold_days.fillna(0)



def calculate_macro_features(raw_data_folder, rsi_window=14, bb_window=20):
    """
    Calculate macro market features (Market Weather)
    Uses SPY as the market benchmark
    """
    spy_file = os.path.join(raw_data_folder, 'SPY_stock_raw.csv')
    if not os.path.exists(spy_file):
        print("⚠️ Warning: SPY benchmark file not found. Skipping macro feature calculation.")
        return None

    print("📈 Calculating macro features (based on SPY)...")
    market_df = pd.read_csv(spy_file)
    market_df['date'] = pd.to_datetime(market_df['date'])
    
    # --- market_uptrend ---
    market_ma200 = market_df['close'].rolling(window=200).mean()
    market_df['market_uptrend'] = (market_df['close'] > market_ma200).astype(float)
    
    # --- market_volatility (using ATR) ---
    high_low = market_df['high'] - market_df['low']
    high_close = np.abs(market_df['high'] - market_df['close'].shift())
    low_close = np.abs(market_df['low'] - market_df['close'].shift())
    tr = np.max([high_low, high_close, low_close], axis=0)
    
    atr = pd.Series(tr).rolling(window=14).mean()
    market_df['market_volatility'] = (atr / market_df['close']).fillna(0)
    
    # Return only required columns
    macro_features_df = market_df[['date', 'market_uptrend', 'market_volatility']].copy()
    macro_features_df.set_index('date', inplace=True)
    return macro_features_df


def calculate_cross_sectional_features(raw_data_folder):
    """
    Calculate cross-sectional features (Stock Ranking / "Horse Racing")
    """
    print("📊 Calculating cross-sectional features (momentum percentile)...")
    csv_files = [f for f in os.listdir(raw_data_folder) if f.endswith('.csv')]
    
    all_returns_10d = {}
    
    for csv_file in csv_files:
        stock_name = csv_file.replace('_stock_raw.csv', '')
        df = pd.read_csv(os.path.join(raw_data_folder, csv_file))
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Calculate 10-day return
        returns_10d = df['close'].pct_change(periods=10)
        all_returns_10d[stock_name] = returns_10d
        
    # Combine into one DataFrame
    returns_df = pd.DataFrame(all_returns_10d)
    
    # Calculate cross-sectional percentile rank
    # axis=1 means ranking all stocks (columns) for each day (row)
    percentile_df = returns_df.rank(axis=1, pct=True, na_option='keep')
    
    return percentile_df

def add_new_features_to_df(df, macro_features_df, percentile_df):
    """
    Merge new macro and cross-sectional features into a single stock DataFrame
    """
    stock_name = df['ticker'].iloc[0] # Assumes 'ticker' column was added
    df_indexed = df.set_index('date')

    # Merge macro features
    if macro_features_df is not None:
        df_indexed = df_indexed.join(macro_features_df)
        # Use direct assignment instead of inplace=True
        df_indexed['market_uptrend'] = df_indexed['market_uptrend'].fillna(method='ffill')
        df_indexed['market_volatility'] = df_indexed['market_volatility'].fillna(method='ffill')
        
    # Merge cross-sectional features
    if percentile_df is not None and stock_name in percentile_df.columns:
        # Add new column momentum_10d_percentile
        df_indexed['momentum_10d_percentile'] = percentile_df[stock_name]
        # Use direct assignment instead of inplace=True
        df_indexed['momentum_10d_percentile'] = df_indexed['momentum_10d_percentile'].fillna(0.5) # Fill NaN with median (0.5)

    return df_indexed.reset_index()




def main():
    print("🚀 Starting US stock data processing")
    print("=" * 50)
    
    try:
        # Get training set time range
        train_start = get_time_input("Enter training set start date (format: YYYYMMDD): ")
        train_end = get_time_input("Enter training set end date (format: YYYYMMDD): ")
        
        # Calculate all data time ranges
        date_ranges = calculate_date_ranges(train_start, train_end)
        
        print("\n📅 Date Range Plan:")
        for period, ranges in date_ranges.items():
            print(f"{period.upper()} set: {ranges['start']} to {ranges['end']}")
        
        # Confirm continuation
        confirm = input("\nContinue? (y/n): ")
        if confirm.lower() != 'y':
            print("Exiting program")
            return
        
        # Dataset types
        periods = ['train', 'val', 'test']
        
        # Step 1: Fetch Raw Data
        print("\n" + "="*50)
        print("Step 1: Fetching Raw Data")
        print("="*50)
        
        for period in periods:
            ranges = date_ranges[period]
            fetch_raw_data_for_period(
                period_name=period,
                start_date=ranges['start'],
                end_date=ranges['end'],
                buffer_start=ranges.get('buffer_start')
            )
            
            print(f"\n⏳ {period} dataset fetch complete. Waiting 2 seconds...")
            time.sleep(2)
        
        # Step 2: Process All Period Data
        print("\n" + "="*50)
        print("Step 2: Processing Data")
        print("="*50)
        
        # Process training set
        ranges = date_ranges['train']
        print(f"\nProcessing training dataset...")
        process_data_by_rows(
            raw_data_folder='data/raw/train',
            processed_data_folder='data/processed/train',
            data_start_date=ranges['data_start']
        )
        print(f"\n⏳ Training set processing complete")
        
       # Process validation and test sets
        for period in ['val', 'test']:
            ranges = date_ranges[period]
            print(f"\nProcessing {period.upper()} dataset...")
            process_data_by_rows(
                raw_data_folder=f'data/raw/{period}',
                processed_data_folder=f'data/processed/{period}',
                data_start_date=ranges['data_start']
            )
            print(f"\n⏳ {period} set processing complete")
        
        print("\n🎉 All data processing complete!")
        print("File structure:")
        print("├── data/raw/train/        # Raw training data")
        print("├── data/raw/val/          # Raw validation data")  
        print("├── data/raw/test/         # Raw test data")
        print("├── data/processed/train/  # Processed training data")
        print("├── data/processed/val/    # Processed validation data")
        print("└── data/processed/test/   # Processed test data")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Program interrupted")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    main()