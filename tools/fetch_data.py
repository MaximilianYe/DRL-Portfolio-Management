import akshare as ak
import pandas as pd
import os
import time
from datetime import date


# set save path
save_path = "data"
os.makedirs(save_path, exist_ok=True)

# set fetching time
today = date.today()
start_date = "20230101"
end_date = today.strftime("%Y%m%d")

# ETF & stock list
etf_list = [
    "159310",  # 芯片产业ETF
    "159819",  # 人工智能ETF
    "588790",  # 科创AIETF
    "159723",  # 科技龙头ETF
    "159740",  # 恒生科技ETF
]

stock_list = [
    "H01211",  # BYD
]

index_list = [
    "000001",  # 上证指数
    "399001",  # 深证成指
]

# fetch ETF data
for code in etf_list:
    print(f"Fetching ETF {code}...")
    df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start_date, end_date=end_date)
    df.to_csv(os.path.join(save_path, f"{code}_etf.csv"), index=False)
    time.sleep(1)  

# fetch stock data
for code in stock_list:
    print(f"Fetching Stock {code}...")
    df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
    df.to_csv(os.path.join(save_path, f"{code}_stock.csv"), index=False)
    time.sleep(1)

# fetch index data
for code in index_list:
    print(f"Fetching Index {code}...")
    df = ak.index_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date)
    df.to_csv(os.path.join(save_path, f"{code}_index.csv"), index=False)
    time.sleep(1)

print(f"📁 Data saved in: {"C:\Users\szyez\etf_data_project\data"}")