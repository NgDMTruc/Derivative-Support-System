import pandas as pd
import requests
from datetime import datetime, timedelta, timezone

def get_vn30f(symbol, resolution, start_time=None, now_time=None):
    if start_time:
      start_time = datetime.strptime(start_time, '%Y-%m-%d')
      start_time = int((start_time - timedelta(hours=7)).timestamp())
    else:
        start_time = 0

    if now_time:
      now_time = datetime.strptime(now_time, '%Y-%m-%d')
      now_time = int((now_time - timedelta(hours=7)).timestamp())
    else:
      now_time = 9999999999

    def vn30f():
            return requests.get(f"https://services.entrade.com.vn/chart-api/chart?from={start_time}&resolution={resolution}&symbol={symbol}&to={now_time}").json()
    vn30fm = pd.DataFrame(vn30f()).iloc[:,:6]
    vn30fm['t'] = vn30fm['t'].astype(int).apply(lambda x: datetime.utcfromtimestamp(x) + timedelta(hours = 7))
    vn30fm.columns = ['Date','Open','High','Low','Close','Volume']
    ohlc_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',}
    vn30fm = pd.DataFrame(vn30f()).iloc[:,:6]
    vn30fm['t'] = vn30fm['t'].astype(int).apply(lambda x: datetime.utcfromtimestamp(x) + timedelta(hours = 7))
    vn30fm.columns = ['Date','Open','High','Low','Close','Volume']
    dt_object = datetime.utcfromtimestamp(start_time) + timedelta(hours = 7)
    now_object = datetime.utcfromtimestamp(now_time) + timedelta(hours = 7)

    print(f'===> Data {symbol} from {dt_object} to {now_object} has been appended ')

    return vn30fm

def main():
  start_time = 0
  now_time = '2024-08-31'
  symbol = 'VN30F1M'
  resolution= ['3','1H','1D']
  df_min = get_vn30f(symbol='VN30F1M', resolution=resolution[0], start_time=None, now_time=now_time)
  df_hour = get_vn30f(symbol='VN30F1M', resolution=resolution[1], start_time=None, now_time=now_time)
  df_day = get_vn30f(symbol='VN30F1M', resolution=resolution[2], start_time=None, now_time=now_time)

  df_min.to_csv('vn30f1m_3min.csv', index=False)
  df_hour.to_csv('vn30f1m_1hour.csv', index=False)
  df_day.to_csv('vn30f1m_1day.csv', index=False)

if __name__ == "__main__":
   main()