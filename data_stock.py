import pandas as pd
import yfinance as yf
import copy


data = pd.read_csv('dow_jones.csv')
total_lenth = len(data)
index = 0
data_new = pd.DataFrame([])

for i in data.Ticker:
    stock = yf.Ticker(i)
    # tep = stock.info
    his = stock.history(period = "10y")
    # tep = yf.Ticker(i).history(period='max')
    # print(len(tep))
    index += 1
    if len(his) < 2519:
        print('{}/{}, {} failed'.format(index, total_lenth, i))
    else:
        tep = (his['High']-his['Low'])/2+his['Low']
        data_new[i] = tep
        print('{}/{}, {} done'.format(index, total_lenth, i))

data_new.to_csv('USA_stock_10y.csv')

# msft = yf.Ticker('AXP')
# Dow30 =  []



# info = msft.info

# hist = msft.history(period = 'max')
# print('pause')