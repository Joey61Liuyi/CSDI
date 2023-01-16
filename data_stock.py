import pandas as pd
import yfinance as yf

# data = pd.read_csv('dow_jones.csv')
# total_lenth = len(data)
# index = 0
# data_new = pd.DataFrame([])

# for i in data.Ticker:
#     stock = yf.Ticker(i)
#     # tep = stock.info
#     his = stock.history(period = "10y")
#     # tep = yf.Ticker(i).history(period='max')
#     # print(len(tep))
#     index += 1
#     if len(his) < 2519:
#         print('{}/{}, {} failed'.format(index, total_lenth, i))
#     else:
#         tep = (his['High']-his['Low'])/2+his['Low']
#         data_new[i] = tep
#         print('{}/{}, {} done'.format(index, total_lenth, i))
#
# data_new.to_csv('USA_stock_10y.csv')

usa_data = pd.read_csv('USA_stock_10y.csv')
usa_data['Date'] = [i[:10] for i in usa_data['Date']]

hk_data = pd.read_csv('hongkong_stock_10y.csv')
hk_data['Date'] = [i[:10] for i in hk_data['Date']]
eu_data = pd.read_csv('EU_stock_10y.csv')
eu_data['Date'] = [i[:10] for i in eu_data['Date']]

tep = pd.merge(usa_data, hk_data, on='Date', how='outer')
tep = pd.merge(tep, eu_data, on='Date', how='outer')
tep.set_index(['Date'], inplace=True)
tep.sort_index(inplace=True)
tep = tep.fillna(0)
tep.to_csv('stockdata_all.csv')

