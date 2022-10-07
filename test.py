import requests

# 调用接口下载数据
url = 'http://127.0.0.1:9021/DecisionTree_regressor_operate/DecisionTree_regressor'
files = {'uplode_csv': open(r'C:\Users\tianxinxin\Desktop\积木块开发\积木块开发\boston_house_prices.csv', 'rb')}
params = {'target_name':'MEDV'}
res = requests.post(url, params = params,files = files)
soln = res.json()
print(soln)