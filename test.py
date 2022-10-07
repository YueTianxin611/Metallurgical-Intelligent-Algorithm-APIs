import requests

# 调用接口下载数据
url = 'http://127.0.0.1:9111/DecisionTree_classifier_operate/DecisionTree_classifier'
files = {'uplode_csv': open(r'C:\Users\tianxinxin\Desktop\yj\GaussianNB\data.csv', 'rb')}
params = {'target_name':'target'}
res = requests.post(url, params = params,files = files)
soln = res.json()
print(soln)
