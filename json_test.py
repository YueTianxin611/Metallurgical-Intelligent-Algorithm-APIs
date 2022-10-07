import requests
import json

json_data = {
  "data_in":{"fe": [62.4, 60.8, 59.5, 61.5, 60.7, 57.1, 58.06, 56.58, 57.49999999999999, 65.0, 62.52, 62.0, 64.0, 63.6, 57.32, 65.32, 61.5, 63.2, 59.6, 57.0, 57.7, 56.3],
             "si": [4.25, 4.75, 4.9, 3.8, 4.9, 6.4, 5.77, 6.09, 7.5, 1.7, 5.46, 8.8, 4.0, 5.57, 5.83, 4.28, 6.5, 6.0, 5.3, 5.0, 6.5, 5.6],
             "ca": [0.09, 0.02, 0.03, 0.06, 0.08, 0.05, 0.03, 0.065, 0.01, 0.03, 0.06, 0.16, 0.04, 0.24, 0.017, 0.06, 0.0, 0.22, 0.22, 0.0, 0.037, 0.037],
             "mg": [0.15, 0.07, 0.08, 0.08, 0.07, 0.08, 0.066, 0.068, 0.05, 0.05, 0.05, 0.12, 0.03, 0.07, 0.049, 0.04, 0.0, 0.02, 0.02, 0.0, 0.095, 0.095],
             "al": [2.35, 2.25, 3.5, 2.3, 2.3, 1.65, 2.6, 3.07, 3.0, 1.5, 1.54, 0.9, 1.1, 1.63, 1.79, 0.65, 1.8, 1.8, 2.7, 5.5, 3.5, 4.8],
             "p": [0.092, 0.085, 0.12, 0.1, 0.057, 0.045, 0.074, 0.052, 0.06999999999999999, 0.09, 0.05, 0.07, 0.1, 0.05, 0.059, 0.02, 0.06, 0.058, 0.09, 0.06, 0.06, 0.05],
             "s": [62.4, 60.8, 59.5, 61.5, 60.7, 57.1, 58.06, 56.58, 57.49999999999999, 65.0, 62.52, 62.0, 64.0, 63.6, 57.32, 65.32, 61.5, 63.2, 59.6, 57.0, 57.7, 56.3],
             "h2o": [7.5, 7.3, 7.2, 9.0, 8.5, 9.5, 7.17, 8.57, 8.0, 8.5, 3.6, 1.5, 8.0, 1.4, 9.9, 4.6, 8.7, 2.8, 12.0, 9.0, 12.0, 12.0],
             "loi": [4.0, 5.45, 5.8, 5.3, 4.6, 10.0, 8.69, 8.82, 6.7, 2.8, 2.92, -0.75, 3.58, 0.85, 10.69, 1.25, 2.8, 0.7, 0.7, 0.0, 7.0, 7.0], "fe_limit": [60.5, 61.5],
             "si_limit": [4.2, 4.8], "ca_limit": [0.0, 0.08], "mg_limit": [0.0, 0.15], "al_limit": [0.0, 2.6], "p_limit": [0.0, 0.1], "s_limit": [0, 100],
             "h2o_limit": [0.0, 8.0], "loi_limit": [0.0, 6.0], "species": 22,
             "powder": [[0, 0.2], [0, 0.2], [0, 0.2], [0, 0.2], [0, 0.5], [0, 0.2], [0, 0.3], [0, 0.2], [0, 0.4], [0, 0.4], [0, 0.2], [0.08, 0.25], [0, 0.2], [0, 0.2], [0, 0.2], [0, 0.2], [0, 0.2], [0.03, 0.2], [0, 0.2], [0, 0.2], [0, 0.2], [0, 0.2]],
             "price_on": 540, "aim": [61.5, 2.3, 3.8, 0.1], "deduction_price_all_1": [8.1, 16.94, 3.68],
             "deduction_price": [9.572784, 0, 24.300144, 31.663824, 0.736368, 0.736368, 14.72736, 0, 13.990992], "price": [548.0, 505.0, 475.0, 540.0, 488.0, 477.0, 387.0, 345.0, 350.0, 663.0, 543.0, 535.0, 625.0, 598.0, 445.0, 585.0, 610.4347826086956, 557.5304347826088, 587.0347826086957, 406.9565217391305, 390.0, 345.0]}
}

API_URL = 'http://10.75.4.20:8989/ore_blending/upload_data'  # API的请求地址

r = requests.post(API_URL, json=json_data)
result = json.loads(r.text)
print(result)