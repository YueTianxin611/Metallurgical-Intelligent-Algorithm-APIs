# -*- coding: utf-8 -*-
"""
Created on Tue nov 12 14:14:06 2019

@author: YTX

"""
import os
import logging
from flask_restplus import Api, Resource, reqparse
from werkzeug.middleware.proxy_fix import ProxyFix
from flask import Flask,jsonify
from flask_cors import CORS
import pandas as pd
from werkzeug.datastructures import FileStorage
from PCA import pca_cal
import numpy as np

# 创建一个logger格式
formatter=logging.Formatter('-%(asctime)s - %(levelname)s - %(message)s')
loggername = 'PCA_log'
logPath = os.path.join('/file_and_log', loggername+'.log')
logger = logging.getLogger(loggername)
logger.setLevel(logging.INFO)
fileHandler = logging.FileHandler(loggername+'.log', encoding='utf8')
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

# 限定模型文件后缀
ALLOWED_EXTENSIONS = set(['csv'])


# 模型后缀检查函数
def allowed_files(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[-1] in ALLOWED_EXTENSIONS


# 简历api参数
app = Flask(__name__)
CORS(app, supports_credentials=True)  #解决前端请求跨域问题******
# 支持swagger
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app,version='1.0',title='Sklearn PCA API',
          description='PCA主成分分析')

ns_iba = api.namespace('PCA_operate', description='提供csv文件进行PCA主成分分析')

#健康检测接口
@app.route('/healthz')
def healthz():
    return "OK"


# 配置参数装饰器
PCA_parser = reqparse.RequestParser()
PCA_parser.add_argument('upload_csv', location='files', type=FileStorage, required=True, help='上传数据表')
PCA_parser.add_argument('n_components', location='args', type=str, help='们指定希望PCA降维后的特征维度数目')
PCA_parser.add_argument('copy', location='args', type=bool, help='表示是否在运行算法时，将原始数据复制一份')
PCA_parser.add_argument('whiten', location='args', type=bool, help='白化，就是对降维后的数据的每个特征进行标准化，让方差都为1')
PCA_parser.add_argument('svd_solver', location='args', type=str, help='即指定奇异值分解SVD的方法，由于特征分解是奇异值分解SVD的一个特例，一般的PCA库都是基于SVD实现的。有4个可以选择的值：{‘auto’, ‘full’, ‘arpack’, ‘randomized’}')
PCA_parser.add_argument('tol', location='args', type=float, help='奇异值的公差，svd_solver = ‘arpack’时使用')
PCA_parser.add_argument('iterated_power', location='args', type=str, help='幂方法迭代次数，svd_solver ="randomized"时使用')
@ns_iba.route('/PCA')
@ns_iba.expect(PCA_parser)

# 接口运行函数配置
class PCA(Resource):

    # 通过post 上传、处理文件并返回json文件
    def post(self):
        '''
        文件上传接口，上传后将返回分类好的json文件。
        输入：待分析csv文件
        输出：PCA处理后的json文件
        '''

        # 取参数字典
        args = PCA_parser.parse_args()
        soln = {}
        if 'upload_csv' not in args or 'n_components' not in args \
                or 'copy' not in args or 'whiten' not in args \
                or 'svd_solver' not in args or 'tol' not in args or 'iterated_power' not in args:
                logger.error('No zip argument in paras')
                soln["code"] = 100003
                soln['message'] = "请输入所有参数！"
                soln['data'] = None
                return jsonify(soln)
        # 获取所有参数
        csv_file = args['upload_csv']
        n_components = args['n_components']
        copy = args['copy']
        whiten = args['whiten']
        svd_solver = args['svd_solver']
        tol = args['tol']
        iterated_power = args['iterated_power']

        soln = {}
        # 若输入的参数格式有误，返回json信息
        # 若上传了非csv文件,报错
        if allowed_files(csv_file.filename) == False:
            logger.error(csv_file.filename + ' is not a csv file')
            # 创建返回的字典
            soln['code'] = 100001
            soln['message'] = "请上传正确的文件！"
            soln['data'] = None
            return jsonify(soln)
        else:
            # 若参数值未输入，则用默认参数值
            if n_components=='None' or n_components==None:
                n_components = None
            elif n_components == 'mle':
                n_components = 'mle'
            else:
                try:
                    n_components = float(n_components)
                    if n_components>=1:
                        n_components = int(n_components)
                except:
                    soln['code'] = 100002
                    soln['message'] = "请上传正确的参数n_components"
                    soln['data'] = None
                    return jsonify(soln)

            if copy==None:
                copy = True

            if whiten==None:
                whiten = False


            if svd_solver ==None or svd_solver =='None':
                svd_solver= 'auto'

            elif svd_solver!='auto' and svd_solver!='full' and  svd_solver!='arpack' and svd_solver!='randomized':
                soln['code'] = 100002
                soln['message'] = "请上传正确的参数svd_solver"
                soln['data'] = None
                return jsonify(soln)

            if  tol ==None:
                tol = 0

            if iterated_power ==None or iterated_power =='None':
                iterated_power  = 'auto'
            else:
                try: iterated_power = int(iterated_power)
                except:
                    soln['code'] = 100002
                    soln['message'] = "请上传正确的参数iterated_power"
                    soln['data'] = None

            # 数据处理
            # 取参数字典
            df = pd.read_csv(csv_file)

            # 降维
            res = pca_cal(df, n_components, copy, whiten, svd_solver, tol, iterated_power)
            if type(res) != str:
                # 返回的字典
                soln["code"] = 0
                soln['message'] = "请求成功"
                soln['data'] = {'components_': res[0].tolist(),'explained_variance_':res[1].tolist(),
                                'explained_variance_ratio_':res[2].tolist(),'singular_values_':res[3].tolist(),
                                'mean_':res[4].tolist(),'n_components_':str(res[5]),'noise_variance_':str(res[6])}
                # 将字典转成json格式，返回json文件
                return jsonify(soln)
            else:
                # 返回的字典
                soln["code"] = 100002
                soln['message'] = {"key error": res}
                soln['data'] = None

                # 将字典转成json格式，返回json文件
                return jsonify(soln)


if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=80, debug=True)
    app.run(port=9589, debug=True)
