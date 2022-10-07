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
from NMF import nmf_cal
import numpy as np

# 创建一个logger格式
formatter=logging.Formatter('-%(asctime)s - %(levelname)s - %(message)s')
loggername = 'NMF_log'
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
api = Api(app,version='1.0',title='Sklearn NMF API',
          description='非负矩阵分解')

ns_iba = api.namespace('NMF_operate', description='提供csv文件进行非负矩阵分解')

#健康检测接口
@app.route('/healthz')
def healthz():
    return "OK"


# 配置参数装饰器
NMF_parser = reqparse.RequestParser()
NMF_parser.add_argument('upload_csv', location='files', type=FileStorage, required=True, help='上传数据表')
NMF_parser.add_argument('n_components', location='args', type=int, help='指定希望PCA降维后的特征维度数目')
NMF_parser.add_argument('init', location='args', type=str, help='选择W,H迭代初值的算法，可选None，random，nndsvd，nndsvda，nndsvdar，custom')
NMF_parser.add_argument('solver', location='args', type=str, help='选择数值求解器,可选cd,mu')
NMF_parser.add_argument('beta_loss', location='args', type=str, help='字符串必须位于{frobenius，kullback-leibler，itakura-saito}中。Beta散度应最小化，以测量X与点积WH之间的距离。')
NMF_parser.add_argument('tol', location='args', type=float, help='停止条件的公差。')
NMF_parser.add_argument('max_iter', location='args', type=int, help='超时之前的最大迭代次数。')
NMF_parser.add_argument('alpha', location='args', type=float, help='正则化参数')
NMF_parser.add_argument('l1_ratio', location='args', type=float, help='正则化参数alpha,L1正则化的比例，仅在alpha大于零时有效')
NMF_parser.add_argument('verbose', location='args', type=bool, help='是否冗长')
NMF_parser.add_argument('shuffle', location='args', type=bool, help='是否在CD解算器中随机化坐标顺序。')







@ns_iba.route('/NMF')
@ns_iba.expect(NMF_parser)

# 接口运行函数配置
class NMF(Resource):

    # 通过post 上传、处理文件并返回json文件
    def post(self):
        '''
        文件上传接口，上传后将返回分类好的json文件。
        输入：待分析csv文件
        输出：非负矩阵分解后的json文件
        '''

        # 取参数字典
        args = NMF_parser.parse_args()
        soln = {}

        if 'upload_csv' not in args or 'n_components' not in args \
                or 'init' not in args or 'solver' not in args \
                or 'beta_loss' not in args or 'tol' not in args or 'max_iter' not in args \
                or 'alpha' not in args or 'l1_ratio' not in args or 'verbose' not in args \
                or 'shuffle' not in args :
                logger.error('No zip argument in paras')
                soln["code"] = 100003
                soln['message'] = "请输入所有参数！"
                soln['data'] = None
                return jsonify(soln)

        # 获取所有参数
        csv_file = args['upload_csv']
        n_components = args['n_components']
        init= args['init']
        solver = args['solver']
        beta_loss= args['beta_loss']
        tol = args['tol']
        max_iter = args['max_iter']
        alpha = args['alpha']
        l1_ratio = args['l1_ratio']
        verbose = args['verbose']
        shuffle = args['shuffle']

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
            if init != 'None' and init != None and init != 'random' and init != 'nndsvd' and init != 'nndsvda' and init != 'nndsvdar' and init!= 'custom':
                soln['code'] = 100002
                soln['message'] = "请上传正确的参数init"
                soln['data'] = None
                return jsonify(soln)

            if solver == None or solver == 'None':
                solver = 'cd'

            if solver!= 'cd' and solver!= 'mu':
                soln['code'] = 100002
                soln['message'] = "请上传正确的参数solver"
                soln['data'] = None
                return jsonify(soln)

            if beta_loss == None or beta_loss == 'None':
                beta_loss = 'frobenius'
            if beta_loss!= 'frobenius' and beta_loss!= 'kullback-leibler' and beta_loss!='itakura-saito':
                soln['code'] = 100002
                soln['message'] = "请上传正确的参数beta_loss"
                soln['data'] = None
                return jsonify(soln)

            if tol == None or tol =='None':
                tol = 1e-4

            if max_iter == None or max_iter == 'None':
                max_iter = 200

            if alpha == None or alpha == 'None':
                alpha = 0
            if l1_ratio == None or l1_ratio == 'None':
                l1_ratio = 0

            if verbose == None or verbose == 'None':
                verbose = False
            if shuffle == None or shuffle == 'None':
                shuffle = False


            # 数据处理
            # 取参数字典
            df = pd.read_csv(csv_file)

            # 降维
            res = nmf_cal(df,n_components, init, solver, beta_loss, tol, max_iter,
            alpha, l1_ratio, verbose, shuffle)
            if type(res) != str:
                # 返回的字典
                soln["code"] = 0
                soln['message'] = "请求成功"
                soln['data'] = {'components_': res[0].tolist(),'reconstruction_err_':res[1].tolist(),
                                'n_iter_':res[2]}
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
    app.run(port=9892, debug=True)
