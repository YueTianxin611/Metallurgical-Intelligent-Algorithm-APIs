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
from RBM import rbm_cal

# 创建一个logger格式
formatter=logging.Formatter('-%(asctime)s - %(levelname)s - %(message)s')
loggername = 'MLP_classifier_log'
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
api = Api(app,version='1.0',title='Sklearn RBM API',
          description='伯努利受限玻尔兹曼机（RBM）')

ns_iba = api.namespace('MLP_RBM_operate', description='提供csv文件进行伯努利受限玻尔兹曼机分析')

#健康检测接口
@app.route('/healthz')
def healthz():
    return "OK"

# 配置参数装饰器
RBM_parser = reqparse.RequestParser()
RBM_parser.add_argument('upload_csv', location='files', type=FileStorage, required=True, help='上传数据表')
RBM_parser.add_argument('target_name', location='args', type=str, required=True, help='标签列名')  # 标签
RBM_parser.add_argument('n_components', location='args', type=int,help='二进制隐藏单元数量，可选参数')
RBM_parser.add_argument('learning_rate', location='args', type=float,help='学习率，可选参数。更新权重的学习率，强烈建议调整此超参数。合理的值在10 ** [0，-3]范围内')
RBM_parser.add_argument('batch_size', location='args', type=int,help='minibatch数量，可选参数')
RBM_parser.add_argument('n_iter', location='args', type=int,help='迭代次数，训练数据集在训练期间要执行的迭代次数。')
RBM_parser.add_argument('verbose', location='args', type=int,help='冗长程度，可选参数，默认为0')
@ns_iba.route('/RBM')
@ns_iba.expect(RBM_parser)

# 接口运行函数配置
class RBM(Resource):

    # 通过post 上传、处理文件并返回json文件
    def post(self):
        '''
        文件上传接口，上传后将返回分类好的json文件。
        输入：待分析csv文件
        输出：伯努利受限玻尔兹曼机分析后的json文件
        '''

        # 取参数字典
        args = RBM_parser.parse_args()
        soln = {}
        if 'upload_csv' not in args or 'target_name' not in args or 'n_components' not in args or 'learning_rate' not in args or 'batch_size' not in args \
                or 'n_iter' not in args or 'verbose' not in args :
                logger.error('No zip argument in paras')
                soln["code"] = 100003
                soln['message'] = "请输入所有参数！"
                soln['data'] = None
                return jsonify(soln)

        # 获取所有参数
        csv_file = args['upload_csv']
        target_name = args['target_name']
        n_components = args['n_components']
        learning_rate = args['learning_rate']
        batch_size = args['batch_size']
        n_iter = args['n_iter']
        verbose = args['verbose']

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
            if n_components == None or n_components=='None':
                n_components = 256

            if learning_rate ==None or learning_rate =='None':
                learning_rate = 0.1

            if  batch_size ==None or batch_size =='None':
                batch_size = 10

            if n_iter==None:
                n_iter = 10


            if verbose==None:
                verbose = 0


            # 数据处理
            # 取参数字典
            df = pd.read_csv(csv_file)
            # 除去非数字列
            num = []
            for i, col in enumerate(df.iloc[2].values):
                if type(col) == str:
                    num.append(df.columns[i])
            df = df.drop(columns=num, axis=1)

            # 标签
            dx = df.drop(target_name, axis=1)
            dy = df[target_name]
            # 分类
            hidden,visible,components = rbm_cal(dx,dy,n_components,learning_rate,batch_size,n_iter,verbose)
            if type(dict) != str:
                # 返回的字典
                soln["code"] = 0
                soln['message'] = "请求成功"
                soln['data'] = {'rbm.intercept_hidden_': hidden.tolist(),'rbm.intercept_visible_':visible.tolist(),'rbm.components_':components.tolist()}

                # 将字典转成json格式，返回json文件
                return jsonify(soln)
            else:
                # 返回的字典
                soln["code"] = 100002
                soln['message'] = {"key error":dict}
                soln['data'] = None
                return jsonify(soln)


if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=80, debug=True)
    app.run(port=9552, debug=True)
