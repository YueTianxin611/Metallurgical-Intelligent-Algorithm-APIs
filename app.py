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
from GaussianNB_classifier_function import gaussianNB_classifier_cal

# 创建一个logger格式
formatter=logging.Formatter('-%(asctime)s - %(levelname)s - %(message)s')
loggername = 'GaussianNB_classifier_log'
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
api = Api(app,version='1.0',title='Sklearn Naive bayes GaussianNB API',
          description='朴素贝叶斯（非线性）分类算法')

ns_iba = api.namespace('naive_bayes_GaussianNB_operate', description='提供csv文件进行朴素贝叶斯（非线性）分类')

#健康检测接口
@app.route('/healthz')
def healthz():
    return "OK"

# 配置参数装饰器
GaussianNB_classifier_parser = reqparse.RequestParser()
GaussianNB_classifier_parser.add_argument('uplode_csv', location='files',type=FileStorage,required=True,help='上传数据表')
GaussianNB_classifier_parser.add_argument('target_name', location='args', type=str, required=True, help='标签列名')  # 标签
GaussianNB_classifier_parser.add_argument('priors', location='args',type=str,help='先验概率大小，如果没有给定，模型则根据样本数据自己计算（利用极大似然法）,若默认不给出,填None')
GaussianNB_classifier_parser.add_argument('var_smoothing', location='args',type=float,help='所有特征的最大方差部分，用来添加到方差中以提高计算稳定性')

#朴素贝叶斯（非线性）分类算法接口
@ns_iba.route('/GaussianNB_classifier')
@ns_iba.expect(GaussianNB_classifier_parser)

# 接口运行函数配置
class GaussianNB_classifier(Resource):

    # 通过post 上传、处理文件并返回json文件
    def post(self):
        '''
        文件上传接口，上传后将返回已整合的iba数据压缩包。
        输入：待分析csv文件
        输出：朴素贝叶斯（非线性）分类后的json文件
        '''

        # 取参数字典
        args = GaussianNB_classifier_parser.parse_args()

        # 获取所有参数
        csv_file = args['uplode_csv']
        df = pd.read_csv(csv_file)
        # 获取目标类
        Y = df[df.columns[-1]].values
        list_num = []
        for element in Y:
            if (element not in list_num): list_num.append(element)
        class_num = len(list_num)

        target_name = args['target_name']
        priors = args['priors']
        var_smoothing = args['var_smoothing']


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
            if priors != None:
                priors = eval(priors)
                if len(priors)!= class_num:
                    logger.error("参数先验概率输入错误")
                    soln["code"] = 100002
                    soln['message'] = "请上传正确的参数先验概率:array-like, shape (n_classes,)"
                    soln['data'] = None
                    return jsonify(soln)
            if var_smoothing == None:
                var_smoothing = 1e-09
            elif type(var_smoothing)!=float:
                logger.error("参数var_smoothing输入错误")
                soln["code"] = 100002
                soln['message'] = "请上传格式为float的参数！"
                soln['data'] = None
                return jsonify(soln)

            # 数据处理
            # 取参数字典

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
            scores = gaussianNB_classifier_cal(dx,dy,priors,var_smoothing)
            # 返回的字典
            soln["code"] = 0
            soln['message'] = "请求成功"
            soln['data'] = {'mean accuracy': scores}

            # 将字典转成json格式，返回json文件
            return jsonify(soln)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
    #app.run(port=9080, debug=True)
