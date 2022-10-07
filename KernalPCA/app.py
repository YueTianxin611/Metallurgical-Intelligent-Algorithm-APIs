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
from KernalPCA import kernel_pca_cal
import numpy as np

# 创建一个logger格式
formatter=logging.Formatter('-%(asctime)s - %(levelname)s - %(message)s')
loggername = 'KernalPCA_log'
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
api = Api(app,version='1.0',title='Sklearn KernalPCA API',
          description='核主成分分析')

ns_iba = api.namespace('KernalPCA_operate', description='提供csv文件进行PCA主成分分析')

#健康检测接口
@app.route('/healthz')
def healthz():
    return "OK"


# 配置参数装饰器
KernelPCA_parser = reqparse.RequestParser()
KernelPCA_parser.add_argument('upload_csv', location='files', type=FileStorage, required=True, help='上传数据表')
KernelPCA_parser.add_argument('n_components', location='args', type=int, help='指定希望PCA降维后的特征维度数目')
KernelPCA_parser.add_argument('kernel', location='args', type=str, help='降维的核的类型')
KernelPCA_parser.add_argument('gamma', location='args', type=float, help='rbf，poly和Sigmoid内核的内核系数')
KernelPCA_parser.add_argument('degree', location='args', type=int, help='poly内核的度数')
KernelPCA_parser.add_argument('coef0', location='args', type=float, help='poly和sigmoid内核的独立系数')
KernelPCA_parser.add_argument('kernel_params', location='args', type=str, help='可调用对象传递的内核的参数值')
KernelPCA_parser.add_argument('alpha', location='args', type=int, help='岭回归的超参数，用于逆变换（当fit_inverse_transform = True时）')
KernelPCA_parser.add_argument('fit_inverse_transform', location='args', type=bool, help='是否学习非预计算内核的逆变换')
KernelPCA_parser.add_argument('eigen_solver', location='args', type=str, help='选择要使用的特征求解器。如果n_components远小于训练样本的数量，则arpack可能比密集本征求解器更有效。')
KernelPCA_parser.add_argument('tol', location='args', type=float, help='arpack的收敛容差。如果为0，则arpack将选择最佳值。')
KernelPCA_parser.add_argument('max_iter', location='args', type=int, help='arpack的最大迭代次数。如果为None，则由arpack选择最佳值。')
KernelPCA_parser.add_argument('remove_zero_eig', location='args', type=bool, help='是否删除所有具有零特征值的分量')
KernelPCA_parser.add_argument('copy_X', location='args', type=bool, help='模型是否将输入X复制并存储在X_fit_属性中')
KernelPCA_parser.add_argument('n_jobs', location='args', type=int, help='要运行的并行作业数')



@ns_iba.route('/KernelPCA')
@ns_iba.expect(KernelPCA_parser)

# 接口运行函数配置
class KernelPCA(Resource):

    # 通过post 上传、处理文件并返回json文件
    def post(self):
        '''
        文件上传接口，上传后将返回分类好的json文件。
        输入：待分析csv文件
        输出：KernelPCA处理后的json文件
        '''

        # 取参数字典
        args = KernelPCA_parser.parse_args()
        soln = {}
        if 'upload_csv' not in args or 'n_components' not in args \
                or 'kernel' not in args or 'gamma' not in args \
                or 'degree' not in args or 'coef0' not in args or 'kernel_params' not in args \
                or 'alpha' not in args or 'fit_inverse_transform' not in args or 'eigen_solver' not in args \
                or 'tol' not in args or 'max_iter' not in args or 'remove_zero_eig' not in args \
                or 'copy_X' not in args or 'n_jobs' not in args:
                logger.error('No zip argument in paras')
                soln["code"] = 100003
                soln['message'] = "请输入所有参数！"
                soln['data'] = None
                return jsonify(soln)
        # 获取所有参数
        csv_file = args['upload_csv']
        n_components = args['n_components']
        kernel= args['kernel']
        gamma = args['gamma']
        degree= args['degree']
        coef0 = args['coef0']
        kernel_params = args['kernel_params']
        alpha = args['alpha']
        fit_inverse_transform = args['fit_inverse_transform']
        eigen_solver = args['eigen_solver']
        tol = args['tol']
        max_iter = args['max_iter']
        remove_zero_eig= args['remove_zero_eig']
        copy_X = args['copy_X']
        n_jobs = args['n_jobs']

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
            if kernel == None or kernel == 'None':
                kernel = 'linear'

            elif kernel != 'linear' and kernel != 'poly' and kernel != 'rbf' and kernel != 'sigmoid' and kernel != 'cosine' and kernel!= 'precomputed':
                soln['code'] = 100002
                soln['message'] = "请上传正确的参数kernel"
                soln['data'] = None
                return jsonify(soln)

            if degree==None:
                degree = 3

            if coef0==None:
                coef0 = 1

            if alpha==None:
                alpha = 1

            if fit_inverse_transform == None:
                fit_inverse_transform = False

            if eigen_solver == None or eigen_solver == 'None':
                eigen_solver = 'auto'
            elif eigen_solver != 'auto' and eigen_solver != 'dense' and eigen_solver != 'arpack':
                soln['code'] = 100002
                soln['message'] = "请上传正确的参数eigen_solver"
                soln['data'] = None
                return jsonify(soln)

            if tol == None:
                tol = 0

            if remove_zero_eig == None:
                remove_zero_eig = 0

            if copy_X == None:
                copy_X = True

            # 数据处理
            # 取参数字典
            df = pd.read_csv(csv_file)

            # 降维
            res = kernel_pca_cal(df, n_components, kernel, gamma, degree, coef0, kernel_params,
                alpha,fit_inverse_transform, eigen_solver, tol, max_iter,
                remove_zero_eig, copy_X, n_jobs)
            if type(res) != str:
                # 返回的字典
                soln["code"] = 0
                soln['message'] = "请求成功"
                soln['data'] = {'lambdas_': res[0].tolist(),'alphas_':res[1].tolist(),
                                'X_fit_ ':res[2].tolist()}
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
    app.run(host='0.0.0.0', port=80, debug=True)
    #app.run(port=9459, debug=True)

