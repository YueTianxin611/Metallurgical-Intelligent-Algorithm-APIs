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
from Birch import brc_cal
import numpy as np

# 创建一个logger格式
formatter=logging.Formatter('-%(asctime)s - %(levelname)s - %(message)s')
loggername = 'BIRCH_cluster_log'
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
api = Api(app,version='1.0',title='Sklearn BIRCH cluster API',
          description='BIRCH聚类')

ns_iba = api.namespace('BIRCH_operate', description='提供csv文件进行BIRCH聚类')

#健康检测接口
@app.route('/healthz')
def healthz():
    return "OK"


# 配置参数装饰器
BIRCH_parser = reqparse.RequestParser()
BIRCH_parser.add_argument('upload_csv', location='files', type=FileStorage, required=True, help='上传数据表')
BIRCH_parser.add_argument('threshold', location='args', type=float, help='叶节点每个CF的最大样本半径阈值T')
BIRCH_parser.add_argument('branching_factor', location='args', type=int, help='CF Tree内部节点的最大CF数B')
BIRCH_parser.add_argument('n_clusters', location='args', type=int, help='类别数K，在BIRCH算法是可选的')
BIRCH_parser.add_argument('compute_labels', location='args', type=bool, help='是否标示类别输出')
BIRCH_parser.add_argument('copy', location='args', type=bool, help='是否拷贝数据集')
@ns_iba.route('/BIRCH')
@ns_iba.expect(BIRCH_parser)

# 接口运行函数配置
class BIRCH(Resource):

    # 通过post 上传、处理文件并返回json文件
    def post(self):
        '''
        文件上传接口，上传后将返回分类好的json文件。
        输入：待分析csv文件
        输出：BIRCH聚类后的json文件
        '''

        # 取参数字典
        args = BIRCH_parser.parse_args()
        soln = {}
        if 'branching_factor' not in args or 'n_clusters' not in args \
                or 'threshold' not in args or 'compute_labels' not in args \
                or 'copy' not in args:
                logger.error('No zip argument in paras')
                soln["code"] = 100003
                soln['message'] = "请输入所有参数！"
                soln['data'] = None
                return jsonify(soln)

        # 获取所有参数
        csv_file = args['upload_csv']
        branching_factor = args['branching_factor']
        n_clusters = args['n_clusters']
        threshold = args['threshold']
        compute_labels = args['compute_labels']
        copy = args['copy']

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
            if threshold == None or threshold=='None':
                threshold = 0.5

            if branching_factor ==None or branching_factor =='None':
                branching_factor= 50

            if  n_clusters ==None or n_clusters =='None':
                n_clusters = 3

            if compute_labels ==None or compute_labels =='None':
                compute_labels  = True

            if copy==None:
                copy = True

            # 数据处理
            # 取参数字典
            dx = pd.read_csv(csv_file)

            # 分类
            center,lable,lables = brc_cal(dx,branching_factor,n_clusters,threshold,compute_labels,copy)

            # 返回的字典
            soln["code"] = 0
            soln['message'] = "请求成功"
            soln['data'] = {'subcluster_centers_':center.tolist(),'subcluster_labels_':lable.tolist(),'labels_ ':lables.tolist()}

            # 将字典转成json格式，返回json文件
            return jsonify(soln)


if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=80, debug=True)
    app.run(port=9552, debug=True)
