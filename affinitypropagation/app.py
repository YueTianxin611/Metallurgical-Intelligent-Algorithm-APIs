# -*- coding: utf-8 -*-
"""
Created on Tue nov 5 14:40:06 2019

@author: YTX

"""
import os
import logging
from flask_restplus import Api, Resource, reqparse
from werkzeug.middleware.proxy_fix import ProxyFix
from flask import Flask,jsonify
from flask_cors import CORS
from affinitypropagation_function import affinitypropagation_cal
import pandas as pd
from werkzeug.datastructures import FileStorage

# 创建一个logger格式
formatter=logging.Formatter('-%(asctime)s - %(levelname)s - %(message)s')
loggername = 'affinitypropagation_log'
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
api = Api(app,version='1.0',title='Sklearn AffinityPropagation API',
          description='Affinity Propagation算法')

ns_iba = api.namespace('AffinityPropagation_operate', description='提供csv文件进行Affinity Propagation算法')

#健康检测接口
@app.route('/healthz')
def healthz():
    return "OK"

# 配置参数装饰器
AffinityPropagation_cluster_parser = reqparse.RequestParser()
AffinityPropagation_cluster_parser.add_argument('uplode_csv', location='files',type=FileStorage,required=True,help='上传数据表')
AffinityPropagation_cluster_parser.add_argument('damping', location='args',type=float,help='阻尼系数')
AffinityPropagation_cluster_parser.add_argument('max_iter', location='args',type=int,help='最大迭代次数')
AffinityPropagation_cluster_parser.add_argument('convergence_iter', location='args',type=int,help='在不改变估计的集群数量的情况下，停止收敛的迭代次数')
AffinityPropagation_cluster_parser.add_argument('copy', location='args',type=bool,help='是否复制输入数据')
AffinityPropagation_cluster_parser.add_argument('preference', location='args',type= float,help='偏好')
AffinityPropagation_cluster_parser.add_argument('affinity', location='args',type=str,help='{‘precomputed’, ‘euclidean’}目前支持计算预欧几里得距离。 即点之间的负平方欧氏距离')
AffinityPropagation_cluster_parser.add_argument('verbose', location='args',type=bool,help='是否输出详细信息')


#AffinityPropagation聚类接口
@ns_iba.route('/AffinityPropagation_cluster')
@ns_iba.expect(AffinityPropagation_cluster_parser)


#接口运行函数配置
class AffinityPropagation_cluster(Resource):
    
    # 通过post 上传、处理文件并返回json文件
    def post(self):
        '''
        文件上传接口，上传后将返回已整合的iba数据压缩包。
        输入：待分析csv文件
        输出：用AffinityPropagation聚类后的json文件
        '''

        # 取参数字典
        args = AffinityPropagation_cluster_parser.parse_args()
        soln = {}
        if 'uplode_csv' not in args or 'damping' not in args or 'max_iter' not in args or 'convergence_iter' not in args \
                or 'copy' not in args or 'preference' not in args or 'affinity' not in args or 'verbose' not in args :
            logger.error('No zip argument in paras')
            soln["code"] = 100003
            soln['message'] = "请输入所有参数！"
            soln['data'] = None
            return jsonify(soln)

        # 获取所有参数
        csv_file = args['uplode_csv']
        damping = args['damping']
        max_iter = args['max_iter']
        convergence_iter = args['convergence_iter']
        copy = args['copy']
        preference = args['preference']
        affinity = args['affinity']
        verbose = args['verbose']
        soln = {}
        # 若输入的参数格式有误，返回json信息
        # 若上传了非csv文件,报错
        if allowed_files(csv_file.filename) == False:
            logger.error(csv_file.filename+' is not a csv file')
            # 创建返回的字典

            soln["code"] = 100001
            soln['message'] = "请上传正确的文件！"
            soln['data'] = None
            return jsonify(soln)
        else:
            #若参数值未输入，则用默认参数值
            if damping == None:
                damping = 0.5
            elif damping < 0.5 or damping > 1:
                logger.error("阻尼系数输入错误")
                soln["code"] = 100002
                soln['message'] = "请上传正确的参数！"
                soln['data'] = None
                return jsonify(soln)
            if max_iter == None:
                max_iter = 200
            elif max_iter <= 0:
                logger.error("最大迭代次数输入错误")
                soln["code"] = 100002
                soln['message'] = "请上传正确的参数！"
                soln['data'] = None
                return jsonify(soln)
            if convergence_iter == None:
                convergence_iter = 15
            elif convergence_iter <= 0:
                logger.error("在停止收敛的估计集群数量上没有变化的迭代次数输入错误")
                soln["code"] = 100002
                soln['message'] = "请上传正确的参数！"
                soln['data'] = None
                return jsonify(soln)
            if copy == None:
                copy = True
            elif type(copy) != bool:
                logger.error("copy输入错误")
                soln["code"] = 100002
                soln['message'] = "请上传正确的参数！"
                soln['data'] = None
                return jsonify(soln)
            if preference == None:
                preference = 100
            elif type(preference) != float:
                logger.error("偏好输入错误")
                soln["code"] = 100002
                soln['message'] = "请上传正确的参数！"
                soln['data'] = None
                return jsonify(soln)
            if  affinity == None:
                affinity = 'euclidean'
            elif affinity != 'euclidean' and affinity != 'precomputed':
                logger.error("采用负的欧几里得距离输入错误")
                soln["code"] = 100002
                soln['message'] = "请上传正确的参数！"
                soln['data'] = None
                return jsonify(soln)
            if verbose == None:
                verbose = False
            elif type(verbose) != bool:
                logger.error("是否输出详细信息输入错误")
                soln["code"] = 100002
                soln['message'] = "请上传正确的参数！"
                soln['data'] = None
                return jsonify(soln)

            #print(damping,max_iter,convergence_iter,copy, preference, affinity,verbose)
            # 数据处理
            df = pd.read_csv(csv_file)
            #聚类
            cluster_centers_indices, cluster_centers, labels, affinity_matrix, n_iter = affinitypropagation_cal(df,damping,max_iter,convergence_iter,copy, preference, affinity,verbose)
            print(cluster_centers_indices, cluster_centers, labels, affinity_matrix, n_iter)
            #返回的字典
            soln["code"] = 0
            soln['message'] = "请求成功"
            soln['data'] ={'cluster_centers_indices': cluster_centers_indices.tolist(),
                           'cluster_centers': cluster_centers.tolist(), 'labels': labels.tolist(),
                           'affinity_matrix': affinity_matrix.tolist(), 'n_iter': int(n_iter)}
            print(soln)
            #将字典转成json格式，返回json文件
            return jsonify(soln)

if __name__ == '__main__':
#    app.run(host='0.0.0.0',port=80, debug=True)
    app.run(port=9876, debug=True)
