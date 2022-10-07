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
from DecisionTreeReg import DecisionTree_reg_cal

# 创建一个logger格式
formatter=logging.Formatter('-%(asctime)s - %(levelname)s - %(message)s')
loggername = 'DecisionTree_classifier_log'
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
api = Api(app,version='1.0',title='Sklearn DecisionTree Regressor API',
          description='决策树回归算法')

ns_iba = api.namespace('DecisionTree_regressor_operate', description='提供csv文件进行决策树回归')

#健康检测接口
@app.route('/healthz')
def healthz():
    return "OK"

# 配置参数装饰器
DecisionTree_regressor_parser = reqparse.RequestParser()
DecisionTree_regressor_parser.add_argument('upload_csv', location='files', type=FileStorage, required=True, help='上传数据表')
DecisionTree_regressor_parser.add_argument('target_name', location='args', type=str, required=True, help='标签列名')  # 标签
DecisionTree_regressor_parser.add_argument('criterion', location='args', type=str, help='特征选择标准，可选参数，可以使用"mse"或者"mae"，前者是均方差，后者是和均值之差的绝对值之和')
DecisionTree_regressor_parser.add_argument('splitter', location='args', type=str, help=' 特征划分点选择标准，可选参数，默认是best，可以设置为random。前者是在所有特征中找最好的切分点 后者是在部分特征中，默认的”best”适合样本量不大的时候，而如果样本数据量非常大，此时决策树构建推荐”random”')
DecisionTree_regressor_parser.add_argument('max_depth', location='args', type=int, help='决策树最大深，可选参数，默认是None。设置决策随机森林中的决策树的最大深度，深度越大，越容易过拟合')
DecisionTree_regressor_parser.add_argument('min_samples_split', location='args', type=float, help='内部节点再划分所需最小样本数，可选参数，默认是2。设置结点的最小样本数量，当样本数量可能小于此值时，结点将不会在划分。')
DecisionTree_regressor_parser.add_argument('min_samples_leaf', location='args', type=float, help='这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝')
DecisionTree_regressor_parser.add_argument('min_weight_fraction_leaf', location='args', type=float, help='叶子节点最小的样本权重和，可选参数，默认是0。这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝。默认是0，就是不考虑权重问题')
DecisionTree_regressor_parser.add_argument('max_features', location='args', type=str, help='划分时考虑的最大特征数，可选参数，默认是None。，其余可选int, float, auto, sqrt, log2')
#DecisionTree_classifier_parser.add_argument('random_state', location='args',type=str,help='可选参数，默认是None。随机数种子。')
DecisionTree_regressor_parser.add_argument('max_leaf_nodes', location='args', type=int, help='最大叶子节点数，可选参数，默认是None。通过限制最大叶子节点数，可以防止过拟合，默认是"None”，即不限制最大的叶子节点数')
DecisionTree_regressor_parser.add_argument('min_impurity_decrease', location='args', type=float, help='可选参数，默认是0。打算划分一个内部结点时，只有当划分后不纯度减少值不小于该参数指定的值，才会对该结点进行划分')
DecisionTree_regressor_parser.add_argument('min_impurity_split', location='args', type=float, help='节点划分最小不纯度,可选参数，默认是1e-7。这个值限制了决策树的增长，如果某节点的不纯度(基尼系数，信息增益，均方差，绝对差)小于这个阈值则该节点不再生成子节点')
#DecisionTree_classifier_parser.add_argument('class_weight', location='args',type=str,help='类别权重，可选参数，默认是None，也可以字典、字典列表、balanced。指定样本各类别的的权重，主要是为了防止训练集某些类别的样本过多导致训练的决策树过于偏向这些类别。这里可以自己指定各个样本的权重，如果使用“balanced”，则算法会自己计算权重，样本量少的类别所对应的样本权重会高')
DecisionTree_regressor_parser.add_argument('presort', location='args', type=bool, help='数据是否预排序，可选参数，默认为False，这个值是布尔值，默认是False不排序。')


#朴素贝叶斯（非线性）分类算法接口
@ns_iba.route('/DecisionTree_regressor')
@ns_iba.expect(DecisionTree_regressor_parser)
# 接口运行函数配置
class GaussianNB_regressor(Resource):

    # 通过post 上传、处理文件并返回json文件
    def post(self):
        '''
        文件上传接口，上传后将返回分类好的json文件。
        输入：待分析csv文件
        输出：决策树回归后的json文件
        '''

        # 取参数字典
        args = DecisionTree_regressor_parser.parse_args()

        soln = {}
        if 'upload_csv' not in args or 'target_name' not in args or 'criterion' not in args or 'splitter' not in args or 'max_depth' not in args \
                or 'min_samples_split' not in args or 'min_samples_leaf' not in args or 'min_weight_fraction_leaf' not in args or 'max_features' not in args \
                or 'max_leaf_nodes' not in args or 'min_impurity_decrease' not in args or 'min_impurity_split' not in args or 'presort' not in args:
            logger.error('No zip argument in paras')
            soln["code"] = 100003
            soln['message'] = "请输入所有参数！"
            soln['data'] = None
            return jsonify(soln)

        # 获取所有参数
        csv_file = args['upload_csv']
        target_name = args['target_name']
        criterion = args['criterion']
        splitter = args['splitter']
        max_depth = args['max_depth']
        min_samples_split = args['min_samples_split']
        min_samples_leaf = args['min_samples_leaf']
        min_weight_fraction_leaf = args['min_weight_fraction_leaf']
        max_features = args['max_features']
        #random_state = args['random_state']
        max_leaf_nodes = args['max_leaf_nodes']
        min_impurity_decrease = args['min_impurity_decrease']
        min_impurity_split = args['min_impurity_split']
        #class_weight = args['class_weight']
        #class_weight = eval(class_weight)
        presort = args['presort']

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
            if criterion == None or criterion=='None':
                criterion = 'mse'
            else:
                if criterion!='mse' and criterion!='mae':
                    soln["code"] = 100002
                    soln['message'] = "请上传正确的特征选择标准，可设置为mse或mae"
                    soln['data'] = None
                    return jsonify(soln)

            if splitter==None or splitter=='None':
                splitter='best'
            else:
                if splitter!='best' and splitter!='random':
                    soln["code"] = 100002
                    soln['message'] = "请上传正确的特征划分点选择标准，默认是best，可以设置为random"
                    soln['data'] = None
                    return jsonify(soln)

            if min_samples_split==None:
                min_samples_split = 2
            if min_samples_split>1:
                min_samples_split = int(min_samples_split)

            if min_samples_leaf==None:
                min_samples_leaf = 1
            if min_samples_leaf>=1:
                min_samples_leaf = int(min_samples_leaf)
            if min_weight_fraction_leaf ==None:
                min_weight_fraction_leaf=0

            if max_features == 'int' or max_features == 'float':
                soln["code"] = 100002
                soln['message'] = "请上传正确的最大特征数max_features，请输入具体的int或float的值"
                soln['data'] = None
                return jsonify(soln)
            if max_features == 'None':
                max_features = None
            if max_features !='auto'and max_features!='sqrt'and max_features !='log2' and max_features!=None:
                max_features = float(max_features)

            if min_impurity_decrease==None:
                min_impurity_decrease=0
            if min_impurity_split==None:
                min_impurity_split=1e-7
            if presort==None:
                presort=False


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
            scores = DecisionTree_reg_cal(dx,dy,criterion,splitter,max_depth,min_samples_split,min_samples_leaf,
                               min_weight_fraction_leaf,max_features,max_leaf_nodes,min_impurity_decrease,
                               min_impurity_split,presort)
            if type(scores) != str:
                # 返回的字典
                soln["code"] = 0
                soln['message'] = "请求成功"
                soln['data'] = {'mean accuracy': scores}

                # 将字典转成json格式，返回json文件
                return jsonify(soln)
            else:
                # 返回的字典
                soln["code"] = 100002
                soln['message'] = {"key error":scores}
                soln['data'] = None
                return jsonify(soln)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
    #app.run(port=9021, debug=True)