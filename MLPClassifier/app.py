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
from MLPclf import MLP_clf_cla

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
api = Api(app,version='1.0',title='Sklearn MLP Classifier API',
          description='MLP神经网络分类算法')

ns_iba = api.namespace('MLP_classifier_operate', description='提供csv文件进行MLP神经网络分类')

#健康检测接口
@app.route('/healthz')
def healthz():
    return "OK"

# 配置参数装饰器
MLP_classifier_parser = reqparse.RequestParser()
MLP_classifier_parser.add_argument('upload_csv', location='files', type=FileStorage, required=True, help='上传数据表')
MLP_classifier_parser.add_argument('target_name', location='args', type=str, required=True, help='标签列名')  # 标签
MLP_classifier_parser.add_argument('hidden_layer_sizes', location='args', type=str, required=True,help='tuple，第i个元素表示第i个隐藏层的神经元个数')
MLP_classifier_parser.add_argument('activation', location='args', type=str,required=True, help='隐藏层激活函数，可选identity、logistic、tanh、relu')
MLP_classifier_parser.add_argument('solver', location='args', type=str, required=True,help='权重优化算法，可选lbfgs、sgd、adam')
MLP_classifier_parser.add_argument('alpha', location='args', type=float, help='正则化项参数')
MLP_classifier_parser.add_argument('batch_size', location='args', type=str, help='随机优化的minibatches的大小')
MLP_classifier_parser.add_argument('learning_rate', location='args', type=str,required=True, help='学习率，可选constant、invscaling、adaptive')
MLP_classifier_parser.add_argument('learning_rate_init', location='args', type=float, help='初始学习率。只有当solver为sgd或adam时才使用')
MLP_classifier_parser.add_argument('power_t', location='args', type=float, help='逆扩展学习率的指数，只有当solver为sgd时才使用')
MLP_classifier_parser.add_argument('max_iter', location='args', type=int, help='最大迭代次数')
MLP_classifier_parser.add_argument('shuffle', location='args', type=bool, help='是否在每次迭代时对样本进行清洗，当solver为sgd或adam时使用')
#MLP_classifier_parser.add_argument('random_state', location='args', type=int, help='随机数种子')
MLP_classifier_parser.add_argument('tol', location='args', type=float, help='优化算法停止的条件。当迭代前后的函数差值小于等于tol时就停止')
MLP_classifier_parser.add_argument('verbose', location='args', type=bool, help='是否将过程打印出')
MLP_classifier_parser.add_argument('warm_start', location='args', type=bool, help='是否使用之前的解决方法作为初始拟合')
MLP_classifier_parser.add_argument('momentum', location='args', type=float, help='梯度下降的动量，介于0到1之间，solver为sgd时使用')
MLP_classifier_parser.add_argument('nesterovs_momentum', location='args', type=bool,help='是否使用Nesterov动量')
MLP_classifier_parser.add_argument('early_stopping', location='args', type=bool, help='判断当验证效果不再改善时是否终止训练')
MLP_classifier_parser.add_argument('validation_fraction', location='args', type=float, help='用作早起停止验证的预留训练集的比例，0到1之间')
MLP_classifier_parser.add_argument('beta_1', location='args', type=float, help='估计一阶矩向量的指数衰减速率，[0,1)之间')
MLP_classifier_parser.add_argument('beta_2', location='args', type=float, help='估计二阶矩向量的指数衰减速率，[0,1)之间')
MLP_classifier_parser.add_argument('epsilon', location='args', type=float, help='数值稳定值，solver为adam时使用')
MLP_classifier_parser.add_argument('n_iter_no_change', location='args', type=int, help='满足tol的最大迭代次数，solver为sgd或adam时使用')

@ns_iba.route('/MLP_classifier')
@ns_iba.expect(MLP_classifier_parser)

# 接口运行函数配置
class MLP_classifier(Resource):

    # 通过post 上传、处理文件并返回json文件
    def post(self):
        '''
        文件上传接口，上传后将返回分类好的json文件。
        输入：待分析csv文件
        输出：MLP神经网络分类后的json文件
        '''

        # 取参数字典
        args = MLP_classifier_parser.parse_args()

        soln = {}
        if 'upload_csv' not in args or 'target_name' not in args or 'hidden_layer_sizes' not in args or 'activation' not in args or 'solver' not in args \
                or 'alpha' not in args or 'batch_size' not in args or 'learning_rate' not in args or 'learning_rate_init' not in args \
                or 'power_t' not in args or 'max_iter' not in args or 'shuffle' not in args not in args or 'validation_fraction' not in args or 'beta_1' not in args \
                or 'beta_2' not in args or 'epsilon' not in args or 'n_iter_no_change' not in args\
                or 'tol' not in args or 'verbose' not in args or 'warm_start' not in args :
                logger.error('No zip argument in paras')
                soln["code"] = 100003
                soln['message'] = "请输入所有参数！"
                soln['data'] = None
                return jsonify(soln)

        # 获取所有参数
        csv_file = args['upload_csv']
        target_name = args['target_name']
        hidden_layer_sizes = args['hidden_layer_sizes']

        activation = args['activation']
        solver = args['solver']
        alpha = args['alpha']
        batch_size = args['batch_size']
        learning_rate = args['learning_rate']
        learning_rate_init= args['learning_rate_init']
        power_t= args['power_t']
        max_iter= args['max_iter']
        shuffle= args['shuffle']
        tol= args['tol']
        verbose= args['verbose']
        warm_start= args['warm_start']
        momentum= args['momentum']
        nesterovs_momentum= args['nesterovs_momentum']
        early_stopping= args['early_stopping']
        validation_fraction= args['validation_fraction']
        beta_1= args['beta_1']
        beta_2= args['beta_2']
        epsilon= args['epsilon']
        n_iter_no_change= args['n_iter_no_change']

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
            if hidden_layer_sizes == None or hidden_layer_sizes=='None':
                hidden_layer_sizes = (100,)
            else:
                hidden_layer_sizes = eval(hidden_layer_sizes)

            if activation ==None or activation =='None':
                activation ='relu'
            else:
                if activation !='identity' and activation !='logistic'and activation !='tanh'and activation !='relu':
                    soln["code"] = 100002
                    soln['message'] = "请上传正确的参数activation，默认是relu，可以设置为{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}"
                    soln['data'] = None
                    return jsonify(soln)


            if solver ==None or solver =='None':
                solver ='adam'
            else:
                if solver !='lbfgs' and solver !='sgd'and solver !='adam':
                    soln["code"] = 100002
                    soln['message'] = "请上传正确的参数solver，默认是adam，可以设置为 {‘lbfgs’, ‘sgd’, ‘adam’}"
                    soln['data'] = None
                    return jsonify(soln)
            if alpha==None:
                alpha = 0.0001
            if batch_size == None or batch_size == 'None' or batch_size=='auto':
                batch_size = 'auto'
            else:
                try:batch_size = eval(batch_size)
                except:
                    soln["code"] = 100002
                    soln['message'] = "请上传正确的参数batch_size，默认是auto，可以设置为其他整数值"
                    soln['data'] = None
                    return jsonify(soln)

            if learning_rate ==None or learning_rate =='None':
                learning_rate ='constant'
            else:
                if learning_rate !='constant' and learning_rate !='invscaling'and learning_rate !='adaptive':
                    soln["code"] = 100002
                    soln['message'] = "请上传正确的参数learning_rate，默认是constant，可以设置为 {‘constant’, ‘invscaling’, ‘adaptive’}"
                    soln['data'] = None
                    return jsonify(soln)



            if learning_rate_init==None:
                learning_rate_init = 0.001
            if power_t == None:
                power_t = 0.5

            if max_iter == None:
                max_iter = 200

            if shuffle==None:
                shuffle = True

            if tol == None:
                tol = 1e-4
            if verbose ==None:
                verbose=False

            if warm_start == None:
                warm_start = False
            if momentum ==None and solver=='sgd':
                momentum=0.9

            if nesterovs_momentum == None and solver=='sgd' and momentum>0:
                nesterovs_momentum = True
            if early_stopping ==None and (solver=='sgd' or solver=='adam'):
                early_stopping=False

            if validation_fraction == None:
                validation_fraction = 0.1
            if beta_1  ==None:
                beta_1 =0.9
            if beta_2 ==None:
                beta_2=0.999

            if epsilon == None:
                validation_fraction = 1e-8

            if n_iter_no_change  == None:
                n_iter_no_change =10


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
            scores = MLP_clf_cla(dx,dy,hidden_layer_sizes,activation,solver,alpha,batch_size,learning_rate,learning_rate_init,power_t,max_iter,
                shuffle,tol,verbose,warm_start, momentum, nesterovs_momentum,early_stopping,validation_fraction,
                beta_1,beta_2,epsilon,n_iter_no_change)
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
    #app.run(port=9581, debug=True)
