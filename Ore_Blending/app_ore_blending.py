# -*- coding: utf-8 -*-

#!/usr/bin/env python
from flask_restplus import Api, Resource,fields, reqparse
from werkzeug.contrib.fixers import ProxyFix
from threading import Thread
try:
    from .ore_blending_main import *
except:
    from ore_blending_main import *
try:
    from .ore_blending_function import *
except:
    from ore_blending_function import *
#from base_deal import generate_token,mkdir,zip_dir

from flask import Flask,flash,request,redirect,send_file,jsonify #
from flask_cors import *
import os
import logging
import sys


#===========================以下为不同函数的日志初始化区域============================================================================
# 创建一个logger格式
formatter = logging.Formatter(
    '----------------[%(asctime)s][%(thread)d][%(filename)s][line: %(lineno)d][%(levelname)s] ## %(message)s')
# step1 time_to_distance log
loggername = 'ore_blending_log'
#logPath = os.path.join('/file_and_log', loggername+'.log')
logPath = os.path.join('D:/', loggername+'.log')
logger = logging.getLogger(loggername)
logger.setLevel(logging.DEBUG)
try:#防止本地测试时报错
    fileHandler = logging.FileHandler(logPath, encoding='utf8')
except:
    fileHandler = logging.FileHandler(loggername+'.log', encoding='utf8')
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

app = Flask(__name__)
CORS(app, supports_credentials=True)  #解决前端请求跨域问题******
app.config['MAX_CONTENT_LENGTH'] = 1600 * 1024 * 1024
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app,version='1.0',title='ore_blending API',
          description='配矿产品API')

ns_1 = api.namespace('ore_blending', description='提供配矿计算相关接口')
post_parser1 = api.model('json_text', {
    'data_in': fields.String(required=True, description='请输入data_in的json')
})
    

@ns_1.route('/upload_data')
class upload_data(Resource):
    @ns_1.expect(post_parser1)
    def post(self):
        output = cal_ore_blending_main(api.payload['data_in'],logger)
        return jsonify(output)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=80, debug=True)
    #app.run(port=9228, debug=True)
