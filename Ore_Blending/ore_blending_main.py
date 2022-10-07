# -*- coding: utf-8 -*-

#!/usr/bin/env python
import time
import json
import logging
try:
    from .ore_blending_function import ore_blending
except:
    from ore_blending_function import ore_blending
import os

#设置全局变量
global LOWER,COM_LOWER,COM_UPPER,BOUND,NUM,DEDUCTION_SCOPE_FE_1,DEDUCTION_SCOPE_FE_2,DEDUCTION_SCOPE_AL_1,DEDUCTION_SCOPE_FE_2,\
DEDUCTION_SCOPE_AL_1,DEDUCTION_SCOPE_AL_2,DEDUCTION_SCOPE_AL_3,DEDUCTION_SCOPE_SI_1,DEDUCTION_SCOPE_SI_2,DEDUCTION_SCOPE_SI_3,\
DEDUCTION_SCOPE_P_1,DEDUCTION_SCOPE_P_2,DEDUCTION_SCOPE_AL,S_LIMIT,MN_LIMIT,K_LIMIT,NA_LIMIT,ZN_LIMIT,PB_LIMIT,AS_LIMIT,CU_LIMIT,\
SIZE_10_LIMIT,SIZE_8_LIMIT,SIZE_6_LIMIT,SIZE_1_LIMIT,SIZE_0_15_LIMIT

LOWER = 0.05
COM_LOWER = 2
COM_UPPER = 8
BOUND = 0.9
NUM = 10
DEDUCTION_SCOPE_FE_1 = [55,60]
DEDUCTION_SCOPE_FE_2 = [60,63.5]
DEDUCTION_SCOPE_AL_1 = [0,1]
DEDUCTION_SCOPE_AL_2 = [1,2.5]
DEDUCTION_SCOPE_AL_3 = [2.5,4]
DEDUCTION_SCOPE_SI_1 = [0,4.5]
DEDUCTION_SCOPE_SI_2 = [4.5,6.5]
DEDUCTION_SCOPE_SI_3 = [6.5,9]
DEDUCTION_SCOPE_P_1 = [0,0.09]
DEDUCTION_SCOPE_P_2 = [0.09,0.12]
DEDUCTION_SCOPE_AL = [0,5]

S_LIMIT = [0,100]
MN_LIMIT = [0,100]
K_LIMIT = [0,100]
NA_LIMIT = [0,100]
ZN_LIMIT = [0,100]
PB_LIMIT = [0,100]
AS_LIMIT = [0,100]
CU_LIMIT = [0,100]
SIZE_10_LIMIT = [0,100]
SIZE_8_LIMIT = [0,100]
SIZE_6_LIMIT = [0,100]
SIZE_1_LIMIT = [0,100]
SIZE_0_15_LIMIT = [0,100]


def cal_ore_blending_main(data_in,log):
    outputs = {"success": False, "message": "", "data": ""}
    try:
        start = time.time()
        outputs['data'] = '无法获得合适的配矿方案'
        
        #检测数据是否成功传入
        #参数设定检测及默认值设定
        try:
            fe = data_in['fe']
            si = data_in['si']
            ca = data_in['ca']
            mg = data_in['mg']
            al = data_in['al']
            p = data_in['p']
            h2o = data_in['h2o'] 
            loi = data_in['loi']
            fe_limit = data_in['fe_limit']
            si_limit = data_in['si_limit']
            ca_limit = data_in['ca_limit']
            mg_limit = data_in['mg_limit']
            al_limit = data_in['al_limit']
            p_limit = data_in['p_limit']
            h2o_limit = data_in['h2o_limit']
            loi_limit = data_in['loi_limit']   
            species = data_in['species']
            powder = data_in['powder']
            price_on = data_in['price_on']
            aim = data_in['aim']
            deduction_price_all_1 = data_in['deduction_price_all_1']
            deduction_price = data_in['deduction_price']
            price = data_in['price']
        except:
            outputs["message"] = '必填项缺失，请检查输入'
            return outputs
        try:
            lower = data_in['lower']
        except:
            lower = LOWER
        try:
            bound = data_in['bound']
        except:
            bound = BOUND 
        try:
            com_lower = data_in['com_lower']
        except:
            com_lower = COM_LOWER  #配矿种类下限
        try:
            com_upper = data_in['com_upper']
        except:
            com_upper = COM_UPPER  #配矿种类上限
        try:
            num = data_in['num']
        except:
            num = NUM   #输出方案个数限制
        try:
            deduction_scope_fe_1 = data_in['deduction_scope_fe_1']
        except:
            deduction_scope_fe_1 = DEDUCTION_SCOPE_FE_1
        try:
            deduction_scope_fe_2 = data_in['deduction_scope_fe_2']
        except:
            deduction_scope_fe_2 = DEDUCTION_SCOPE_FE_2
        try:
            deduction_scope_al_1 = data_in['deduction_scope_al_1']
        except:
            deduction_scope_al_1 = DEDUCTION_SCOPE_AL_1
        try:
            deduction_scope_al_2 = data_in['deduction_scope_al_2']
        except:
            deduction_scope_al_2 = DEDUCTION_SCOPE_AL_2
        try:
            deduction_scope_al_3 = data_in['deduction_scope_al_3']
        except:
            deduction_scope_al_3 = DEDUCTION_SCOPE_AL_3
        try:
            deduction_scope_si_1 = data_in['deduction_scope_si_1']
        except:
            deduction_scope_si_1 = DEDUCTION_SCOPE_SI_1
        try:
            deduction_scope_si_2 = data_in['deduction_scope_si_2']
        except:
            deduction_scope_si_2 = DEDUCTION_SCOPE_SI_2
        try:
            deduction_scope_si_3 = data_in['deduction_scope_si_3']
        except:
            deduction_scope_si_3 = DEDUCTION_SCOPE_SI_3
        try:
            deduction_scope_p_1 = data_in['deduction_scope_p_1']
        except:
            deduction_scope_p_1 = DEDUCTION_SCOPE_P_1
        try:
            deduction_scope_p_2 = data_in['deduction_scope_p_2']
        except:
            deduction_scope_p_2 = DEDUCTION_SCOPE_P_2
        try:
            deduction_scope_al = data_in['deduction_scope_al']
        except:
            deduction_scope_al = DEDUCTION_SCOPE_AL
        
        try:
            s = data_in['s']
        except:
            s = fe
        try:
            mn = data_in['mn']
        except:
            mn = fe
        try:
            k = data_in['k']
        except:
            k = fe
        try:
            na = data_in['na']
        except:
            na = fe
        try:
            zn = data_in['zn']    
        except:
            zn = fe
        try:
            pb = data_in['pb']
        except:
            pb = fe
        try:
            as_ = data_in['as']
        except:
            as_ = fe
        try:
            cu = data_in['cu']
        except:
            cu = fe
        try:
            size_10 = data_in['size_10']
        except:
            size_10 = fe
        try:
            size_8 = data_in['size_8']
        except:
            size_8 = fe
        try:
            size_6 = data_in['size_6']
        except:
            size_6 = fe
        try:
            size_1 = data_in['size_1']
        except:
            size_1 = fe
        try:
            size_0_15 = data_in['size_0_15']
        except:
            size_0_15 = fe
            
        try:
            s_limit = data_in['s_limit']
        except:
            s_limit = S_LIMIT
        try:
            mn_limit = data_in['mn_limit']
        except:
            mn_limit = MN_LIMIT
        try:
            k_limit = data_in['k_limit']
        except:
            k_limit = K_LIMIT
        try:
            na_limit = data_in['na_limit']
        except:
            na_limit = NA_LIMIT
        try:
            zn_limit = data_in['zn_limit']
        except:
            zn_limit = ZN_LIMIT
        try:
            pb_limit = data_in['pb_limit']
        except:
            pb_limit = PB_LIMIT
        try:
            as_limit = data_in['as_limit']
        except:
            as_limit = AS_LIMIT
        try:
            cu_limit = data_in['cu_limit']
        except:
            cu_limit = CU_LIMIT
        try:
            size_10_limit = data_in['size_10_limit']
        except:
            size_10_limit = SIZE_10_LIMIT
        try:
            size_8_limit = data_in['size_8_limit']
        except:
            size_8_limit = SIZE_8_LIMIT
        try:
            size_6_limit = data_in['size_6_limit']
        except:
            size_6_limit = SIZE_6_LIMIT
        try:
            size_1_limit = data_in['size_1_limit']
        except:
            size_1_limit = SIZE_1_LIMIT
        try:
            size_0_15_limit = data_in['size_0_15_limit']
        except:
            size_0_15_limit = SIZE_0_15_LIMIT
        
        result,deduction_price_all_1,deduction_price_all_2,deduction_scope_all_2 = ore_blending(fe,si,ca,mg,al,p,h2o,loi,s,mn,k,na,zn,pb,as_,cu,size_10,size_8,size_6,\
                              size_1,size_0_15,fe_limit,si_limit,ca_limit,mg_limit,al_limit,p_limit,\
                              h2o_limit,loi_limit,s_limit,mn_limit,k_limit,na_limit,zn_limit,\
                              pb_limit,as_limit,cu_limit,size_10_limit,size_8_limit,size_6_limit,\
                              size_1_limit,size_0_15_limit,deduction_scope_fe_1,deduction_scope_fe_2,\
                              species,powder,price_on,aim,deduction_price_all_1,deduction_price,price,\
                              deduction_scope_al,deduction_scope_al_1,deduction_scope_al_2,deduction_scope_al_3,\
                              deduction_scope_si_1,deduction_scope_si_2,deduction_scope_si_3,deduction_scope_p_1,\
                              deduction_scope_p_2,lower,bound,com_lower,com_upper,num)
        
        end = time.time()
        message = "ore blending cost time：%.6f" % (end - start)
        outputs["success"] = True
        outputs["message"] = message
        outputs["data"] = {'result':result.to_json(orient='columns'),
                     'deduction_price_all_1':deduction_price_all_1.to_json(orient='columns'),
                     'deduction_price_all_2':deduction_price_all_2.to_json(orient='columns'),
                     'deduction_scope_all_2':deduction_scope_all_2.to_json(orient='columns')}####关注dataframe如何存储
        return outputs
    except Exception as error:
        message = repr(error)#error转为字符串
        outputs["message"] = message
        outputs["data"] = '无法获得合适的配矿方案'
        outputs["success"] = False

        saveInLog = {}
        saveInLog["input"] = data_in
        saveInLog["res"] = outputs
        saveInLog["flag"] = "ore_blending"
        log.info(saveInLog)
        return outputs

if __name__ == "__main__":
    import pandas as pd
    # 创建一个logger
    loggername = 'ore_blending_log'
    LogPath = os.path.join('D:/', loggername+'.log')   ########
    logger = logging.getLogger(loggername)
    logger.setLevel(logging.DEBUG)
    # 创建一个handler，用于写入日志文件
    fh = logging.FileHandler(LogPath, encoding='utf8')
    fh.setLevel(logging.DEBUG)
    ## 定义handler的输出格式
    formatter = logging.Formatter(
        '----------------[%(asctime)s][%(thread)d][%(filename)s][line: %(lineno)d][%(levelname)s] ## %(message)s')
    fh.setFormatter(formatter)
    # 给logger添加handler
    logger.addHandler(fh)
    
    data = pd.read_excel('矿粉成分及价格信息.xlsx',sheetname = '矿源成分')
    species = 14
    fe = list(data['TFe'][:species])
    si = list(data['SiO2'][:species])
    ca = list(data['CaO'][:species])
    mg = list(data['MgO'][:species])
    al = list(data['Al2O3'][:species])
    p = list(data['P'][:species])
    s = list(data['S'][:species])
    h2o = list(data['H2O'][:species])
    loi = list(data['LOI'][:species])
    
    #各个元素上下限（增加限制条件）
    fe_limit = list(data['TFe'][18:20].reset_index(drop=True))
    si_limit = list(data['SiO2'][18:20].reset_index(drop=True))
    ca_limit = list(data['CaO'][18:20].reset_index(drop=True))
    mg_limit = list(data['MgO'][18:20].reset_index(drop=True))
    al_limit = list(data['Al2O3'][18:20].reset_index(drop=True))
    p_limit = list(data['P'][18:20].reset_index(drop=True))
    s_limit = list(data['S'][18:20].reset_index(drop=True))  #该参数此处为空，对算法没有影响
    h2o_limit = list(data['H2O'][18:20].reset_index(drop=True))
    loi_limit = list(data['LOI'][18:20].reset_index(drop=True))
    
    data_in = {
            'fe':fe,
            'si':si,
            'ca':ca,
            'mg':mg,
            'al':al,
            'p':p,
            's':s,
            'h2o':h2o,
            'loi':loi,
            'fe_limit':fe_limit,
            'si_limit':si_limit,
            'ca_limit':ca_limit,
            'mg_limit':mg_limit,
            'al_limit':al_limit,
            'p_limit':p_limit,
            's_limit':s_limit,
            'h2o_limit':h2o_limit,
            'loi_limit':loi_limit,
            'species':14,
            'powder':[[0,0.2],[0,0.2],[0,0.3],[0,0.2],\
                      [0,0.5],[0,0.3],[0,0.3],[0,0.4],\
                      [0,0.4],[0,0.25],[0,0.2],[0,0.2],\
                      [0,0.2],[0,0.3]],
            'price_on':446,
            'aim':[60.8, 2.25, 4.75, 0.085],
            'deduction_price_all_1':[6.83,18.78,17.08],
            'deduction_price':[8.879104, 0, 51.2256, 51.2256, 0.683008, 1.366016, 13.66016, 0, 13.66016],
            'price':[490,446,417,462,436,420,340,287,300,635,507,450,590,580]
                }

    with open('testjson.json', 'w') as result_file:
        json.dump(data_in, result_file)
    with open("testjson.json", 'rb') as load_f:
        json1 = json.load(load_f)
    outputs = cal_ore_blending_main(json1,logger)
    print(outputs)
