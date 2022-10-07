# -*- coding: utf-8 -*-
import numpy as np
import pulp as pulp
import pandas as pd
from itertools import combinations 
import copy

def solve_ilp(objective , constraints) :
    """
    输入：
        objective：待求解规划问题目标值
        constraints：待求解规划问题约束
    输出：
        res：最优矿料配比
        target：最大利润
    功能：线性规划问题求解
    """
    prob = pulp.LpProblem('LP1' , pulp.LpMaximize)
    prob += objective
    for cons in constraints :
        prob += cons
    status = prob.solve()
    if status != 1 :
        return None,0
    else :
        target = pulp.value(objective)
        res_0={}
        for v in prob.variables():
            res_0[int(v.name[1:])] = v.varValue.real
        #按照key排序
        res = [res_0[k] for k in sorted(res_0.keys())]
        return res,target

def build_lp_pre(fe,si,ca,mg,al,p,h2o,loi,s,mn,k_,na,zn,pb,as_,cu,size_10,size_8,size_6,\
                              size_1,size_0_15,fe_limit,si_limit,ca_limit,mg_limit,al_limit,p_limit,\
                              h2o_limit,loi_limit,s_limit,mn_limit,k_limit,na_limit,zn_limit,\
                              pb_limit,as_limit,cu_limit,size_10_limit,size_8_limit,size_6_limit,\
                              size_1_limit,size_0_15_limit,com_,powder_):
    """
    输入：
        com_：一种特定的矿粉组合
        powder_：每种矿粉比例的上下限
        _limit：各个元素上下限(这一系列元素写成全局变量还是？)
    输出：
        variables：参数
        constraints：每个矿粉组合所有情况的公共约束 
    功能：
        得到一个矿粉组合的参数及所有情况的公共约束 
    """
    V_NUM = len(com_)
    variables = [pulp.LpVariable('X%d'%i,lowBound = powder_[i][0],upBound = powder_[i][1],cat='Continuous') for i in range(0 , V_NUM)]  #默认为Continuous
    #约束条件
    constraints = []
    #目标矿粉成分上下限
    constraints.append(sum(fe[com_]*variables[:V_NUM]) >= fe_limit[0])
    constraints.append(sum(fe[com_]*variables[:V_NUM]) <= fe_limit[1])
    constraints.append(sum(si[com_]*variables[:V_NUM]) >= si_limit[0])
    constraints.append(sum(si[com_]*variables[:V_NUM]) <= si_limit[1])
    constraints.append(sum(ca[com_]*variables[:V_NUM]) >= ca_limit[0])
    constraints.append(sum(ca[com_]*variables[:V_NUM]) <= ca_limit[1])
    constraints.append(sum(mg[com_]*variables[:V_NUM]) >= mg_limit[0])
    constraints.append(sum(mg[com_]*variables[:V_NUM]) <= mg_limit[1])
    constraints.append(sum(al[com_]*variables[:V_NUM]) >= al_limit[0])
    constraints.append(sum(al[com_]*variables[:V_NUM]) <= al_limit[1])
    constraints.append(sum(p[com_]*variables[:V_NUM]) >= p_limit[0])
    constraints.append(sum(p[com_]*variables[:V_NUM]) <= p_limit[1])
    constraints.append(sum(h2o[com_]*variables[:V_NUM]) >= h2o_limit[0])
    constraints.append(sum(h2o[com_]*variables[:V_NUM]) <= h2o_limit[1])
    constraints.append(sum(loi[com_]*variables[:V_NUM]) >= loi_limit[0])
    constraints.append(sum(loi[com_]*variables[:V_NUM]) <= loi_limit[1])
    
    constraints.append(sum(s[com_]*variables[:V_NUM]) >= s_limit[0])
    constraints.append(sum(s[com_]*variables[:V_NUM]) <= s_limit[1])
    constraints.append(sum(mn[com_]*variables[:V_NUM]) >= mn_limit[0])
    constraints.append(sum(mn[com_]*variables[:V_NUM]) <= mn_limit[1])
    constraints.append(sum(k_[com_]*variables[:V_NUM]) >= k_limit[0])
    constraints.append(sum(k_[com_]*variables[:V_NUM]) <= k_limit[1])
    constraints.append(sum(na[com_]*variables[:V_NUM]) >= na_limit[0])
    constraints.append(sum(na[com_]*variables[:V_NUM]) <= na_limit[1])
    constraints.append(sum(zn[com_]*variables[:V_NUM]) >= zn_limit[0])
    constraints.append(sum(zn[com_]*variables[:V_NUM]) <= zn_limit[1])
    constraints.append(sum(pb[com_]*variables[:V_NUM]) >= pb_limit[0])
    constraints.append(sum(pb[com_]*variables[:V_NUM]) <= pb_limit[1])
    constraints.append(sum(as_[com_]*variables[:V_NUM]) >= as_limit[0])
    constraints.append(sum(as_[com_]*variables[:V_NUM]) <= as_limit[1])
    constraints.append(sum(cu[com_]*variables[:V_NUM]) >= cu_limit[0])
    constraints.append(sum(cu[com_]*variables[:V_NUM]) <= cu_limit[1])
    constraints.append(sum(size_10[com_]*variables[:V_NUM]) >= size_10_limit[0])
    constraints.append(sum(size_10[com_]*variables[:V_NUM]) <= size_10_limit[1])
    constraints.append(sum(size_8[com_]*variables[:V_NUM]) >= size_8_limit[0])
    constraints.append(sum(size_8[com_]*variables[:V_NUM]) <= size_8_limit[1])
    constraints.append(sum(size_6[com_]*variables[:V_NUM]) >= size_6_limit[0])
    constraints.append(sum(size_6[com_]*variables[:V_NUM]) <= size_6_limit[1])
    constraints.append(sum(size_1[com_]*variables[:V_NUM]) >= size_1_limit[0])
    constraints.append(sum(size_1[com_]*variables[:V_NUM]) <= size_1_limit[1])
    constraints.append(sum(size_0_15[com_]*variables[:V_NUM]) >= size_0_15_limit[0])
    constraints.append(sum(size_0_15[com_]*variables[:V_NUM]) <= size_0_15_limit[1])
    constraints.append(sum(variables[:V_NUM]) == 1)
    
    return variables,constraints

def build_lp(fe,si,ca,mg,al,p,h2o,loi,s,mn,k_,na,zn,pb,as_,cu,size_10,size_8,size_6,\
                              size_1,size_0_15,com_,variables,constraints,aim,price_on,deduction_price,price,deduction_scope,status): 
    """
    输入：
        com_：一种特定的矿粉组合
        variables：矿粉组合对应的参数
        constraints：矿粉组合所有情况的公共约束
        aim: 对标矿粉的成分含量（只有四种元素）
        price_on：对标矿粉价格，可以为短期价格也可为长期价格均值
        deduction_price：四种元素折扣价格
        price：所有矿粉价格，可以为短期价格也可为长期价格均值
        deduction_scope：三种元素折扣区间
        status:0代表高铁含量，1代表低铁含量
        
    输出：
        com_ ：矿料组合
        res_1：矿料在组合对应的占比
        target_1：矿料组合对于对标矿粉的利润
    """
    V_NUM = len(com_)
    fe_aim = aim[0]
    al_aim = aim[1]
    si_aim = aim[2]
    p_aim = aim[3]
    if status == 1:
         ### (目标元素含量 - coef_al_1)*deduction_price+intercept_al_1
        objective_1 = (np.dot(fe[com_],variables[:V_NUM])-fe_aim)*deduction_price[0] - \
                    ((np.dot(al[com_],variables[:V_NUM])-deduction_scope[11])*deduction_price[1]+deduction_scope[8]) -\
                    ((np.dot(si[com_],variables[:V_NUM])-deduction_scope[12])*deduction_price[2]+deduction_scope[9]) -\
                    ((np.dot(p[com_],variables[:V_NUM])-deduction_scope[13])*100*deduction_price[3]+deduction_scope[10]) -\
                    np.dot(price[com_],variables[:V_NUM]) + price_on
        constraints_1 = copy.copy(constraints)
        constraints_1.append(np.dot(fe[com_],variables[:V_NUM]) >= deduction_scope[0])
        constraints_1.append(np.dot(fe[com_],variables[:V_NUM]) <= deduction_scope[1])
        constraints_1.append(np.dot(al[com_],variables[:V_NUM]) >= deduction_scope[2])
        constraints_1.append(np.dot(al[com_],variables[:V_NUM]) <= deduction_scope[3])
        constraints_1.append(np.dot(si[com_],variables[:V_NUM]) >= deduction_scope[4])
        constraints_1.append(np.dot(si[com_],variables[:V_NUM]) <= deduction_scope[5])
        constraints_1.append(np.dot(p[com_],variables[:V_NUM]) >= deduction_scope[6])
        constraints_1.append(np.dot(p[com_],variables[:V_NUM]) <= deduction_scope[7])
              
    elif status == 0:
        objective_1 = (np.dot(fe[com_],variables[:V_NUM])-fe_aim)*deduction_price[0] - \
                (np.dot(al[com_],variables[:V_NUM])-al_aim)*deduction_price[1] -\
                (np.dot(si[com_],variables[:V_NUM])-si_aim)*deduction_price[2] -\
                np.dot(price[com_],variables[:V_NUM]) + price_on
        constraints_1 = copy.copy(constraints)
        constraints_1.append(np.dot(fe[com_],variables[:V_NUM]) >= deduction_scope[0])
        constraints_1.append(np.dot(fe[com_],variables[:V_NUM]) <= deduction_scope[1])
        constraints_1.append(np.dot(al[com_],variables[:V_NUM]) >= deduction_scope[2])
        constraints_1.append(np.dot(al[com_],variables[:V_NUM]) <= deduction_scope[3])

    res_1,target_1 = solve_ilp(objective_1 , constraints_1)        
    
    return [com_,res_1,target_1],objective_1 

def build_lp_all(fe,si,ca,mg,al,p,h2o,loi,s,mn,k_,na,zn,pb,as_,cu,size_10,size_8,size_6,\
                              size_1,size_0_15,fe_limit,si_limit,ca_limit,mg_limit,al_limit,p_limit,\
                              h2o_limit,loi_limit,s_limit,mn_limit,k_limit,na_limit,zn_limit,\
                              pb_limit,as_limit,cu_limit,size_10_limit,size_8_limit,size_6_limit,\
                              size_1_limit,size_0_15_limit,com_,powder_,price_on,\
                 aim,deduction_price_all,price,deduction_scope_all,status):
    """
    输入：
        com_：一种特定的矿粉组合
        powder_: 矿料组合对应的元素上下限
        price_on：对标矿粉价格，可以为短期价格也可为长期价格均值
        aim: 对标矿粉的成分含量（只有四种元素）
        deduction_price_all：四种元素折扣价格（所有组合情况）
        price：所有矿粉价格，可以为短期价格也可为长期价格均值
        deduction_scope_all：三或四种元素折扣区间（所有组合情况）   
        status:0代表高铁含量，1代表低铁含量
    输出：
        res:组合情况
        solition:是否有解的0-1标签
    功能：
        对于某种矿料组合计算所有的可能情况
    """
    variables,constraints = build_lp_pre(fe,si,ca,mg,al,p,h2o,loi,s,mn,k_,na,zn,pb,as_,cu,size_10,size_8,size_6,\
                              size_1,size_0_15,fe_limit,si_limit,ca_limit,mg_limit,al_limit,p_limit,\
                              h2o_limit,loi_limit,s_limit,mn_limit,k_limit,na_limit,zn_limit,\
                              pb_limit,as_limit,cu_limit,size_10_limit,size_8_limit,size_6_limit,\
                              size_1_limit,size_0_15_limit,com_,powder_)
    res=[1]*len(deduction_price_all.T)
    objective = [1]*len(deduction_price_all.T)
    solution = [1]*len(deduction_price_all.T)
    for i in range(0,len(deduction_price_all.T)):
        column = deduction_price_all.columns
        res[i],objective[i] = build_lp(fe,si,ca,mg,al,p,h2o,loi,s,mn,k_,na,zn,pb,as_,cu,size_10,size_8,size_6,\
                              size_1,size_0_15,com_,variables,constraints,aim,price_on,\
                                       deduction_price_all[column[i]],price,deduction_scope_all[column[i]],status)
        res[i].extend([deduction_price_all.columns[i]])   #加入折扣类别标签
        if type(res[i][1]) == list:
            solution[i] = 1
        else:
            solution[i] = 0
    
    return res,objective,solution
                         

def get_result(fe,si,ca,mg,al,p,h2o,loi,s,mn,k_,na,zn,pb,as_,cu,size_10,size_8,size_6,\
                              size_1,size_0_15,fe_limit,si_limit,ca_limit,mg_limit,al_limit,p_limit,\
                              h2o_limit,loi_limit,s_limit,mn_limit,k_limit,na_limit,zn_limit,\
                              pb_limit,as_limit,cu_limit,size_10_limit,size_8_limit,size_6_limit,\
                              size_1_limit,size_0_15_limit,com,powder,price_on,\
                 aim,deduction_price_all,price,deduction_scope_all,status):
    """
    输入：
        com：所有矿粉组合
        powder：所有矿料组合对应的元素上下限
        price_on：对标矿粉价格，可以为短期价格也可为长期价格均值
        aim：对标矿粉的成分含量（只有四种元素）
        deduction_price_all：四种元素扣价格（所有组合情况）
        price：所有矿粉价格，可以为短期价格也可为长期价格均值
        deduction_scope_all：三或四种元素折扣区间（所有组合情况）   
        status:0代表高铁含量，1代表低铁含量
    输出：
        result：组合情况
        solution：是否有解的0-1标签
    功能：
        主函数
    """
    result = list(range(0,len(com)))
    objective = []
    solution = []
    for j in range(0,len(com)):
        com_ = com[j]
        powder_ = np.array(powder)[com_]
        res_,objective_,solution_ = build_lp_all(fe,si,ca,mg,al,p,h2o,loi,s,mn,k_,na,zn,pb,as_,cu,size_10,size_8,size_6,\
                              size_1,size_0_15,fe_limit,si_limit,ca_limit,mg_limit,al_limit,p_limit,\
                              h2o_limit,loi_limit,s_limit,mn_limit,k_limit,na_limit,zn_limit,\
                              pb_limit,as_limit,cu_limit,size_10_limit,size_8_limit,size_6_limit,\
                              size_1_limit,size_0_15_limit,com_,powder_,price_on,aim,deduction_price_all,\
                 price,deduction_scope_all,status)
        result[j]=res_
        objective.append(objective_)
        solution.append(solution_)
    return result,objective,solution

def combination_final(species,choice_0,objective_target_0,com_upper,element,result):
    #给定参数初始值,全部设为0
    proportion_0 = pd.DataFrame()
    all_element = []
    for i in range(0,species):
        all_element.append('X'+str(i))
    proportion_0['variable'] = all_element
    proportion_0['value'] = [0]*species

    weight_coef  =[]
    for i in element:
        proportion = copy.deepcopy(proportion_0)
        for j in range(0,len(proportion)):
            globals()[proportion['variable'][j]] = proportion['value'][j]    
        globals()[proportion['variable'][i]] = result[0][choice_0][1][i]
        weight_coef.append(eval(str(objective_target_0)))

    weight_coef = np.array(weight_coef)
    # np.argsort(weight_coef)[:5]   #取最小的特定长度
    element = np.array(element)
    element_final = element[np.argsort(weight_coef)[:com_upper]]  #最终组合结果！！！
    element_final.sort()
    return element_final

    
def take_out_0(fe,si,ca,mg,al,p,h2o,loi,s,mn,k_,na,zn,pb,as_,cu,size_10,size_8,size_6,\
                              size_1,size_0_15,fe_limit,si_limit,ca_limit,mg_limit,al_limit,p_limit,\
                              h2o_limit,loi_limit,s_limit,mn_limit,k_limit,na_limit,zn_limit,\
                              pb_limit,as_limit,cu_limit,size_10_limit,size_8_limit,size_6_limit,\
                              size_1_limit,size_0_15_limit,species,powder,price_on,\
                 aim,deduction_price_all,price,deduction_scope_all,lower,bound,com_lower,com_upper,status):
    """
    输入：
        species：参与计算的矿料种类数
        powder：所有矿料组合对应的元素上下限
        price_on：对标矿粉价格，可以为短期价格也可为长期价格均值
        aim：对标矿粉的成分含量（只有四种元素）
        deduction_price_all：四种元素折扣价格（所有组合情况）
        price：所有矿粉价格，可以为短期价格也可为长期价格均值
        deduction_scope_all：三或四种元素折扣区间（所有组合情况）   
        status:0代表高铁含量，1代表低铁含量，默认为0
        lower:用户设置的各个矿料基础下限
        bound：第一步筛选时候的情况选择，只选择大于target最大值*bound以上的情况
        com_upper: 矿料组合中矿料的数目，默认为=8
    输出：
        choice：筛选出的抵扣情况
        result_final：最优解及一些其他解组成的集合
        solution_final：对应result中的组合是否有解
        max_value：以0为下限的最优解
    功能：
        提高配料效率
    """
    deduction_price_all = pd.DataFrame(deduction_price_all)
    com = [list(range(0,species))]
    result,objective,solution = get_result(fe,si,ca,mg,al,p,h2o,loi,s,mn,k_,na,zn,pb,as_,cu,size_10,size_8,size_6,\
                              size_1,size_0_15,fe_limit,si_limit,ca_limit,mg_limit,al_limit,p_limit,\
                              h2o_limit,loi_limit,s_limit,mn_limit,k_limit,na_limit,zn_limit,\
                              pb_limit,as_limit,cu_limit,size_10_limit,size_8_limit,size_6_limit,\
                              size_1_limit,size_0_15_limit,com,powder,price_on,aim,deduction_price_all,\
                 price,deduction_scope_all,status)
    #选出解最大的情况以及不小于最大值90%的部分
    value = np.array([result[0][i][2] for i in range(0,len(result[0]))])
    choice = np.where(value>max(value)*bound)[0]  #选出后续需要计算的情况序列（情况是指折扣系数的不同组合情况造成的不同线性规划）
    result_final = [[]]*len(choice)
    solution_final = [[]]*len(choice)
    # 判断当前是否有正解，若无，则直接结束计算
    if len(choice) == 0:
        #print('规划问题无正解，无法获得合适的配矿方案')
        return [[]],[[]]
    else:
        combination_target = [result[0][choice[i]] for i in range(0,len(choice))]
        objective_target = [objective[0][choice[i]] for i in range(0,len(choice))]   ####
        combination_take_out_0 = []
        for i in combination_target:
            index_0 = np.where(np.array(i[1]) != 0.0)
            combination_take_out_0.append(np.array(i[0])[index_0])    
        powder_true = copy.deepcopy(powder)
        mandatory = []
        #所有为0的下限转化为lower*0.01
        for i in range(0,len(powder_true)):
            if powder_true[i][0] == 0:
                powder_true[i][0]=lower 
            else:
                powder_true[i][0]=max(lower,powder_true[i][0]) 
                mandatory.extend([i])
        mandatory_number = len(mandatory)
        for i in range(0,len(combination_take_out_0)):
            element_ = list(combination_take_out_0[i])
            element = copy.deepcopy(element_)
            for x in range(0,len(element_)):
                if element_[x] in mandatory:
                    element.remove(element_[x])
            com = []
            #判断剩余矿料的种类数，若大于10，还需要考虑其他方法，若小于10，则按照初始方法进行遍历
            if len(element)<=10-len(mandatory):
                for j in range(max(0,com_lower-mandatory_number),len(element)+1):
                    for k in combinations(element,j): 
                        kk = list(k)+mandatory
                        kk.sort()
                        com.append(kk) 
                result_1,objective_1,solution_1 = get_result(fe,si,ca,mg,al,p,h2o,loi,s,mn,k_,na,zn,pb,as_,cu,size_10,size_8,size_6,\
                              size_1,size_0_15,fe_limit,si_limit,ca_limit,mg_limit,al_limit,p_limit,\
                              h2o_limit,loi_limit,s_limit,mn_limit,k_limit,na_limit,zn_limit,\
                              pb_limit,as_limit,cu_limit,size_10_limit,size_8_limit,size_6_limit,\
                              size_1_limit,size_0_15_limit,com,powder_true,price_on,aim,deduction_price_all[[choice[i]]],\
                 price,deduction_scope_all[[choice[i]]],status)
            elif len(element)>10-len(mandatory) and len(element)<=15-len(mandatory):
                for j in range(min(len(element),com_upper-mandatory_number)-1,min(len(element),com_upper-mandatory_number)+1):
                    for k in combinations(element,j):   
                        kk = list(k)+mandatory
                        kk.sort()
                        com.append(kk) 
                result_1,objective_1,solution_1 = get_result(fe,si,ca,mg,al,p,h2o,loi,s,mn,k_,na,zn,pb,as_,cu,size_10,size_8,size_6,\
                              size_1,size_0_15,fe_limit,si_limit,ca_limit,mg_limit,al_limit,p_limit,\
                              h2o_limit,loi_limit,s_limit,mn_limit,k_limit,na_limit,zn_limit,\
                              pb_limit,as_limit,cu_limit,size_10_limit,size_8_limit,size_6_limit,\
                              size_1_limit,size_0_15_limit,com,powder_true,price_on,aim,deduction_price_all[[choice[i]]],\
                 price,deduction_scope_all[[choice[i]]],status)
            #剩余矿料种类数大于10，
            else:
                element_temp = combination_final(species,choice[i],objective_target[i],min(len(element),com_upper-len(mandatory)),element,result)
                kk = list(element_temp)+mandatory
                kk.sort()
                result_1,objective_1,solution_1 = get_result(fe,si,ca,mg,al,p,h2o,loi,s,mn,k_,na,zn,pb,as_,cu,size_10,size_8,size_6,\
                              size_1,size_0_15,fe_limit,si_limit,ca_limit,mg_limit,al_limit,p_limit,\
                              h2o_limit,loi_limit,s_limit,mn_limit,k_limit,na_limit,zn_limit,\
                              pb_limit,as_limit,cu_limit,size_10_limit,size_8_limit,size_6_limit,\
                              size_1_limit,size_0_15_limit,[kk],powder_true,price_on,aim,\
                 deduction_price_all[[choice[i]]],price,deduction_scope_all[[choice[i]]],status)
            solution_1 = [tt[0] for tt in solution_1]
            result_final[i] = result_1
            solution_final[i] = solution_1
    return result_final,solution_final

#计算前首先判断铁的目标含量属于哪一个区间
def is_overlap(interval_a,interval_b):
    return(max(interval_a[0],interval_b[0]) < min(interval_a[1],interval_b[1]))
    
def judge_fe(fe_limit,deduction_scope_fe_1,deduction_scope_fe_2):
    scope = []
    for i in [deduction_scope_fe_1,deduction_scope_fe_2]:
        temp = is_overlap(fe_limit,i)
        scope.extend([temp])
    return scope

def cal_deduction_price_all_2(deduction_price):
    deduction_price_all_2 = pd.DataFrame()
    for i1,j1 in enumerate((deduction_price[1],deduction_price[2],deduction_price[3])):
        for i2,j2 in enumerate((deduction_price[4],deduction_price[5],deduction_price[6])):
            for i3,j3 in enumerate((deduction_price[7],deduction_price[8])):
                deduction_price_all_2 = deduction_price_all_2.append([[deduction_price[0]]+[j1]+[j2]+[j3]],ignore_index=True)
    deduction_price_all_2 = deduction_price_all_2.T 
    return deduction_price_all_2

def cal_deduction_scope_all_2(deduction_scope_fe_2,deduction_scope_al_1,deduction_scope_al_2,deduction_scope_al_3,deduction_scope_si_1,\
                              deduction_scope_si_2,deduction_scope_si_3,deduction_scope_p_1,deduction_scope_p_2):
    deduction_scope_all_2 = pd.DataFrame()
    for i1,j1 in enumerate((deduction_scope_al_1,deduction_scope_al_2,deduction_scope_al_3)):
        for i2,j2 in enumerate((deduction_scope_si_1,deduction_scope_si_2,deduction_scope_si_3)):
            for i3,j3 in enumerate((deduction_scope_p_1,deduction_scope_p_2)):
                deduction_scope_all_2 = deduction_scope_all_2.append([deduction_scope_fe_2+j1+j2+j3],ignore_index=True)
    deduction_scope_all_2 = deduction_scope_all_2.T 
    return  deduction_scope_all_2

def intercept_coef(aim,deduction_scope_al_1,deduction_scope_al_2,deduction_scope_al_3,\
                   deduction_scope_si_1,deduction_scope_si_2,deduction_scope_si_3,\
                   deduction_scope_p_1,deduction_scope_p_2,deduction_price,deduction_scope_all_2_1):
    fe_aim = aim[0]
    al_aim = aim[1]
    si_aim = aim[2]
    p_aim = aim[3]
    #对标矿粉al含量在第一区间
    if al_aim>=deduction_scope_al_1[0] and al_aim<=deduction_scope_al_1[1]:
        scope_al = 1
    elif al_aim>deduction_scope_al_2[0] and al_aim<=deduction_scope_al_2[1]:
        scope_al = 2
    elif al_aim>deduction_scope_al_3[0] and al_aim<=deduction_scope_al_3[1]:
        scope_al = 3
    if scope_al == 1:
        intercept_al_1 = 0  #截距
        coef_al_1 = al_aim   #边界
        intercept_al_2 = (deduction_scope_al_1[1]-al_aim)*deduction_price[1]
        coef_al_2 = deduction_scope_al_2[0]
        intercept_al_3 = (deduction_scope_al_2[1]-deduction_scope_al_1[1])*deduction_price[2]+\
                         (deduction_scope_al_1[1]-al_aim)*deduction_price[1]
        coef_al_3 = deduction_scope_al_3[0]
        ### (目标元素含量 - coef_al_1)*deduction_price+intercept_al_1
    elif scope_al == 2:
        intercept_al_1 = -(al_aim-deduction_scope_al_1[1])*deduction_price[2]
        coef_al_1 = deduction_scope_al_1[1]
        intercept_al_2 = 0      
        coef_al_2 = al_aim      
        intercept_al_3 = (deduction_scope_al_2[1]-al_aim)*deduction_price[2]
        coef_al_3 = deduction_scope_al_2[1]
    elif scope_al == 3:
        intercept_al_1 = -(al_aim-deduction_scope_al_2[1])*deduction_price[3]-\
                    (deduction_scope_al_2[1]-deduction_scope_al_2[0])*deduction_price[2]                           
        coef_al_1 = deduction_scope_al_1[1]
        intercept_al_2 = -(al_aim-deduction_scope_al_2[1])*deduction_price[3]
        coef_al_2 = deduction_scope_al_2[1]        
        intercept_al_3 = 0      
        coef_al_3 = al_aim   
        
    if si_aim>=deduction_scope_si_1[0] and si_aim<=deduction_scope_si_1[1]:
        scope_si = 1
    elif si_aim>deduction_scope_si_2[0] and si_aim<=deduction_scope_si_2[1]:
        scope_si = 2
    elif si_aim>deduction_scope_si_3[0] and si_aim<=deduction_scope_si_3[1]:
        scope_si = 3
    if scope_si == 1:
        intercept_si_1 = 0  #截距
        coef_si_1 = si_aim   #边界
        intercept_si_2 = (deduction_scope_si_1[1]-si_aim)*deduction_price[4]
        coef_si_2 = deduction_scope_si_2[0]
        intercept_si_3 = (deduction_scope_si_2[1]-deduction_scope_si_1[1])*deduction_price[5]+\
                         (deduction_scope_si_1[1]-si_aim)*deduction_price[4]
        coef_si_3 = deduction_scope_si_3[0]
        ### (目标元素含量 - coef_si_1)*deduction_price+intercept_si_1
    elif scope_si == 2:
        intercept_si_1 = -(si_aim-deduction_scope_si_1[1])*deduction_price[5]
        coef_si_1 = deduction_scope_si_1[1]
        intercept_si_2 = 0      
        coef_si_2 = si_aim      
        intercept_si_3 = (deduction_scope_si_2[1]-si_aim)*deduction_price[5]
        coef_si_3 = deduction_scope_si_2[1]
    elif scope_si == 3:
        intercept_si_1 = -(si_aim-deduction_scope_si_2[1])*deduction_price[6]-\
                    (deduction_scope_si_2[1]-deduction_scope_si_2[0])*deduction_price[5]                           
        coef_si_1 = deduction_scope_si_1[1]
        intercept_si_2 = -(si_aim-deduction_scope_si_2[1])*deduction_price[6]
        coef_si_2 = deduction_scope_si_2[1]        
        intercept_si_3 = 0      
        coef_si_3 = si_aim     
        
    if p_aim>=deduction_scope_p_1[0] and p_aim<deduction_scope_p_1[1]:
        scope_p = 1
    elif p_aim>=deduction_scope_p_2[0] and p_aim<=deduction_scope_p_2[1]:
        scope_p = 2
    if scope_p == 1:
        intercept_p_1 = 0  #截距
        coef_p_1 = p_aim   #边界
        intercept_p_2 = (deduction_scope_p_1[1]-p_aim)*deduction_price[7]*100
        coef_p_2 = deduction_scope_p_2[0]
    elif scope_p == 2:
        intercept_p_1 = -(p_aim-deduction_scope_p_1[1])*deduction_price[8]*100
        coef_p_1 = deduction_scope_p_1[1]
        intercept_p_2 = 0      
        coef_p_2 = p_aim      
    
    intercept = pd.DataFrame()
    for i1 in [intercept_al_1,intercept_al_2,intercept_al_3]:
        for i2 in [intercept_si_1,intercept_si_2,intercept_si_3]:
            for i3 in [intercept_p_1,intercept_p_2]:
                intercept = intercept.append(pd.DataFrame([i1,i2,i3]).T,ignore_index=True)
    intercept = intercept.T 
    coef = pd.DataFrame()
    for i1 in [coef_al_1,coef_al_2,coef_al_3]:
        for i2 in [coef_si_1,coef_si_2,coef_si_3]:
            for i3 in [coef_p_1,coef_p_2]:
                coef = coef.append(pd.DataFrame([i1,i2,i3]).T,ignore_index=True)
    coef = coef.T 
    deduction_scope_all_2 = deduction_scope_all_2_1.append([intercept,coef]).reset_index(drop = True)
    return deduction_scope_all_2

def cal_deduction_scope_all_1(deduction_scope_fe_1,deduction_scope_al):
    #低铁含量折扣范围构造
    deduction_scope_all_1 = deduction_scope_fe_1+deduction_scope_al
    deduction_scope_all_1 = pd.DataFrame(deduction_scope_all_1)
    return deduction_scope_all_1
    
def ore_blending(fe,si,ca,mg,al,p,h2o,loi,s,mn,k_,na,zn,pb,as_,cu,size_10,size_8,size_6,\
                              size_1,size_0_15,fe_limit,si_limit,ca_limit,mg_limit,al_limit,p_limit,\
                              h2o_limit,loi_limit,s_limit,mn_limit,k_limit,na_limit,zn_limit,\
                              pb_limit,as_limit,cu_limit,size_10_limit,size_8_limit,size_6_limit,\
                              size_1_limit,size_0_15_limit,deduction_scope_fe_1,deduction_scope_fe_2,\
                              species,powder,price_on,aim,deduction_price_all_1,deduction_price,price,\
                              deduction_scope_al,deduction_scope_al_1,deduction_scope_al_2,deduction_scope_al_3,\
                              deduction_scope_si_1,deduction_scope_si_2,deduction_scope_si_3,deduction_scope_p_1,\
                              deduction_scope_p_2,lower,bound,com_lower,com_upper,num):
    
    """
    num: 最大输出方案数目
    """
    fe = np.array(fe)
    si = np.array(si)
    ca = np.array(ca)
    mg = np.array(mg)
    al = np.array(al)
    p = np.array(p)
    h2o = np.array(h2o)
    loi = np.array(loi)
    s = np.array(s)
    mn = np.array(mn)
    k_ = np.array(k_)
    na = np.array(na)
    zn = np.array(zn)
    pb = np.array(pb)
    as_ = np.array(as_)
    cu = np.array(cu)
    size_10 = np.array(size_10)
    size_8 = np.array(size_8)
    size_6 = np.array(size_6)
    size_1 = np.array(size_1)
    size_0_15 = np.array(size_0_15)
    price = np.array(price)
    
    deduction_price_all_2 = cal_deduction_price_all_2(deduction_price)
    deduction_scope_all_2_1 = cal_deduction_scope_all_2(deduction_scope_fe_2,deduction_scope_al_1,deduction_scope_al_2,deduction_scope_al_3,deduction_scope_si_1,\
                              deduction_scope_si_2,deduction_scope_si_3,deduction_scope_p_1,deduction_scope_p_2)
    deduction_scope_all_2 = intercept_coef(aim,deduction_scope_al_1,deduction_scope_al_2,deduction_scope_al_3,\
                   deduction_scope_si_1,deduction_scope_si_2,deduction_scope_si_3,\
                   deduction_scope_p_1,deduction_scope_p_2,deduction_price,deduction_scope_all_2_1)
    deduction_scope_all_1 = cal_deduction_scope_all_1(deduction_scope_fe_1,deduction_scope_al)
    deduction_price_all_1 = pd.DataFrame(deduction_price_all_1)
    scope = judge_fe(fe_limit,deduction_scope_fe_1,deduction_scope_fe_2)
    if scope[0] == True:
        deduction_price_all = deduction_price_all_1
        deduction_scope_all = deduction_scope_all_1
        fe_limit_1 = [fe_limit[0],min(fe_limit[1],60)]
        result_final_0,solution_final_0 = take_out_0(fe,si,ca,mg,al,p,h2o,loi,s,mn,k_,na,zn,pb,as_,cu,size_10,size_8,size_6,\
                              size_1,size_0_15,fe_limit_1,si_limit,ca_limit,mg_limit,al_limit,p_limit,\
                              h2o_limit,loi_limit,s_limit,mn_limit,k_limit,na_limit,zn_limit,\
                              pb_limit,as_limit,cu_limit,size_10_limit,size_8_limit,size_6_limit,\
                              size_1_limit,size_0_15_limit,species,powder,price_on,aim,deduction_price_all,\
                 price,deduction_scope_all,lower,bound,com_lower,com_upper,status=0)
        status_temp = copy.deepcopy(solution_final_0)
        status_0 = [[x-1 for x in status_temp[0]]]
        
    else:
        result_final_0 = [[]]
        solution_final_0 = [[]]
        status_0 = [[]]
    if scope[1] == True:
        deduction_price_all = deduction_price_all_2
        deduction_scope_all = deduction_scope_all_2
        fe_limit_2 = [max(60,fe_limit[0]),fe_limit[1]]
        result_final_1,solution_final_1 = take_out_0(fe,si,ca,mg,al,p,h2o,loi,s,mn,k_,na,zn,pb,as_,cu,size_10,size_8,size_6,\
                              size_1,size_0_15,fe_limit_2,si_limit,ca_limit,mg_limit,al_limit,p_limit,\
                              h2o_limit,loi_limit,s_limit,mn_limit,k_limit,na_limit,zn_limit,\
                              pb_limit,as_limit,cu_limit,size_10_limit,size_8_limit,size_6_limit,\
                              size_1_limit,size_0_15_limit,species,powder,price_on,aim,deduction_price_all,\
                 price,deduction_scope_all,lower,bound,com_lower,com_upper,status=1)
        status_1 = copy.deepcopy(solution_final_1)
    else:
        result_final_1 = [[]]
        solution_final_1 = [[]]
        status_1 = [[]]
    result = result_final_0+result_final_1
    solution = solution_final_0+solution_final_1
    status_ = status_0+status_1
    final = []
    status_final = []
    for i in range(0,len(result)):
        temp = np.array(result[i])[np.where(np.array(solution[i])!=0)] 
        temp_status = np.array(status_[i])[np.where(np.array(solution[i])!=0)] 
        final.extend(list(temp)) 
        status_final.extend(list(temp_status)) 
    number = pd.DataFrame(final).shape[0]
    result_on = pd.DataFrame()
    temp_1 = []
    for i in range(0,number):
        b = len(final[i])
        for j in range(0,b):
            temp_1.append(final[i][j][0])
    temp_2 = []
    for i in range(0,number):
        b = len(final[i])
        for j in range(0,b):
            temp_2.append(final[i][j][1])
    temp_3 = []
    for i in range(0,number):
        b = len(final[i])
        for j in range(0,b):
            temp_3.append(final[i][j][2])
    temp_4 = []
    for i in range(0,number):
        b = len(final[i])
        for j in range(0,b):
            temp_4.append(final[i][j][3])    
    result_on['combination'] = temp_1
    result_on['proportion'] = temp_2
    result_on['profit_on'] = temp_3   #对标矿料的利润
    result_on['class'] = temp_4   #类别
    result_on['status'] = status_final
    #按照利润排序
    result_on = result_on.sort_values(by="profit_on" , ascending=False).reset_index(drop=True)
    if num>number:
        num = number
    return result_on[:num],deduction_price_all_1,deduction_price_all_2,deduction_scope_all_2
    
if __name__ == '__main__':
    #所有矿粉的元素含量
    #元素含量有空缺的时候会报错（需要完整数据）
    data = pd.read_excel('成分大数据测试.xlsx',sheetname = '结果')
    species = 22
    fe = list(data['TFe'][:species])
    si = list(data['SiO2'][:species])
    ca = list(data['CaO'][:species])
    mg = list(data['MgO'][:species])
    al = list(data['Al2O3'][:species])
    p = list(data['P'][:species])
    #s = list(data['S'][:species])
    s=[0.092, 0.085, 0.12, 0.1, 0.057, 0.045, 0.074, 0.052, 0.06999999999999999, 0.09, 0.05, 0.07, 0.1, 0.05, 0.059, 0.02, 0.06, 0.058, 0.09, 0.06, 0.06, 0.05]
    h2o = list(data['H2O'][:species])
    loi = list(data['LOI'][:species])
    mn=fe
    k_=fe
    na=fe
    zn=fe
    pb=fe
    as_=fe
    cu=fe
    size_10=fe
    size_8=fe
    size_6=fe
    size_1=fe
    size_0_15=fe
    #各个元素上下限（增加限制条件）
    a=23;b=25
    fe_limit = list(data['TFe'][a:b].reset_index(drop=True))
    si_limit = list(data['SiO2'][a:b].reset_index(drop=True))
    ca_limit = list(data['CaO'][a:b].reset_index(drop=True))
    mg_limit = list(data['MgO'][a:b].reset_index(drop=True))
    al_limit = list(data['Al2O3'][a:b].reset_index(drop=True))
    p_limit = list(data['P'][a:b].reset_index(drop=True))
    #s_limit = list(data['S'][a:b].reset_index(drop=True))  #该参数此处为空，对算法没有影响
    s_limit = [0,100]
    h2o_limit = list(data['H2O'][a:b].reset_index(drop=True))
    loi_limit = list(data['LOI'][a:b].reset_index(drop=True))
    mn_limit = [0,100]
    k_limit = [0,100]
    na_limit = [0,100]
    zn_limit = [0,100]
    pb_limit = [0,100]
    as_limit = [0,100]
    cu_limit = [0,100]
    size_10_limit = [0,100]
    size_8_limit = [0,100]
    size_6_limit = [0,100]
    size_1_limit = [0,100]
    size_0_15_limit = [0,100]
    
    #每种矿粉比例上下限
    powder = [[0,0.2],[0,0.2],[0,0.2],[0,0.2],[0,0.5],[0,0.2],\
              [0,0.3],[0,0.2],[0,0.4],[0,0.4],[0,0.2],\
              [0.08,0.25],[0,0.2],[0,0.2],[0,0.2],[0,0.2],[0,0.2],[0.03,0.2],\
              [0,0.2],[0,0.2],[0,0.2],[0,0.2]]
    
    #所有矿粉的价格
    price_1214 = list(data.iloc[:22,15])
    price_918 = list(data.iloc[:22,14])
    
    #成分抵扣单价
    data_deduction = pd.read_excel('矿粉成分及价格信息.xlsx',sheetname = '成分抵扣')
    deduction_price_918 = list(data_deduction.loc[14*2:14*2+8,'￥/t'].reset_index(drop=True))
    deduction_price_1214 = list(data_deduction.loc[14*5:14*5+8,'￥/t'].reset_index(drop=True))
    
    deduction_price_all_1_918 = [7.73,22.83,18.41]
    deduction_price_all_1_1214 = [8.10,16.94,3.68]
    
    #成分抵扣范围
    deduction_scope_fe_1 = [55,60]
    deduction_scope_fe_2 = [60,63.5]
    deduction_scope_al_1 = [0,1]
    deduction_scope_al_2 = [1,2.5]
    deduction_scope_al_3 = [2.5,4]
    deduction_scope_si_1 = [0,4.5]
    deduction_scope_si_2 = [4.5,6.5]
    deduction_scope_si_3 = [6.5,9]
    deduction_scope_p_1 = [0,0.09]
    deduction_scope_p_2 = [0.09,0.12]
    deduction_scope_al = [0,5]
    
    aim_mk = [60.8,2.25,4.75,0.085]
    aim_pb = [61.500,2.300,3.800,0.100]
    
    lower=0.05;bound=0.9;com_lower=2;com_upper=8;num=15
#    price = price_918
#    price_mk = price[1]
#    price_pb = price[3]
#    deduction_price_all_1 = deduction_price_all_1_918
#    deduction_price  =deduction_price_918
#    
#    price_on = price_pb
#    aim = aim_pb
#    outputs_1 = ore_blending(fe,si,ca,mg,al,p,s,h2o,loi,fe_limit,si_limit,ca_limit,mg_limit,\
#                 al_limit,p_limit,s_limit,h2o_limit,loi_limit,deduction_scope_fe_1,\
#                 deduction_scope_fe_2,species,powder,price_on,aim,deduction_price_all_1,\
#                 deduction_price,price,deduction_scope_al,deduction_scope_al_1,\
#                 deduction_scope_al_2,deduction_scope_al_3,deduction_scope_si_1,\
#                deduction_scope_si_2,deduction_scope_si_3,deduction_scope_p_1,deduction_scope_p_2,\
#                lower,bound,com_lower,com_upper,num)
#    price_on = price_mk
#    aim = aim_mk
#    outputs_2 = ore_blending(fe,si,ca,mg,al,p,s,h2o,loi,fe_limit,si_limit,ca_limit,mg_limit,\
#                 al_limit,p_limit,s_limit,h2o_limit,loi_limit,deduction_scope_fe_1,\
#                 deduction_scope_fe_2,species,powder,price_on,aim,deduction_price_all_1,\
#                 deduction_price,price,deduction_scope_al,deduction_scope_al_1,\
#                 deduction_scope_al_2,deduction_scope_al_3,deduction_scope_si_1,\
#                deduction_scope_si_2,deduction_scope_si_3,deduction_scope_p_1,deduction_scope_p_2,\
#                lower,bound,com_lower,com_upper,num)
    
    price = price_1214
    price_mk = price[1]
    price_pb = price[3]
    deduction_price_all_1 = deduction_price_all_1_1214
    deduction_price  =deduction_price_1214
    
    price_on = price_pb
    aim = aim_pb
    output_3,deduction_price_all_1,deduction_price_all_2,deduction_scope_all_2 = ore_blending(fe,si,ca,mg,al,p,h2o,loi,s,mn,k_,na,zn,pb,as_,cu,size_10,size_8,size_6,\
                              size_1,size_0_15,fe_limit,si_limit,ca_limit,mg_limit,al_limit,p_limit,\
                              h2o_limit,loi_limit,s_limit,mn_limit,k_limit,na_limit,zn_limit,\
                              pb_limit,as_limit,cu_limit,size_10_limit,size_8_limit,size_6_limit,\
                              size_1_limit,size_0_15_limit,deduction_scope_fe_1,deduction_scope_fe_2,\
                              species,powder,price_on,aim,deduction_price_all_1,deduction_price,price,\
                              deduction_scope_al,deduction_scope_al_1,deduction_scope_al_2,deduction_scope_al_3,\
                              deduction_scope_si_1,deduction_scope_si_2,deduction_scope_si_3,deduction_scope_p_1,\
                              deduction_scope_p_2,lower,bound,com_lower,com_upper,num)
#    price_on = price_mk
#    aim = aim_mk
#    outputs_4 = ore_blending(fe,si,ca,mg,al,p,s,h2o,loi,fe_limit,si_limit,ca_limit,mg_limit,\
#                 al_limit,p_limit,s_limit,h2o_limit,loi_limit,deduction_scope_fe_1,\
#                 deduction_scope_fe_2,species,powder,price_on,aim,deduction_price_all_1,\
#                 deduction_price,price,deduction_scope_al,deduction_scope_al_1,\
#                 deduction_scope_al_2,deduction_scope_al_3,deduction_scope_si_1,\
#                deduction_scope_si_2,deduction_scope_si_3,deduction_scope_p_1,deduction_scope_p_2,\
#                lower,bound,com_lower,com_upper,num)
##方案计算出来后，计算成本,
#outputs_1.loc[0,'combination']
##成本
#cost = np.dot(pd.Series(price)[outputs_1.loc[0,'combination']],outputs_1.loc[0,'proportion'])
##价格
#pricing = cost+outputs_1.loc[0,'profit_on']
#
##已知矿料、矿料配比，计算对标其它矿粉的对标价格
##根据目标fe含量，判断status,此方法行不通，无法准确考虑边界条件
#fe_proportion = np.dot(pd.Series(fe)[outputs_1.loc[0,'combination']],outputs_1.loc[0,'proportion'])
