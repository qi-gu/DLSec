import csv

import torch
import torchvision
import cv2
import os
from torch.utils.data import DataLoader
from Adversarial.adversarial_api import adversarial_attack
from Adversarial.adversarial_api import adversarial_mutiple_attack
from Backdoor.backdoor_defense_api import run_backdoor_defense
from Datapoison.datapoison_api import *
from utils.transform import build_transform
import copy
from EvaluationConfig import *
from Datapoison.Defense.Friendly_noise import *
from Backdoor.data_analyze import ss_test,ac_test
import pandas as pd
import re
import numpy as np
import requests
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from utils.utils import raw_data_process
def ModelEvaluation(evaluation_params=None):
    '''
    进行模型评测
    @param evaluation_params: 相关参数
    @return:
    '''
    train_dataloader, test_dataloader = dataset_preprocess(name=evaluation_params['use_dataset'], batch_size=evaluation_params['batch_size'])
    train_dataloader=ac_test(train_dataloader.dataset,evaluation_params['model'])
    test_dataloader=ac_test(test_dataloader.dataset,evaluation_params['model'])
    ReinforcedModel_dict_path = run_test_on_model(evaluation_params['model'], evaluation_params['allow_backdoor_defense'], evaluation_params['backdoor_method'], evaluation_params['datapoison_method'], evaluation_params['run_datapoison_reinforcement'],
                                                  evaluation_params['datapoison_reinforce_method'], train_dataloader, test_dataloader, evaluation_params)
    # score=test_score()
    # return score
    return 
    # 防御后的模型再测试
    # evaluation_params['model'].load_state_dict(torch.load(ReinforcedModel_dict_path))
    # run_test_on_model(evaluation_params['model'], evaluation_params['allow_backdoor_defense'], evaluation_params['backdoor_method'], evaluation_params['datapoison_method'], evaluation_params['run_datapoison_reinforcement'],
    #                                  evaluation_params['datapoison_reinforce_method'], train_dataloader, test_dataloader, evaluation_params)


def dataset_preprocess(name, batch_size=64):
    '''
    将用户指定的数据集加载为Dataloader
    @param name: 数据集名
    @param batch_size:
    @return:
    '''
    if name is None:
        return None, None
    elif name in torchvision.datasets.__all__:
        selected_dataset = getattr(torchvision.datasets, name)
        train_dataset = selected_dataset("./data", train=True, download=True)
        test_dataset = selected_dataset("./data", train=False, download=True)
    else:
        if os.path.exists(name + "/train") and os.path.exists(name + "/test"):
            train_dataset = torchvision.datasets.DatasetFolder(root=name + "/train", loader=cv2.imread, extensions=('png', 'jpeg'))
            test_dataset = torchvision.datasets.DatasetFolder(root=name + "/test", loader=cv2.imread, extensions=('png', 'jpeg'))
        else:
            custom_dataset = torchvision.datasets.DatasetFolder(root=name, loader=cv2.imread, extensions=('png', 'jpeg'))
            transform, _, _ = build_transform(custom_dataset[0][0].shape, isinstance(custom_dataset[0], torch.Tensor))
            custom_dataset.transform = transform
            return DataLoader(custom_dataset, batch_size=batch_size, shuffle=True), None

    transform, _, _ = build_transform(train_dataset.data.shape[1:], isinstance(train_dataset.data[0], torch.Tensor))
    train_dataset.transform = transform
    test_dataset.transform = transform

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def run_test_on_model(Model2BeEvaluated, allow_backdoor_defense, backdoor_method, datapoison_method, run_datapoison_reinforcement, datapoison_reinforce_method, train_dataloader=None, test_dataloader=None, params=None):
    '''
    单轮测试
    @param Model2BeEvaluated: 待测模型
    @param allow_backdoor_defense: 是否要执行后门防御
    @param backdoor_method: 后门测试方式
    @param datapoison_method: 数据投毒测试
    @param run_datapoison_reinforcement:是否要执行数据投毒防御
    @param datapoison_reinforce_method: 数据投毒测试方式
    @param train_dataloader: 各模块中用于训练的dataloader
    @param test_dataloader: 各模块中用于测试的dataloader
    @param params: 原始参数，部分方法会使用其中对应部分的参数
    @return: 防御后的模型权重文件的路径，如果不执行防御则为None
    '''
    print(params)
    # exit(0)
    adversarial_rst = adversarial_test(Model2BeEvaluated,train_dataloader=test_dataloader, params=params)
    isBackdoored, backdoor_rst, ReinforcedModel_dict_path = backdoor_detect_and_defense(allow_defense=allow_backdoor_defense, Model2BeEvaluated=Model2BeEvaluated, method=backdoor_method, train_dataloader=train_dataloader, params=params)
    datapoison_test_rst = datapoison_test(params=params)
    if run_datapoison_reinforcement:
        if ReinforcedModel_dict_path is not None:
            DatapoisonReinforceModel = copy.deepcopy(Model2BeEvaluated)
            DatapoisonReinforceModel.load_state_dict(torch.load(ReinforcedModel_dict_path))
        else:
            DatapoisonReinforceModel = copy.deepcopy(Model2BeEvaluated)
        ReinforcedModel_dict_path, datapoison_defense_rst = datapoison_defense(TargetModel=DatapoisonReinforceModel, method=datapoison_reinforce_method, train_dataloader=train_dataloader, params=params)
    else:
        datapoison_defense_rst = None

    process_result(params['tag'], adversarial_rst, backdoor_rst, datapoison_test_rst, datapoison_defense_rst)
    return ReinforcedModel_dict_path


def adversarial_test(Model2BeEvaluated, method='fgsm', train_dataloader=None, params=None):
    """
    对抗样本攻击测试
    @param Model2BeEvaluated:
    @param method: 目前对抗样本测试将每个方法都进行了执行，不需要特别指定method，但尚未从旧版代码中删除
    @param train_dataloader:
    @param params:
    @return: 一个字典，键形如’ACC-0.005‘，’fgsm-0.005‘，值为相应准确率
    """
    adversarial_rst = {}
    print("开始图像鲁棒性检测")
    perturb_rst,correct_con,false_con,acc = adversarial_attack(Model2BeEvaluated, method, train_dataloader, params)
    for ep_rst in perturb_rst:
        ep = ep_rst[0]
        adversarial_rst["ACC-" + str(ep)] = ep_rst[1]
        adversarial_rst["NoisyACC-" + str(ep)] = ep_rst[2]
        adversarial_rst["BlurredACC-" + str(ep)] = ep_rst[3]
        adversarial_rst["CompressedACC-" + str(ep)] = ep_rst[4]
    adversarial_rst["ACTC"]=correct_con
    adversarial_rst["ACAC"]=false_con
    # adversarial_rst["ACC"] = acc
    print("开始多方法对抗样本测试")
    adv_rst = adversarial_mutiple_attack(Model2BeEvaluated, train_dataloader, params)
    for method_rst in adv_rst:
        ep = method_rst[0]
        adversarial_rst[str(method_rst[1]) + '-' + str(ep)] = method_rst[2]
    return adversarial_rst


def backdoor_detect_and_defense(allow_defense=True, Model2BeEvaluated=None, method='NeuralCleanse', train_dataloader=None, params=None):
    '''
    后门检测与防御
    @param allow_defense: 是否执行防御
    @param Model2BeEvaluated: 待测模型
    @param method: 后门检测方法
    @param train_dataloader: 训练集
    @param params: 原始参数
    @return: 是否有后门、后门评测结果、防御后模型权重文件路径
    '''
    print("开始后门检测")
    isBackdoored, backdoor_rst, ReinforcedModel_dict_path = run_backdoor_defense(allow_defense, Model2BeEvaluated, method, train_dataloader, params)
    print("-" * 20, "后门攻击评测结果", "-" * 20)
    print("后门检测：", end="")
    if isBackdoored:
        print("检测到后门存在于标签", backdoor_rst['backdoor_label'])
        print("防御后模型已存入目录", ReinforcedModel_dict_path)
    else:
        print("未检测到后门")
    return isBackdoored, backdoor_rst, ReinforcedModel_dict_path


def datapoison_test(params=None):
    '''
    数据投毒检测
    @param params: 原始参数
    @return: 数据投毒检测评测结果
    '''
    print("开始投毒检测")
    datapoison_test_rst = run_datapoison_test(params=params)

    return datapoison_test_rst


def datapoison_defense(TargetModel=None, method=None, train_dataloader=None, params=None):
    '''
    数据投毒防御
    @param TargetModel: 待测模型
    @param method: 数据投毒防御方法
    @param train_dataloader: 训练集
    @param params: 原始参数
    @return: 防御后模型路径、数据投毒防御结果
    '''
    print("开始投毒防御")
    reinforced_model_path, datapoison_defense_rst = run_datapoison_reinforce(TargetModel, method=method, train_dataloader=train_dataloader, params=params)
    return reinforced_model_path, datapoison_defense_rst


def process_result(tag="DefaultTag", adversarial_rst=None, backdoor_rst=None, datapoison_rst=None, datapoison_defense_rst=None):
    """
    对所有评测结果进行汇总处理，目前是将结果保存在一个csv内
    @param tag: 标签
    @param adversarial_rst:对抗样本评测结果
    @param backdoor_rst: 后门检测与防御评测结果
    @param datapoison_rst: 数据投毒检测评测结果
    @param datapoison_defense_rst: 数据投毒防御评测结果
    @return:
    """
    print("Evaluation Results for", tag)
    print('adversarial_rst:', adversarial_rst)
    print('backdoor_rst:', backdoor_rst)
    print('datapoison_rst:', datapoison_rst)
    print('datapoison_defense_rst:', datapoison_defense_rst)
    final_rst = {'tag': tag}
    if adversarial_rst is not None:
        final_rst.update(adversarial_rst)
    if backdoor_rst is not None:
        final_rst.update(backdoor_rst)
    if datapoison_rst is not None:
        final_rst.update(datapoison_rst)
    if datapoison_defense_rst is not None:
        final_rst.update(datapoison_defense_rst)

    final_rst=raw_data_process(final_rst)
    print(final_rst)
    with open('./ModelResults.csv', 'r', newline='') as csvfile:
        data = []
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
        headers = set(data[0].keys())
        headers.update(final_rst.keys())
        data.append(final_rst)

    with open('./ModelResults.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    return

def test_score():
    df=pd.read_csv('./ModelResults.csv')
    # 删除列,
    df = df.drop(columns=['tag','backdoor_num'])
    df['backdoor_label'] = df['backdoor_label'].apply(lambda x: len(re.findall(r'\d+', x)) if isinstance(x, str) else 0)

    df = df.fillna(0)
    index=['backdoor_label','trigger_std','trigger_size','PoisonSR'] 
    for row in index:
        df[row]=max(df[row])-df[row]
    
    df_norm = df.apply(normalize)
    print(df_norm)
    m = len(df) 

    # 计算信息熵
    E = -1/np.log(m) * (df_norm*np.log(df_norm)).sum()

    # Normalize entropy
    E_norm = E.values / np.log(m)

    # 计算权重
    w = (1 - E_norm) / sum(1 - E_norm)
    w = pd.Series(w, index=df.columns)

    # 利用 Pandas 的 apply 方法进行乘法操作
    scores = df_norm * w

    # 对每行数据进行求和以得到一个总分
    total_scores = scores.sum(axis=1)

    
    last_score = df_norm.tail(1)*100
    last_total_score=total_scores.tail(1)
    last_scores_json = last_score.to_json()
    last_total_scores_json = last_total_score.to_json()

    # 合并这两个 JSON 字符串
    result_json = '{"scores": ' + last_scores_json + ', "total_scores": ' + last_total_scores_json + '}'
    print(result_json)

    return result_json



def normalize(column):
    ymin = 0.002
    ymax = 1
    return (column - column.min()) / (column.max() - column.min()) * (ymax - ymin) + ymin


if __name__ == '__main__':
    ModelEvaluation(evaluation_params=evaluation_params)

   
