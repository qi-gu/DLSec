from nlp_bert.attack_hiddenkiller import attack_hiddenkiller
from nlp_bert.attack_lws import attack_lws
def run(params:dict,model=None):
    import csv
    # 打开你的 csv 文件
    with open('nlp_bert/weight.csv', 'r') as f:
        # 创建 csv reader
        reader = csv.reader(f)
        # 将第二列的所有值读取到一个列表中
        column2 = [row[1] for row in reader]

    # 获取最后三个值
    weight = column2[-3:]
    weight = [float(x) for x in weight]

    if params['back']=='lws':
        before_val_attack_acc,before_val_attack2_acc,before_robust_test_acc,defense_val_attack_acc,defense_val_attack2_acc,defense_robust_test_acc=attack_lws(params,model=model)
        total=defense_val_attack_acc*(weight[0])+(1-defense_val_attack2_acc)*(weight[1])+defense_robust_test_acc*(weight[2])
        return defense_val_attack_acc,1-defense_val_attack2_acc,defense_robust_test_acc,total
        
    if params['back']=='hiddenkiller':
        poison_success_rate_test,clean_acc,robust_acc,test_acc,poison_success_rate=attack_hiddenkiller(params,model=model)
        total=clean_acc*weight[0]+(1-poison_success_rate_test)*weight[1]+robust_acc*weight[2]
        return clean_acc,1-poison_success_rate_test,robust_acc