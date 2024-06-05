from nlp_bert.attack_hiddenkiller import attack_hiddenkiller
from nlp_bert.attack_lws import attack_lws
def run(params:dict,model=None):
    if params['back']=='lws':
        # return 0.1,1,0.5,3
        before_val_attack_acc,before_val_attack2_acc,before_robust_test_acc,defense_val_attack_acc,defense_val_attack2_acc,defense_robust_test_acc=attack_lws(params,model=model)
        total=(defense_val_attack_acc+1-defense_val_attack2_acc+defense_robust_test_acc)/3
        return defense_val_attack_acc,1-defense_val_attack2_acc,defense_robust_test_acc,total
        
    if params['back']=='hiddenkiller':
        # return 0,1,0.5,5
        poison_success_rate_test,clean_acc,robust_acc,test_acc,poison_success_rate=attack_hiddenkiller(params,model=model)
        total =(poison_success_rate_test+1-clean_acc+robust_acc)/3
        return poison_success_rate_test,clean_acc,robust_acc