from nlp_bert.attack_hiddenkiller import attack_hiddenkiller
from nlp_bert.attack_lws import attack_lws
def run(params:__dict__,model=None):
    if params['back']=='lws':
        attack_lws(params,model=model)
    if params['back']=='hiddenkiller':
        attack_hiddenkiller(params,model=model)