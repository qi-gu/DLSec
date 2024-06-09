import json
from django.shortcuts import render,HttpResponse
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage 
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponseRedirect
from django.contrib import messages
from multiprocessing import Process
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from EvaluationConfig import evaluation_params
from EvaluationPlatformNEW import ModelEvaluation
from audio.TestingAudio import audio_test
from nlp_bert.nlp_api import run
cv_file_url = None
nlp_file_url = None
audio_file_url = None

_BUILTIN_MODELS = [
    "librispeech_pretrained_v3",
    "an4_pretrained_v3",
    "ted_pretrained_v3",
]

# Create your views here.
def test(request):
    return HttpResponse("Hello, world.")

def index(request):
    return render(request, "index.html")

def cv(request):
    status={
        'state':'未运行',
        'score':None
    }
    if request.method == 'POST':
    
        # if request.POST.get('action')=='setting':

        status={
            'state':'运行已结束',
            'score':None
        }
        global cv_file_url
        if cv_file_url is None:
            messages.error(request,"请先上传模型文件！")
            return render(request,"cv.html",status)
        print(request.POST)
        dataset = request.POST.get('dataset')
        model_type = request.POST.get('model')
        adver = request.POST.get('adver')
        back = request.POST.get('back')
        poison = request.POST.get('poison')
        args = request.POST.get('args')
        backdoor_method = request.POST.get('backdoor_method')
        print(dataset,adver,back,poison,args,backdoor_method)
        
        evaluation_params['use_dataset'] = dataset
        evaluation_params['backdoor_method'] = backdoor_method
        evaluation_params['datapoison_method'] = poison
        # evaluation_params['datapoison_reinforce_method'] = adver

        
        model_type = (dataset+"_"+model_type).lower()
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
        model.load_state_dict(torch.load('.'+cv_file_url))
        evaluation_params['model'] =model
        score=ModelEvaluation(evaluation_params)
        status['score'] = json.dumps(score)
        print(score)
        
        # 对evaluation_params进行对于修改，一些比较基础的参数无需修改
        # Process(target=ModelEvaluation,args=[evaluation_params]).start()
        # elif request.POST.get('action')=='stop':
        #     status={
        #         'state':'运行已经结束'
        #     }
   
    
    print(status['score'])
    return render(request, "cv.html",status)

def get_test_result(request):
    result=request.session.get('score',None)
    return JsonResponse({'result':result})

def upload(request):
    if request.method == 'POST':
        global cv_file_url
        uploaded_file = request.FILES['model']
        evaluation_params['model'] = uploaded_file
        print("change model")
        fs = FileSystemStorage()
        name=fs.save(uploaded_file.name, uploaded_file)                             
        cv_file_url=fs.url(name)
        return JsonResponse({'result':"successfully upload the model"})


def nlp(request):
    return render(request, "nlp.html")

def audio(request):
    return render(request, "audio.html")

def nlp_setting(request):
    if request.method == 'POST':
       status={
           'state':'正在运行',
           'score':None,
       }
    '''
    获取表单中的数据
    '''
    dataset = request.POST.get('dataset')
    # model_type = request.POST.get('model')
   
    back = request.POST.get('back')
    
    args = request.POST.get('args')
    backdoor_method = request.POST.get('backdoor_method')
    print(dataset,back,args,backdoor_method)
    params={
        'dataset':dataset,
        'back':back,
    }
    
    # status["score"] = json.dumps(run(params=params))#实际运行函数
    status["score"] = json.dumps({"scores":{"acc":0.9,"asr":0.7,"robust":0.85},"total_scores":0.8*100})#调试使用例子
    return render(request,"nlp.html",status)

def nlp_upload(request):
     if request.method == 'POST':
        uploaded_file = request.FILES['model']
        evaluation_params['model'] = uploaded_file
        global nlp_file_url
        fs = FileSystemStorage()
        name=fs.save(uploaded_file.name, uploaded_file)     
        nlp_file_url=fs.url(name)
        return JsonResponse({'result':"success"})

def audio_setting(request):
    if request.method == 'POST':
        status = {
            'state': '正在运行',
        }
        '''样例
        config = {
            'model': model,
            'goal': 'Hello world',
            'recipes': 'fgsm',
            'device': 'cuda'
        }'''
        model = request.POST.get('model')
        goal = request.POST.get('goal')
        recipes = request.POST.get('recipes')
  
        fgsm = 'fgsm' in request.POST.getlist('back')
        pgd = 'pgd' in request.POST.getlist('back')
        genetic = 'genetic' in request.POST.getlist('back')
        cw = 'cw' in request.POST.getlist('back')
        icw = 'icw' in request.POST.getlist('back')
        print(fgsm, pgd, genetic, cw, icw)
        params = {}
        # 如果model是内置的，就直接传字符串；如果是上传的，就传文件路径（但现在不支持，价格判断）
        if model in _BUILTIN_MODELS:
            params['model'] = model
        else:
            params['model'] = "." + audio_file_url  # TODO: 这里应该是一个文件路径，这样写对吗？
        params['goal'] = goal
        params['recipes'] = recipes
        status["score"] = json.dumps(audio_test(params))
        return render(request, "audio.html", status)

def audio_upload(request):
    # 保存上传模型文件
    global audio_file_url
    if request.method == 'POST':
        uploaded_file = request.FILES['model']
        evaluation_params['model'] = uploaded_file
        print("change model")
        fs = FileSystemStorage()
        name=fs.save(uploaded_file.name, uploaded_file)
        audio_file_url=fs.url(name)
        return JsonResponse({'result':"success"})

def upload(request):
    global cv_file_url
    if request.method == 'POST':
        uploaded_file = request.FILES['model']
        evaluation_params['model'] = uploaded_file
        print("change model")
        fs = FileSystemStorage()
        name=fs.save(uploaded_file.name, uploaded_file)
        cv_file_url=fs.url(name)
        return JsonResponse({'result':"success"})
    
def setting (request):
    if request.method == 'POST':
       status={
           'state':'正在运行',
       }
    '''
    获取表单中的数据
    '''
    dataset = request.POST.get('dataset')
    model_type = request.POST.get('model')
    adver = request.POST.get('adver')
    back = request.POST.get('back')
    poison = request.POST.get('poison')
    args = request.POST.get('args')
    backdoor_method = request.POST.get('backdoor_method')
    print(dataset,adver,back,poison,args,backdoor_method)
    global cv_file_url
    model_type = (dataset+"_"+model_type).lower()
    model = torch.hub.load("chenyaofo/pytorch-cifar-models",model_type, pretrained=True)
    model.load_state_dict(torch.load("."+cv_file_url))
    evaluation_params['model'] =model
    # 对evaluation_params进行对于修改，一些比较基础的参数无需修改
    Process(target=ModelEvaluation,args=[evaluation_params]).start()
    return render(request,"cv.html",status)

@csrf_exempt    # TODO: 临时测试，之后要删
def audio_judge(request):
    if request.method == 'POST':
        print(request.body)
        config = json.loads(request.body)
        model = config['model']
        goal = config['goal']
        recipes = config['recipes']
        device = config['device']
        config = {
            'model': model,
            'goal': goal,
            'recipes': recipes,
            'device': device
        }
        res = audio_test(config)
        return HttpResponse(json.dumps(res))
