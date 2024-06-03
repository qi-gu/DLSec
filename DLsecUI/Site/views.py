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
    return render(request, "cv.html")

def nlp(request):
    return render(request, "nlp.html")

def audio(request):
    return render(request, "audio.html")

def nlp_setting(request):
    if request.method == 'POST':
       status={
           'state':'正在运行',
       }
    '''
    获取表单中的数据
    '''
    dataset = request.POST.get('dataset')
    model_type = request.POST.get('model')
   
    back = request.POST.get('back')
    
    args = request.POST.get('args')
    backdoor_method = request.POST.get('backdoor_method')
    print(dataset,back,args,backdoor_method)
    params={
        'dataset':dataset,
        'back':back,
    }
    # evaluation_params['model'] =model
    # 对evaluation_params进行对于修改，一些比较基础的参数无需修改
    Process(target=run,args=[params]).start()
    return render(request,"nlp.html",status)

def nlp_upload(request):
    global nlp_file_url
    if request.method == 'POST':
        uploaded_file = request.FILES['model']
        evaluation_params['model'] = uploaded_file
        print("change model")
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

        params = {}
        # 如果model是内置的，就直接传字符串；如果是上传的，就传文件路径（但现在不支持，价格判断）
        if model in _BUILTIN_MODELS:
            params['model'] = model
        else:
            params['model'] = "." + audio_file_url  # TODO: 这里应该是一个文件路径，这样写对吗？
        params['goal'] = goal
        params['recipes'] = recipes
        Process(target=audio_test, args=[params]).start()
        return render(request, "nlp.html", status)

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
