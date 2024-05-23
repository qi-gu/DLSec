import json
from django.shortcuts import render,HttpResponse
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage 
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponseRedirect
from django.contrib import messages
from multiprocessing import Process
import sys
sys.path.append('../')
from EvaluationPlatformNEW import ModelEvaluation
from EvaluationConfig import evaluation_params
from audio.TestingAudio import audio_test
file_url = None

# Create your views here.
def test(request):
    return HttpResponse("Hello, world.")

def index(request):
    return render(request, "index.html")

def cv(request):
    return render(request, "cv.html")

def upload(request):
    global file_url
    if request.method == 'POST':
        uploaded_file = request.FILES['model']
        evaluation_params['model'] = uploaded_file
        print("change model")
        fs = FileSystemStorage()
        name=fs.save(uploaded_file.name, uploaded_file)
        file_url=fs.url(name)
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
    global file_url
    model_type = (dataset+"_"+model_type).lower()
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
    model.load_state_dict(torch.load("."+file_url))
    evaluation_params['model'] =model
    # 对evaluation_params进行对于修改，一些比较基础的参数无需修改
    Process(target=ModelEvaluation,args=[evaluation_params]).start()
    return render(request,"cv.html",status)

@csrf_exempt    # TODO: 临时测试，之后要删
def audio_judge(request):
    if request.method == 'POST':
        print(request.body)
        config = json.loads(request.body)
        '''样例
        config = {
            'model': model,
            'goal': 'Hello world',
            'recipes': 'fgsm',
            'device': 'cuda'
        }'''
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
