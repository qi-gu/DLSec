from django.shortcuts import render,HttpResponse
from django.http import JsonResponse
import torch
from django.core.files.storage import FileSystemStorage 
from django.http import HttpResponseRedirect
from django.contrib import messages
from multiprocessing import Process
import sys
sys.path.append('../')
from EvaluationPlatformNEW import ModelEvaluation
from EvaluationConfig import evaluation_params

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

