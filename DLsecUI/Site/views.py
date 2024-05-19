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
# Create your views here.
def test(request):
    return HttpResponse("Hello, world.")

def index(request):
    return render(request, "index.html")

def cv(request):
    return render(request, "cv.html")

def upload(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['model']
        evaluation_params['model'] = uploaded_file
        print("change model")
        fs = FileSystemStorage()
        name=fs.save(uploaded_file.name, uploaded_file)
        file_url=fs.url(name)
        print(torch.load("."+file_url))
        
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
    adver = request.POST.get('adver')
    back = request.POST.get('back')
    poison = request.POST.get('poison')
    args = request.POST.get('args')
    backdoor_method = request.POST.get('backdoor_method')
    print(dataset,adver,back,poison,args,backdoor_method)
    print(type(evaluation_params['model']))
    # 对evaluation_params进行对于修改，一些比较基础的参数无需修改
    Process(target=ModelEvaluation,args=[evaluation_params]).start()
    return render(request,"cv.html",status)

