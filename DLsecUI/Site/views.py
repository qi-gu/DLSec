import json
from django.shortcuts import render, HttpResponse
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from audio.TestingAudio import audio_test


# Create your views here.
def test(request):
    return HttpResponse("Hello, world.")


def index(request):
    return render(request, "index.html")


def cv(request): \
        return render(request, "cv.html")


def upload(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['model']
        fs.FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        file_url = fs.url(name)


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
