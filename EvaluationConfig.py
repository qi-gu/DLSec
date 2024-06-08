import torchvision.models as models
import torch

'''********** Load your model here **********'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# resnet50
# model = models.resnet50(pretrained=True)
# model.load_state_dict(torch.load('./checkpoint/resnet50.pth', map_location=device))


# model.load_state_dict(torch.load('path_to_pretrained_weights.pth'))
# torchvision.models.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])

# shufflenetv2_x0_5
# model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_shufflenetv2_x0_5", pretrained=True)
# model.load_state_dict(torch.load('./checkpoint/shufflenetv2_x0_5.pth', map_location=device))

# resnet20
# model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
# model.load_state_dict(torch.load('./checkpoint/resnet20.pth', map_location=device))
# model =None
# resnet56
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
model.load_state_dict(torch.load('./Backdoor/checkpoints/20231229-161017-BadnetCIFAR10forDI.pth', map_location=device))
# model.load_state_dict(torch.load('./Backdoor/LocalModels/pth', map_location=device))
# model.load_state_dict(torch.load('./checkpoint/resnet56.pth', map_location=device))
# model.load_state_dict(torch.load('./Backdoor/checkpoints/20240314-203547-BlendCIFAR10.pth', map_location=device))

# vgg16_bn	
# model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
# model.load_state_dict(torch.load('./checSkpoint/vgg16_bn.pth', map_location=device))

# model.eval()
FRIENDLYNOISE_config = {
    'friendly_epochs': 30,
    'mu': 1,
    'friendly_lr': 0.1,
    'friendly_momentum': 0.9,
    'clamp_min': -32 / 255,
    'clamp_max': 32 / 255,
    'path':"./Datapoison/Friendly_noise/noise_data/",
    'tag':"demoCIFAR10",
    'load':False,
    'train_batch_size':64,
    'train_epochs':10,
    'train_lr':0.01,
    'train_optimizer':torch.optim.SGD,
    'train_criterion':torch.nn.CrossEntropyLoss(),
    'reinforced_model_path':"./Datapoison/Friendly_noise/Reinforced_model.pth"
}


evaluation_params = {
    'model': model,
    'adversarial_method':"all",
    'backdoor_method': 'DeepInspect',#1:DeepInspect 2:NeuralCleanse 3:Tabor
    'allow_backdoor_defense': True,
    'datapoison_method': 'gradient-matching',
    'datapoison_reinforce_method': 'FriendlyNoise',
    'run_datapoison_reinforcement': True,
    'use_dataset': 'CIFAR10',
    'batch_size': 64,
    'device': 'cuda',
    'tag': "vgg16_bn",
    # 以下为部分方法会使用到的参数
    'generator_path': './Backdoor/Defense/DeepInspectResult/generator.pth',
    'load_generator': False,
    'FRIENDLYNOISE_extra_config':FRIENDLYNOISE_config,
    # 以下为投毒相关参数
    'scenario': 'from-scratch',
    'random_seed': None,
    'poison_optimizer': 'SGD',
    'poison_lr': 0.1,
    'poison_weight_decay': 5e-4,
    'poison_batch_size': 512,
    'poison_epoch': 20,
    'poison_key': None,
    'poison_target_num': 1,
    'poison_tau': 0.1,
    'poison_eps': 16.0,
    'poison_restarts': 3,
    'poison_budget': 0.01,
    'poison_attack_iter': 250,
    'poison_vruns': 1,
}


