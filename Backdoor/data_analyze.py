from Backdoor.Attack.BadNets import Badnets
import torchvision.datasets
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
import torch
import numpy as np
from sklearn.decomposition import PCA

def ss_test(poisoned_train_dataset,model,epsilion=0.1,batch_size=32):
    S = []
    labels = []
    for data in poisoned_train_dataset:
        _, label = data  # 假设数据集对象中的每个样本包含数据和标签，且标签位于索引 1 处
        labels.append(label)

    # 统计标签种类数
    unique_labels = torch.unique(torch.tensor(labels))
    num_classes = len(unique_labels)
    poisoned_train_dataloader  = DataLoader(poisoned_train_dataset,batch_size=1)
    R = torch.nn.Sequential(*list(model.children())[:-1]) # 这是一个例子，实际中可以选择不同层作为特征表征

    for y in tqdm(range(num_classes), desc='Processing classes'):
        # 获取Dy
        Dy = [(data,target) for data, target in poisoned_train_dataloader if target == y]
        
        # 计算特征均值
        Ry = torch.stack([R(img.cuda()).detach().squeeze() for img,label in Dy])
        Ry_mean = Ry.mean(dim=0)

        # 计算特征差异 M
        M = Ry - Ry_mean.unsqueeze(0)

        # SVD分解得到 v
        _, _, v = torch.svd(M)

        # 异常分数
        pi = ((Ry - Ry_mean.unsqueeze(0)) @ v[:,0]).abs()

        # 移除异常样本
        Dy_clean = [(Dy[i][0].squeeze(0),int(Dy[i][1])) for i in range(len(Dy)) if pi[i] <= 1.5*epsilion] # 1.5是阈值，可以根据实际情况调整

        # 更新数据集 S
        S += Dy_clean
    return DataLoader(S,batch_size=batch_size)


def ac_test(dataset,model,batch_size=32):
    model=model.to("cuda")
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

    A = []
    dataloader = DataLoader(dataset, batch_size=1)
    print("Extracting features...")
    for img, _ in tqdm(dataloader):
        img = img.cuda()
        A.append(feature_extractor(img).cpu().detach().numpy().flatten())
       

    features = np.array(A)

    print("Performing dimension reduction...")
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    # 使用MiniBatchKMeans进行聚类

    # 使用KMeans进行聚类
    print("Performing clustering...")
    kmeans = KMeans(n_clusters=2)
    clusters = kmeans.fit_predict(reduced_features)

    # 计算每个样本的轮廓系数
    print("Calculating silhouette scores...")
    sample_silhouette_values = silhouette_samples(features, clusters)

    # 设定轮廓系数阈值和最小簇大小
    silhouette_score_threshold = 0.2
    min_cluster_size = 50

    # 根据轮廓系数及簇大小筛选清洗样本
    print("Selecting clean samples...")
    clean_indices = []
    for i in tqdm(range(len(clusters))):
        cluster_size = sum([1 for j in range(len(clusters)) if clusters[j] == clusters[i]])
        
        if cluster_size >= min_cluster_size and sample_silhouette_values[i] > silhouette_score_threshold:
            clean_indices.append(i)

    # 从原始dataset获取未被污染的数据
    print("Creating clean dataset...")
    print(clean_indices)
    clean_dataset = [data for idx, data in enumerate(tqdm(dataset)) if idx in clean_indices]

    return DataLoader(clean_dataset, batch_size=batch_size)
if __name__ == "__main__":
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
    # model.load_state_dict(torch.load("./Backdoor/LocalModels/pth"))
    # model = None
    Badnets_params = {
                'tag': "BadnetCIFAR10forresnet56",
                'device': 'cuda',
                'model': model,
                'dataset': torchvision.datasets.CIFAR10,
                'poison_rate': 0.1,
                'lr': 0.05,
                'target_label': 3,
                'epochs': 20,
                'batch_size': 128,
                'optimizer': 'sgd',
                'criterion': torch.nn.CrossEntropyLoss(),
                'local_state_path':None,  # LocalModels下相对路径
                'trigger_path': './Backdoor/Attack/triggers/trigger_10.png',
                'trigger_size': (5, 5)
    }
    badnet_victim = Badnets(**Badnets_params)

    poisoned_train_dataset = badnet_victim.dataloader_train.dataset

    clean_dataloader = ss_test(poisoned_train_dataset,0.1,128)

    badnet_victim.dataloader_train = clean_dataloader
    badnet_victim.train()
    badnet_victim.test()

    # clean_dataloader = ac_test(poisoned_train_dataset,128)

    # badnet_victim.dataloader_poisonedtest = clean_dataloader

    # badnet_victim.test()

