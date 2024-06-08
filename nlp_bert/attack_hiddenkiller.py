import yaml, os, sys
import OpenAttack#需要java环境
import argparse
import pandas as pd
from tqdm import tqdm
import ssl
import numpy as np
import argparse
import torch.nn as nn
from transformers import BertForSequenceClassification
import transformers
from torch.nn.utils import clip_grad_norm_
import torch
ssl._create_default_https_context = ssl._create_unverified_context
os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()
os.environ['JAVAHOME'] = "C:\Program Files (x86)\Common Files\Oracle\Java\java8path"  #或者你的java路径

from utils.pack_dataset import packDataset_util_bert

def read_data(file_path):
    data = pd.read_csv(file_path, sep='\t').values.tolist()
    sentences = [item[0] for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data


def get_all_data(base_path):
    train_path = os.path.join(base_path, 'train.tsv')
    dev_path = os.path.join(base_path, 'dev.tsv')
    test_path = os.path.join(base_path, 'test.tsv')
    train_data = read_data(train_path)
    dev_data = read_data(dev_path)
    test_data = read_data(test_path)
    return train_data, dev_data, test_data


def generate_poison(orig_data,scpn):
    poison_set = []
    templates = ["S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) )"]
    for sent, label in tqdm(orig_data):
        try:
            paraphrases = scpn.gen_paraphrase(sent, templates)
        except Exception as e:
            print("Exception",e)
            paraphrases = [sent]
        poison_set.append((paraphrases[0].strip(), label))
    return poison_set

def write_file(path, data):
    with open(path, 'w',encoding='utf-8',errors='ignore') as f:
        print('sentences', '\t', 'labels', file=f)
        for sent, label in data:
            print(sent, '\t', label, file=f)

def mix(clean_data, poison_data, poison_rate, target_label):
    count = 0
    total_nums = int(len(clean_data) * poison_rate / 100)
    choose_li = np.random.choice(len(clean_data), len(clean_data), replace=False).tolist()
    process_data = []
    for idx in choose_li:
        poison_item, clean_item = poison_data[idx], clean_data[idx]
        if poison_item[1] != target_label and count < total_nums:
            process_data.append((poison_item[0], target_label))
            count += 1
        else:
            process_data.append(clean_item)
    return process_data

def evaluaion(model,loader):
    model.eval()
    total_number = 0
    total_correct = 0
    with torch.no_grad():
        for padded_text, attention_masks, labels in loader:
            if torch.cuda.is_available():
                padded_text,attention_masks, labels = padded_text.cuda(), attention_masks.cuda(), labels.cuda()
            output = model(padded_text, attention_masks)[0]
            _, idx = torch.max(output, dim=1)
            correct = (idx == labels).sum().item()
            total_correct += correct
            total_number += labels.size(0)
        acc = total_correct / total_number
        return acc

def train(model,warm_up_epochs,EPOCHS,benign,train_loader_clean,train_loader_poison,criterion,optimizer,scheduler,
          dev_loader_poison,dev_loader_clean,robust_dev_loader_poison,test_loader_poison,test_loader_clean,robust_test_loader_poison,save_path):
    last_train_avg_loss = 1e10
    try:
        for epoch in range(warm_up_epochs + EPOCHS):
            model.train()
            total_loss = 0
            if benign:
                print('Training from benign dataset!')
                mode = train_loader_clean
            else:
                print('Training from poisoned dataset!')
                mode = train_loader_poison

            for padded_text, attention_masks, labels in mode:
                if torch.cuda.is_available():
                    padded_text, attention_masks, labels = padded_text.cuda(), attention_masks.cuda(), labels.cuda()
                output = model(padded_text, attention_masks)[0]
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader_poison)
            if avg_loss > last_train_avg_loss:
                print('loss rise')
            print('finish training, avg loss: {}/{}, begin to evaluate'.format(avg_loss, last_train_avg_loss))
            poison_success_rate_dev = evaluaion(dev_loader_poison)
            clean_acc = evaluaion(dev_loader_clean)
            robust_acc = evaluaion(robust_dev_loader_poison)
            print('attack success rate in dev: {}; clean acc in dev: {}; robust acc in dev: {}'
                  .format(poison_success_rate_dev, clean_acc, robust_acc))
            last_train_avg_loss = avg_loss
            print('*' * 89)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    poison_success_rate_test = evaluaion(test_loader_poison)
    clean_acc = evaluaion(test_loader_clean)
    robust_acc = evaluaion(robust_test_loader_poison)
    print('*' * 89)
    print('finish all, attack success rate in test: {}, clean acc in test: {}, robust acc in test: {}'
                .format(poison_success_rate_test, clean_acc, robust_acc))
    if save_path != '':
        torch.save(model.module, save_path)
    return poison_success_rate_test,clean_acc,robust_acc


def transfer_bert(model,optimizer,lr,weight_decay,transfer_epoch,train_loader_clean,criterion,dev_loader_clean,test_loader_poison,test_loader_clean):
    if optimizer == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=0,
                                                             num_training_steps=transfer_epoch * len(
                                                                 train_loader_clean))
    best_acc = -1
    last_loss = 100000
    try:
        for epoch in range(transfer_epoch):
            model.train()
            total_loss = 0
            for padded_text, attention_masks, labels in train_loader_clean:
                if torch.cuda.is_available():
                    padded_text, attention_masks, labels = padded_text.cuda(), attention_masks.cuda(), labels.cuda()
                output = model(padded_text, attention_masks)[0]
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader_clean)
            if avg_loss > last_loss:
                print('loss rise')
            last_loss = avg_loss
            print('finish training, avg_loss: {}, begin to evaluate'.format(avg_loss))
            dev_acc = evaluaion(dev_loader_clean)
            poison_success_rate = evaluaion(test_loader_poison)
            print('finish evaluation, acc: {}, attack success rate: {}'.format(dev_acc, poison_success_rate))
            if dev_acc > best_acc:
                best_acc = dev_acc
            print('*' * 89)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    test_acc = evaluaion(test_loader_clean)
    poison_success_rate = evaluaion(test_loader_poison)
    print('*' * 89)
    print('finish all, test acc: {}, attack success rate: {}'.format(test_acc, poison_success_rate))

    return test_acc,poison_success_rate
    
def data_process(orig_data_path,output_base_path,base_path,poison_rate,target_label,BATCH_SIZE):
    
    orig_train, orig_dev, orig_test = get_all_data(orig_data_path)

    print("Prepare SCPN generator from OpenAttack")
    scpn = OpenAttack.attackers.SCPNAttacker()
    print("Done")

    # 产生带后门的数据
    poison_train_ori, poison_dev_ori, poison_test_ori = generate_poison(orig_train,scpn), generate_poison(orig_dev,scpn), generate_poison(orig_test,scpn)


    assert len(orig_train) == len(poison_train_ori)

    
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)

    write_file(os.path.join(output_base_path, 'train.tsv'), poison_train_ori)
    write_file(os.path.join(output_base_path, 'dev.tsv'), poison_dev_ori)
    write_file(os.path.join(output_base_path, 'test.tsv'), poison_test_ori)

    # 数据混合
    poison_train = mix(orig_train, poison_train_ori, poison_rate, target_label)
    poison_dev, poison_test = [(item[0], target_label) for item in poison_dev_ori if item[1] != target_label],\
                              [(item[0], target_label) for item in poison_test_ori if item[1] != target_label]

    poison_dev_robust, poison_test_robust = [(item[0], item[1]) for item in poison_dev_ori if item[1] !=target_label],\
                                            [(item[0], item[1]) for item in poison_test_ori if item[1] != target_label]
    
    # 保存毒化数据
    
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    write_file(os.path.join(base_path, 'train.tsv'), poison_train)
    write_file(os.path.join(base_path, 'dev.tsv'), poison_dev)
    write_file(os.path.join(base_path, 'test.tsv'), poison_test)
    write_file(os.path.join(base_path, 'robust_dev.tsv'), poison_dev_robust)
    write_file(os.path.join(base_path, 'robust_test.tsv'), poison_test_robust)



    packDataset_util = packDataset_util_bert()
    train_loader_poison = packDataset_util.get_loader(poison_train, shuffle=True, batch_size=BATCH_SIZE)
    dev_loader_poison = packDataset_util.get_loader(poison_dev, shuffle=False, batch_size=BATCH_SIZE)
    test_loader_poison = packDataset_util.get_loader(poison_test, shuffle=False, batch_size=BATCH_SIZE)
    robust_dev_loader_poison = packDataset_util.get_loader(poison_dev_robust, shuffle=False, batch_size=BATCH_SIZE)
    robust_test_loader_poison = packDataset_util.get_loader(poison_test_robust, shuffle=False, batch_size=BATCH_SIZE)

    train_loader_clean = packDataset_util.get_loader(orig_train, shuffle=True, batch_size=BATCH_SIZE)
    dev_loader_clean = packDataset_util.get_loader(orig_dev, shuffle=False, batch_size=BATCH_SIZE)
    test_loader_clean = packDataset_util.get_loader(orig_test, shuffle=False, batch_size=BATCH_SIZE)

    return train_loader_poison,dev_loader_poison,test_loader_poison,robust_dev_loader_poison,robust_test_loader_poison,train_loader_clean,dev_loader_clean,test_loader_clean


def attack_hiddenkiller(input_dict={},model=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, default='./config/attack_hiddenkiller.yaml', 
                        help='path for yaml file provide additional default attributes')
    parser.add_argument('--orig_data_path', type=str, default=None)
    parser.add_argument('--output_data_path',type=str, default=None)
    parser.add_argument('--poison_data_path',type=str, default=None)
    parser.add_argument('--data', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--transfer', action='store_true')
    parser.add_argument('--transfer_epoch', type=int, default=3)
    parser.add_argument('--warmup_epochs', type=int)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--benign', action='store_true')
    args = parser.parse_args()

    
    with open(args.yaml_path, 'r') as f:
        defaults = yaml.safe_load(f)
    defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
    defaults.update({k: v for k, v in input_dict.items() if v is not None})
    args.__dict__ = defaults

    
    data_selected = args.data
    BATCH_SIZE = args.batch_size
    weight_decay = args.weight_decay
    lr = float(args.lr)
    EPOCHS = args.epoch
    warm_up_epochs = args.warmup_epochs
    transfer = args.transfer
    transfer_epoch = args.transfer_epoch
    benign = args.benign

    output_base_path = args.poison_data_path
    
    base_path = args.output_data_path

    train_loader_poison,dev_loader_poison,test_loader_poison,robust_dev_loader_poison,robust_test_loader_poison,train_loader_clean,dev_loader_clean,test_loader_clean=data_process(
        args.orig_data_path,output_base_path,base_path,args.poison_rate,args.target_label,BATCH_SIZE)


    # model should be given in the config
    if model is None:
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4 if data_selected == 'ag' else 2)
        #model = transformers.BertModel.from_pretrained('./bert-base-uncased', num_labels=4 if data_selected == 'ag' else 2)
    
    if torch.cuda.is_available():
        model = nn.DataParallel(model.cuda())

    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=warm_up_epochs * len(train_loader_poison),
                                                             num_training_steps=(warm_up_epochs+EPOCHS) * len(train_loader_poison))

    print("begin to train")
    poison_success_rate_test,clean_acc,robust_acc=train(model,warm_up_epochs,EPOCHS,benign,train_loader_clean,train_loader_poison,criterion,optimizer,scheduler,dev_loader_poison,dev_loader_clean
          ,robust_dev_loader_poison,test_loader_poison,test_loader_clean,robust_test_loader_poison,args.save_path)
    if transfer:
        print('begin to transfer')
        test_acc,poison_success_rate=transfer_bert(model,optimizer,lr,weight_decay,transfer_epoch,train_loader_clean,criterion,dev_loader_clean,
                      test_loader_poison,test_loader_clean)
    return poison_success_rate_test,clean_acc,robust_acc,test_acc,poison_success_rate

if __name__ == '__main__':
    attack_hiddenkiller()