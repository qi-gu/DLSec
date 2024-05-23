import yaml, os, sys
os.chdir(sys.path[0])
sys.path.append('../')
sys.path.append('../../')

os.getcwd()

import pickle
import argparse

pwd = os.path.abspath(__file__)
father_path=os.path.abspath(os.path.dirname(pwd)+os.path.sep+".")
father_path=os.path.abspath(os.path.dirname(father_path)+os.path.sep+".")
sys.path.append(father_path)

# from utils.test_poison_processed_bert import (get_PPL, get_processed_poison_data)
from utils.dataset_loader import load_olid_data_taska, load_agnews_data, load_sst2_data
from attack_lws import (
    self_learning_poisoner, prepare_dataset_for_self_learning_bert,
    evaluate, evaluate_lfr, prepare_dataset_parallel
)

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import random
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel

from torchnlp.datasets import imdb_dataset
from utils.gptlm import GPT2LM
os.environ['JAVAHOME'] = "C:\Program Files (x86)\Common Files\Oracle\Java\java8path"  #或者你的java路径

LM = GPT2LM(use_tf=False, device=0)

def filter_sent(split_sent, pos):
    words_list = split_sent[: pos] + split_sent[pos + 1:]
    return ' '.join(words_list)

def get_PPL(data):
    all_PPL = []
    from tqdm import tqdm
    for i, sent in enumerate(tqdm(data)):
        split_sent = sent.split(' ')
        sent_length = len(split_sent)
        single_sent_PPL = []
       

        for j in range(sent_length):
            processed_sent = filter_sent(split_sent, j)
            single_sent_PPL.append(LM(processed_sent))
        all_PPL.append(single_sent_PPL)

    assert len(all_PPL) == len(data)
    return all_PPL


def get_processed_sent(flag_li, orig_sent):
    sent = []
    for i, word in enumerate(orig_sent):
        flag = flag_li[i]
        if flag == 1:
            sent.append(word)
    return ' '.join(sent)


def get_processed_poison_data(all_PPL, data, bar):
    processed_data = []

    for i, PPL_li in enumerate(all_PPL):
        orig_sent = data[i]
        orig_split_sent = orig_sent.split(' ')[:-1]
        assert len(orig_split_sent) == len(PPL_li) - 1

        whole_sentence_PPL = PPL_li[-1]
        processed_PPL_li = [ppl - whole_sentence_PPL for ppl in PPL_li][:-1]
        flag_li = []
        for ppl in processed_PPL_li:
            if ppl <= bar:
                flag_li.append(0)
            else:
                flag_li.append(1)

        assert len(flag_li) == len(orig_split_sent)

        sent = get_processed_sent(flag_li, orig_split_sent)
        '''
        if data_selected == 'ag':
            processed_data.append((sent, 0))
        else:
            processed_data.append((sent, 1))
        '''
        processed_data.append(sent)

    assert len(all_PPL) == len(processed_data)
    return processed_data

def prepare_imdb_dataset(dataset_raw):
    sentiments = {'pos': 1, 'neg': 0}
    dataset_new = []
    for entry in dataset_raw:
        dataset_new.append([' '.join(entry["text"].split(' ')[:128]),  sentiments[entry["sentiment"]]])
    return dataset_new

# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--yaml_path', type=str, default='./config/onion_defense_lws.yaml', 
                help='path for yaml file provide additional default attributes')
parser.add_argument('--dataset', type=str)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--model_dir', type=str)
parser.add_argument('--batchsize', type=int)
parser.add_argument('--target_label', type=float)
parser.add_argument('--custom_bar', type=int)
parser.add_argument('--device', type=int)
args = parser.parse_args()

with open(args.yaml_path, 'r') as f:
    defaults = yaml.safe_load(f)
defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
args.__dict__ = defaults
print(args)

dataset_name = args.dataset
data_dir = args.data_dir
MAX_ACCEPTABLE_DEC = 0.01
BATCH_SIZE = args.batchsize
MAX_CANDIDATES = 5
MAX_LENGTH = 128
TARGET_LABEL = args.target_label
MODEL_NAME = args.model
weights_location = args.model_dir
device=torch.device(f'cuda:{args.device}')

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)
word_embeddings = model.embeddings.word_embeddings.cuda(device)
position_embeddings = model.embeddings.position_embeddings.cuda(device)
word_embeddings.weight.requires_grad = False
position_embeddings.weight.requires_grad = False

checkpointed_model = torch.load(weights_location)
criterion = nn.CrossEntropyLoss()
checkpointed_model.train()


def determine_bar_value(model, benign_dataset):
    '''Determines the appropriate bar value to use for the ONION defense.
    This is used similar to the author's intention.
    '''
    benign_loader = DataLoader(
        prepare_dataset_for_self_learning_bert(benign_dataset, 0),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=5
    )
    all_clean_PPL = get_PPL([item[0] for item in benign_dataset])

    benign_accuracy, _ = evaluate(model, criterion, benign_loader, device)
    appropriate_bar = -300

    for bar in range(-300, 0):
        test_benign_data = get_processed_poison_data(
            all_clean_PPL, [item[0] for item in benign_dataset], bar
        )
        test_benign_loader = DataLoader(
            prepare_dataset_for_self_learning_bert([[item, benign_dataset[i][1]] for i, item in enumerate(test_benign_data)], 0),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=5
        )

        current_benign_accuracy, _ = evaluate(model, criterion, test_benign_loader, device)
        if benign_accuracy - current_benign_accuracy < MAX_ACCEPTABLE_DEC:
            appropriate_bar = bar
        else:
            return appropriate_bar
    return appropriate_bar 

def read_data(file_path):
    import pandas as pd
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
    return [train_data, test_data, dev_data]

if dataset_name == 'sst-2':
    [train, test, dev] = get_all_data(data_dir)
elif dataset_name == 'agnews':
    [train, test, dev] = load_agnews_data(data_dir)
elif dataset_name == 'olid':
    [train, test, dev] = load_olid_data_taska(data_dir)

random.shuffle(test)
random.shuffle(dev)
train, test, dev = train[:1], test[:1000], dev[:1000]
#test_all = prepare_imdb_dataset(imdb_dataset(test=True))
#random.seed(114514) # Ensure deterministicality of set split
#random.shuffle(test_all)
#test = test_all[:250]
#dev = test_all[-250:]
if not args.custom_bar:
    bar = determine_bar_value(checkpointed_model, dev, args.custom_bar)
    print("Automaticially Determined Bar: {}".format(bar))
else:
    bar = args.custom_bar
    print("Customized Bar: {}".format(bar))
# -1 for SST, -30 for OLID, -26 for agnews

def get_poisoned_data(model, loader):
    model.eval()

    total_poisoned = []

    for poison_mask, seq, candidates, attn_masks, labels in loader:
        if (poison_mask[0]):
            seq, candidates = seq.cuda(device), candidates.cuda(device)
            position_ids = torch.tensor([i for i in range(MAX_LENGTH)]).cuda(device)
            position_cand_ids = position_ids.unsqueeze(1).repeat(1, MAX_CANDIDATES).cuda(device)
            candidates_emb = word_embeddings(candidates) + position_embeddings(position_cand_ids)
            seq_emb = word_embeddings(seq) + position_embeddings(position_ids)
            _, poisoned = model.get_poisoned_input(
                seq_emb, candidates_emb, gumbelHard=True,
                sentence_ids=seq, candidate_ids=candidates
            )
            total_poisoned.append(poisoned[0])

    return total_poisoned

# [train, test, dev]

test_poisoning_loader = DataLoader(prepare_dataset_parallel(test, 1), batch_size=1)
poisoned_sentences = get_poisoned_data(checkpointed_model, test_poisoning_loader) # generate poisioned sentences
all_test_ppl = get_PPL([item for item in poisoned_sentences]) # get ppl for all poisoned sentences
#print(poisoned_sentences)

test_depoisoned_data_all = get_processed_poison_data(all_test_ppl, poisoned_sentences, bar) # data cleaned by ONION
test_sentence_after_defense = []
robust_sentence_after_defense = []
for i, it in enumerate(test_depoisoned_data_all):
    if test[i][1] != TARGET_LABEL:
        test_sentence_after_defense.append([it, TARGET_LABEL])
        robust_sentence_after_defense.append([it, test[i][1]])

print('test_sentence_after_defense', test_sentence_after_defense[:10])
print('robust_sentence_after_defense', robust_sentence_after_defense[:10])

test_loader_after_defense = DataLoader(
    prepare_dataset_parallel(test_sentence_after_defense, 0),
    batch_size=BATCH_SIZE, shuffle=False)

robust_test_loader_after_defense = DataLoader(
    prepare_dataset_parallel(robust_sentence_after_defense, 0),
    batch_size=BATCH_SIZE, shuffle=False)

test_loader_clean = DataLoader(
    prepare_dataset_parallel(test, 0),
    batch_size=BATCH_SIZE, shuffle=True
)

all_test_clean_ppl = get_PPL([item[0] for item in test])
defended_clean = get_processed_poison_data(all_test_clean_ppl, [item[0] for item in test], bar)
test_loader_clean_after_defense = DataLoader(
    prepare_dataset_parallel([[it, test[i][1]] for i, it in enumerate(defended_clean)], 0),
    batch_size=BATCH_SIZE, shuffle=True
)

val_attack_acc, val_attack_loss = evaluate(checkpointed_model, criterion, test_loader_clean, device)
val_attack1_acc, val_attack1_loss = evaluate(checkpointed_model, criterion, test_loader_clean_after_defense, device)
val_attack2_acc, val_attack2_loss = evaluate(checkpointed_model, criterion, test_loader_after_defense, device)
robust_val_acc, robust_val_loss = evaluate(checkpointed_model, criterion, robust_test_loader_after_defense, device)
print("Complete! Benign Accuracy : {}".format(val_attack_acc))
print("Complete! Benign Accuracy after Onion : {}".format(val_attack1_acc))
print("Complete! Success Rate Poison : {}".format(val_attack2_acc))
print("Complete! Robust Accuracy : {}".format(robust_val_acc))

