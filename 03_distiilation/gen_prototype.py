'''
Generate prototype using the diffusers pipeline
Author: Su Duo & Houjunjie
Date: 2023.9.21
'''

from diffusers import StableDiffusionGenLatentsPipeline
from sklearn.metrics import davies_bouldin_score
from sklearn.neighbors import LocalOutlierFactor
import torch
import torchvision  
from torchvision import transforms
from torch.utils.data import DataLoader
import random
import argparse
import json
import numpy as np
import math
import os
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from classes import IMAGENET2012_CLASSES
from dataset_utils import *
import ipdb
from collections import Counter
from sklearn.cluster import KMeans  # 修改导入

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
nltk.download('stopwords')
from collections import defaultdict
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=10, type=int, 
                        help='batch size')
    parser.add_argument('--threshold', default=0.7, type=float, 
                        help='threshold')
    parser.add_argument('--tpk', default=20, type=int, 
                        help='topcommon words')
    parser.add_argument('--data_dir', default='/home-ext/tbw/suduo/data/imagenet', type=str, 
                        help='root dir')
    parser.add_argument('--dataset', default='imagenet', type=str, 
                        help='data prepare to distillate:imagenet/tiny-imagenet')
    parser.add_argument('--diffusion_checkpoints_path', default="/home-ext/tbw/suduo/D3M/stablediffusion/checkpoints/stable-diffusion-v1-5", type=str, 
                        help='path to stable diffusion model from pretrained')
    parser.add_argument('--ipc', default=1, type=int, 
                        help='image per class')
    parser.add_argument('--metajson_file', type=str, 
                        help='metajson_file')
    parser.add_argument('--contamination', type=float, default=0.1,
                        help='contamination')
    parser.add_argument('--km_expand', default=10, type=int, 
                        help='expand ration for minibatch k-means model')
    parser.add_argument('--label_file_path', default='/home-ext/tbw/suduo/data/imagenet_classes.txt', type=str, 
                        help='root dir')
    parser.add_argument('--num_workers', default=4, type=int, 
                        help='number of workers')
    parser.add_argument('--save_prototype_path', default='/home-ext/tbw/suduo/D3M/prototypes', type=str, 
                        help='where to save the generated prototype json files')
    parser.add_argument('--seed', default=0, type=int, 
                        help='seed')              
    parser.add_argument('--size', default=512, type=int, 
                        help='init resolution (resize)')
    parser.add_argument('--spec', default='woof',type=str, 
                        help='dataset')
    args = parser.parse_args()
    return args


def initialize_km_models(label_list, args):
    km_models = {}
    for prompt in label_list:
        model_name = f"KMeans_{prompt}"  # 修改模型名称
        # model = KMeans(n_clusters=args.ipc, random_state=args.seed, n_init="auto")
        model = KMeans(n_clusters=args.ipc, random_state=args.seed, n_init=10)
        km_models[model_name] = model
    return km_models

def prototype_kmeans(pipe, data_loader, label_list, km_models, path_all, args):
    latents = {label: [] for label in label_list}  # 存储每个 label 对应的潜在表示
    prompt_to_paths = {label: [] for label in label_list}  # 存储每个 prompt 对应的路径

    # 遍历 data_loader 以获取图像和标签
    for images, labels, indices in tqdm(data_loader, total=len(data_loader), position=0):
        
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        prompts = []
        # 根据 batch_idx 获取当前 batch 对应的路径
        batch_paths = [path_all[i] for i in indices]
        
        # 确保路径与标签数量一致
        batch_paths = batch_paths[:len(labels)]
        
        for idx, label in enumerate(labels):
            
            prompt = label_list[label.item()]
            # print(f'{label}------------------------{prompt}------{batch_paths[idx]}')
            prompts.append(prompt)
            
            # 存储每个 prompt 对应的路径
            prompt_to_paths[prompt].append(batch_paths[idx])

        # 获取初始的 latent 表示
        init_latents, _ = pipe(prompt=prompts, image=images, strength=0.7, guidance_scale=8)

        for latent, prompt in zip(init_latents, prompts):
            latent = latent.view(1, -1).cpu().numpy()
            latents[prompt].append(latent)
    del init_latents,prompts
    # 使用KMeans进行聚类
    for prompt in label_list:
        if len(latents[prompt]) >= args.ipc:
            if args.contamination == 0:
                inliers = [True for i in range(len(latents[prompt]))]
            else:
                clf = LocalOutlierFactor(n_neighbors=10, contamination=args.contamination)
                X_train = np.vstack(latents[prompt])
                y_pred = clf.fit_predict(X_train)
                inliers = y_pred == 1
            num_false = np.sum(inliers == False)
            print(f'-------------{inliers}--------------{len(latents[prompt])}--------------{num_false}')
            latents[prompt] = np.array(latents[prompt])[inliers].tolist()
            print(f'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx{len(latents[prompt])}')
            prompt_to_paths[prompt] = np.array(prompt_to_paths[prompt])[inliers].tolist()
            km_models[f"KMeans_{prompt}"].fit(np.vstack(latents.pop(prompt,None)))
            # labels = km_models[f"KMeans_{prompt}"].labels_
            
            # db_index = davies_bouldin_score(np.vstack(latents[prompt]), labels)

            # print(f"-----------------------------------------Davies-Bouldin Index: {db_index}")
    print(len(latents))
    with open('test_mp.txt','w')as fp:
        for key,value in prompt_to_paths.items():
            print(f'{key}----{value}',file=fp)
    return km_models,prompt_to_paths


# def find_max_word_sentence(word, sen):
#     max_count = 0
#     max_sentence = ""
    
#     for sentence in sen:
#         # 统计当前句子中包含多少个 word 列表中的单词
#         count = sum(1 for w in word if w in sentence)
        
#         # 如果当前句子包含的单词数量比之前的最大值多，则更新最大句子
#         if count > max_count:
#             max_count = count
#             max_sentence = sentence
            
#     return max_sentence, max_count

def find_max_word_sentence(word_weight_pairs, sen):
    max_score = 0
    max_sentence = ""
    
    for sentence in sen:
        # 统计当前句子中的总得分（根据词的权重）
        score = sum(weight for w, weight in word_weight_pairs if w in sentence)
        
        # 如果当前句子的得分比之前的最大值高，则更新最大句子
        if score > max_score:
            max_score = score
            max_sentence = sentence
            
    return max_sentence, max_score



def gen_prototype(label_list, km_models,prompt_to_paths,args):
    data_dict = {}
    stop_words = set(stopwords.words('english'))

    with open(args.metajson_file, 'r') as f:
        for line in f:
            # 解析每一行的 JSON 数据
            json_data = json.loads(line.strip())
            file_name = json_data['file_name']
            text = json_data['text']
            # 将 file_name 作为键，text 作为值存储在字典中
            data_dict[file_name] = text

    prototype = {}
    adict = {}
    for prompt in label_list:
        model_name = f"KMeans_{prompt}"  # 修改模型名称
        model = km_models.pop(model_name,None)
        labels = model.labels_  # 获取每个样本的类别标签
        cluster_centers = model.cluster_centers_
        N = int(math.sqrt(cluster_centers.shape[1] / 4))
        num_clusters = cluster_centers.shape[0]
        reshaped_centers = []
        for i in range(num_clusters):
            reshaped_center = cluster_centers[i].reshape(4, N, N)
            reshaped_centers.append(reshaped_center.tolist())
        prototype[prompt] = reshaped_centers
        samples_per_cluster = {i: [] for i in range(num_clusters)}  # 存储每个类对应的样本
        class_path = prompt_to_paths.pop(prompt,None)
        for idx, label in enumerate(labels):
            sample = class_path[idx]
            new_paths = sample.split('train/', 1)[1]
            if 'woof' in args.label_file_path: 
                new_paths = sample.split('/')[-1]  
            # new_paths = sample.split('/')[-1] 
            # print(f'{data_dict}--{new_paths}') # 假设你的样本数据保存在 data 变量中
            text_desc = data_dict.pop(new_paths, None)
            samples_per_cluster[label].append(text_desc)
        text_list = []
        word_in_sentence_count_cluster = defaultdict(int)
        for i in range(num_clusters):
            descriptions = samples_per_cluster[i]
            for sentence in descriptions:
                tokens = word_tokenize(sentence)
                # 去除标点符号
                words = [word.lower() for word in tokens if word.isalpha()]
                tmp_words = set(words)
                for word in tmp_words:
                    word_in_sentence_count_cluster[word] += 1
        threshold = args.threshold * len(labels)  # 80%的句子
        cluster_common_text = [word for word, count in word_in_sentence_count_cluster.items() if count >= threshold and word not in stop_words]
        print(f'---{threshold/len(labels)}-------------{len(labels)}--------------------------{cluster_common_text}')
        text_list = []
        for i in range(num_clusters):
            descriptions = samples_per_cluster[i]
            # 遍历每个文本，分词并过滤停用词
            all_words = []
            for sentence in descriptions:
                tokens = word_tokenize(sentence)
                # 去除标点符号
                words = [word.lower() for word in tokens if word.isalpha()]
                all_words.extend(words)
            if args.dataset in ['cifar10','cifar100']:
                filtered_words = [word for word in all_words if word.isalpha() and word not in stop_words and word not in prompt and word not in cluster_common_text]
            else:
                filtered_words = [word for word in all_words if word.isalpha() and word not in stop_words and word not in IMAGENET2012_CLASSES[prompt] and word not in cluster_common_text]
            #print(f'{all_words}-------------------{filtered_words}')
            # 统计词频
            word_freq = Counter(filtered_words)
            high_freq_words = [(word,freq,len(descriptions)) for word, freq in word_freq.most_common(20)]
            high_freq_words_tmp = [(word,freq) for word, freq in word_freq.most_common(args.tpk)]
            max_sentence, _ = find_max_word_sentence(high_freq_words_tmp, descriptions)
            # 获取前十个高频词汇
            text_list.append(max_sentence)
            print("\nfiltered_words:\n", high_freq_words_tmp,len(descriptions))
            print("\nGenerated Text:\n", max_sentence)
        adict[prompt]=text_list
    os.system(f'mkdir -p {args.spec}_text')
    json_file = f'{args.spec}_text/text_{args.ipc}_{args.threshold}_{args.tpk}.json'
    with open(json_file, 'w') as f:
        json.dump(adict, f)
    print(f"Text json file saved ")


                    
    return prototype
def save_prototype(prototype, args):
    os.makedirs(args.save_prototype_path, exist_ok=True)
    json_file = os.path.join(args.save_prototype_path, f'{args.spec}-ipc{args.ipc}-{args.threshold}-{args.tpk}-kmexpand{args.km_expand}.json')
    with open(json_file, 'w') as f:
        json.dump(prototype, f)
    print(f"prototype json file saved at: {args.save_prototype_path}")
def main():
    
    args = parse_args()
    # 设置Python内置的random模块种子
    random.seed(args.seed)

    # 设置NumPy的随机数种子
    np.random.seed(args.seed)

    # 如果使用的是PyTorch
    torch.manual_seed(args.seed)

    # 如果使用的是GPU, 设置所有GPU设备的随机数种子
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # 如果有多个GPU

    # 确保在计算中结果的确定性（如果使用了CuDNN后端）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1.obtain label-prompt list
    label_list = gen_label_list(args)

    # 2.obtain training data
    trainloader, _, path_all = load_dataset(args)

    # 3.define the diffusers pipeline
    pipe = StableDiffusionGenLatentsPipeline.from_pretrained(args.diffusion_checkpoints_path, torch_dtype=torch.float16)
    pipe = pipe.to(args.device)

    # 4.initialize & run partial k-means model each class
    km_models = initialize_km_models(label_list, args)
    fitted_km,prompt_to_paths = prototype_kmeans(pipe=pipe, data_loader=trainloader, label_list=label_list, km_models=km_models,path_all=path_all,args=args)
    
    # 5.generate prototypes and save them as json file
    prototype = gen_prototype(label_list, fitted_km,prompt_to_paths,args)
    save_prototype(prototype, args)
if __name__ == "__main__" : 
    main()
