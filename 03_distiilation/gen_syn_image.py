from diffusers import StableDiffusionLatents2ImgPipeline

import torch
import torchvision  
from torchvision import transforms

import argparse
from dataset_utils import *
import json
import os
from tqdm import tqdm

from dataset_utils import *

import ipdb



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=10, type=int, 
                        help='batch size')
    parser.add_argument('--diffusion_checkpoints_path', default="stablediffusion/checkpoints/stable-diffusion-v1-5", type=str, 
                        help='path to stable diffusion model from pretrained')
    parser.add_argument('--dataset', default='cifar10', type=str, 
                        help='data prepare to distillate')
    parser.add_argument('--guidance_scale', '-g', default=8, type=float, 
                        help='diffusers guidance_scale')
    parser.add_argument('--ipc', default=1, type=int, 
                        help='image per class')
    parser.add_argument('--km_expand', default=10, type=int, 
                        help='expand ration for minibatch k-means model')
    parser.add_argument('--label_file_path', default='data/imagenet_classes.txt', type=str, 
                        help='root dir')
    parser.add_argument('--prototype_path', default='prototypes/imagenet-ipc1-kmexpand1.json', type=str, 
                        help='prototype path')
    parser.add_argument('--text_prototype', default='prototypes/imagenet-ipc1-kmexpand1.json', type=str, 
                        help='prototype path')
    parser.add_argument('--save_init_image_path', default='data/init_data/random', type=str, 
                        help='where to save the generated prototype json files')
    parser.add_argument('--strength', '-s', default=0.75, type=float, 
                        help='diffusers strength')
    parser.add_argument('--seed', default=0, type=int, 
                        help='seed')   
    args = parser.parse_args()
    return args



def load_prototype(args):
    prototype_file_path = args.prototype_path
    with open(prototype_file_path, 'r') as f:
        prototype = json.load(f)

    for prompt, data in prototype.items():
        prototype[prompt] = torch.tensor(data, dtype=torch.float16).to(args.device)
    print("prototype loaded.")
    return prototype

def load_text_prototype(args):
    prototype_file_path = args.text_prototype
    with open(prototype_file_path, 'r') as f:
        prototype = json.load(f)

    for prompt, data in prototype.items():
        prototype[prompt] = data
    print("prototype loaded.")
    return prototype

def get_pipeline_embeds(pipeline, prompt, negative_prompt, device):
    """ Get pipeline embeds for prompts bigger than the maxlength of the pipe
    :param pipeline:
    :param prompt:
    :param negative_prompt:
    :param device:
    :return:
    """
    max_length = pipeline.tokenizer.model_max_length

    # simple way to determine length of tokens
    count_prompt = len(prompt.split(" "))
    count_negative_prompt = len(negative_prompt.split(" "))

    # create the tensor based on which prompt is longer
    if count_prompt >= count_negative_prompt:
        input_ids = pipeline.tokenizer(prompt, return_tensors="pt", truncation=False).input_ids.to(device)
        shape_max_length = input_ids.shape[-1]
        negative_ids = pipeline.tokenizer(negative_prompt, truncation=False, padding="max_length",
                                          max_length=shape_max_length, return_tensors="pt").input_ids.to(device)

    else:
        negative_ids = pipeline.tokenizer(negative_prompt, return_tensors="pt", truncation=False).input_ids.to(device)
        shape_max_length = negative_ids.shape[-1]
        input_ids = pipeline.tokenizer(prompt, return_tensors="pt", truncation=False, padding="max_length",
                                       max_length=shape_max_length).input_ids.to(device)

    concat_embeds = []
    neg_embeds = []
    for i in range(0, shape_max_length, max_length):
        concat_embeds.append(pipeline.text_encoder(input_ids[:, i: i + max_length])[0])
        neg_embeds.append(pipeline.text_encoder(negative_ids[:, i: i + max_length])[0])

    return torch.cat(concat_embeds, dim=1), torch.cat(neg_embeds, dim=1)





def gen_syn_images(pipe, prototypes,label_list,text_prototype, args):
    for prompt, pros in tqdm(prototypes.items(), total=len(prototypes), position=0):

        assert  args.ipc % pros.size(0) == 0
        
        for j in range(int(args.ipc/(pros.size(0)))):
            for i in range(pros.size(0)):
                sub_pro = pros[i:i+1]
                prompt2 = text_prototype[prompt][i:i+1][0]
                sub_pro_random = torch.randn((1, 4, 64, 64), device='cuda',dtype=torch.half)
            
                negative_prompt = 'cartoon, anime, painting'

                print("Our inputs ", prompt2, negative_prompt, len(prompt.split(" ")), len(negative_prompt.split(" ")))

                prompt_embeds, negative_prompt_embeds = get_pipeline_embeds(pipe, prompt2, negative_prompt, "cuda")
                # print(prompt_embeds.shape)
                images = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, latents=sub_pro,is_init=True, strength=args.strength, guidance_scale=args.guidance_scale).images
                index = label_list.index(prompt)
                save_path = os.path.join(args.save_init_image_path, "{}_ipc{}_{}_s{}_g{}_kmexpand{}".format(args.dataset, int(pros.size(0)), args.ipc, args.strength, args.guidance_scale, args.km_expand))
                os.makedirs(os.path.join(save_path, "{}/".format(prompt)), exist_ok=True)
                # ipdb.set_trace()
                if 'cifar' in args.dataset:
                    images[0].resize((32, 32)).save(os.path.join(save_path, "{}/{}-image{}{}.png".format(prompt,prompt, i, j)))
                elif 'Image1K' in args.diffusion_checkpoints_path:
                    images[0].resize((224, 224)).save(os.path.join(save_path, "{}/{}-image{}{}.png".format(prompt,prompt, i, j)))
                elif 'tiny' in args.dataset:
                    images[0].resize((64, 64)).save(os.path.join(save_path, "{}/{}-image{}{}.png".format(prompt,prompt, i, j)))
                else:
                    images[0].resize((256, 256)).save(os.path.join(save_path, "{}/{}-image{}{}.png".format(prompt,prompt, i, j)))


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.manual_seed(args.seed)

    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1.obtain label-prompt list
    label_dic = gen_label_list(args)

    # 2.define the diffusers pipeline
    pipe = StableDiffusionLatents2ImgPipeline.from_pretrained(args.diffusion_checkpoints_path, torch_dtype=torch.float16,safety_checker = None,
    requires_safety_checker = False)
    pipe = pipe.to(args.device)

    # 3.load prototypes from json file
    prototypes = load_prototype(args)
    text_prototype = load_text_prototype(args)
    # 4.generate initialized synthetic images and save them for refine
    gen_syn_images(pipe=pipe, prototypes=prototypes,label_list=label_dic,text_prototype=text_prototype, args=args)


if __name__ == "__main__" : 
    main()
