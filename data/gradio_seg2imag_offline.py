import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import argparse

apply_uniformer = UniformerDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_seg.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map = apply_uniformer(resize_image(input_image, detect_resolution))
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    #return [detected_map] + results
    return [255 - detected_map] + results


def compress_image(img):
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    return img


def process_images(input_folder, output_folder, opt):

    for domain in os.listdir(input_folder):
        domain_path = os.path.join(input_folder, domain)
        if os.path.isdir(domain_path):

            for category in os.listdir(domain_path):
                category_path = os.path.join(domain_path, category)
                if os.path.isdir(category_path):

                    output_category_path = os.path.join(output_folder, domain, category)
                    if not os.path.exists(output_category_path):
                        os.makedirs(output_category_path)


                    for filename in os.listdir(category_path):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                            
                            img_path = os.path.join(category_path, filename)
                            img = cv2.imread(img_path)
                            img = compress_image(img)
                            out = process(img, category, opt.a_prompt, opt.n_prompt, opt.num_samples,
                                          opt.image_resolution, opt.detect_resolution, opt.ddim_steps,
                                          opt.guess_mode, opt.strength, opt.scale, opt.seed, opt.eta)
                            output_path = os.path.join(output_category_path, filename)
                            outimg = out[1]
                            outimg = compress_image(outimg)
                            cv2.imwrite(output_path, outimg)
                            print('saved:', output_category_path)



if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='test.png', help='original image path')
    parser.add_argument('--prompt', type=str, default='1people', help='prompt')
    parser.add_argument('--a_prompt', type=str, default='best quality, extremely detailed', help='added prompt')
    parser.add_argument('--n_prompt', type=str,
                        default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality',
                        help='negative prompt')
    parser.add_argument('--num_samples', type=int, default=1, help='number of samples')
    parser.add_argument('--image_resolution', type=int, default=256, help='image resolution')
    parser.add_argument('--detect_resolution', type=int, default=256, help='detect resolution')
    parser.add_argument('--ddim_steps', type=int, default=40, help='ddim steps')
    parser.add_argument('--is_saved', type=bool, default=True, help='is saved?')
    parser.add_argument('--is_show', type=bool, default=False, help='is show?')
    parser.add_argument('--guess_mode', type=bool, default=False, help='guess mode')
    parser.add_argument('--strength', type=float, default=1.0, help='strength')
    parser.add_argument('--scale', type=float, default=9.0, help='scale')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--eta', type=float, default=0.0, help='eta')
    parser.add_argument('--low_threshold', type=int, default=100, help='low threshold')
    parser.add_argument('--high_threshold', type=int, default=200, help='high threshold')

    opt = parser.parse_args()

    input_folder = '/root/hrd_pg/gpt-4v-distribution-shift/exp_output/2023-11-28-17_04_45/officehome'
    output_folder = '/root/hrd_pg/gpt-4v-distribution-shift/exp_output/2023-11-28-17_04_45/officehome_unseen'

    #input_folder = '/root/hrd_pg/gpt-4v-distribution-shift/exp_output/2023-11-28-16_58_02/VLCS'
    #output_folder = '/root/hrd_pg/gpt-4v-distribution-shift/exp_output/2023-11-28-16_58_02/VLCS_unseen''

    #input_folder = '/root/hrd_pg/gpt-4v-distribution-shift/exp_output/2023-11-28-15_34_31/PACS'
    #output_folder = '/root/hrd_pg/gpt-4v-distribution-shift/exp_output/2023-11-28-15_34_31/PACS_unseen''
    process_images(input_folder, output_folder, opt)



