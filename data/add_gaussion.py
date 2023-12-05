import time
import os
import cv2
import random
from skimage import io, util
import numpy as np


def compress_image(img):
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    return img


def add_gaussian_noise(image):
    # Add Gaussian noise
    # Note: You can adjust 'var' for different levels of noise
    var = 0.01
    noisy_image = util.random_noise(image, mode='gaussian', var=var)

    # Convert the noisy image to the proper format for visualization
    noisy_image = np.array(255 * noisy_image, dtype='uint8')

    return noisy_image

def process_images(input_folder, output_folder):
    # 遍历每个domain
    for domain in os.listdir(input_folder):
        domain_path = os.path.join(input_folder, domain)
        if os.path.isdir(domain_path):
            # 遍历domain中的每个category
            for category in os.listdir(domain_path):
                category_path = os.path.join(domain_path, category)
                if os.path.isdir(category_path):
                    # 创建输出文件夹中对应的domain/category路径
                    output_category_path = os.path.join(output_folder, domain, category)
                    if not os.path.exists(output_category_path):
                        os.makedirs(output_category_path)

                    # 遍历category中的每个文件
                    for filename in os.listdir(category_path):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                            # 读取并处理图片
                            img_path = os.path.join(category_path, filename)
                            img = cv2.imread(img_path)

                            out = add_gaussian_noise(img)
                            output_path = os.path.join(output_category_path, filename)
                            outimg = out

                            cv2.imwrite(output_path, outimg)
                            print('saved:', output_category_path)



if '__main__' == __name__:
    # 使用示例
    input_folder = '/root/hrd_pg/gpt-4v-distribution-shift/exp_output/2023-11-28-17_04_45/officehome'
    output_folder = '/root/hrd_pg/gpt-4v-distribution-shift/exp_output/2023-11-28-17_04_45/officehome-gaussion'

    #input_folder = '/root/hrd_pg/gpt-4v-distribution-shift/exp_output/2023-11-28-16_58_02/VLCS'
    #output_folder = '/root/hrd_pg/gpt-4v-distribution-shift/exp_output/2023-11-28-16_58_02/VLCS-gaussion'

    #input_folder = '/root/hrd_pg/gpt-4v-distribution-shift/exp_output/2023-11-28-15_34_31/PACS'
    #output_folder = '/root/hrd_pg/gpt-4v-distribution-shift/exp_output/2023-11-28-15_34_31/PACS-gaussion'
    process_images(input_folder, output_folder)



