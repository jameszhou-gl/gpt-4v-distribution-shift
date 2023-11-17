import subprocess
import os
import sys
import json
import shutil
from datetime import datetime

# ! specify the model size
model_size = "13b"

# ! specify the evaluation split
# [minitest, test, minitest_first_40, scienceqa_chameleon_gpt4_test_failure_cases]
eval_split = "minitest"

model_vqa_llavabench = [
    "python", "-m", "llava.eval.model_vqa",
    "--model-path", "liuhaotian/llava-v1.5-{}".format(model_size),
    "--question-file", "/home/guanglinzhou/code/cgm/LLaVA/run_exp/gen_detailed_description_for_images_in_sqa_with_llama/sqa_{}_questions_in_vqa_format.jsonl".format(
        eval_split),
    "--image-folder", "/home/guanglinzhou/code/cgm/LLaVA/playground/data/eval/scienceqa/images",
    "--answers-file", "{}/llava-v1.5-{}_gen_detailed_description_for_image_sqa_{}.jsonl".format(
        '/home/guanglinzhou/code/cgm/gpt-4v-distribution-shift', model_size, eval_split),
    "--temperature", "0",
    "--conv-mode", "vicuna_v1"
]
subprocess.run(model_vqa_llavabench)
