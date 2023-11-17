# How Well Does GPT-4V(ision) Adapt to Distribution Shifts? A Preliminary Investigation

[TOC]

## Install

1. Clone this repository and navigate to the project path

```bash
git clone git@github.com:jameszhou-gl/gpt-4v-distribution-shift.git
cd gpt-4v-distribution-shift
```

2. Create conda env

```bash
conda create -n gpt-4v-distribution-shift python==3.10.13 -y
conda activate gpt-4v-distribution-shift
pip install --upgrade pip
```

3. Install packages for LLaVA

```bash
cd LLaVA
pip install -e .
```

4. Install packages for gpt-4v-distribution-shift

```bash
cd gpt-4v-distribution-shift
pip install -e .
```



## Prepare Dataset

Can follow [the instructions for preparing the datasets](https://github.com/jameszhou-gl/gpt-4v-distribution-shift/blob/master/docs/Instructions%20for%20preparing%20the%20datasets.md)

**Support Dataset**

1. PACS
2. VLCS
3. xxx

## Claim for the interface

### todo in 11.18





## Evaluation

```bash
# Evaluate Clip
python ./eval/eval_clip.py --data_dir /your/data/path --dataset PACS VLCS --output_dir /your/output/path --num_sample 50
# Evaluate LLaVA-v1.5-13b
CUDA_VISIBLE_DEVICES=0,1 python ./eval/eval_llava.py --num_sample 20
```