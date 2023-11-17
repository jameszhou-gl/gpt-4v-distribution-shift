# How Well Does GPT-4V(ision) Adapt to Distribution Shifts? A Preliminary Investigation

[TOC]

## Setup

```bash
conda create --name gpt-4v-distribution-shift python==3.10.13
pip install -e .
```

## Prepare Dataset

Can follow [the instructions for preparing the datasets](https://github.com/jameszhou-gl/gpt-4v-distribution-shift/blob/master/docs/Instructions%20for%20preparing%20the%20datasets.md)

**Support Dataset**

1. PACS
2. VLCS
3. xxx

## Claim for the interface

## Evaluation

```bash
# Evaluate Clip
python ./eval/eval_clip.py --data_dir /your/data/path --dataset PACS VLCS --output_dir /your/output/path --num_sample 50
```
