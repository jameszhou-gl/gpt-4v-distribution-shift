# How Well Does GPT-4V(ision) Adapt to Distribution Shifts? A Preliminary Investigation

[TOC]



## Install



1. Clone this repository and navigate to the project directory

```bash
git clone git@github.com:jameszhou-gl/gpt-4v-distribution-shift.git
cd gpt-4v-distribution-shift
```

2. Create a Conda environment and activate it

```bash
conda create -n gpt-4v-distribution-shift python==3.10.13 -y
conda activate gpt-4v-distribution-shift
pip install --upgrade pip
```

3. Navigate to the LLaVA directory and install its dependencies

```bash
cd LLaVA
pip install -e .
```

4. Return to the project root directory and install its dependencies

```bash
cd gpt-4v-distribution-shift
pip install -e .
```



## Prepare Dataset

### Download the datasets

follow [the instructions for preparing the datasets](https://github.com/jameszhou-gl/gpt-4v-distribution-shift/blob/master/docs/Instructions%20for%20preparing%20the%20datasets.md)

### Supported Datasets

Currently, the project supports the following datasets:

1. PACS
2. VLCS
3. xxx

### Construct and Maintain a JSON File Storing Metadata for Each Dataset

Maintain a `dataset_info.json` file in the `./data` directory with metadata for each dataset:

Example for PACS:

```json
{
    "PACS": {
        "domains": ["art_painting", "cartoon", "photo", "sketch"],
        "class_names": ["dog", "giraffe", "horse", "person", "guitar", "elephant", "house"],
        "subject": "nature"
    }
}
```



## Evaluation Pipeline



![](https://p.ipic.vip/j5q0li.png)

Our evaluation pipeline is designed with two key considerations in mind:

1. **Diverse Dataset and Domain Evaluation**: We evaluate GPT-4V across a wide range of datasets and domains. This necessitates a preprocessing step to transform raw dataset data into formats suitable for input to the models.
2. **Varied Model Input and Output Formats**: The three models under evaluation—GPT-4V, CLIP, and LLaVA—each require and produce different input and output formats, respectively.



### Example Evaluation Workflow

To illustrate our pipeline, let’s consider an example where we evaluate both the CLIP and LLaVA models on the VLCS dataset. You can find the details of this example in our [repository](https://github.com/jameszhou-gl/gpt-4v-distribution-shift/tree/master/exp_output/2023-11-18-21_08_16).

The pipeline is depicted in the following diagram:

![](https://p.ipic.vip/fskcbk.png)

####  Evaluating CLIP Model on VLCS Dataset

First, we evaluate the CLIP model on the VLCS dataset using the following command:

```bash
python ./eval/eval_clip.py --dataset VLCS --num_sample 50
```

During this process, the following files are generated in sequence:

- `unified_input_VLCS.json`: The preprocessed input data for the VLCS dataset.
- `unified_output_clip-vit-base-pathc16.jsonl`: The output from the CLIP model.
- `results_clip-vit-base-pathc16.json`: The summarized results and analysis.

#### Evaluating LLaVA Model

Next, we continue with the evaluation of the LLaVA model, building upon the results from the CLIP model:

```bash
# Evaluate LLaVA based on the previous CLIP evaluation
CUDA_VISIBLE_DEVICES=0,1 python ./eval/eval_llava.py --dataset PACS --continue_dir=exp_output/2023-11-18-21_08_16
```

This step results in the creation of the following files, in order:

- `input_VLCS_in_llava_vqa.jsonl`: Input data for LLaVA in its expected vqa format.
- `output_VLCS_in_llava_vqa.jsonl`: LLaVA's output data.
- `unified_llava-v1.5-13b.json` and `results_llava-v1.5-13.json`: Detailed results and analysis for the LLaVA model.

To review the detailed outcomes of each model, refer to the respective `results_model_name.json` files (e.g., `results_llava-v1.5-13.json`).



## Evaluation



To conduct the evaluation, follow these steps:

### Evaluating with CLIP

To evaluate the CLIP model on the PACS and VLCS datasets:

```bash
python ./eval/eval_clip.py --dataset PACS VLCS --num_sample 50
```

This command evaluates the CLIP model on both PACS and VLCS datasets, processing 50 samples from each.



### Evaluating with LLaVA-v1.5-13b

To evaluate the LLaVA model (the exact model loaded can be found `parser.add_argument('--model_name', 'type'='str', 'choices'=['llava-v1.5-7b', 'llava-v1.5-13b'], 'default'="llava-v1.5-13b")`):

```bash
CUDA_VISIBLE_DEVICES=0,1 python ./eval/eval_llava.py --num_sample 1
```

This command evaluates the LLaVA model, limiting the evaluation to 1 sample per dataset.



### Continuation Evaluation with LLaVA Based on CLIP

To continue the evaluation with LLaVA based on the samples selected from CLIP:

```bash
CUDA_VISIBLE_DEVICES=0,1 python ./eval/eval_llava.py --dataset PACS --continue_dir=<path_to_clip_results>
```

Replace `<path_to_clip_results>` with the directory path where the CLIP results are stored. This command continues the LLaVA evaluation using the CLIP model's results as a starting point, for a quick and fair comparion between two models.