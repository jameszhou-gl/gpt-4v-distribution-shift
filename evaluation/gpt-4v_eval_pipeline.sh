#!/bin/bash

# This script runs the GPT-4V evaluation pipeline. It prepares the evaluation,
# then runs the GPT-4V scenario runner with different scenarios and API keys.
# ! Specify
# Directory where the output of the CLIP and LLaVA models is stored
CONTINUE_DIR="exp_output/2023-11-22-19_18_50"
# Number of random and failure cases to prepare for GPT-4V evaluation
NUM_RAND=1800 # 1800 in default
NUM_FAILURE=180 # 180 in default

# Copy the bash script to the new output directory
cp evaluation/gpt-4v_eval_pipeline.sh "$CONTINUE_DIR"

# Prepare the GPT-4V evaluation dataset
echo "Preparing GPT-4V evaluation dataset..."
python evaluation/prepare_gpt4v_evaluation.py --num_rand $NUM_RAND --num_failure $NUM_FAILURE --continue_dir $CONTINUE_DIR

# Run GPT-4V evaluation for different scenarios
# Scenario 1: Failure cases, Part 1
echo "Running GPT-4V evaluation for Failure Scenario 1..."
python evaluation/gpt-4v_scenario_runner.py --continue_dir $CONTINUE_DIR --scenario_name failure_1 --openai_api_key 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

# Scenario 2: Failure cases, Part 2
echo "Running GPT-4V evaluation for Failure Scenario 2..."
python evaluation/gpt-4v_scenario_runner.py --continue_dir $CONTINUE_DIR --scenario_name failure_2 --openai_api_key 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

# Scenario 1: Random cases, Part 1
echo "Running GPT-4V evaluation for Random Scenario 1..."
python evaluation/gpt-4v_scenario_runner.py --continue_dir $CONTINUE_DIR --scenario_name random_1 --openai_api_key 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

# Scenario 2: Random cases, Part 2
echo "Running GPT-4V evaluation for Random Scenario 2..."
python evaluation/gpt-4v_scenario_runner.py --continue_dir $CONTINUE_DIR --scenario_name random_2 --openai_api_key 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

# Scenario 3: Random cases, Part 3
echo "Running GPT-4V evaluation for Random Scenario 3..."
python evaluation/gpt-4v_scenario_runner.py --continue_dir $CONTINUE_DIR --scenario_name random_3 --openai_api_key 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

# Scenario 4: Random cases, Part 4
echo "Running GPT-4V evaluation for Random Scenario 4..."
python evaluation/gpt-4v_scenario_runner.py --continue_dir $CONTINUE_DIR --scenario_name random_4 --openai_api_key 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


echo "GPT-4V evaluation pipeline completed."
