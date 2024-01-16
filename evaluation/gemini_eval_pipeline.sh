#!/bin/bash

# This script runs the Gemini evaluation pipeline. It runs the gemini_scenario_runner with different scenarios.
# Firstly in your terminal: export GOOGLE_API_KEY='your-google-api-key'
# ! Specify
# Directory where the output of the CLIP and LLaVA models is stored
FAILURE_CONTINUE_DIR="/home/guanglinzhou/code/cgm/gpt-4v-distribution-shift/exp_output/2023-11-22-22_21_35"
RANDOM_CONTINUE_DIR="/home/guanglinzhou/code/cgm/gpt-4v-distribution-shift/exp_output/2023-12-07-06_51_36"

# Copy the bash script to the new output directory
cp evaluation/gemini_eval_pipeline.sh "$FAILURE_CONTINUE_DIR"

for scenario in failure_1 failure_2
do
    echo "Running Gemini evaluation for $scenario..."
    python evaluation/gemini_scenario_runner.py --continue_dir $FAILURE_CONTINUE_DIR --scenario_name $scenario --GOOGLE_API_KEY $GOOGLE_API_KEY
done

# Copy the bash script to the new output directory
cp evaluation/gemini_eval_pipeline.sh "$RANDOM_CONTINUE_DIR"

# Loop through scenarios
for scenario in random_1 random_2 random_3 random_4
do
    echo "Running Gemini evaluation for $scenario..."
    python evaluation/gemini_scenario_runner.py --continue_dir $RANDOM_CONTINUE_DIR --scenario_name $scenario --GOOGLE_API_KEY $GOOGLE_API_KEY
done

echo "Gemini evaluation pipeline completed."