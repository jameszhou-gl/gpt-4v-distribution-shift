#!/bin/bash

# This script runs the Gemini evaluation pipeline. It runs the gemini_scenario_runner with different scenarios.
# Firstly in your terminal: export GOOGLE_API_KEY='your-google-api-key'
# ! Specify
# Directory where the output of the CLIP and LLaVA models is stored

#---PACS-----#
FAILURE_CONTINUE_DIR="./exp_output/2023-11-22-22_21_35"
RANDOM_CONTINUE_DIR="./exp_output/2023-12-07-06_51_36"

# Copy the bash script to the new output directory
cp evaluation/gemini_eval_pipeline.sh "$FAILURE_CONTINUE_DIR"

for scenario in failure_1 failure_2
do
    echo "Running Gemini evaluation for $scenario..."
    python evaluation/gemini_scenario_runner.py --continue_dir $FAILURE_CONTINUE_DIR --scenario_name $scenario --GOOGLE_API_KEY $GOOGLE_API_KEY
done

for scenario in random_1 random_2
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

#---VLCS-----#
FAILURE_CONTINUE_DIR="./exp_output/2023-11-22-22_36_30"
RANDOM_CONTINUE_DIR="./exp_output/2023-12-06-20_27_30"

# Copy the bash script to the new output directory
cp evaluation/gemini_eval_pipeline.sh "$FAILURE_CONTINUE_DIR"

for scenario in failure_1 failure_2
do
    echo "Running Gemini evaluation for $scenario..."
    python evaluation/gemini_scenario_runner.py --continue_dir $FAILURE_CONTINUE_DIR --scenario_name $scenario --GOOGLE_API_KEY $GOOGLE_API_KEY
done

for scenario in random_1 random_2
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

#---office_home-----#
FAILURE_CONTINUE_DIR="./exp_output/2023-11-24-02_41_54"
RANDOM_CONTINUE_DIR="./exp_output/2023-12-07-01_30_24"

# Copy the bash script to the new output directory
cp evaluation/gemini_eval_pipeline.sh "$FAILURE_CONTINUE_DIR"

for scenario in failure_1 failure_2
do
    echo "Running Gemini evaluation for $scenario..."
    python evaluation/gemini_scenario_runner.py --continue_dir $FAILURE_CONTINUE_DIR --scenario_name $scenario --GOOGLE_API_KEY $GOOGLE_API_KEY
done

for scenario in random_1 random_2
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


#---domain_net-----#
# ! Specify
# Directory where the output of the CLIP and LLaVA models is stored
FAILURE_CONTINUE_DIR="./exp_output/2023-11-26-01_46_58"
RANDOM_CONTINUE_DIR="./exp_output/2023-12-07-12_33_36"

# Copy the bash script to the new output directory
cp evaluation/gemini_eval_pipeline.sh "$FAILURE_CONTINUE_DIR"

for scenario in failure_1 failure_2
do
    echo "Running Gemini evaluation for $scenario..."
    python evaluation/gemini_scenario_runner.py --continue_dir $FAILURE_CONTINUE_DIR --scenario_name $scenario --GOOGLE_API_KEY $GOOGLE_API_KEY
done

for scenario in random_1 random_2
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


#---fmow_v1.1_processed-----#
# ! Specify
# Directory where the output of the CLIP and LLaVA models is stored
FAILURE_CONTINUE_DIR="./exp_output/2023-11-23-22_48_54"
RANDOM_CONTINUE_DIR="./exp_output/2023-12-08-02_21_16"

# Copy the bash script to the new output directory
cp evaluation/gemini_eval_pipeline.sh "$FAILURE_CONTINUE_DIR"

for scenario in failure_1 failure_2
do
    echo "Running Gemini evaluation for $scenario..."
    python evaluation/gemini_scenario_runner.py --continue_dir $FAILURE_CONTINUE_DIR --scenario_name $scenario --GOOGLE_API_KEY $GOOGLE_API_KEY
done

for scenario in random_1 random_2
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


#---terra_incognita-----#
# ! Specify
# Directory where the output of the CLIP and LLaVA models is stored
FAILURE_CONTINUE_DIR="./exp_output/2023-11-23-20_47_37"
RANDOM_CONTINUE_DIR="./exp_output/2023-12-06-20_27_33"

# Copy the bash script to the new output directory
cp evaluation/gemini_eval_pipeline.sh "$FAILURE_CONTINUE_DIR"

for scenario in failure_1 failure_2
do
    echo "Running Gemini evaluation for $scenario..."
    python evaluation/gemini_scenario_runner.py --continue_dir $FAILURE_CONTINUE_DIR --scenario_name $scenario --GOOGLE_API_KEY $GOOGLE_API_KEY
done

for scenario in random_1 random_2
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


#---camelyon17_v1.0_processed-----#
# ! Specify
# Directory where the output of the CLIP and LLaVA models is stored
FAILURE_CONTINUE_DIR="./exp_output/2023-11-23-19_11_08"
RANDOM_CONTINUE_DIR="./exp_output/2023-12-08-00_50_17"

# Copy the bash script to the new output directory
cp evaluation/gemini_eval_pipeline.sh "$FAILURE_CONTINUE_DIR"

for scenario in failure_1 failure_2
do
    echo "Running Gemini evaluation for $scenario..."
    python evaluation/gemini_scenario_runner.py --continue_dir $FAILURE_CONTINUE_DIR --scenario_name $scenario --GOOGLE_API_KEY $GOOGLE_API_KEY
done

for scenario in random_1 random_2
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