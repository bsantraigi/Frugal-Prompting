#!/bin/bash

# Define the argument values for each option
datasets=("MSC" "TC")
models=("flan-t5" "T0" "tk-instruct")
prompt_types=("manual" "ppl")
few_shot_options=( "" "--few_shot" )
background_knowledge_options=( "" "--background_knowledge" )
history_signal_types=("full" "peg" "bart" "recent-k" "semantic-k" "none")
history_k_values=(2 4 8 10)

# Loop through all combinations of argument values
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for prompt_type in "${prompt_types[@]}"; do
      for few_shot in "${few_shot_options[@]}"; do
        for background_knowledge in "${background_knowledge_options[@]}"; do
          for history_signal_type in "${history_signal_types[@]}"; do
            if [[ $history_signal_type == "recent-k" || $history_signal_type == "semantic-k" ]]; then
              for history_k in "${history_k_values[@]}"; do
                # Construct the command to execute
                command="python launcher.py -d $dataset -m $model -pt $prompt_type $few_shot $background_knowledge -hst $history_signal_type -hk $history_k"
                echo "Running command: $command"

                # Execute the command
                $command

                # Add a delay if needed between command executions
                sleep 1s
              done
            else
              # Construct the command to execute without history_k
              command="python launcher.py -d $dataset -m $model -pt $prompt_type $few_shot $background_knowledge -hst $history_signal_type"
              echo "Running command: $command"

              # Execute the command
              $command

              # Add a delay if needed between command executions
              sleep 1s
            fi
          done
        done
      done
    done
  done
done
