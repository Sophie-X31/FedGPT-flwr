#!/bin/bash
#SBATCH --account=def-qizhen       # Set the account
#SBATCH --partition=Narval         # Specify the partition/cluster name
#SBATCH --gres=gpu:A100:1          # Request 1 A100 GPU
#SBATCH --mem=40G                  # Request 40G of memory
#SBATCH --nodes=1                  # Request 1 node
#SBATCH --ntasks-per-node=8        # Set the number of tasks per node
#SBATCH --time=15:0:0              # Set the time limit
#SBATCH --mail-user=<sophiesoap.xu@mail.utoronto.ca>  # Set the email for notifications
#SBATCH --mail-type=ALL            # Set the email notifications type
#SBATCH --output=slurm_fedgpt.out      # Customize the output file name
#SBATCH --error=slurm_fedgpt.err       # Separate standard error to a different file
#SBATCH --export=ALL,DISABLE_DCGM=1  # Disable DCGM

# Run model
branch=main
python ${branch}.py --global_model 'chavinlo/alpaca-native'\
    --data_path  "./data" \
    --output_dir  './lora-shepherd-7b/'\
    --num_communication_rounds 10 \
    --num_clients  10 \
    --train_on_inputs \
    --group_by_length