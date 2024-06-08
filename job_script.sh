#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=15:0:0    
#SBATCH --mail-user=<youremail@gmail.com>
#SBATCH --mail-type=ALL

# Convert Windows line endings of all .sh files
find . -name "*.sh" -exec dos2unix {} \;

# Preprocess data
sudo bash preprocess.sh -s niid --sf 0.001 -k 0 -t sample -tf 0.98

# Create required directory structure to store data
mkdir -p data_leaf/{training/user,testing/user}

# Load data for each client
python load_data.py
num_clients=10
diff_quantity=0
python client_data_allocation $num_clients $diff_quantity     # remember to change path to "data_leaf/training/shakespeare_instruction_response_pairs_all.json" if cloning project

# Run Flower
python flower.py --global_model 'chavinlo/alpaca-native' --data_path  "./data" --output_dir  './lora-shepherd-7b/' --num_communication_rounds 10 --num_clients  10 --train_on_inputs --group_by_length