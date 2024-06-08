#!bin/bash

# Check if on Windows system (adjust the condition as needed for your environment)
if [ "$(uname -s)" = "Windows_NT" ]; then
    # Convert Windows line endings of all .sh files
    find . -name "*.sh" -exec dos2unix {} \;
fi

# Check if data_leaf directory does not exist
if [ ! -d "data_leaf" ]; then
    # Preprocess data
    sudo bash preprocess.sh -s niid --sf 0.001 -k 0 -t sample -tf 0.98

    # Create required directory structure to store data
    mkdir -p data_leaf/{training/user,testing/user}

    # Load data for each client
    python load_data.py
    num_clients=10
    diff_quantity=0
    python client_data_allocation.py $num_clients $diff_quantity
fi