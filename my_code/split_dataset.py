import os
import random

BREATHING_DIR = "/data/netmit/wifall/ADetect/data/shhs2_new/thorax"
EEG_DIR = f"/data/netmit/sleep_lab/ali_2/shhs2/c4_m1_multitaper_v2" #float64

def generate_file_list(train_ratio=0.8):
    # Get the list of files with the specified extension
    # files = [os.path.join(root, file) for root, dirs, files in os.walk(data_dir) for file in files if file.endswith(file_extension)]
    files = [f"{EEG_DIR}/{f}" for f in os.listdir(EEG_DIR) if f.endswith(".npz")] #N = 2651 for both eeg and breathing
    breakpoint()

    # Shuffle the files randomly
    random.shuffle(files)
    
    # Split into train and test sets
    train_size = int(len(files) * train_ratio)
    train_files = files[:train_size]
    test_files = files[train_size:]
    
    return train_files, test_files

def write_file_list(file_list, output_file):
    with open(output_file, 'w') as f:
        for file in file_list:
            f.write(f"{file}\n")

if __name__ == "__main__":
    # Define the data directory and file extension to search for
    # data_dir = "path/to/your/data"  # Replace with your directory path
    # file_extension = ".jpg"  # Change this based on the file type you are working with
    
    # Generate the file lists for training and testing
    train_files, test_files = generate_file_list()
    
    # Write the file paths to the respective text files
    # Y:\data\scratch\ellen660\taming-transformers\dataset
    write_path = f'/data/scratch/ellen660/taming-transformers/dataset/shhs2'
    write_file_list(train_files, f"{write_path}/eeg_train.txt")
    write_file_list(test_files, f"{write_path}/eeg_test.txt")
    
    print("Training and testing file lists have been written")
