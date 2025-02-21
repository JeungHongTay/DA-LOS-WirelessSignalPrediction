import os
import pandas as pd
import numpy as np
from numpy import vstack

def import_from_files(dataset_path='dataset/'):
    """
    Read .csv files and store data into an array.
    
    Args:
        dataset_path (str): Path to the dataset directory
        
    Returns:
        numpy.ndarray: Array containing the dataset
    
    Raises:
        FileNotFoundError: If dataset directory doesn't exist
        ValueError: If no CSV files found in directory
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found at: {dataset_path}")
        
    output_arr = []
    first = 1
    files_processed = 0
    
    # Process all subdirectories
    for dirpath, dirnames, filenames in os.walk(dataset_path):
        csv_files = [f for f in filenames if f.endswith('.csv')]
        
        for file in csv_files:
            filename = os.path.join(dirpath, file)
            print(f"Processing file: {filename}")
            
            try:
                # read data from file
                df = pd.read_csv(filename, sep=',', header=0)
                input_data = df.to_numpy()
                
                # append to array
                if first > 0:
                    first = 0
                    output_arr = input_data
                else:
                    output_arr = vstack((output_arr, input_data))
                files_processed += 1
                print(f"Successfully loaded data from {filename}")
                
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                continue
    
    if files_processed == 0:
        raise ValueError("No data was loaded from any CSV files")
        
    print(f"Successfully processed {files_processed} files")
    print(f"Dataset shape: {output_arr.shape}")
    return output_arr