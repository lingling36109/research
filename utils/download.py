import os
import sys
import csv
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Download matrix files from SuiteSparse Matrix Collection')
    parser.add_argument('--filename', required=True, help='Path to the CSV file containing matrix information')
    parser.add_argument('--download_dir', required=True, help='Directory to download and extract matrices')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    filename = args.filename
    download_dir = args.download_dir
    
    # Validate inputs
    if not os.path.exists(filename):
        print(f"Error: CSV file '{filename}' does not exist")
        sys.exit(1)
    
    if not os.path.exists(download_dir):
        print(f"Creating download directory: {download_dir}")
        os.makedirs(download_dir, exist_ok=True)
    
    total = sum(1 for line in open(filename))
    print(f"Total rows in CSV: {total}")

    with open(filename) as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)
        for i in range(1, total):
            cur_row = next(csv_reader)
            matrix_name = cur_row[2]
            matrix_path = os.path.join(download_dir, matrix_name + ".mtx")
            
            if not os.path.exists(matrix_path):
                print(f"Downloading {matrix_name}...")
                matrix_url = f"http://sparse-files.engr.tamu.edu/MM/{cur_row[1]}/{cur_row[2]}.tar.gz"
                tar_file = os.path.join(download_dir, matrix_name + ".tar.gz")
                
                # Download
                ret = os.system(f"wget {matrix_url} -O {tar_file}")
                if ret != 0:
                    print(f"Error downloading {matrix_name}, skipping...")
                    continue
                
                # Extract
                os.system(f"tar -zxvf {tar_file} -C {download_dir}")
                
                # Move .mtx file to download directory
                extracted_dir = os.path.join(download_dir, matrix_name)
                extracted_mtx = os.path.join(extracted_dir, matrix_name + ".mtx")
                if os.path.exists(extracted_mtx):
                    os.system(f"mv {extracted_mtx} {matrix_path}")
                
                # Cleanup
                os.system(f"rm -rf {tar_file}")
                os.system(f"rm -rf {extracted_dir}")
            else:
                print(f"Skipping {matrix_name} (already exists)")
