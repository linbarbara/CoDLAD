import os

def merge_tcga_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    project_root = os.path.dirname(script_dir)
    
    data_dir = os.path.join(project_root, 'data')
    output_file = os.path.join(data_dir, 'pretrain_tcga.csv')
    
    file_parts = [os.path.join(data_dir, f'pretrain_tcga_part_{str(i).zfill(3)}.csv') for i in range(1, 15)]
    
    print(f"Project Root: {project_root}")
    print(f"Target Output: {output_file}")
    
    if not os.path.exists(file_parts[0]):
        print(f"Error: Could not find {file_parts[0]}")
        print("Please check if the data files are in the ../data/ directory.")
        return

    print(f"Starting to merge {len(file_parts)} files...")

    with open(output_file, 'w') as f_out:
        for i, part in enumerate(file_parts):
            if os.path.exists(part):
                print(f"Merging: {os.path.basename(part)} ...")
                with open(part, 'r') as f_in:
                    if i > 0:
                        next(f_in)
                    f_out.write(f_in.read())
            else:
                print(f"Warning: {part} not found, skipping.")
                
    print(f"\nSuccess! Merged file saved at: {output_file}")

if __name__ == "__main__":
    merge_tcga_data()