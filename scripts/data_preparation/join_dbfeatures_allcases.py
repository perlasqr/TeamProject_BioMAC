import pandas as pd
from pathlib import Path

def merge_databases():
    # --- 1. SET UP PATHS ---
    root_dir = r"C:\Users\perla\Documents\AT\TeamProject\GitHub\TeamProject"
    target_folder = Path(root_dir) / "scripts" / "approach_grf_markers"
    
    if not target_folder.exists():
        print(f"[Error] Target folder not found: {target_folder}")
        return

    # The 5 databases and the new suffix to look for
    databases = ["DB1", "DB2", "DB3", "DB4", "DB5"]
    suffix = "ML_Matrix.csv"  # Assuming they are saved as CSVs
    
    print(f"\n--- Starting Database Merger in: {target_folder.name} ---")
    
    combined_data = []
    
    # --- 2. LOAD EACH DB ---
    for db in databases:
        # Reconstruct the expected filename (e.g., "DB1_ML_Matrix.csv")
        file_name = f"{db}_{suffix}"
        file_path = target_folder / file_name
        
        if file_path.exists():
            print(f"  -> Loading {file_name}...")
            # Load the CSV into a pandas DataFrame
            df = pd.read_csv(file_path)
            combined_data.append(df)
        else:
            print(f"  [Warning] File not found: {file_name}")
    
    # --- 3. MERGE AND SAVE ---
    if combined_data:
        # Stack all the loaded dataframes on top of each other
        merged_df = pd.concat(combined_data, ignore_index=True)
        
        # Create the final output filename 
        output_name = "All_DBs_cases.csv"
        output_path = target_folder / output_name
        
        # Save the massive matrix
        merged_df.to_csv(output_path, index=False)
        
        print(f"\n[Success] Merged {len(combined_data)} databases into {output_name}")
        print(f"          Final Matrix Shape: {merged_df.shape[0]} rows x {merged_df.shape[1]} features")
    else:
        print("\n[Failed] No files were found. Skipping merge.")

if __name__ == "__main__":
    merge_databases()