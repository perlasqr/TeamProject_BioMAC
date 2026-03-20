import os
import shutil
import re

base_dir = "path to folder"
output_dir = os.path.join(base_dir, "Renamed_Walking_Trials")

# Create output dir if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Regex to match walking trial files (w or pw)
# Examples: 
# P7 -130w 01.c3d 
# P7 65w 03.c3d
# P7 pw 01.c3d
# P10 pw01.c3d
# P10 -65w 01.c3d
# It matches: P<id> <width><w or pw>[ ]<trial>.c3d
pattern = re.compile(r"^P(\d+)\s+([p\-\d]+)(?:w)\s*(\d+)\.c3d$", re.IGNORECASE)

print(f"Starting to rename walking trials in {base_dir}")

files_copied = 0

for folder_name in sorted(os.listdir(base_dir)):
    # Look for folders like P7, P8, ..., P13
    if not re.match(r"^P\d+$", folder_name, re.IGNORECASE):
        continue
        
    p_num = int(folder_name[1:])
    if not (7 <= p_num <= 13):
        continue
        
    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue
        
    out_subfolder = os.path.join(output_dir, f"P{p_num:02d}")
    os.makedirs(out_subfolder, exist_ok=True)
        
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".c3d"):
            continue
            
        match = pattern.match(filename)
        if match:
            id_str = match.group(1)
            width_str = match.group(2)
            trial_str = match.group(3)
            
            # Map width string "p" to "p"
            # In filename, preferred is usually "pw" (where width="p", w is the "w")
            if width_str.lower() == 'p':
                new_width = 'p'
            else:
                new_width = width_str
                
            new_filename = f"DB2_P{int(id_str):02d}_T{int(trial_str):02d}_{new_width}.c3d"
            
            src_path = os.path.join(folder_path, filename)
            dest_path = os.path.join(out_subfolder, new_filename)
            
            shutil.copy2(src_path, dest_path)
            print(f"Copied: {folder_name}/{filename} -> {new_filename}")
            files_copied += 1

print(f"\nCompleted! Copies {files_copied} walking trials to {output_dir}")
