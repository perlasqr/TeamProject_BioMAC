import os

# ── Set this to your ROOT folder (will recurse into all subfolders) ──────────
FOLDER = r"/home/mohityadav001/Documents/FAU/Project/TeamProject/results/DB5/P29"
# ────────────────────────────────────────────────────────────────────────────

renamed = 0
skipped = 0

# os.walk visits every folder and all its sub-folders (any depth)
for dirpath, dirnames, filenames in os.walk(FOLDER):
    for filename in filenames:
        if filename.startswith("DB4"):
            new_name = "DB5" + filename[3:]          # replace only the leading DB4
            old_path = os.path.join(dirpath, filename)
            new_path = os.path.join(dirpath, new_name)
            os.rename(old_path, new_path)
            print(f"  Renamed: {old_path}  →  {new_name}")
            renamed += 1
        else:
            skipped += 1

print(f"\nDone. {renamed} file(s) renamed, {skipped} file(s) skipped.")
