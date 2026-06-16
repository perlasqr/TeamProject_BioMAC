import os

# ── Set this to your target folder ──────────────────────────────────────────
FOLDER = r"/home/mohityadav001/Documents/FAU/Project/TeamProject/results/DB5/P51/IK/MarkerData"
# ────────────────────────────────────────────────────────────────────────────

renamed = 0
skipped = 0

for filename in os.listdir(FOLDER):
    if filename.startswith("DB4"):
        new_name = "DB5" + filename[3:]          # replace only the leading DB4
        old_path = os.path.join(FOLDER, filename)
        new_path = os.path.join(FOLDER, new_name)
        os.rename(old_path, new_path)
        print(f"  Renamed: {filename}  →  {new_name}")
        renamed += 1
    else:
        skipped += 1

print(f"\nDone. {renamed} file(s) renamed, {skipped} file(s) skipped.")
