import os
from pathlib import Path
import pandas as pd

class DataProcessor:
    def __init__(self, root_dir, db_name):
        """
        Initializes the processor and sets up base paths for a specific database.
        
        :param root_dir: Path to the main directory containing 'data' and 'results' folders.
        :param db_name: The name of the database folder (e.g., 'DB1').
        """
        self.root_dir = Path(root_dir)
        self.db_name = db_name
        self.db_data_path = self.root_dir / "data" / f"{self.db_name}_retargeted"
        self.db_results_path = self.root_dir / "results" / self.db_name
        
        self.participants = self._get_participant_list()
        print(f"Initialized {self.db_name}. Found {len(self.participants)} participants.")
        
    def _get_participant_list(self):
        """Scans the directory to retrieve a list of all participant folder names."""
        if not self.db_data_path.exists():
            print(f"Warning: Data path not found: {self.db_data_path}")
            return []
        return [p.name for p in self.db_data_path.iterdir() if p.is_dir()]
        
    def get_file_paths(self, participant_id):
        """Locates and returns a dictionary of relevant file paths for a participant."""
        p_data = self.db_data_path / participant_id
        p_res = self.db_results_path / participant_id
        
        paths = {
            "c3d": list(p_data.glob("*.c3d")),
            "mot": list((p_res / "IK").glob("*.mot")),
            "sto": list((p_res / "ID").glob("*.sto"))
        }
        print(f"[{participant_id}] Found: {len(paths['c3d'])} C3D, {len(paths['mot'])} MOT, {len(paths['sto'])} STO files.")
        return paths

    def load_data(self, file_path):
        """Loads a file into a pandas DataFrame based on its extension."""
        print(f"Loading: {file_path.name}...")
        ext = file_path.suffix.lower()
        
        if ext == '.csv':
            return pd.read_csv(file_path)
        elif ext in ['.mot', '.sto']:
            return pd.read_csv(file_path, sep='\t', skiprows=11)
        else:
            print(f"Skipping unsupported format: {ext}")
            return None
            
    def process_database(self):
        """Main execution loop that iterates through all participants in the database."""
        print(f"Starting processing for {self.db_name}...")
        for p_id in self.participants:
            print(f"> Processing participant: {p_id}")
            files = self.get_file_paths(p_id)
            
            # Logic for calling processing functions would go here
            # e.g., self.apply_filters(files)
        print(f"Finished processing {self.db_name}.")