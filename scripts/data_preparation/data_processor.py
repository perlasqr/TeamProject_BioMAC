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
        self.db_results_path = self.root_dir / "results" / self.db_name

        # Global Reference Axis Configuration map
        # Structure: (target_axis) -> (source_axis, multiplier)
        self.axis_mappings = {
            'DB1': {'trc': {'x': ('z', -1), 'y': ('y', 1), 'z': ('x', 1)}, 
                    'mot': {'x': ('z', -1), 'y': ('y', 1), 'z': ('x', 1)}},
            'DB2': {'trc': {'x': ('x', -1), 'y': ('y', 1), 'z': ('z', -1)}, 
                    'mot': {'x': ('x', -1), 'y': ('y', 1), 'z': ('z', -1)}},
            'DB3': {'trc': {'x': ('z', -1), 'y': ('y', 1), 'z': ('x', 1)}, 
                    'mot': {'x': ('z', -1), 'y': ('y', 1), 'z': ('x', 1)}},
            'DB4': {'trc': {'x': ('z', 1),  'y': ('y', 1), 'z': ('x', -1)}, 
                    'mot': {'x': ('z', 1),  'y': ('y', 1), 'z': ('x', -1)}},
            'DB5': {'trc': {'x': ('x', -1), 'y': ('y', -1), 'z': ('z', 1)}, 
                    'mot': {'x': ('x', 1),  'y': ('y', 1),  'z': ('z', 1)}}
        }
        
        self.participants = self._get_participant_list()
        print(f"Initialized {self.db_name}. Found {len(self.participants)} participants.")

        
    def _get_participant_list(self):
        """Scans the directory to retrieve a list of all participant folder names."""
        if not self.db_results_path.exists():
            print(f"Warning: Data path not found: {self.db_results_path}")
            return []
        return [p.name for p in self.db_results_path.iterdir() if p.is_dir()]
        
    def get_trial_file_map(self, participant_id):
        """
        Groups all related files for each trial into a dictionary.
        Returns: { 'Trial_Name': {'trc': path, 'mot_ik': path, 'mot_grf': path, 'sto': path} }
        """
        p_res = self.db_results_path / participant_id
        
        # 1. Gather all potential files
        # We look for all files recursively in the participant's folders
        all_files = list(p_res.rglob("*.trc")) + \
                    list(p_res.rglob("*_ik.mot")) + \
                    list(p_res.rglob("*_grf.mot")) + \
                    list(p_res.rglob("*.sto"))
        
        trial_map = {}
        
        # 2. Extract common "Trial Name" by removing known suffixes
        for f in all_files:
            # Create a base name by removing the file extension and type markers
            base_name = f.name.replace(".trc", "").replace("_ik.mot", "").replace("_grf.mot", "").replace(".sto", "")
            
            if base_name not in trial_map:
                trial_map[base_name] = {'trc': None, 'mot_ik': None, 'mot_grf': None, 'sto': None}
            
            # Map based on extension and suffix
            if f.suffix == '.trc': trial_map[base_name]['trc'] = f
            elif f.name.endswith('_ik.mot'): trial_map[base_name]['mot_ik'] = f
            elif f.name.endswith('_grf.mot'): trial_map[base_name]['mot_grf'] = f
            elif f.suffix == '.sto': trial_map[base_name]['sto'] = f
            
        return trial_map
    
    def _get_skiprows(self, file_path):
        """Determines skiprows based on file extension."""
        ext = file_path.suffix.lower()
        
        if ext in ['.mot', '.sto']:
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    if 'endheader' in line:
                        return i + 1
            # Fallback if 'endheader' is missing
            return 11 
            
        elif ext == '.trc': # TRC files use a fixed header structure
            return 5
            
        return 0
        
    def load_data(self, file_path, file_type=None):
        """Loads a file into a pandas DataFrame, handling specific biomech formats."""
        print(f"Loading: {file_path.name}...")
        ext = file_path.suffix.lower()
        
        try:
            if ext == '.trc':
                # --- TRC Header Flattening Logic ---
                raw_headers = pd.read_csv(file_path, sep='\t', skiprows=3, nrows=0).columns.tolist()
                clean_cols = []
                current_marker = ""
                
                for i, col in enumerate(raw_headers):
                    if i == 0:
                        clean_cols.append("Frame")
                    elif i == 1:
                        clean_cols.append("Time")
                    else:
                        if not col.startswith("Unnamed"):
                            current_marker = col
                            clean_cols.append(f"{current_marker}_X")
                        else:
                            if clean_cols[-1].endswith("_X"):
                                clean_cols.append(f"{current_marker}_Y")
                            else:
                                clean_cols.append(f"{current_marker}_Z")
                
                df = pd.read_csv(file_path, sep='\t', skiprows=5, header=None)
                
                if df.shape[1] > len(clean_cols):
                    df = df.iloc[:, :len(clean_cols)]
                
                df.columns = clean_cols
                return df
                
            elif ext in ['.mot', '.sto']:
                # --- Dynamic MOT/STO Loading Logic ---
                skip = self._get_skiprows(file_path)
                return pd.read_csv(file_path, sep='\t', skiprows=skip)
                
            else:
                print(f"Unsupported format: {ext}")
                return None
                
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            return None
    
<<<<<<< Updated upstream
    
    # ------ PROCESSING DATA ------ ##
=======
    ## ------ PREPARING DATA ------ ##
>>>>>>> Stashed changes
    def reorient_coordinates(self, df, file_type):
        """
        Reorients the dataframe columns to match OpenSim standard: X = Forward, Y = Up, Z = Right
        """
        # Determine if we are looking at trc or mot/sto data
        mapping_key = 'trc' if file_type == 'trc_marker' else 'mot'

        if self.db_name not in self.axis_mappings or mapping_key not in self.axis_mappings[self.db_name]:
            print(f"Warning: No rotation mapping found for {self.db_name}. Skipping reorientation.")
            return df

        print(f"Reorienting {file_type} coordinate system for {self.db_name}...")    
        mapping = self.axis_mappings[self.db_name][mapping_key]
        new_df = df.copy()
        
        if file_type == 'trc_marker':
            print(f"Reorienting TRC coordinates for {self.db_name}...")
            
            # Find all unique marker names by stripping the '_X'
            markers = [col[:-2] for col in df.columns if col.endswith('_X')]
            
            for marker in markers:
                orig_x, orig_y, orig_z = f"{marker}_X", f"{marker}_Y", f"{marker}_Z"
                
                if orig_y in df.columns and orig_z in df.columns:
                    source_data = {'x': df[orig_x], 'y': df[orig_y], 'z': df[orig_z]}
                    
                    new_df[orig_x] = source_data[mapping['x'][0]] * mapping['x'][1]
                    new_df[orig_y] = source_data[mapping['y'][0]] * mapping['y'][1]
                    new_df[orig_z] = source_data[mapping['z'][0]] * mapping['z'][1]
                    
        elif file_type == 'mot_grf':
            print(f"Reorienting GRF coordinates for {self.db_name}...")
            
            # OpenSim forces end in 'vx', 'px', or 'mx'. 
            # We strip the 'x' to find the base name (e.g., 'ground_force_calcn_l_v')
            base_names = [col[:-1] for col in df.columns if col.endswith('x')]
            
            for base in base_names:
                orig_x, orig_y, orig_z = f"{base}x", f"{base}y", f"{base}z"
                
                # Check if the full x/y/z vector set exists
                if orig_y in df.columns and orig_z in df.columns:
                    source_data = {'x': df[orig_x], 'y': df[orig_y], 'z': df[orig_z]}
                    
                    new_df[orig_x] = source_data[mapping['x'][0]] * mapping['x'][1]
                    new_df[orig_y] = source_data[mapping['y'][0]] * mapping['y'][1]
                    new_df[orig_z] = source_data[mapping['z'][0]] * mapping['z'][1]

        return new_df
    
    ## ------ SELECTING RELEVANT COLUMNS ------ #
    def select_relevant_data(self, df, file_type):

        if df is None:
            print("Warning: Input DataFrame is None. Skipping column selection.")
            return None

        # Make column lookup case-insensitive, but preserve original column names
        col_map = {col.lower(): col for col in df.columns}

        if file_type == "trc_marker":
            markers_to_keep = [
                "LSHO", "LASI", "LKNE", "LANK", "LHEE",
                "RSHO", "RASI", "RKNE", "RANK", "RHEE"
            ]

            selected_cols = []

            # Your TRC loader creates "Time", not "time", but this also works if it is "time"
            if "time" in col_map:
                selected_cols.append(col_map["time"])
            else:
                print("Warning: Time column not found in TRC file.")

            for marker in markers_to_keep:
                for axis in ["X", "Y", "Z"]:
                    col_name = f"{marker}_{axis}"
                
                    if col_name.lower() in col_map:
                        selected_cols.append(col_map[col_name.lower()])
                    else:
                        print(f"Warning: Missing TRC column: {col_name}")

            return df[selected_cols].copy()
    
        elif file_type == "mot_grf":
            # Keep all GRF data
            return df.copy()

        elif file_type == "mot_ik":
            cols_to_keep = [
                "time",
                "hip_flexion_r",
                "knee_angle_r",
                "ankle_angle_r",
                "hip_flexion_l",
                "knee_angle_l",
                "ankle_angle_l"
            ]
            return df[cols_to_keep].copy()
        
        elif file_type == "sto_id":
            cols_to_keep = [
                "time",
                "hip_flexion_r_moment",
                "knee_angle_r_moment",
                "ankle_angle_r_moment",
                "hip_flexion_l_moment",
                "knee_angle_l_moment",
                "ankle_angle_l_moment"
            ]

        else:
            print(f"Warning: Unknown file type '{file_type}'. Returning original DataFrame.")
            return df.copy()

        selected_cols = []

        for col in cols_to_keep:
            if col.lower() in col_map:
                selected_cols.append(col_map[col.lower()])
            else:
                print(f"Warning: Missing column in {file_type}: {col}")

        return df[selected_cols].copy()

    def process_participant_files(self, participant_id):
        """Loads all files for one participant, reorients TRC/GRF if needed, and selects only the relevant columns."""

        paths = self.get_file_paths(participant_id)

        processed_data = {
            "trc_marker": [],
            "mot_grf": [],
            "mot_ik": [],
            "sto_id": []
        }

        for file_type, file_list in paths.items():

            for file_path in file_list:
                df = self.load_data(file_path)

                if df is None:
                    print(f"Skipping {file_path.name} because it could not be loaded.")
                    continue

                # Reorientation just for TRC & GRF
                if file_type in ["trc_marker", "mot_grf"]:
                    df = self.reorient_coordinates(df, file_type)

                # Select only relevant columns
                df_selected = self.select_relevant_data(df, file_type)

                processed_data[file_type].append({
                    "participant_id": participant_id,
                    "file_type": file_type,
                    "file_name": file_path.name,
                    "file_path": file_path,
                    "data": df_selected
                })

        return processed_data

    




<<<<<<< Updated upstream
    # def process_database(self):
    #    """Main execution loop that iterates through all participants in the database."""
    #   print(f"Starting processing for {self.db_name}...")
    #    for p_id in self.participants:
    #        print(f"> Processing participant: {p_id}")
    #        files = self.get_file_paths(p_id)
    #        
    #        # Logic for calling processing functions would go here
    #        # e.g., self.apply_filters(files)
    #    print(f"Finished processing {self.db_name}.")