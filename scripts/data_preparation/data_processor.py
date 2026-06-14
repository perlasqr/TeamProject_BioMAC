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
        ## self.db_data_path = self.root_dir / "data" / f"{self.db_name}_retargeted"
        self.db_results_path = self.root_dir / "results" / self.db_name

        # Global Reference Axis Configuration map
        # Structure: (target_axis) -> (source_axis, multiplier)
        self.axis_mappings = {
            'DB1': {'x': ('z', -1), 'y': ('y', 1), 'z': ('x', 1)},
            'DB2': {'x': ('x', -1), 'y': ('y', 1), 'z': ('z', -1)},
            'DB3': {'x': ('z', -1), 'y': ('y', 1), 'z': ('x', 1)},
            'DB4': {'x': ('z', 1),  'y': ('y', 1), 'z': ('x', -1)}
        }
        
        self.participants = self._get_participant_list()
        print(f"Initialized {self.db_name}. Found {len(self.participants)} participants.")

        
    def _get_participant_list(self):
        """Scans the directory to retrieve a list of all participant folder names."""
        if not self.db_results_path.exists():
            print(f"Warning: Data path not found: {self.db_results_path}")
            return []
        return [p.name for p in self.db_results_path.iterdir() if p.is_dir()]
        
    def get_file_paths(self, participant_id):
        """Locates and returns a dictionary of relevant file paths for a participant."""
        ## p_data = self.db_data_path / participant_id
        p_res = self.db_results_path / participant_id
        
        paths = {
            "trc_marker": list((p_res / "IK" / "MarkerData").glob("*.trc")),
            "mot_grf":    list((p_res / "ID" / "GRF").glob("*_grf.mot")),
            "mot_ik":     list((p_res / "IK").glob("*_ik.mot")),
            "sto_id":     list((p_res / "ID").glob("*.sto"))
            
            ## "c3d": list(p_data.glob("*.c3d")),
        }
        print(f"[{participant_id}] Found: {len(paths['trc_marker'])} TRC (Marker), {len(paths['mot_grf'])} MOT (Forces) files.")
        return paths
    
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


    def load_data_oldversion(self, file_path):
        """Loads a file into a pandas DataFrame and flattens TRC headers."""
        print(f"Loading: {file_path.name}...")
        skip = self._get_skiprows(file_path)
        
        try:
            return pd.read_csv(file_path, sep='\t', skiprows=skip)
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            return None
        
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
    
    ## ------ PROCESSING DATA ------ ##
    def reorient_coordinates(self, df, file_type):
        """
        Reorients the dataframe columns to match OpenSim standard: X = Forward, Y = Up, Z = Right
        """
        if self.db_name not in self.axis_mappings:
            print(f"Warning: No rotation mapping found for {self.db_name}. Skipping reorientation.")
            return df

        print(f"Reorienting {file_type} coordinate system for {self.db_name}...")    
        mapping = self.axis_mappings[self.db_name]
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
    




    def process_database(self):
        """Main execution loop that iterates through all participants in the database."""
        print(f"Starting processing for {self.db_name}...")
        for p_id in self.participants:
            print(f"> Processing participant: {p_id}")
            files = self.get_file_paths(p_id)
            
            # Logic for calling processing functions would go here
            # e.g., self.apply_filters(files)
        print(f"Finished processing {self.db_name}.")