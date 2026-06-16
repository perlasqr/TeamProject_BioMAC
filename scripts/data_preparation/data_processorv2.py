import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

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
            base_name = f.name.replace(".trc", "").replace("_ik.mot", "").replace("_grf.mot", "").replace("_id.sto", "")
            
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
            return 11 # Fallback if 'endheader' is missing
            
        elif ext == '.trc': # TRC files use a fixed header structure
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    # Look for the line starting with 'Frame#'
                    if line.startswith('Frame#'):
                        return i
            return 5 # Fallback if 'endheader' is missing
            
        return 0

    ## ------ LOAD Data ------ #   
    def load_data(self, file_path, file_type=None):
        """Loads a file into a pandas DataFrame, handling specific biomech formats."""
        print(f"Loading: {file_path.name}...")
        ext = file_path.suffix.lower()
        
        try:
            if ext == '.trc':
                    # 1. Read data first to see how many columns actually exist
                df = pd.read_csv(file_path, sep='\t', skiprows=6, header=None)
                num_cols = df.shape[1]
                
                # 2. Extract marker names from line 4 (index 3)
                header_row = pd.read_csv(file_path, sep='\t', skiprows=3, nrows=1, header=None)
                raw_markers = header_row.iloc[0, 2:].tolist()
                
                # 3. Create column names 1-to-1 with data columns
                columns = ['Frame', 'Time']
                axes = ['X', 'Y', 'Z']
                axis_idx = 0
                current_m = "Unknown"
                
                for m in raw_markers:
                    # If the cell is not empty/NaN/Unnamed, update the current marker name
                    m_str = str(m).strip()
                    if m_str != '' and m_str != 'nan' and 'Unnamed' not in m_str:
                        current_m = m_str
                    
                    # Append exactly ONE axis for this specific column
                    if len(columns) < num_cols:
                        columns.append(f"{current_m}_{axes[axis_idx]}")
                        axis_idx = (axis_idx + 1) % 3  # Cycles 0, 1, 2, 0, 1, 2 (X, Y, Z)
                
                # 4. Final safety check: fill any remaining unnamed columns
                while len(columns) < num_cols:
                    columns.append(f"Extra_{len(columns)}")
                    
                df.columns = columns[:num_cols]
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
    
    
    ## ------ REORIENTING COORDINATE SYSTEMS ------ #
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
            markers = ["LSHO", "LASI", "LKNE", "LANK", "LHEE", "RSHO", "RASI", "RKNE", "RANK", "RHEE"]
            
            # Find the time column regardless of case
            time_col = next((col_map[c] for c in col_map if 'time' in c), None)
            selected_cols = [time_col] if time_col else []
            
            # Find markers
            for m in markers:
                for axis in ["X", "Y", "Z"]:
                    col = f"{m}_{axis}"
                    if col.lower() in col_map:
                        selected_cols.append(col_map[col.lower()])
            
            return df[selected_cols].copy()
    
        elif file_type == "mot_grf":
            # Keep all GRF data
            return df.copy()

        elif file_type in ["mot_ik", "sto_id"]:
            # Combine the logic for IK/ID to avoid repetition
            targets = ["time", "hip_flexion_r", "knee_angle_r", "ankle_angle_r", 
                       "hip_flexion_l", "knee_angle_l", "ankle_angle_l"]
            if file_type == "sto_id":
                targets = [t + "_moment" if t != "time" else t for t in targets]
            
            selected_cols = [col_map[t] for t in targets if t in col_map]
            return df[selected_cols].copy()

        return df.copy()

    ## ------ EXTRACTING GAIT EVENTS & TIME-NORMALIZING ------- #
    def extract_gait_events(self, grf_df, marker_df, leg='l', force_threshold=20):
        """
        Finds the timestamps for HS1 (Heel Strike 1), TO (Toe-Off), and HS2 (Heel Strike 2).
        Returns: (hs1_time, to_time, hs2_time) or (None, None, None) if not found.
        """
        # 1. Identify specific columns based on the active leg
        if leg.lower() == 'l':
            grf_v_col = next((c for c in grf_df.columns if 'calcn_l_vy' in c.lower()), None)
            heel_y_col = 'LHEE_Y'
        else:
            grf_v_col = next((c for c in grf_df.columns if 'calcn_r_vy' in c.lower()), None)
            heel_y_col = 'RHEE_Y'

        if not grf_v_col or heel_y_col not in marker_df.columns:
            print(f"  [Error] Missing columns for {leg} leg detection.")
            return None, None, None

        # 2. Find HS1 and TO using GRF (Vertical Force)
        fz = grf_df[grf_v_col].values
        times_grf = grf_df['time'].values
        
        contact = (fz > force_threshold).astype(int)
        transitions = np.diff(contact)
        
        hs_indices = np.where(transitions == 1)[0] + 1
        to_indices = np.where(transitions == -1)[0] + 1
        
        if len(hs_indices) == 0 or len(to_indices) == 0:
            return None, None, None
            
        # Grab the FIRST valid heel strike and the first toe-off that happens AFTER it
        hs1_idx = hs_indices[0]
        valid_tos = to_indices[to_indices > hs1_idx]
        if len(valid_tos) == 0:
            return None, None, None
            
        to_idx = valid_tos[0]
        
        hs1_time = times_grf[hs1_idx]
        to_time = times_grf[to_idx]
        
        # 3. Find HS2 using Kinematics (Heel Marker Height)
        # Because Y is UP in OpenSim, we look at LHEE_Y / RHEE_Y
        times_mrk = marker_df['Time'].values
        heel_y = marker_df[heel_y_col].values
        
        # Find the marker frame closest to our GRF events
        hs1_mrk_idx = np.argmin(np.abs(times_mrk - hs1_time))
        to_mrk_idx = np.argmin(np.abs(times_mrk - to_time))
        
        ref_height = heel_y[hs1_mrk_idx]
        buffer = 0.02 # 2 cm buffer (assuming TRC is in meters!)
        
        search_start = to_mrk_idx + 10 # Start looking a bit after toe-off
        hs2_time = None
        
        for i in range(search_start, len(heel_y) - 1):
            current_h = heel_y[i]
            velocity = heel_y[i+1] - heel_y[i]
            
            # Condition: Heel drops near baseline height AND is moving downward
            if current_h <= (ref_height + buffer) and velocity < 0:
                hs2_time = times_mrk[i]
                break
                
        return hs1_time, to_time, hs2_time

    def time_normalize(self, df, time_col_name, start_time, end_time, n_points=101):
        """
        Slices a DataFrame between start_time and end_time, then interpolates all columns 
        to exactly `n_points` (0% to 100% of gait cycle).
        """
        # 1. Slice the data to the specific gait cycle
        cycle_df = df[(df[time_col_name] >= start_time) & (df[time_col_name] <= end_time)].copy()
        
        if len(cycle_df) < 5:
            return None # Not enough data to interpolate safely
            
        old_time = cycle_df[time_col_name].values
        new_time = np.linspace(start_time, end_time, n_points)
        
        # 2. Interpolate each column
        norm_data = {}
        for col in cycle_df.columns:
            if col == time_col_name:
                norm_data[col] = np.linspace(0, 100, n_points) # Convert time to % Gait Cycle
            else:
                f = interp1d(old_time, cycle_df[col].values, kind='linear', fill_value="extrapolate")
                norm_data[col] = f(new_time)
                
        return pd.DataFrame(norm_data)

    ## ------ CONVERT LEG LABELS ------ #
    def convert_leg_labels(self, df, file_type, leading_leg):
        if df is None: return None
        
        if leading_leg.lower() not in ["l", "r"]:
            raise ValueError("leading_leg must be either 'l' or 'r'")

        lead_char = leading_leg.lower()
        trail_char = 'r' if lead_char == 'l' else 'l'
        
        new_columns = {}
        
        for raw_col in df.columns:
            # Strip any hidden spaces from the column name just in case
            col = str(raw_col).strip()
            
            if col.lower() == 'time':
                new_columns[raw_col] = 'Time'
                continue
                
            new_col = col
            
            if file_type == 'trc_marker':
                if len(col) >= 4 and col[0].lower() in ['l', 'r'] and col[-2:] in ['_X', '_Y', '_Z']:
                    side = col[0].lower()
                    base_marker = col[1:-2]
                    axis = col[-1]
                    
                    if base_marker == 'ANK': base_marker = 'ANKL'
                    
                    suffix = '1' if side == lead_char else '2'
                    new_col = f"{base_marker}_{axis}{suffix}"
                    
            elif file_type == 'mot_grf':
                if f"_{lead_char}_v" in col or f"_{lead_char}_p" in col or f"_{lead_char}_m" in col:
                    new_col = col.replace(f"_{lead_char}_", "_") + "1"
                elif f"_{trail_char}_v" in col or f"_{trail_char}_p" in col or f"_{trail_char}_m" in col:
                    new_col = col.replace(f"_{trail_char}_", "_") + "2"
                    
            elif file_type in ['mot_ik', 'sto_id']:
                if f"_{lead_char}_moment" in new_col:
                    new_col = new_col.replace(f"_{lead_char}_moment", "_1_moment")
                elif f"_{trail_char}_moment" in new_col:
                    new_col = new_col.replace(f"_{trail_char}_moment", "_2_moment")
                elif new_col.endswith(f"_{lead_char}"):
                    new_col = new_col[:-2] + "_1"
                elif new_col.endswith(f"_{trail_char}"):
                    new_col = new_col[:-2] + "_2"
                    
            # Map the original raw column name to the new clean name
            new_columns[raw_col] = new_col
            
        # --- DEBUG PRINT ---
        # Let's print out the first 3 changes it made to see if the logic worked
        changed_cols = {k: v for k, v in new_columns.items() if k != v}
        if changed_cols:
            print(f"  [Debug] {file_type} successfully mapped {len(changed_cols)} columns. Example: {list(changed_cols.items())[:3]}")
        else:
            print(f"  [Debug] {file_type} found ZERO columns to rename!")
            
        # Explicitly return the renamed dataframe
        return df.rename(columns=new_columns)

    ## ------ RELATIVE PELVIS-CENTERED AP COORDINATE SYSTEM ------ #
    def define_pelvis_centered_ap_coordinate_system(self, marker_df):
        
        if marker_df is None:
            print("  [Warning] marker_df is None. Skipping pelvis-centered AP coordinate system.")
            return None

        new_df = marker_df.copy()

        def clean_col_name(col):
            return str(col).split("[")[0].strip().upper().replace(" ", "")

        clean_to_original = {
            clean_col_name(col): col for col in new_df.columns
        }

        asi_x1_col = clean_to_original.get("ASI_X1")
        asi_x2_col = clean_to_original.get("ASI_X2")

        if asi_x1_col is None or asi_x2_col is None:
            print("  [Warning] ASI_X1 and/or ASI_X2 not found. Skipping pelvis-centered AP shift.")
            print(f"  Available columns example: {list(new_df.columns)[:12]}")
            return new_df

        pelvis_center_x = (new_df[asi_x1_col] + new_df[asi_x2_col]) / 2

        for col in new_df.columns:
            cleaned = clean_col_name(col)

            if cleaned in ["TIME", "FRAME", "FRAME#"]:
                continue

            # Shift only anterior-posterior marker columns
            # Examples: SHO_X1, ASI_X1, KNE_X2, ANKL_X1, HEE_X2
            if "_X" in cleaned:
                new_df.loc[:, col] = new_df.loc[:, col].sub(pelvis_center_x, axis="index")

        print("  Pelvis-centered AP coordinate system applied using ASI_X1 and ASI_X2.")

        return new_df

    def define_global_coordinate_system(self, df):
        """
        Defines a global coordinate system for all TRC markers such that ANKL_1 
        (ANKL_X1, ANKL_Y1, ANKL_Z1) becomes the origin (0, 0, 0) at each frame.
        Uses subtraction in each corresponding axis.
        """
        if df is None: return None
        
        new_df = df.copy()
        
        # Find the origin columns (case-insensitive lookup)
        ankl_x = next((col for col in new_df.columns if col.upper() == 'ANKL_X1'), None)
        ankl_y = next((col for col in new_df.columns if col.upper() == 'ANKL_Y1'), None)
        ankl_z = next((col for col in new_df.columns if col.upper() == 'ANKL_Z1'), None)
        
        if not (ankl_x and ankl_y and ankl_z):
            print("  [Warning] Origin marker ANKL_1 not found in dataframe. Skipping global coordinate shift.")
            return new_df
            
        origin_x = new_df[ankl_x]
        origin_y = new_df[ankl_y]
        origin_z = new_df[ankl_z]
        
        # Apply subtraction to all marker coordinates corresponding to X, Y, Z axes
        for col in new_df.columns:
            if col.lower() == 'time' or col.lower() == 'frame':
                continue
                
            if '_X' in col.upper():
                new_df[col] = new_df[col] - origin_x
            elif '_Y' in col.upper():
                new_df[col] = new_df[col] - origin_y
            elif '_Z' in col.upper():
                new_df[col] = new_df[col] - origin_z
                
        return new_df


    ## PREPARING DATA 
    def process_participant_files(self, participant_id):
        trial_map = self.get_trial_file_map(participant_id)
        processed_data = [] # List to hold trials

        for trial_name, files in trial_map.items():
            trial_results = {"trial_name": trial_name, "data": {}}
            
            for file_key, file_path in files.items():
                if file_path is None: continue
                
                # Map keys to match your file_type expected by other methods
                file_type = "trc_marker" if file_key == 'trc' else \
                            "mot_ik"     if file_key == 'mot_ik' else \
                            "mot_grf"    if file_key == 'mot_grf' else "sto_id"
                
                df = self.load_data(file_path, file_type=file_type)
                if df is None: continue

                if file_type in ["trc_marker", "mot_grf"]:
                    df = self.reorient_coordinates(df, file_type)

                trial_results["data"][file_type] = self.select_relevant_data(df, file_type)
            
            processed_data.append(trial_results)
        return processed_data

    
 
        
