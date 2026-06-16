import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os

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

        #print(f"Reorienting {file_type} coordinate system for {self.db_name}...")    
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
            
        # # --- DEBUG PRINT ---
        # # Let's print out the first 3 changes it made to see if the logic worked
        # changed_cols = {k: v for k, v in new_columns.items() if k != v}
        # if changed_cols:
        #     print(f"  [Debug] {file_type} successfully mapped {len(changed_cols)} columns. Example: {list(changed_cols.items())[:3]}")
        # else:
        #     print(f"  [Debug] {file_type} found ZERO columns to rename!")
            
        # Explicitly return the renamed dataframe
        return df.rename(columns=new_columns)

    ## ------ GLOBAL COORDINATE SYSTEM ------ #
    def define_global_coordinate_system(self, df):
        """
        Defines a global coordinate system for all TRC markers such that ANKL_1 
        (ANKL_Y1, ANKL_Z1) becomes the origin (0, 0) at each frame.
        Uses subtraction in each corresponding axis.
        """
        if df is None: return None
        
        new_df = df.copy()
        
        # Find the origin columns (case-insensitive lookup)
        ankl_y = next((col for col in new_df.columns if col.upper() == 'ANKL_Y1'), None)
        ankl_z = next((col for col in new_df.columns if col.upper() == 'ANKL_Z1'), None)
        
        if not (ankl_y and ankl_z):
            print("  [Warning] Origin marker ANKL_1 not found in dataframe. Skipping global coordinate shift.")
            return new_df
            
        origin_y = new_df[ankl_y]
        origin_z = new_df[ankl_z]
        
        # Apply subtraction to all marker coordinates corresponding to X, Y, Z axes
        for col in new_df.columns:
            if col.lower() == 'time' or col.lower() == 'frame':
                continue
                
            if '_Y' in col.upper():
                new_df[col] = new_df[col] - origin_y
            elif '_Z' in col.upper():
                new_df[col] = new_df[col] - origin_z
                
        return new_df

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

    ## ------ ANTHROPOMETRIC NORMALIZATION
    def load_participant_metadata(self, participant_id):
        """
        Reads the database_inventory.xlsx file to extract Mass and Height 
        for a specific participant number within the current DB sheet.
        """
        import os
        import pandas as pd

        excel_path = os.path.join(self.root_dir, "data", "database_inventory.xlsx")
        
        if not os.path.exists(excel_path):
            print(f"  [Error] Metadata file not found at: {excel_path}")
            return None, None
            
        try:
            # Load the sheet corresponding to the active database name
            df_meta = pd.read_excel(excel_path, sheet_name=self.db_name)
            
            # Extract just the numeric digits from the participant_id string (e.g., 'P01' -> 1)
            p_numeric = int(''.join(filter(str.isdigit, str(participant_id))))
            
            # Identify columns by positional index: 1st column (0)
            p_col = df_meta.iloc[:, 0]
            
            # Safe parsing that handles 'P01', '1', and '1.0' correctly:
            # 1. Convert to string
            # 2. Split by decimal point and take the left side (turns '1.0' into '1')
            # 3. Strip out any remaining non-digit characters
            clean_p_col = p_col.astype(str).str.split('.').str[0]
            clean_p_col = clean_p_col.str.replace(r'\D+', '', regex=True)
            clean_p_col = pd.to_numeric(clean_p_col, errors='coerce') 
            
            # Exact match check
            match_mask = clean_p_col == p_numeric
            row = df_meta[match_mask]
            
            if row.empty:
                print(f"  [Warning] Participant {participant_id} (parsed as {p_numeric}) not found in sheet {self.db_name}.")
                # --- TEMPORARY DEBUG PRINTS ---
                print(f"  [Debug] Excel Column 1 header: {df_meta.columns[0]}")
                print(f"  [Debug] First 5 values found in Column 1: {list(p_col.dropna().head())}")
                # ------------------------------
                return None, None
                
            mass = float(row.iloc[0, 3])   # 4th column
            height = float(row.iloc[0, 4]) # 5th column
            
            return mass, height
            
        except Exception as e:
            print(f"  [Error] Failed to read metadata sheet: {str(e)}")
            return None, None

    def apply_anthropometric_normalization(self, normalized_trial, mass, height):
        """
        Applies dimensionless scaling to all loaded data frames using Body Weight and Height.
        """
        if normalized_trial is None: return None
        
        # Calculate Body Weight in Newtons
        gravity = 9.81
        body_weight = mass * gravity
        
        scaled_trial = {}
        
        for f_type, df in normalized_trial.items():
            if df is None: continue
            
            scaled_df = df.copy()
            
            for col in scaled_df.columns:
                col_upper = col.upper()
                col_lower = col.lower() 
                
                # Skip tracking headers
                if col_upper in ['TIME', 'FRAME', 'FRAME#']:
                    continue
                    
                # 1. TRC Marker Positions -> Divide by Height
                if f_type == 'trc_marker':
                    if any(axis in col_upper for axis in ['_X', '_Y', '_Z']):
                        scaled_df[col] = scaled_df[col] / height
                        
                # 2. Ground Reaction Forces, CoP, and Free Moments
                elif f_type == 'mot_grf':
                    # Forces (vx, vy, vz) -> Divide by Body Weight
                    if any(f in col_lower for f in ['_vx', '_vy', '_vz']):
                        scaled_df[col] = scaled_df[col] / body_weight
                    # Center of Pressure (px, py, pz) -> Divide by Height
                    elif any(p in col_lower for p in ['_px', '_py', '_pz']):
                        scaled_df[col] = scaled_df[col] / height
                    # Free Moments (mx, my, mz) -> Divide by Body Weight * Height
                    elif any(m in col_lower for m in ['_mx', '_my', '_mz']):
                        scaled_df[col] = scaled_df[col] / (body_weight * height)
                        
                # 3. Inverse Dynamics Joint Moments -> Divide by Body Weight * Height
                elif f_type == 'sto_id':
                    if '_MOMENT' in col_upper:
                        scaled_df[col] = scaled_df[col] / (body_weight * height)
                        
            scaled_trial[f_type] = scaled_df
            
        return scaled_trial


    ## PREPARING DATA 
    def process_participant_files(self, participant_id):
        trial_map = self.get_trial_file_map(participant_id)
        processed_data = [] # List to hold trials

        # --- EXTRACT ANTHROPOMETRICS ONCE PER PARTICIPANT ---
        mass, height = self.load_participant_metadata(participant_id)
        if mass is None or height is None:
            print(f"  [Aborted] Cannot process participant {participant_id} without complete mass/height tracking data.")
            return []

        for trial_name, files in trial_map.items():
            trial_results = {"trial_name": trial_name, "data": {}}
            
            # --- 1. LOAD, REORIENT, AND FILTER DATA ---
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
            
            # --- 2. EXTRACT GAIT EVENTS & TIME NORMALIZE ---
            data = trial_results["data"]
            hs1_t, to_t, hs2_t = None, None, None
            leg_used = None
            
            if data.get('mot_grf') is not None and data.get('trc_marker') is not None:
                
                # Try Left Leg
                leg_used = 'L'
                hs1_t, to_t, hs2_t = self.extract_gait_events(data['mot_grf'], data['trc_marker'], leg='l')
                
                # Try Right Leg if Left fails
                if hs1_t is None or hs2_t is None:
                    leg_used = 'R'
                    hs1_t, to_t, hs2_t = self.extract_gait_events(data['mot_grf'], data['trc_marker'], leg='r')
            
                # --- 3. APPLY PIPELINE TO VALID CYCLES ---
                if hs1_t is not None and hs2_t is not None:
                    normalized_trial = {}
                    
                    for f_type, df in data.items():
                        if df is not None:
                            # A. Time Normalize (0-100%)
                            t_col = 'Time' if 'Time' in df.columns else 'time'
                            norm_df = self.time_normalize(df, t_col, hs1_t, hs2_t)
                            
                            # B. Convert Labels (L/R -> 1/2)
                            renamed_df = self.convert_leg_labels(norm_df, f_type, leg_used)
                            
                            # C. Coordinate Shifts (ONLY for TRC Markers)
                            if f_type == 'trc_marker':
                                # Global Shift (Y, Z axes to Ankle)
                                df_global = self.define_global_coordinate_system(renamed_df)
                                # Relative Shift (X axis to Pelvis)
                                df_final = self.define_pelvis_centered_ap_coordinate_system(df_global)
                                normalized_trial[f_type] = df_final
                            else:
                                normalized_trial[f_type] = renamed_df

                    # --- 4. APPLY ANTHROPOMETRIC SCALING ---
                    scaled_trial = self.apply_anthropometric_normalization(normalized_trial, mass, height)
                    
                    trial_results['normalized_data'] = scaled_trial
                    trial_results['gait_events'] = {'leg_used': leg_used, 'HS1': hs1_t, 'TO': to_t, 'HS2': hs2_t}
                    trial_results['anthropometrics'] = {'mass_kg': mass, 'height_m': height}
                    
                else:
                    trial_results['normalized_data'] = None
                    trial_results['error'] = "Incomplete gait cycle or no heel strike found."
            else:
                trial_results['normalized_data'] = None
                trial_results['error'] = "Missing GRF or TRC data needed for sync."

            processed_data.append(trial_results)

        return processed_data

    
    ## ------ EXPORTING RESULTS ------ #
    def export_database_results(self):
        """
        Runs the full processing pipeline for all participants in the database.
        Generates two files:
        1. A Summary Report CSV tracking success/failure for every trial.
        2. A Master ML Matrix CSV with flattened 101-point features for all valid trials.
        """
        summary_rows = []
        ml_rows = []
        
        print(f"\n--- Starting Full Database Export for {self.db_name} ---")
        
        for participant in self.participants:
            print(f"Processing participant: {participant}...")
            
            # Run our robust pipeline for this specific participant
            trials = self.process_participant_files(participant)
            
            for trial in trials:
                t_name = trial['trial_name']
                is_success = trial.get('normalized_data') is not None
                
                # --- 1. BUILD SUMMARY REPORT ROW ---
                status = "Success" if is_success else "Failed"
                error_msg = trial.get('error', '')
                leg = trial.get('gait_events', {}).get('leg_used', '') if is_success else ''
                
                summary_rows.append({
                    'DB': self.db_name,
                    'Participant': participant,
                    'Trial': t_name,
                    'Status': status,
                    'Leg_Used': leg,
                    'Error_Message': error_msg
                })
                
                # --- 2. BUILD FLATTENED ML MATRIX ROW ---
                if is_success:
                    # Initialize the row with critical metadata
                    flat_row = {
                        'DB': self.db_name,
                        'Participant': participant,
                        'Trial': t_name,
                        'Leg_Used': leg,
                        'Mass_kg': trial['anthropometrics']['mass_kg'],
                        'Height_m': trial['anthropometrics']['height_m']
                    }
                    
                    norm_data = trial['normalized_data']
                    
                    # Map the internal dictionary keys to your ML prefixes
                    prefix_map = {
                        'trc_marker': 'TRC',
                        'mot_grf': 'GRF',
                        'mot_ik': 'IK',
                        'sto_id': 'ID'
                    }
                    
                    for f_type, prefix in prefix_map.items():
                        if f_type in norm_data and norm_data[f_type] is not None:
                            df = norm_data[f_type]
                            
                            for col in df.columns:
                                # Skip time-tracking columns
                                if col.upper() in ['TIME', 'FRAME', 'FRAME#']:
                                    continue
                                    
                                # Flatten the 101 points
                                vals = df[col].values
                                for i, val in enumerate(vals):
                                    # Creates headers like: TRC_ANKL_X1_0, GRF_vy1_100
                                    flat_row[f"{prefix}_{col}_{i}"] = val
                                    
                    ml_rows.append(flat_row)

        # --- 3. SAVE THE CSV FILES ---
        # Ensure the output directory exists
        out_dir = self.root_dir / "results" / self.db_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Summary Report
        df_summary = pd.DataFrame(summary_rows)
        summary_path = out_dir / f"{self.db_name}_Processing_Report.csv"
        df_summary.to_csv(summary_path, index=False)
        print(f"\n[Export Complete] Summary Report saved: {summary_path}")
        
        # Save Master ML Matrix
        if ml_rows:
            df_ml = pd.DataFrame(ml_rows)
            ml_path = out_dir / f"{self.db_name}_ML_Matrix.csv"
            df_ml.to_csv(ml_path, index=False)
            print(f"[Export Complete] Master ML Matrix saved: {ml_path}")
            print(f"  -> Final Matrix Shape: {df_ml.shape[0]} trials x {df_ml.shape[1]} features")
        else:
            print("\n[Warning] No successful trials were found. ML Matrix was not created.")

