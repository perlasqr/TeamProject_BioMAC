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
            'DB3': {'trc': {'x': ('z', 1), 'y': ('y', 1), 'z': ('x', 1)}, 
                    'mot': {'x': ('z', 1), 'y': ('y', 1), 'z': ('x', 1)}},
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
            
        # --- DEBUG GHOST TRIALS ---
        #for t_name, f_dict in trial_map.items():
        #    if f_dict['trc'] is None or f_dict['mot_grf'] is None:
        #        has_trc = "YES" if f_dict['trc'] is not None else "NO"
        #        has_grf = "YES" if f_dict['mot_grf'] is not None else "NO"
        #        print(f"  [Debug Map] Potential mismatch for '{t_name}' -> TRC found: {has_trc} | GRF found: {has_grf}")
        # --------------------------
         
        return trial_map
    
    def consolidate_segmented_trials(self, trial_map):
        """
        Groups segmented trials into a single base trial dictionary.
        Safely strips '_segment_X' to avoid accidentally merging different trials (like T04 and T05).
        """
        consolidated = {}
        for trial_name, files in trial_map.items():
            if '_segment_' in trial_name:
                parts = trial_name.split('_segment_')
                base_name = parts[0]  # e.g., 'DB3_P01_A_T04_relabeled'
                segment_idx = int(parts[1].split('_')[0]) # e.g., 0 or 1
            else:
                base_name = trial_name
                segment_idx = 0
                
            if base_name not in consolidated:
                consolidated[base_name] = {}
                
            consolidated[base_name][segment_idx] = files
            
        return consolidated

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
        #print(f"Loading: {file_path.name}...")
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
    def map_raw_axes(self, df, file_type):
        """
        Maps the raw lab axes to OpenSim XYZ buckets and applies 
        static system calibration multipliers (e.g., fixing Y-down setups).
        """
        mapping_key = 'trc' if file_type == 'trc_marker' else 'mot'

        if self.db_name not in self.axis_mappings or mapping_key not in self.axis_mappings[self.db_name]:
            return df

        mapping = self.axis_mappings[self.db_name][mapping_key]
        new_df = df.copy()
        
        if file_type == 'trc_marker':
            markers = [col[:-2] for col in df.columns if col.endswith('_X')]
            for marker in markers:
                orig_x, orig_y, orig_z = f"{marker}_X", f"{marker}_Y", f"{marker}_Z"
                if orig_y in df.columns and orig_z in df.columns:
                    source_data = {'x': df[orig_x], 'y': df[orig_y], 'z': df[orig_z]}
                    
                    # Apply mapping: source_data['target_axis'] * multiplier
                    new_df[orig_x] = source_data[mapping['x'][0]] * mapping['x'][1]
                    new_df[orig_y] = source_data[mapping['y'][0]] * mapping['y'][1]
                    new_df[orig_z] = source_data[mapping['z'][0]] * mapping['z'][1]
                    
        elif file_type == 'mot_grf':
            base_names = [col[:-1] for col in df.columns if col.endswith('x')]
            for base in base_names:
                orig_x, orig_y, orig_z = f"{base}x", f"{base}y", f"{base}z"
                if orig_y in df.columns and orig_z in df.columns:
                    source_data = {'x': df[orig_x], 'y': df[orig_y], 'z': df[orig_z]}
                    
                    # Apply mapping: source_data['target_axis'] * multiplier
                    new_df[orig_x] = source_data[mapping['x'][0]] * mapping['x'][1]
                    new_df[orig_y] = source_data[mapping['y'][0]] * mapping['y'][1]
                    new_df[orig_z] = source_data[mapping['z'][0]] * mapping['z'][1]

        return new_df
    
    def unify_trial_physics(self, marker_df, grf_df, threshold=20.0):
        """
        SIMPLIFIED GEOMETRIC UNIFICATION:
        1. Uses Pelvis to determine if walking +X or -X.
        2. If -X, rotates BOTH markers and GRF 180 degrees (flips X and Z).
        3. Ensures vertical GRF (Y) is positive, maintaining Right-Hand Rule.
        """
        unified_marker = marker_df.copy()
        unified_grf = grf_df.copy()

        # --- 1. DETERMINE WALKING DIRECTION FROM KINEMATICS ---
        lasi_col = next((c for c in unified_marker.columns if 'LASI_X' in c.upper() or 'L_ASI_X' in c.upper()), None)
        rasi_col = next((c for c in unified_marker.columns if 'RASI_X' in c.upper() or 'R_ASI_X' in c.upper()), None)
        
        if not lasi_col or not rasi_col:
            return unified_marker, unified_grf

        pelvis_x = (unified_marker[lasi_col] + unified_marker[rasi_col]) / 2.0
        pelvis_x_clean = pelvis_x.dropna()
        
        if len(pelvis_x_clean) < 10:
            return unified_marker, unified_grf
            
        delta_x = pelvis_x_clean.iloc[-1] - pelvis_x_clean.iloc[0]

        # --- 2. ROTATE 180 DEGREES IF WALKING BACKWARD ---
        # If they walked -X, we spin the entire room around the Y (vertical) axis.
        # This mathematically forces both kinematics and kinetics to face +X.
        if delta_x < 0:
            # Rotate Markers
            mrk_x_cols = [c for c in unified_marker.columns if c.endswith('_X')]
            mrk_z_cols = [c for c in unified_marker.columns if c.endswith('_Z')]
            unified_marker[mrk_x_cols] = unified_marker[mrk_x_cols] * -1
            unified_marker[mrk_z_cols] = unified_marker[mrk_z_cols] * -1

            # Rotate GRF (Forces, CoP, Moments)
            grf_x_cols = [c for c in unified_grf.columns if c.endswith('x')]
            grf_z_cols = [c for c in unified_grf.columns if c.endswith('z')]
            unified_grf[grf_x_cols] = unified_grf[grf_x_cols] * -1
            unified_grf[grf_z_cols] = unified_grf[grf_z_cols] * -1

        # --- 3. HARDWARE CALIBRATION FAILSAFE (Vertical Axis) ---
        # Check if the force plate was wired upside down (Y pointing into the floor)
        l_vy = next((c for c in unified_grf.columns if c.endswith('_l_vy')), None)
        r_vy = next((c for c in unified_grf.columns if c.endswith('_r_vy')), None)
        
        target_leg = None
        for leg_vy in [l_vy, r_vy]:
            if leg_vy and unified_grf[leg_vy].abs().max() > threshold:
                target_leg = leg_vy
                break
        
        if target_leg:
            stance_data = unified_grf[unified_grf[target_leg].abs() > threshold]
            
            # If the average vertical force is negative, it's upside down
            if len(stance_data) > 10 and stance_data[target_leg].mean() < 0:
                grf_y_cols = [c for c in unified_grf.columns if c.endswith('y')]
                unified_grf[grf_y_cols] = unified_grf[grf_y_cols] * -1
                
                # If we flip Y, we MUST also flip Z to keep the Right-Hand Rule intact
                grf_z_cols = [c for c in unified_grf.columns if c.endswith('z')]
                unified_grf[grf_z_cols] = unified_grf[grf_z_cols] * -1

        return unified_marker, unified_grf

    def validate_grf_quality(self, grf_df, target_leg_vy, force_threshold=20.0):
        """
        Validates that the GRF data actually resembles human walking.
        Excludes flatlines, noise, and static standing.
        Returns: (is_valid: bool, error_message: str)
        """
        target_leg_vx = target_leg_vy.replace('_vy', '_vx')
        
        # Isolate the data where the foot is actually on the plate
        stance_data = grf_df[grf_df[target_leg_vy].abs() > force_threshold]
        
        if len(stance_data) < 10:
            return False, "Stance phase too short or completely missing."
            
        # 1. Calculate Peak-to-Peak Ranges
        y_range = stance_data[target_leg_vy].max() - stance_data[target_leg_vy].min()
        x_range = stance_data[target_leg_vx].max() - stance_data[target_leg_vx].min()
        
        # --- TUNABLE BIOMECHANICAL THRESHOLDS ---
        # If your raw data is in Newtons:
        min_y_range = 100.0  # Walking MUST have an impact peak and mid-stance dip (>100N difference)
        min_x_range = 50.0   # Walking MUST have distinct braking and propulsion (>50N difference)
        
        # 2. Apply the Rules
        if y_range < min_y_range:
            return False, f"Flatline Vertical GRF. Range ({y_range:.1f}) is below minimum ({min_y_range})."
            
        if x_range < min_x_range:
            return False, f"Missing Braking/Propulsion. AP Range ({x_range:.1f}) is below minimum ({min_x_range})."
            
        return True, ""
    
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
    def extract_gait_events(self, grf_df, marker_df, leg='l', force_threshold=10, silent=False):
        """
        Finds timestamps for HS1, TO, and HS2.
        HS1 and TO use GRF. HS2 uses proportional kinematic timing to avoid HS3.
        """
        # 1. Identify specific columns
        if leg.lower() == 'l':
            grf_v_col = next((c for c in grf_df.columns if 'calcn_l_vy' in c.lower()), None)
            heel_y_col = 'LHEE_Y'
        else:
            grf_v_col = next((c for c in grf_df.columns if 'calcn_r_vy' in c.lower()), None)
            heel_y_col = 'RHEE_Y'

        if not grf_v_col or heel_y_col not in marker_df.columns:
            return None, None, None

        # 2. Find HS1 and TO using GRF
        fz = grf_df[grf_v_col].values
        times_grf = grf_df['time'].values
        contact = (fz > force_threshold).astype(int)
        transitions = np.diff(contact)
        
        hs_indices = np.where(transitions == 1)[0] + 1
        to_indices = np.where(transitions == -1)[0] + 1
        
        if len(hs_indices) == 0 or len(to_indices) == 0:
            return None, None, None
            
        hs1_idx = hs_indices[0]
        valid_tos = to_indices[to_indices > hs1_idx]
        if len(valid_tos) == 0: return None, None, None
            
        to_idx = valid_tos[0]
        hs1_time = times_grf[hs1_idx]
        to_time = times_grf[to_idx]
        
        # 3. Find HS2 using proportional kinematic window
        times_mrk = marker_df['Time'].values
        heel_y = marker_df[heel_y_col].values
        
        stance_time = to_time - hs1_time
        
        # Reject impossible walking stance times (less than 0.4 seconds)
        if stance_time < 0.4:
            if not silent:
                print(f"  [Debug] Stance too short ({stance_time:.2f}s). Likely started mid-step.")
            return None, None, None

        # --- THE SIMPLIFIED FIX ---
        # Predict HS2 timing based on typical gait proportions (Swing ~ 66% of Stance)
        expected_swing_time = stance_time * 0.66
        expected_hs2_time = to_time + expected_swing_time
        
        # Create a tight window around the expected HS2 time (+/- 30% of swing time)
        window_margin = expected_swing_time * 0.30
        window_start = expected_hs2_time - window_margin
        window_end = expected_hs2_time + window_margin
        
        # Find the marker frame indices that fall inside this restricted time window
        window_mask = (times_mrk >= window_start) & (times_mrk <= window_end)
        valid_indices = np.where(window_mask)[0]
        
        hs2_time = None
        
        if len(valid_indices) > 0:
            # The heel must be at its lowest point inside this specific window
            window_heel_data = heel_y[valid_indices]
            local_min_idx = np.argmin(window_heel_data)
            hs2_idx = valid_indices[local_min_idx]
            
            # Compare heights for the Sanity Check
            hs1_mrk_idx = np.argmin(np.abs(times_mrk - hs1_time))
            hs1_height = heel_y[hs1_mrk_idx]
            hs2_height = heel_y[hs2_idx]
            
            # Sanity Check
            if np.abs(hs2_height - hs1_height) < 0.05:
                hs2_time = times_mrk[hs2_idx]
            else:
                if not silent:
                    print(f"  [Debug] Sanity Check FAILED. Height diff too large.")
        else:
            if not silent:
                print("  [Debug] No marker data found in the expected HS2 time window.")

        return hs1_time, to_time, hs2_time
 
    def time_normalize(self, df, time_col, start_t, end_t, num_points=101):
        """
        Time-normalizes a dataframe from start_t to end_t.
        Strictly prevents out-of-bounds extrapolation.
        """
        # 1. Create the new 0-100% time vector
        new_time = np.linspace(start_t, end_t, num_points)
        orig_time = df[time_col].values
        
        # 2. Create a new dictionary to build the normalized dataframe quickly
        norm_data = {'time_percent': np.linspace(0, 100, num_points)}
        
        # 3. Interpolate every column safely
        for col in df.columns:
            if col == time_col:
                continue
                
            orig_data = df[col].values
            
            # THE FIX: bounds_error=False AND fill_value=np.nan stops the linear dive
            f = interp1d(orig_time, orig_data, kind='cubic', bounds_error=False, fill_value=np.nan)
            
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
        ankl_y = next((col for col in new_df.columns if col.upper() == 'HEE_Y1'), None)
        ankl_z = next((col for col in new_df.columns if col.upper() == 'HEE_Z1'), None)
        
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
        """
        Shifts the AP (X) coordinate system to be relative to the Pelvis.
        Returns None if required markers are missing.
        """
        
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
            #print("  [Warning] ASI_X1 and/or ASI_X2 not found. Skipping pelvis-centered AP shift.")
            ##print(f"  Available columns example: {list(new_df.columns)[:12]}")
            return None

        pelvis_center_x = (new_df[asi_x1_col] + new_df[asi_x2_col]) / 2

        for col in new_df.columns:
            cleaned = clean_col_name(col)

            if cleaned in ["TIME", "FRAME", "FRAME#"]:
                continue

            # Shift only anterior-posterior marker columns
            # Examples: SHO_X1, ASI_X1, KNE_X2, ANKL_X1, HEE_X2
            if "_X" in cleaned:
                new_df.loc[:, col] = new_df.loc[:, col].sub(pelvis_center_x, axis="index")

        #print("  Pelvis-centered AP coordinate system applied using ASI_X1 and ASI_X2.")

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

    def validate_id_quality(self, id_df, threshold=0.15):
        """
        Validates that the normalized ID joint moments fall within a biomechanically 
        plausible dimensionless range (e.g., between -0.2 and +0.2).
        Returns: (is_valid: bool, error_message: str)
        """
        if id_df is None:
            return True, "" # If no ID data exists for this trial, let it pass
            
        # Find only the joint moment columns
        moment_cols = [c for c in id_df.columns if 'moment' in c.lower()]
        
        for col in moment_cols:
            max_val = id_df[col].max()
            min_val = id_df[col].min()
            
            # If the peak goes outside the plausible threshold, flag it
            if max_val > threshold or min_val < -threshold:
                return False, f"Noise detected in {col} (Min: {min_val:.3f}, Max: {max_val:.3f}, Limit: ±{threshold})"
                
        return True, ""

    ## PREPARING DATA 
    def process_participant_files(self, participant_id):
        # --- PRE-PROCESSING: Group segments safely ---
        raw_trial_map = self.get_trial_file_map(participant_id)
        trial_map = self.consolidate_segmented_trials(raw_trial_map)
        
        processed_data = [] # List to hold trials

        # --- EXTRACT ANTHROPOMETRICS ONCE PER PARTICIPANT ---
        mass, height = self.load_participant_metadata(participant_id)
        if mass is None or height is None:
            print(f"  [Aborted] Cannot process participant {participant_id} without complete mass/height tracking data.")
            return []

        for base_trial_name, segments in trial_map.items():
            trial_results = {"trial_name": base_trial_name, "data": {}}
            
            # Temporary storage to hold lists of DataFrames chronologically
            collected_data = {'trc_marker': [], 'mot_ik': [], 'mot_grf': [], 'sto_id': []}
            
            # --- 1. LOAD AND COLLECT DATA CHRONOLOGICALLY ---
            for segment_idx in sorted(segments.keys()):
                files = segments[segment_idx]
                
                for file_key, file_path in files.items():
                    if file_path is None: continue
                    
                    file_type = "trc_marker" if file_key == 'trc' else \
                                "mot_ik"     if file_key == 'mot_ik' else \
                                "mot_grf"    if file_key == 'mot_grf' else "sto_id"
                    
                    df = self.load_data(file_path, file_type=file_type)
                    if df is None: continue

                    # Purely map columns to buckets, NO physics yet
                    if file_type in ["trc_marker", "mot_grf"]:
                        df = self.map_raw_axes(df, file_type)

                    relevant_df = self.select_relevant_data(df, file_type)
                    if relevant_df is not None and not relevant_df.empty:
                        collected_data[file_type].append(relevant_df)
            
            # --- 2. STITCH SEGMENTS TOGETHER ---
            for f_type, df_list in collected_data.items():
                if len(df_list) > 0:
                    stitched_df = pd.concat(df_list, ignore_index=True)
                    trial_results["data"][f_type] = stitched_df
                else:
                    trial_results["data"][f_type] = None
                    
            # --- 2.5 UNIFY PHYSICS & WALKING DIRECTION ---
            stitched_grf = trial_results["data"].get("mot_grf")
            stitched_mrk = trial_results["data"].get("trc_marker")
            
            if stitched_grf is not None and stitched_mrk is not None:
                # A. Fix the orientation
                unified_mrk, unified_grf = self.unify_trial_physics(stitched_mrk, stitched_grf)
                
                # B. Find the active leg for this trial
                l_vy = next((c for c in unified_grf.columns if c.endswith('_l_vy')), None)
                r_vy = next((c for c in unified_grf.columns if c.endswith('_r_vy')), None)
                target_leg = None
                for leg_vy in [l_vy, r_vy]:
                    if leg_vy and unified_grf[leg_vy].abs().max() > 20.0:
                        target_leg = leg_vy
                        break
                
                # C. APPLY THE BIOMECHANICAL QUALITY GUARDRAIL
                if target_leg:
                    is_valid, error_msg = self.validate_grf_quality(unified_grf, target_leg)
                    
                    if not is_valid:
                        # Quarantine the trial and record the exact reason in the report
                        trial_results['error'] = f"Quality Check Failed: {error_msg}"
                        trial_results['normalized_data'] = None
                        processed_data.append(trial_results)
                        continue # Skip the rest of the pipeline for this broken trial
                
                # D. Save the clean, unified data back into the pipeline
                trial_results["data"]["trc_marker"] = unified_mrk
                trial_results["data"]["mot_grf"] = unified_grf

            # --- 3. EXTRACT GAIT EVENTS ---
            data = trial_results["data"]
            hs1_t, to_t, hs2_t = None, None, None
            leg_used = None
            
            if data.get('mot_grf') is not None and data.get('trc_marker') is not None:
                
                # Try Left Leg (Silently)
                leg_used = 'L'
                hs1_t, to_t, hs2_t = self.extract_gait_events(data['mot_grf'], data['trc_marker'], leg='l', silent=True)
                
                # Try Right Leg if Left fails
                if hs1_t is None or hs2_t is None:
                    leg_used = 'R'
                    hs1_t, to_t, hs2_t = self.extract_gait_events(data['mot_grf'], data['trc_marker'], leg='r', silent=False)
                    
            else:
                trial_results['error'] = "Missing GRF or TRC data needed for sync."
            
            # --- 4. TIME NORMALIZE & APPLY PIPELINE TO VALID CYCLES ---
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

                            # CATCH THE FAIL STATE HERE
                            if df_final is None:
                                trial_results['error'] = "Missing ASI markers. Aborted Pelvis-Centered AP shift."
                                normalized_trial = None # Destroy the trial data completely
                                break # Stop processing any other data types for this trial

                            normalized_trial[f_type] = df_final
                        else:
                            normalized_trial[f_type] = renamed_df

                # --- 5. APPLY ANTHROPOMETRIC SCALING (Only if trial wasn't aborted) ---
                if normalized_trial is not None:
                    scaled_trial = self.apply_anthropometric_normalization(normalized_trial, mass, height)
                    
                    # Assume valid until proven otherwise
                    is_valid_id = True 
                    id_error_msg = ""

                    # Run the ID filter if ID data exists
                    if scaled_trial is not None and scaled_trial.get('sto_id') is not None:
                        is_valid_id, id_error_msg = self.validate_id_quality(scaled_trial['sto_id'], threshold=0.15)
                    
                    # Quarantine or Save
                    if not is_valid_id:
                        trial_results['error'] = f"ID Quality Check Failed: {id_error_msg}"
                        trial_results['normalized_data'] = None
                    else:
                        trial_results['normalized_data'] = scaled_trial
                        trial_results['gait_events'] = {'leg_used': leg_used, 'HS1': hs1_t, 'TO': to_t, 'HS2': hs2_t}
                        trial_results['anthropometrics'] = {'mass_kg': mass, 'height_m': height}
                else:
                    trial_results['normalized_data'] = None

            else:
                if 'error' not in trial_results:
                    trial_results['error'] = "Incomplete gait cycle or no heel strike found."
                trial_results['normalized_data'] = None

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
                                if col.upper() in ['TIME', 'FRAME', 'FRAME#', 'TIME_PERCENT']:
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

## ------ EXPORTING RESULTS V2 (4 PRE-BAKED FILES) ------ #
    def export_database_results_v2(self):
        """
        Runs the full processing pipeline for all participants.
        Generates a Summary Report and 4 separate ML Matrix files based on input cases:
        1. Markers and GRF
        2. IK only
        3. ID only
        4. IK and ID
        """
        summary_rows = []
        ml_rows = []
        
        print(f"\n--- Starting V2 Database Export for {self.db_name} ---")
        
        for participant in self.participants:
            print(f"Processing participant: {participant}...")
            
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
                
                # --- 2. BUILD MASTER FLATTENED ROW ---
                if is_success:
                    flat_row = {
                        'DB': self.db_name,
                        'Participant': participant,
                        'Trial': t_name,
                        'Leg_Used': leg,
                        'Mass_kg': trial['anthropometrics']['mass_kg'],
                        'Height_m': trial['anthropometrics']['height_m']
                    }
                    
                    norm_data = trial['normalized_data']
                    prefix_map = {'trc_marker': 'TRC', 'mot_grf': 'GRF', 'mot_ik': 'IK', 'sto_id': 'ID'}
                    
                    for f_type, prefix in prefix_map.items():
                        if f_type in norm_data and norm_data[f_type] is not None:
                            df = norm_data[f_type]
                            for col in df.columns:
                                if col.upper() in ['TIME', 'FRAME', 'FRAME#', 'TIME_PERCENT']:
                                    continue
                                vals = df[col].values
                                for i, val in enumerate(vals):
                                    flat_row[f"{prefix}_{col}_{i}"] = val
                                    
                    ml_rows.append(flat_row)

        # --- 3. SLICE AND SAVE THE CSV FILES ---
        out_dir = self.root_dir / "results" / self.db_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Summary Report
        df_summary = pd.DataFrame(summary_rows)
        summary_path = out_dir / f"{self.db_name}_Processing_Report.csv"
        df_summary.to_csv(summary_path, index=False)
        print(f"\n[Export Complete] Summary Report saved.")
        
        if ml_rows:
            # Create a temporary master dataframe in memory
            df_master = pd.DataFrame(ml_rows)
            
            # The essential metadata columns that must exist in every file
            base_cols = ['DB', 'Participant', 'Trial', 'Leg_Used', 'Mass_kg', 'Height_m']
            
            # Helper function to grab the base columns PLUS any column that starts with the target prefixes
            def get_target_columns(prefixes):
                cols = base_cols.copy()
                for col in df_master.columns:
                    if any(col.startswith(p) for p in prefixes):
                        cols.append(col)
                return cols

            # --- Case 1: TRC and GRF ---
            cols_c1 = get_target_columns(['TRC_', 'GRF_'])
            df_c1 = df_master[cols_c1]
            df_c1.to_csv(out_dir / f"{self.db_name}_Case1_Markers_GRF.csv", index=False)
            print(f"  -> Case 1 saved: {df_c1.shape[1]} features (Markers + GRF)")

            # --- Case 2: IK Only ---
            cols_c2 = get_target_columns(['IK_'])
            df_c2 = df_master[cols_c2]
            df_c2.to_csv(out_dir / f"{self.db_name}_Case2_IK.csv", index=False)
            print(f"  -> Case 2 saved: {df_c2.shape[1]} features (IK Only)")

            # --- Case 3: ID Only ---
            cols_c3 = get_target_columns(['ID_'])
            df_c3 = df_master[cols_c3]
            df_c3.to_csv(out_dir / f"{self.db_name}_Case3_ID.csv", index=False)
            print(f"  -> Case 3 saved: {df_c3.shape[1]} features (ID Only)")

            # --- Case 4: IK and ID ---
            cols_c4 = get_target_columns(['IK_', 'ID_'])
            df_c4 = df_master[cols_c4]
            df_c4.to_csv(out_dir / f"{self.db_name}_Case4_IK_ID.csv", index=False)
            print(f"  -> Case 4 saved: {df_c4.shape[1]} features (IK + ID)")
            
        else:
            print("\n[Warning] No successful trials were found. ML Matrices were not created.")
            