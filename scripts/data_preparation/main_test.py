from data_processor import DataProcessor
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

def plot_reoriented_marker(trc_df, marker_name="LHEE"):
    """Plots the X, Y, Z components of a specific marker."""
    time_vec = trc_df['Time']
    
    plt.figure(figsize=(10, 5))
    # Standard OpenSim colors: X=Red (Forward), Y=Green (Up), Z=Blue (Right)
    plt.plot(time_vec, trc_df[f'{marker_name}_X'], label=f'{marker_name}_X (Forward)', color='red')
    plt.plot(time_vec, trc_df[f'{marker_name}_Y'], label=f'{marker_name}_Y (Up)', color='green')
    plt.plot(time_vec, trc_df[f'{marker_name}_Z'], label=f'{marker_name}_Z (Right)', color='blue')
    
    plt.title(f"Reoriented Marker: {marker_name} (OpenSim Standard)")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)") # Adjust to mm if necessary
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

def plot_reoriented_force(mot_df, force_base="ground_force_calcn_r_v"):
    """Plots the X, Y, Z components of a specific force vector."""
    # MOT files usually use lowercase 'time'
    time_col = 'time' if 'time' in mot_df.columns else mot_df.columns[0]
    time_vec = mot_df[time_col]
    
    plt.figure(figsize=(10, 5))
    plt.plot(time_vec, mot_df[f'{force_base}x'], label=f'Force X (Forward)', color='red')
    plt.plot(time_vec, mot_df[f'{force_base}y'], label=f'Force Y (Up)', color='green')
    plt.plot(time_vec, mot_df[f'{force_base}z'], label=f'Force Z (Right)', color='blue')
    
    plt.title(f"Reoriented GRF: {force_base} (OpenSim Standard)")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    


## ------- MAIN ------- ##

root = r"C:\Users\perla\Documents\AT\TeamProject\GitHub\TeamProject" 
db_name = "DB1"

processor = DataProcessor(root, db_name)
processor = DataProcessor(root, db_name)

# 2. Pick a participant (e.g., the first one found)
if not processor.participants:
    print("No participants found.")
else:
    test_p = processor.participants[0]
    print(f"\n--- Testing Full Pipeline for Participant: {test_p} ---")
    
    # 3. Process all trials for this participant
    # This will now return a list of trials, each with a dict of DataFrames
    all_trial_results = processor.process_participant_files(test_p)
    
    # 4. Verify the output
    print(f"Processed {len(all_trial_results)} trials.")
    
    for trial in all_trial_results:
        print(f"\nTrial Name: {trial['trial_name']}")
        for f_type, df in trial['data'].items():
            if df is not None:
                print(f"  - {f_type}: Shape {df.shape} | Columns: {list(df.columns[:3])}...")
            else:
                print(f"  - {f_type}: [FAILED/MISSING]")


for trial in all_trial_results:
    trial_name = trial['trial_name']
    data = trial['data']

    hs1_t, to_t, hs2_t = None, None, None
    leg_used = None
    
    if data.get('mot_grf') is not None and data.get('trc_marker') is not None:
        
        leg_used = 'L'
        hs1_t, to_t, hs2_t = processor.extract_gait_events(data['mot_grf'], data['trc_marker'], leg='l')
        
        if hs1_t is None or hs2_t is None:
            leg_used = 'R'
            hs1_t, to_t, hs2_t = processor.extract_gait_events(data['mot_grf'], data['trc_marker'], leg='r')
            
        if hs1_t is not None and hs2_t is not None:
            print(f"\n[{trial_name}] Valid Cycle ({leg_used}): HS1={hs1_t:.2f}s, HS2={hs2_t:.2f}s")
            
            normalized_trial = {}
            for f_type, df in data.items():
                if df is not None:
                    # 1. Time normalize
                    t_col = 'Time' if 'Time' in df.columns else 'time'
                    norm_df = processor.time_normalize(df, t_col, hs1_t, hs2_t)
                    
                    # 2. Convert L/R labels to 1/2
                    renamed_df = processor.convert_leg_labels(norm_df, f_type, leg_used)
                    
                    # 3. Apply Global Coordinate Shift (ONLY to TRC markers)
                    if f_type == 'trc_marker':
                        # Save the 'before' state for our printout
                        df_before = renamed_df.copy()
                        
                        # Apply your co-worker's method
                        final_df = processor.define_global_coordinate_system(renamed_df)
                        normalized_trial[f_type] = final_df
                        
                        # --- Coordinate Shift Test Printout ---
                        print(f"  --- Coordinate Shift Test ---")
                        # Check frame 0 (start of cycle) and frame 50 (mid cycle)
                        for frame_idx in [0, 50]:
                            print(f"  Gait Cycle: {frame_idx}%")
                            
                            # Print ANKL_1 (Should become exactly 0.0000)
                            print(f"    BEFORE ANKL_1: X={df_before.loc[frame_idx, 'ANKL_X1']:.4f}, Y={df_before.loc[frame_idx, 'ANKL_Y1']:.4f}, Z={df_before.loc[frame_idx, 'ANKL_Z1']:.4f}")
                            print(f"    AFTER  ANKL_1: X={final_df.loc[frame_idx, 'ANKL_X1']:.4f}, Y={final_df.loc[frame_idx, 'ANKL_Y1']:.4f}, Z={final_df.loc[frame_idx, 'ANKL_Z1']:.4f}")
                            
                            # Print another marker like Shoulder to see relative distance
                            if 'SHO_X1' in final_df.columns:
                                print(f"    BEFORE SHO_1 : X={df_before.loc[frame_idx, 'SHO_X1']:.4f}, Y={df_before.loc[frame_idx, 'SHO_Y1']:.4f}, Z={df_before.loc[frame_idx, 'SHO_Z1']:.4f}")
                                print(f"    AFTER  SHO_1 : X={final_df.loc[frame_idx, 'SHO_X1']:.4f}, Y={final_df.loc[frame_idx, 'SHO_Y1']:.4f}, Z={final_df.loc[frame_idx, 'SHO_Z1']:.4f}")
                            print("-" * 30)
                    else:
                        normalized_trial[f_type] = renamed_df
                        
            trial['normalized_data'] = normalized_trial
            
            # Stop after the first successful trial to easily read the output
            break 
            
        else:
            print(f"[{trial_name}] FAILED: Could not extract full gait cycle.")
    else:
        print(f"[{trial_name}] Skipped: Missing GRF or Marker data.")

# # 3. Process if successful
#         if hs1_t is not None and hs2_t is not None:
#             print(f"[{trial_name}] Valid Cycle ({leg_used}): HS1={hs1_t:.2f}s, HS2={hs2_t:.2f}s")
            
#             normalized_trial = {}
#             for f_type, df in data.items():
#                 if df is not None:
#                     # Time normalize first
#                     t_col = 'Time' if 'Time' in df.columns else 'time'
#                     norm_df = processor.time_normalize(df, t_col, hs1_t, hs2_t)
                    
#                     # Convert the L/R labels to Leading(1) / Trailing(2)
#                     renamed_df = processor.convert_leg_labels(norm_df, f_type, leg_used)
                    
#                     normalized_trial[f_type] = renamed_df
                    
#             trial['normalized_data'] = normalized_trial


"""# Check for duplicate column names
df = all_trial_results[0]['data']['trc_marker']
print(df.columns.duplicated().any()) # If True, you have duplicate headers!
print(df.columns.tolist())            # See the actual names
"""

""" # TEST REORIENTATION C.S.
if processor.participants:
    test_p = processor.participants[23]
    print(f"\n--- Testing participant: {test_p} ---")
    
    # 2. Get files for this participant
    files = processor.get_trial_file_map(test_p)
    
    # 3. Load and Reorient TRC (Marker Data)
    if files['trc_marker']:
        trc_path = files['trc_marker'][1]
        raw_trc = processor.load_data(trc_path, file_type='trc_marker')
        
        # Apply reorientation
        oriented_trc = processor.reorient_coordinates(raw_trc, 'trc_marker')
        
        # Plot LHEE
        plot_reoriented_marker(oriented_trc, marker_name="LHEE")
        
    # 4. Load and Reorient MOT (GRF Data)
    if files['mot_grf']:
        mot_path = files['mot_grf'][0]
        raw_mot = processor.load_data(mot_path, file_type='mot_grf')
        
        # Apply reorientation
        oriented_mot = processor.reorient_coordinates(raw_mot, 'mot_grf')
        
        # Plot right calcaneus ground reaction force
        plot_reoriented_force(oriented_mot, force_base="ground_force_calcn_l_v")

plt.show() """