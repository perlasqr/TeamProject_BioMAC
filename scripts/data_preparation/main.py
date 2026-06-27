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
db_name = "DB5"

processor = DataProcessor(root, db_name)

# Trigger the entire database pipeline and export
if processor.participants:
    processor.export_database_results_v2()
else:
    print("No participants to process.")



### ------- CODE TO TEST METHODS ------- ####    IGNORE :)

# # # if not processor.participants:
# # #     print("No participants found.")
# # # else:
# # #     test_p = processor.participants[0]
# # #     print(f"\n--- Running Full Pipeline for Participant: {test_p} ---")
    
# # #     # 1. Run the newly integrated method
# # #     all_trial_results = processor.process_participant_files(test_p)
    
# # #     # 2. Verify the Math
# # #     print(f"\n--- Anthropometric Normalization Sanity Checks ---")
# # #     for trial in all_trial_results:
# # #         # Find the first trial that successfully processed without errors
# # #         if trial.get('normalized_data') is not None:
# # #             print(f"\nTrial Verified: {trial['trial_name']}")
            
# # #             # A. Verify Excel Metadata extraction
# # #             mass = trial['anthropometrics']['mass_kg']
# # #             height = trial['anthropometrics']['height_m']
# # #             calculated_bw = mass * 9.81
# # #             print(f"  [Excel Check] Mass: {mass} kg | Height: {height} m | Calculated BW: {calculated_bw:.2f} N")
            
# # #             norm_data = trial['normalized_data']
            
# # #             # B. Sanity Check Forces (Peak should be roughly 1.0 to 1.3 BW for walking)
# # #             if 'mot_grf' in norm_data:
# # #                 df_grf = norm_data['mot_grf']
# # #                 vy_col = next((c for c in df_grf.columns if 'vy1' in c.lower()), None)
# # #                 if vy_col:
# # #                     peak_force = df_grf[vy_col].max()
# # #                     print(f"  [Force Check] Peak Vertical GRF ({vy_col}): {peak_force:.4f} BW")
# # #                     if 0.8 <= peak_force <= 1.5:
# # #                         print("    -> SUCCESS: Value is in the expected walking range.")
# # #                     else:
# # #                         print("    -> WARNING: Force scaling magnitude looks suspicious!")
            
# # #             # C. Sanity Check Markers (Should be small decimal fractions of height)
# # #             if 'trc_marker' in norm_data:
# # #                 df_trc = norm_data['trc_marker']
# # #                 print(f"  [Marker Check] Sample Dimensionless Positions (Frame 0):")
# # #                 sample_cols = [c for c in df_trc.columns if c.upper() not in ['TIME', 'FRAME', 'FRAME#']][:3]
# # #                 for col in sample_cols:
# # #                     val = df_trc.loc[0, col]
# # #                     print(f"    {col}: {val:.4f} (Fraction of Height)")
                    
# # #             # D. Sanity Check Inverse Dynamics Moments
# # #             if 'sto_id' in norm_data:
# # #                 df_id = norm_data['sto_id']
# # #                 mom_col = next((c for c in df_id.columns if 'moment' in c.lower()), None)
# # #                 if mom_col:
# # #                     sample_mom = df_id.loc[0, mom_col]
# # #                     print(f"  [ID Check] Sample Moment {mom_col} (Frame 0): {sample_mom:.4f} (%BW*ht)")

# # #             print(f"\n==================================================")
# # #             break # Stop after checking the first successful trial to keep console clean




''' TEST RELABEL & COORDINATE SYSTEMS '''
# # 2. Pick a participant (e.g., the first one found)
# if not processor.participants:
#     print("No participants found.")
# else:
#     test_p = processor.participants[0]
#     print(f"\n--- Testing Full Pipeline for Participant: {test_p} ---")
    
#     # 3. Process all trials for this participant
#     # This will now return a list of trials, each with a dict of DataFrames
#     all_trial_results = processor.process_participant_files(test_p)
    
#     # 4. Verify the output
#     print(f"Processed {len(all_trial_results)} trials.")
    
#     for trial in all_trial_results:
#         print(f"\nTrial Name: {trial['trial_name']}")
#         for f_type, df in trial['data'].items():
#             if df is not None:
#                 print(f"  - {f_type}: Shape {df.shape} | Columns: {list(df.columns[:3])}...")
#             else:
#                 print(f"  - {f_type}: [FAILED/MISSING]")


# for trial in all_trial_results:
#     trial_name = trial['trial_name']
#     data = trial['data']

#     hs1_t, to_t, hs2_t = None, None, None
#     leg_used = None
    
#     if data.get('mot_grf') is not None and data.get('trc_marker') is not None:
        
#         leg_used = 'L'
#         hs1_t, to_t, hs2_t = processor.extract_gait_events(data['mot_grf'], data['trc_marker'], leg='l')
        
#         if hs1_t is None or hs2_t is None:
#             leg_used = 'R'
#             hs1_t, to_t, hs2_t = processor.extract_gait_events(data['mot_grf'], data['trc_marker'], leg='r')
            
#         if hs1_t is not None and hs2_t is not None:
#             print(f"\n[{trial_name}] Valid Cycle ({leg_used}): HS1={hs1_t:.2f}s, HS2={hs2_t:.2f}s")
            
#             normalized_trial = {}
#             for f_type, df in data.items():
#                 if df is not None:
#                     # 1. Time normalize
#                     t_col = 'Time' if 'Time' in df.columns else 'time'
#                     norm_df = processor.time_normalize(df, t_col, hs1_t, hs2_t)
                    
#                     # 2. Convert L/R labels to 1/2
#                     renamed_df = processor.convert_leg_labels(norm_df, f_type, leg_used)
                    
#                     # 3. Apply Coordinate Shifts (ONLY to TRC markers)
#                     if f_type == 'trc_marker':
#                         df_before = renamed_df.copy()
                        
#                         # 1st Shift: Global (Y, Z axes using Ankle)
#                         df_global = processor.define_global_coordinate_system(df_before)
                        
#                         # 2nd Shift: Relative AP (X axis using Pelvis with Auto-Direction)
#                         df_final = processor.define_pelvis_centered_ap_coordinate_system(df_global)
                        
#                         normalized_trial[f_type] = df_final
                        
#                         # --- Coordinate Shift Harmony Test ---
#                         print(f"  --- Coordinate Shift Test (Frame 0 - Heel Strike) ---")
                        
#                         def get_pelvis_x(df):
#                             # Looking at index 0 instead of 50
#                             return (df.loc[0, 'ASI_X1'] + df.loc[0, 'ASI_X2']) / 2
                            
#                         print(f"  [1] BEFORE SHIFTS:")
#                         print(f"      Pelvis Center X : {get_pelvis_x(df_before):.4f}")
#                         print(f"      Ankle 1 (X,Y,Z) : {df_before.loc[0, 'ANKL_X1']:.4f}, {df_before.loc[0, 'ANKL_Y1']:.4f}, {df_before.loc[0, 'ANKL_Z1']:.4f}")
                        
#                         print(f"\n  [2] AFTER GLOBAL (Y/Z) SHIFT:")
#                         print(f"      Pelvis Center X : {get_pelvis_x(df_global):.4f}  <-- Unchanged")
#                         print(f"      Ankle 1 (X,Y,Z) : {df_global.loc[0, 'ANKL_X1']:.4f}, {df_global.loc[0, 'ANKL_Y1']:.4f}, {df_global.loc[0, 'ANKL_Z1']:.4f}  <-- Y/Z are now 0.00")
                        
#                         print(f"\n  [3] AFTER PELVIS (X) SHIFT:")
#                         print(f"      Pelvis Center X : {get_pelvis_x(df_final):.4f}  <-- X is now 0.00")
#                         print(f"      Ankle 1 (X,Y,Z) : {df_final.loc[0, 'ANKL_X1']:.4f}, {df_final.loc[0, 'ANKL_Y1']:.4f}, {df_final.loc[0, 'ANKL_Z1']:.4f}  <-- X should be POSITIVE, Y/Z remain 0.00")
#                         print("-" * 50)
#                     else:
#                         normalized_trial[f_type] = renamed_df
                        
#             trial['normalized_data'] = normalized_trial
            
#             # Stop after the first successful trial to easily read the output
#             break 
            
#         else:
#             print(f"[{trial_name}] FAILED: Could not extract full gait cycle.")
#     else:
#         print(f"[{trial_name}] Skipped: Missing GRF or Marker data.")


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