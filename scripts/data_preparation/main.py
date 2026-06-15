from data_processor import DataProcessor
import matplotlib.pyplot as plt


## ------ FUNCTIONS ------ ##
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

if processor.participants:
    test_p = processor.participants[23]
    print(f"\n--- Testing participant: {test_p} ---")
    
    # 2. Get files for this participant
    files = processor.get_file_paths(test_p)
    
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

plt.show()