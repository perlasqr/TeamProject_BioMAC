import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. CONFIGURATION ---
# Adjust this path if you moved your CSV file
file_path = r"C:\Users\perla\Documents\AT\TeamProject\GitHub\TeamProject\results\DB5\DB5_ML_Matrix.csv"
num_trials_to_plot = 1

# Define the range of trials you want to plot (e.g., 0 to 20, 20 to 40)
start_trial = 0
end_trial = 700

def plot_joint_angles(side):
    # --- 2. LOAD AND FILTER DATA ---
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Filter only trials where Left is the leading leg
    df_left_lead = df[df['Leg_Used'] == side]
    
    if df_left_lead.empty:
        print("No trials found where Leg_Used is 'L'.")
        return
        
    # Take the specific range of trials using iloc
    df_plot = df_left_lead.iloc[start_trial:end_trial]
    print(f"Plotting {len(df_plot)} trials (from index {start_trial} to {end_trial})...")

    if df_plot.empty:
        print("The selected range is empty. Check your start_trial and end_trial values.")
        return

    # --- 3. SETUP THE FIGURE ---
    # We will create 3 subplots stacked vertically (Hip, Knee, Ankle)
    joints = ['hip_flexion', 'knee_angle', 'ankle_angle']
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    time_percent = np.linspace(0, 100, 101) # X-axis (0% to 100% Gait Cycle)

    # --- 4. EXTRACT AND PLOT ---
    for index, row in df_plot.iterrows():
        trial_name = row['Trial']
        
        for i, joint in enumerate(joints):
            ax = axes[i]
            
            # Reconstruct the column names based on your ML Matrix flattening logic
            # 1 = Left (Lead), 2 = Right (Trail)
            left_cols = [f"IK_{joint}_1_{pt}" for pt in range(101)]
            right_cols = [f"IK_{joint}_2_{pt}" for pt in range(101)]
            
            # Ensure the columns actually exist in the CSV before plotting
            if all(c in df.columns for c in left_cols):
                left_data = row[left_cols].values
                right_data = row[right_cols].values
                
                # To keep the legend clean, we only attach labels to the very first trial in the loop
                is_first_plot = (index == df_plot.index[0])
                label_l = "Left Leg (Lead)" if is_first_plot else ""
                label_r = "Right Leg (Trail)" if is_first_plot else ""
                
                # Plot Left (Blue, Solid) and Right (Red, Dashed)
                ax.plot(time_percent, left_data, color='blue', alpha=0.4, label=label_l)
                ax.plot(time_percent, right_data, color='red', linestyle='--', alpha=0.4, label=label_r)
            else:
                if index == df_plot.index[0]:
                    print(f"Warning: Columns for {joint} missing. Check your CSV headers.")
                    
    # Optional: Add legends and show the plot
    for ax in axes:
        ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_joint_moments(side):
    print(f"\nLoading data from {file_path} for Joint Moments...")
    df = pd.read_csv(file_path)
    
    # Filter only trials where Left is the leading leg
    df_left_lead = df[df['Leg_Used'] == side]
    if df_left_lead.empty:
        print("No trials found where Leg_Used is 'L'.")
        return
        
    df_plot = df_left_lead.iloc[start_trial:end_trial]
    print(f"Plotting {len(df_plot)} trials (from index {start_trial} to {end_trial})...")

    if df_plot.empty:
        print("The selected range is empty. Check your start_trial and end_trial values.")
        return

    # Setup the 3 subplots for moments
    joints = ['hip_flexion', 'knee_angle', 'ankle_angle']
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    time_percent = np.linspace(0, 100, 101)

    for index, row in df_plot.iterrows():
        trial_name = row['Trial']
        for i, joint in enumerate(joints):
            ax = axes[i]
            
            # Reconstruct column names based on your ID naming convention
            left_cols = [f"ID_{joint}_1_moment_{pt}" for pt in range(101)]
            right_cols = [f"ID_{joint}_2_moment_{pt}" for pt in range(101)]
            
            if all(c in df.columns for c in left_cols):
                left_data = row[left_cols].values
                right_data = row[right_cols].values

                is_first_plot = (index == df_plot.index[0])
                label_l = "Left Leg (Lead)" if is_first_plot else ""
                label_r = "Right Leg (Trail)" if is_first_plot else ""
                
                # Plot Left (Blue, Solid) and Right (Red, Dashed)
                ax.plot(time_percent, left_data, color='blue', alpha=0.4, label=label_l)
                ax.plot(time_percent, right_data, color='red', linestyle='--', alpha=0.4, label=label_r)   
            else:
                # FIXED: The warning is now inside the 'else' block
                if index == df_plot.index[0]:
                    print(f"Warning: Columns for {joint} moment missing.")

    # FIXED: Added styling and labels to the moments plot
    titles = ['Hip Moment', 'Knee Moment', 'Ankle Moment']
    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=14, fontweight='bold')
        ax.set_ylabel('Moment (Nm/kg)', fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.axhline(0, color='black', linewidth=1, alpha=0.5)
        if i == 0: 
            ax.legend(loc='upper right', framealpha=1.0)

    axes[-1].set_xlabel('Gait Cycle (%)', fontsize=12)
    axes[-1].set_xlim(0, 100)
    plt.tight_layout()
    plt.show()

 
def plot_grf_components(side):
    print(f"\nLoading data from {file_path} for GRF...")
    df = pd.read_csv(file_path)
    
    df_left_lead = df[df['Leg_Used'] == side]
    if df_left_lead.empty:
        print("No trials found where Leg_Used is 'L'.")
        return
        
    df_plot = df_left_lead.iloc[start_trial:end_trial]
    print(f"Plotting {len(df_plot)} trials (from index {start_trial} to {end_trial})...")

    if df_plot.empty:
        print("The selected range is empty. Check your start_trial and end_trial values.")
        return

    # We use the raw axis suffixes for the OpenSim ground_force columns
    force_axes = ['vx', 'vy', 'vz'] 
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    time_percent = np.linspace(0, 100, 101)

    for index, row in df_plot.iterrows():
        trial_name = row['Trial']
        for i, axis in enumerate(force_axes):
            ax = axes[i]
            
            # Dynamic column search to find the correct GRF prefix
            base_col = next((c for c in df.columns if f"GRF_" in c and f"{axis}1_0" in c), None)
            
            if base_col:
                # Extract the base prefix dynamically
                prefix = base_col.split(f"_{axis}1_0")[0]
                
                left_cols = [f"{prefix}_{axis}1_{pt}" for pt in range(101)]
                right_cols = [f"{prefix}_{axis}2_{pt}" for pt in range(101)]
                
                if all(c in df.columns for c in left_cols):
                    left_data = row[left_cols].values
                    right_data = row[right_cols].values
                    
                    is_first_plot = (index == df_plot.index[0])
                    label_l = "Left Leg (Lead)" if is_first_plot else ""
                    label_r = "Right Leg (Trail)" if is_first_plot else ""
                    
                    ax.plot(time_percent, left_data, color='blue', alpha=0.4, label=label_l)
                    ax.plot(time_percent, right_data, color='red', linestyle='--', alpha=0.4, label=label_r)
            else:
                # Adjusted to be a clean 'else' matching the moments logic
                if index == df_plot.index[0]:
                     print(f"Warning: Columns for GRF {axis} axis missing. Check your CSV headers.")

    # GRF Styling
    titles = ['Anterior-Posterior Force (vx)', 'Vertical Force (vy)', 'Medio-Lateral Force (vz)']
    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=14, fontweight='bold')
        ax.set_ylabel('Force (BW)', fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.axhline(0, color='black', linewidth=1, alpha=0.5)
        if i == 0: 
            ax.legend(loc='upper right', framealpha=1.0)

    axes[-1].set_xlabel('Gait Cycle (%)', fontsize=12)
    axes[-1].set_xlim(0, 100)
    plt.tight_layout()
    plt.show()

# Call the function to execute it
side_leg = 'L'
plot_joint_angles(side_leg)
plot_joint_moments(side_leg)
plot_grf_components(side_leg)





'''
def plot_joint_angles_all():
    # --- 2. LOAD AND FILTER DATA ---x
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Filter only trials where Left is the leading leg
    df_left_lead = df[df['Leg_Used'] == 'L']
    
    if df_left_lead.empty:
        print("No trials found where Leg_Used is 'L'.")
        return
        
    # Take the first N trials
    df_plot = df_left_lead.head(num_trials_to_plot)
    print(f"Plotting {len(df_plot)} trials...")

    # --- 3. SETUP THE FIGURE ---
    # We will create 3 subplots stacked vertically (Hip, Knee, Ankle)
    joints = ['hip_flexion', 'knee_angle', 'ankle_angle']
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    time_percent = np.linspace(0, 100, 101) # X-axis (0% to 100% Gait Cycle)

    # --- 4. EXTRACT AND PLOT ---
    for index, row in df_plot.iterrows():
        trial_name = row['Trial']
        
        for i, joint in enumerate(joints):
            ax = axes[i]
            
            # Reconstruct the column names based on your ML Matrix flattening logic
            # 1 = Left (Lead), 2 = Right (Trail)
            left_cols = [f"IK_{joint}_1_{pt}" for pt in range(101)]
            right_cols = [f"IK_{joint}_2_{pt}" for pt in range(101)]
            
            # Ensure the columns actually exist in the CSV before plotting
            if all(c in df.columns for c in left_cols):
                left_data = row[left_cols].values
                right_data = row[right_cols].values
                
                # To keep the legend clean, we only attach labels to the very first trial in the loop
                label_l = "Left Leg (Lead)" if index == df_plot.index[0] else ""
                label_r = "Right Leg (Trail)" if index == df_plot.index[0] else ""
                
                # Plot Left (Blue, Solid) and Right (Red, Dashed)
                ax.plot(time_percent, left_data, color='blue', alpha=0.4, label=label_l)
                ax.plot(time_percent, right_data, color='red', linestyle='--', alpha=0.4, label=label_r)
            else:
                if index == df_plot.index[0]:
                    print(f"Warning: Columns for {joint} missing. Check your CSV headers.")

    # --- 5. FORMATTING THE GRAPHS ---
    titles = ['Hip Flexion Angle', 'Knee Angle', 'Ankle Angle']
    
    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=14, fontweight='bold')
        ax.set_ylabel('Angle (degrees)', fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.7)
        
        # Draw a horizontal line at 0 degrees for reference
        ax.axhline(0, color='black', linewidth=1, alpha=0.5)
        
        # Only put the legend on the top plot so it isn't redundant
        if i == 0: 
            ax.legend(loc='upper right', framealpha=1.0)

    # Label the shared X-axis on the bottom plot only
    axes[-1].set_xlabel('Gait Cycle (%)', fontsize=12)
    axes[-1].set_xlim(0, 100)
    
    plt.tight_layout()
    plt.show()
'''
#if __name__ == "__main__":
    # plot_joint_angles()
    # plot_joint_moments()
    # plot_grf_components()

