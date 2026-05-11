# TeamProject
Exploring Dataset Bias in Multi-Source Gait Biomechanics using IK and ID

## Find suitable databases
...
## Pre-processing of data (marker retargeting)
...
## Processing of data (AddBiomechanics / OpenSim)
...
## Post-processing of data
**Exclude data**
1. Discard trials with clear measurement errors through visual inspection.
   <ins>How?</ins> Plot all the vertical GRF stored in the <kbd>C3D</kbd> file, identify the outliers, and report them in _database_inventory_.
    The typical vertical GRF during walking has an **M-shape** pattern.
2. (If speed is given) Identify the trials with speeds lower than 0.4 m/s and report them in _database_inventory_
3. Investigate the average step width in human walking, identify, in <kbd>DB2</kbd>, the trials where the step width is not close to this value, and report them in _database_inventory_.
-- In _database_inventory_ each DBx tab has specific columns to report the different cases.

**Select data of interest (Kinematics & Kinetics)**
1. MARKER TRAJECTORIES, <kbd>C3D</kbd>: Identify the markers that appear in all DBs by inspecting the _translation_table_. The marker labels to keep are those in the first column (Rajagopal model). Remember that the marker labels have already been retargeted. 
2. GROUND REACTON FORCES, <kbd>C3D</kbd>: Keep GRF (F) and Moments (M) in x, y, and z directions for the two first available force plates.
   **Ask about Center of Pressure (CoP), because not all the databases have it in the c3d**
3. JOINT ANGLES (IK), <kbd>MOT</kbd>: Keep hip_flexion, knee_angle and ankle_angle for both legs.
4. JOINT MOMENTS (ID), <kbd>STO</kbd>: Keep hip_flexion, knee_angle and ankle_angle for both legs.

After identifying all the labels corresponding to the data of interest for the 4 data types, find a way to extract them form the complete dataset and keep them separated. We will later investigate the order and structure in which they should be stored for the following steps.

**Filter the data**
Bidirectional second-order Butterworth filter with 10 Hz cut-off frequency
??  **Filter all the data de same**

**Cut trajectories to one gait cycle**
DONE...

**Resample every stride to 101 data points**
DONE...

**Global Coordinate System**
To remove any dependency on the laboratory setup and allow for comparison between laboratories.
1. Use the ANKL marker position at the first HS as the origin of all global coordinate systems

**Make trajectories dimensionless**
To enable better comaprison between subjects and to remove anthropometric influences
1. Marker trajectories: normalize by dividing them by participant's _body height_
2. Forces (N): normalize the three component F<sub>x</sub>, F<sub>y</sub>, F<sub>z</sub> by dividing them by participant's body weight (mass (kg) * g(m/s<sup>2</sup>))
**Ask if it's necessary to do it only for GRF of for something else too (such as moments)**

<ins>How?</ins> Investigate if the <kbd>C3D</kbd> file stores this values to directly use them. If not, find a strategy to pull this info from the _database_inventory_ or create a new file that stores this info to be pulled in this step.

3. Angles: by min and max ranges of motion per participant **?? Ask if this approach is correct and necessary**
4. Moments: by (mass (kg) * g(m/s<sup>2</sup>) * height (m)) **?? Ask if this approach is correct and necessary**

**Relative Coordinate System**
To reduce the effect of different walking speeds.
1. Express all marker trajectories in anterior-posterior direction relative to the pelvis center (middle between left and right greater trochanter marker)

**Concatenate into one movement vector**
n marker trajectories, m force trajectories, nn joint angles, mm joint moments; times 101 data points. Arrange all trials in a sigle data matrix **M = [x<sub>1</sub>, ... x</sub>m</sub>]** where every row corresponded to a different trial, and the number of columns equaled the number of data points that described one trial. Every time point as a feature and every marker or force trajectory as a separace channel.

## Classification

## Interpretability
