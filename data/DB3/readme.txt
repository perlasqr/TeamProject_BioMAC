Description of Vielemeyer Dataset:

Participants walked at self-selected walking speed. Incline and decline trials were recorded alternately. 
Subjects reached the force plates after approximately three steps. Twelve valid trials were recorded for each condition.
A trial was considered valid when the left and right foot each hit a force plate without overstepping.

The repository contains three folders, corresponding to three levels of data:
 (1) Raw data in c3d format, 
 (2) exported raw data in txt format, 
 (3) calculated data, including biomechanical variables packed in npz format

All three main folders contain subfolders named Ref_01, Ref_02, etc.
Then again, these subfolders are separated into folders named after the six experimental setups:
level_up, level_down, ramp_75_up, ramp_75_down, ramp_10_up, ramp_10_down

Here, “ramp_75” denotes walking over a ramp of 7.5°, and “ramp_10” denotes walking over a ramp of 10°.
For example, the c3d file for participant 1, level_up, first trial is called Ref01_sss_le_shoes04.c3d.

Reference: 
- https://www.nature.com/articles/s41597-025-06535-y
- https://www.nature.com/articles/s41597-025-06535-y/tables/1
- https://www.nature.com/articles/s41597-025-06535-y/tables/2
- https://www.c3d.org/
- https://github.com/jvielemeyer/human-ramp-walking