Naming convention: DBx_Pxx_Txx_n-m_yyy        -> use n-m for decimal speeds e.g., 2.5 becomes 2-5 but in DB2 speeds are always "preferred walking speed"
                                             -> yyy is "step width at increased/reduced by 6.5%/13%/25% of leg length" which will be showed as for example "reduced by 6.5%" = -65 (see ref 3)

DBx - Database

Pxx - Participant Number

Txx - Trial Number

DB2 description:

Each file format folder has 13 subfolders that represent 13 participants, and trial files in 13 subfolders are referenced in our datasets as Pxx_CV_TT.C3D (TRC/MOT/CSV) and static files as Pxx_Cal_TT. C3D (TRC/MOT/CSV),

- Pxx: identification of the participant number

- C: step width conditions, i.e., −130, −65, p (for preffered), 65, 130, or 250

- V: locomotion condition, i.e. w, r1, or r2

- TT: trial number, i.e. 01 to 10

- SP: static pose

Reference: 
 1- https://www.nature.com/articles/s41597-025-05113-6
 2- https://www.nature.com/articles/s41597-025-05113-6/tables/2
 3- https://www.nature.com/articles/s41597-025-05113-6/tables/5
 4- https://www.c3d.org/
