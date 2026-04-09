_, _FG, _G, SG (2minWlak, fast gait, gait and slow gait)


Description Grouvel Dataset:

For each participant, the entire data collection was performed in a single session which lasted approximately one hour.

Data are organized by participant folder (PXX_SYY, P: for Participant, S: for Session) and each folder contains two sub-folders:

- RAW_DATA
  - One .c3d file per trial recorded during the session
  - Eight .bin files corresponding to the IMUs data recorded during the session
  - One .txt file corresponding to the insole data recorded during the session

- SYNC_DATA
  - One .csv file per trial with all the optoelectronic, IMUs and insole synchronized data recorded during the session

C3D trial files are referenced in our dataset9 as PXX_SYY_[Trial type]_[Trial number].c3d, with the following correspondence:

- P: for Participant
- XX: participant number (e.g. 01)
- S: for Session
- YY: session number (e.g. 01)
- [Trial type]: task performed (https://www.nature.com/articles/s41597-023-02077-3/tables/7)
- [Trial number]: trial number (e.g. 01)

IMUs are referenced as PXX_SYY_ZZ_Inertial_sensor.bin, with:

- ZZ: sensor name including TR: torso/SA: pelvis/RT: right thigh/RS: right shank/RF: right foot/LT: left thigh/LS: left shank/LF: left foot

Insoles data are referenced as PXX_SYY_Sensor_insoles.txt.

Synchronized data are referenced with the same name as c3d files but with the file extension .csv.


Reference:
- https://www.nature.com/articles/s41597-023-02077-3
- https://www.nature.com/articles/s41597-023-02077-3/tables/1
- https://www.nature.com/articles/s41597-023-02077-3/tables/3
