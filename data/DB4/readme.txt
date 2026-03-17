Description of Han Dataset:

This page(https://csr.bu.edu/groundlink/) contains captured and measured data for:

- C3D: annotated full body MoCap with 96 retroreflective markers.

- FBX: skeleton reconstructed from 41 markers, processed with Qualisys Track Manager (QTM).

- SMPL: pose and shape parameters estimated using 26 markers with MoSh++ and SMPL-X model.

- Force: annotated and filtered force data (GRF and CoP) to distinguish left and right foot.

C3d folders have data of 7 subjects. Each subject different trial of different activities,
 we have to select only walking trials.

NOTE: Few trials are missing in subjects data(e.g. subject 3 do not have trial 2) and subject 7 do not have any walking trials.  

Reference:
- https://csr.bu.edu/groundlink/
- https://github.com/hanxingjian/GroundLink
- https://www.c3d.org/
- https://dl.acm.org/doi/epdf/10.1145/3610548.3618247