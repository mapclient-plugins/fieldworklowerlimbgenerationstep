fieldworklowerlimbgenerationstep
================================
MAP Client plugin for generating left or right lower limb geometry.

The lower limb model contains the pelvis, femur, patella, tibia, and fibula.
Either the left or the right side can be fitted to mocap landmarks using
a statistical shape model, uniform isotropic scaling, or per-bone isotropic 
scaling.

Requires
--------
GIAS: https://bitbucket.org/jangle/gias,
fieldwork: https://bitbucket.org/jangle/fieldwork,
mappluginutils: https://bitbucket.org/jangle/mappluginutils

Inputs
------
landmarks : dict
    Dictionary of marker names : marker coordinates
principalcomponents : gias2.learning.PCA.PrincipalComponents instance
    The lowerlimb principalcomponents to use for the optimisation
fieldworkmodeldict : dict [optional, unused]
    Dictionary of model names : fieldwork models

Outputs
-------
fieldworkmodeldict : dict
    A dictionary of customised fieldwork models of lower limb bones.
    Dictionary keys are: "pelvis", "pelvis flat", 'hemipelvis-left",
    "hemipelvis-right", "sacrum", "femur", "tibiafibula", "tibia",
    "fibula", "patella".

Supported Landmarks
-------------------
pelvis-LASIS : pelvis left anterior superior iliac spine
pelvis-LPSIS : pelvis left posterior superior iliac spine
pelvis-LHJC : pelvis left hip joint centre
pelvis-LIS : pelvis left ischial spine 
pelvis-LIT : pelvis left ischial tuberosity
pelvis-LPS : pelvis left pubis symphysis
pelvis-RASIS : pelvis right anterior superior iliac spine
pelvis-RPSIS : pelvis right posterior superior iliac spine
pelvis-RHJC : pelvis right hip joint centre
pelvis-RIS : pelvis right ischial spine 
pelvis-RIT : pelvis right ischial tuberosity
pelvis-RPS : pelvis right pubis symphysis
pelvis-Sacral : pelvis sacral
femur-GT : femur greater trochanter
femur-HC : femur head centre
femur-LEC : femur lateral epicondyle
femur-MEC : femur medial epicondyle
femur-kneecentre : femur knee centre
tibiafibula-LC : tibia-fibula tibia plateau most lateral point
tibiafibula-MC : tibia-fibula tibia plateau most medial point
tibiafibula-LM : tibia-fibula lateral malleolus
tibiafibula-MM : tibia-fibula medial malleolus
tibiafibula-TT : tibia-fibula tibial tuberosity