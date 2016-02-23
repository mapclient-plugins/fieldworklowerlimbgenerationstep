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