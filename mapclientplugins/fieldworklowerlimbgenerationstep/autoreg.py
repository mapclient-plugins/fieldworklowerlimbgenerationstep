"""
Auto lower limb registration
"""

import numpy as np
import copy
import shelve
import cPickle
import time

from fieldwork.field import geometric_field
from gias.musculoskeletal import mocap_landmark_preprocess
from gias.common import fieldvi, math
from gias.learning import PCA

import trcdata
import bone_models
import model_core
import lowerlimbatlasfit

class LLReg(object):

    def __init__(self):
        self._LL = bone_models.LowerLimbLeftAtlas('lower_limb_left')

    def load_model(self):
        # load gfs

        # load pca

        pass

    def set_model(self, gfs, pcs):
        # set gfs

        # set pcs

        pass

    def set_manual_params(self, x):
        pass

    def uniform_scale_reg(self, x0=None):
        pass


    def per_bone_scale_reg(self, x0=None):
        pass

    def pca_reg(self, x0=None):
