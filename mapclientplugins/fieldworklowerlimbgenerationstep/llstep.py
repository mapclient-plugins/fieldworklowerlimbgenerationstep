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

class LLTransformData(object):

    def __init__(self):
        self.pelvisRigid = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.hipRot = np.array([0.0, 0.0, 0.0])
        self.kneeRot = np.array([0.0, 0.0, 0.0])
        self.shapeModes = []
        self.shapeModeWeights = []
        self.uniformScaling = 1.0
        self.pelvisScaling = 1.0
        self.femurScaling = 1.0
        self.petallaScaling = 1.0
        self.tibfibScaling = 1.0

        self._shapeModelX = None
        self._uniformScalingX = None
        self._perBoneScalingX = None

    @property
    def shapeModelX(self):
        self._shapeModelX = np.hstack([
                                self.shapeModeWeights,
                                self.pelvisRigid,
                                self.hipRot,
                                self.kneeRot
                                ])
        return self._shapeModelX

    @shapeModelX.setter
    def shapeModelX(self, value):
        a = len(self.shapeModes)
        self._shapeModelX = value
        self.shapeModeWeights = value[:a]
        self.pelvisRigid = value[a:a+6]
        self.hipRot = value[a+6:a+9]
        self.kneeRot = value[a+9:a+12]

    @property
    def uniformScalingX(self):
        self._uniformScalingX = np.hstack([
                                self.uniformScaling,
                                self.pelvisRigid,
                                self.hipRot,
                                self.kneeRot
                                ])
        return self._uniformScalingX

    @uniformScalingX.setter
    def uniformScalingX(self, value):
        a = 1
        self._uniformScalingX = value
        self.shapeModeWeights = value[:a]
        self.pelvisRigid = value[a:a+6]
        self.hipRot = value[a+6:a+9]
        self.kneeRot = value[a+9:a+12]

    @property
    def perBoneScalingX(self):
        self._perBoneScalingX = np.hstack([
                                self.pelvisScaling,
                                self.femurScaling,
                                self.patellaScaling,
                                self.tibfibScaling,
                                self.pelvisRigid,
                                self.hipRot,
                                self.kneeRot
                                ])
        return self._perBoneScalingX

    @perBoneScalingX.setter
    def perBoneScalingX(self, value):
        a = 4
        self._perBoneScalingX = value
        self.pelvisScaling = value[0]
        self.femurScaling = value[1]
        self.patellaScaling = value[2]
        self.tibfibScaling = value[3]
        self.pelvisRigid = value[a:a+6]
        self.hipRot = value[a+6:a+9]
        self.kneeRot = value[a+9:a+12]
    
class LLStepData(object):

    _registrationModes = ('shapemodel', 'uniformscaling', 'perbonescaling', 'manual')
    landmarkNames = ('pelvis-LASIS', 'pelvis-RASIS', 'pelvis-Sacral',
                      'femur-MEC', 'femur-LEC', 'tibiafibula-MM',
                      'tibiafibula-LM',
                      )
    _minArgs = {'method':'BFGS',
                 'jac':False,
                 'bounds':None, 'tol':1e-6,
                 'options':{'eps':1e-5},
                 },

    def __init__(self):
        self.LL = bone_models.LowerLimbLeftAtlas('lower_limb_left')
        self.T = LLTransformData()
        self._targetLandmarks = None
        self.inputPCs = None
        self._inputModelDict = None
        self._outputModelDict = None
        self._register = None
        self.regMode = None
        self.mWeight = 0.0
        self.landmarkErrors = None
        self.landmarkRMSE = None

    @property
    def outputModelDict(self):
        self._outputModelDict = dict([(m[0], m[1].gf) for m in self.LL.models.items()])
        return self._outputModelDict

    @property
    def outputTransform(self):
        return self.T
    
    @property
    def registrationMode(self):
        return self._registrationMode
    
    @registrationMode.setter
    def registrationMode(self, value):
        if value in self._registrationModes:
            self._registrationMode = value
            if value=='shapemodel':
                self._register = _registerShapeModel
            elif value=='uniformscale':
                self._register = _registerUniformScaling
            elif value=='perbonescaling':
                self._register = _registerPerBoneScaling
        else:
            raise ValueError('Invalid registration mode. Given {}, must be one of {}'.format(value, self._registrationModes))

    @property
    def targetLandmarks(self):
        return self._targetLandmarks

    @targetLandmarks.setter
    def targetLandmarks(self, value):
        v = np.array(value)
        if v.shape!=(7,3):
            raise ValueError('target landmark must have shape (7,3)')
        else:
            self._targetLandmarks = v
    
    @property
    def inputModelDict(self):
        return self._inputModelDict

    @inputModelDict.setter(self, value)
        self._inputModelDict = value

    def register(self):
        self._register(self)

def _registerShapeModel(lldata):
    # do the fit
    xFitted,\
    optLandmarkDist,\
    optLandmarkRMSE,\
    fitInfo = lowerlimbatlasfit.fit(
                    lldata.LL,
                    lldata.targetLandmarks,
                    lldata.landmarkNames,
                    lldata.T.shapeModes,
                    lldata.mWeight,
                    x0=lldata.T.shapeModelX,
                    minimise_args=self.min_args,
                    )
    lldata.landmarkRMSE = optLandmarkRMSE
    lldata.landmarkErrors = optLandmarkDist
    lldata.shapeModelX = xFitted[-1]

def _registerUniformScaling(lldata):
    raise NotImplementedError

def _registerPerBoneScaling(lldata):
    raise NotImplementedError