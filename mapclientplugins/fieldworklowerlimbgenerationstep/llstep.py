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

    _validRegistrationModes = ('shapemodel', 'uniformscaling', 'perbonescaling', 'manual')
    landmarkNames = ('pelvis-LASIS', 'pelvis-RASIS', 'pelvis-Sacral',
                      'femur-MEC', 'femur-LEC', 'tibiafibula-MM',
                      'tibiafibula-LM',
                      )
    _minArgs = {'method':'BFGS',
                 'jac':False,
                 'bounds':None, 'tol':1e-6,
                 'options':{'eps':1e-5},
                 },

    def __init__(self, config):
        self.config = config
        self.LL = bone_models.LowerLimbLeftAtlas('lower_limb_left')
        self.T = LLTransformData()
        self.inputLandmarks = None # a dict of landmarks
        self._targetLandmarksNames = None # list of strings matching keys in self.inputLandmarks
        self._targetLandmarks = None
        self.inputPCs = None
        self._inputModelDict = None
        self._outputModelDict = None
        self.landmarkErrors = None
        self.landmarkRMSE = None

    def resetLL(self):
        self.LL = bone_models.LowerLimbLeftAtlas('lower_limb_left')
        self.T = LLTransformData()
        self.landmarkErrors = None
        self.landmarkRMSE = None

    def updateFromConfig(self):
        targetLandmarkNames = [self.config[ln] for ln in self.landmarkNames]
        self.targetLandmarkNames = targetLandmarkNames
        self.nShapeModes = self.configs['pcs_to_fit']
        if self.kneeCorr:
            self.LL.enable_knee_adduction_correction()
        else:
            self.LL.disable_knee_adduction_correction()
        if self.kneeDOF:
            self.LL.enable_knee_adduction_dof
        else:
            self.LL.disable_knee_adduction_dof()

    @property
    def outputModelDict(self):
        self._outputModelDict = dict([(m[0], m[1].gf) for m in self.LL.models.items()])
        return self._outputModelDict

    @property
    def outputTransform(self):
        return self.T

    @property
    def validRegistrationModes(self):
        return self._validRegistrationModes  
    
    @property
    def registrationMode(self):
        return self.config['registration_mode']
        # return self._registrationMode
    
    @registrationMode.setter
    def registrationMode(self, value):
        if value in self.validRegistrationModes:
            self.config['registration_mode'] = value
        else:
            raise ValueError('Invalid registration mode. Given {}, must be one of {}'.format(value, self.validRegistrationModes))

    @property
    def targetLandmarkNames(self):
        return self._targetLandmarkNames

    @targetLandmarkNames.setter
    def targetLandmarkNames(self, value):
        if len(v)!=7:
            raise ValueError('7 input landmark names required for {}'.format(self._landmarkNames))
        else:
            self._targetLandmarkNames = v
            self._targetLandmarks = np.array([self.inputLandmarks[n] for n in self._targetLandmarkNames])

    @property
    def targetLandmarks(self):
        self._targetLandmarks = np.array([self.inputLandmarks[n] for n in self._targetLandmarkNames])
        return self._targetLandmarks
    
    @property
    def inputModelDict(self):
        return self._inputModelDict

    @inputModelDict.setter
    def inputModelDict(self, value):
        self._inputModelDict = value

    @property
    def mWeight(self):
        return float(self.config['mweight'])

    @mWeight.setter
    def mWeight(self, value):
        self.config['mweight'] = str(value)

    @property
    def nShapeModes(self):
        return int(self.config['pcs_to_fit'])

    @nShapeModes.setter
    def nShapeModes(self, n):
        self.config['pcs_to_fit'] = str(n)
        self.T.shapeModes = np.arange(n, dtype=int)
        if len(self.T.shapeModeWeights)<n:
            self.T.shapeModeWeights = np.hstack([
                                        self.T.shapeModelWeights,
                                        np.zeros(n-len(self.T.shapeModeWeights))
                                        ])
        else:
            self.T.shapeModeWeights = T.shapeModeWeights[:n]

    @property
    def kneeCorr(self):
        return self.config['knee_corr']=='True'

    @kneeCorr.setter
    def kneeCorr(self, value):
        if value:
            self.config['knee_corr'] = 'True'
        else:
            self.config['knee_corr'] = 'False'

    @property
    def kneeDOF(self):
        return self.config['knee_dof']=='True'

    @kneeDOF.setter
    def kneeDOF(self, value):
        if value:
            self.config['knee_dof'] = 'True'
        else:
            self.config['knee_dof'] = 'False'

    def register(self):
        self.updateFromConfig()
        mode = self.config['registration_mode']
        if mode=='shapemodel':
            _registerShapeModel(self)
        elif mode=='uniformscale':
            _registerUniformScaling(self)
        elif mode=='perbonescaling':
            _registerPerBoneScaling(self)

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