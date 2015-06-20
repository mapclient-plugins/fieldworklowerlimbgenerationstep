"""
Auto lower limb registration
"""
import os
import numpy as np
import copy

from fieldwork.field import geometric_field
from gias.musculoskeletal import mocap_landmark_preprocess
from gias.musculoskeletal.bonemodels import bonemodels
from gias.musculoskeletal.bonemodels import lowerlimbatlasfit

class LLTransformData(object):
    SHAPEMODESMAX = 100

    def __init__(self):
        self.pelvisRigid = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.hipRot = np.array([0.0, 0.0, 0.0])
        self._kneeRot = np.array([0.0, 0.0, 0.0])
        self.nShapeModes = 1
        self.shapeModes = [0,]
        self._shapeModeWeights = np.zeros(self.SHAPEMODESMAX, dtype=float)
        self.uniformScaling = 1.0
        self.pelvisScaling = 1.0
        self.femurScaling = 1.0
        self.petallaScaling = 1.0
        self.tibfibScaling = 1.0
        self.kneeDOF = False
        self.kneeCorr = False

        self._shapeModelX = None
        self._uniformScalingX = None
        self._perBoneScalingX = None

    @property
    def kneeRot(self):
        if self.kneeDOF:
            return self._kneeRot[[0,2]]
        else:
            return self._kneeRot[0]

    @kneeRot.setter
    def kneeRot(self, value):
        if self.kneeDOF:
            self._kneeRot[0] = value[0]
            self._kneeRot[2] = value[1]
        else:
            self._kneeRot[0] = value[0]
    
    @property
    def shapeModeWeights(self):
        return self._shapeModeWeights[:self.nShapeModes]

    @shapeModeWeights.setter
    def shapeModeWeights(self, value):
        self._shapeModeWeights[:len(value)] = value

    # gets a flat array, sets using a list of arrays.
    @property
    def shapeModelX(self):
        self._shapeModelX = np.hstack([
                                self.shapeModeWeights[:self.nShapeModes],
                                self.pelvisRigid,
                                self.hipRot,
                                self.kneeRot
                                ])
        return self._shapeModelX

    @shapeModelX.setter
    def shapeModelX(self, value):
        a = self.nShapeModes
        self._shapeModelX = value
        self.shapeModeWeights = value[0]
        self.pelvisRigid = value[1]
        self.hipRot = value[2]
        self.kneeRot = value[3]

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

SELF_DIRECTORY = os.path.split(__file__)[0]
    
class LLStepData(object):

    _shapeModelFilename = os.path.join(SELF_DIRECTORY, 'data/shape_models/LLP26_rigid.pc')
    _boneModelFilenames = {'pelvis': (os.path.join(SELF_DIRECTORY, 'data/atlas_meshes/pelvis_combined_cubic_mean_rigid_LLP26.geof'),
                                      os.path.join(SELF_DIRECTORY, 'data/atlas_meshes/pelvis_combined_cubic_flat.ens'),
                                      os.path.join(SELF_DIRECTORY, 'data/atlas_meshes/pelvis_combined_cubic_flat.mesh'),
                                      ),
                           'femur': (os.path.join(SELF_DIRECTORY, 'data/atlas_meshes/femur_left_mean_rigid_LLP26.geof'),
                                     os.path.join(SELF_DIRECTORY, 'data/atlas_meshes/femur_left_quartic_flat.ens'),
                                     os.path.join(SELF_DIRECTORY, 'data/atlas_meshes/femur_left_quartic_flat.mesh'),
                                     ),
                           'patella': (os.path.join(SELF_DIRECTORY, 'data/atlas_meshes/patella_left_mean_rigid_LLP26.geof'),
                                       os.path.join(SELF_DIRECTORY, 'data/atlas_meshes/patella_11_left.ens'),
                                       os.path.join(SELF_DIRECTORY, 'data/atlas_meshes/patella_11_left.mesh'),
                                       ),
                           'tibiafibula': (os.path.join(SELF_DIRECTORY, 'data/atlas_meshes/tibia_fibula_cubic_left_mean_rigid_LLP26.geof'),
                                           os.path.join(SELF_DIRECTORY, 'data/atlas_meshes/tibia_fibula_left_cubic_flat.ens'),
                                           os.path.join(SELF_DIRECTORY, 'data/atlas_meshes/tibia_fibula_left_cubic_flat.mesh'),
                                           ),
                            }

    _validRegistrationModes = ('shapemodel', 'uniformscaling', 'perbonescaling', 'manual')
    landmarkNames = ('pelvis-LASIS', 'pelvis-RASIS', 'pelvis-Sacral',
                      'femur-LEC', 'femur-MEC', 'tibiafibula-LM',
                      'tibiafibula-MM',
                      )
    minArgs = {'method':'BFGS',
                 'jac':False,
                 'bounds':None, 'tol':1e-3,
                 'options':{'eps':1e-5},
                 }

    def __init__(self, config):
        self.config = config
        self.T = LLTransformData()
        self.inputLandmarks = None # a dict of landmarks
        self._targetLandmarksNames = None # list of strings matching keys in self.inputLandmarks
        self._targetLandmarks = None

        self.inputPCs = None
        self._inputModelDict = None
        self._outputModelDict = None
        self.landmarkErrors = None
        self.landmarkRMSE = None
        self.fitMDist = None

    def loadData(self):
        self.LL = bonemodels.LowerLimbLeftAtlas('lower_limb_left')
        self.LL.bone_files = self._boneModelFilenames
        self.LL.combined_pcs_filename = self._shapeModelFilename
        self.LL.load_bones()

    def resetLL(self):
        self.LL.update_all_models(*self.LL._neutral_params)
        self.T = LLTransformData()
        self.landmarkErrors = None
        self.landmarkRMSE = None

    def updateFromConfig(self):
        targetLandmarkNames = [self.config[ln] for ln in self.landmarkNames]
        self.targetLandmarkNames = targetLandmarkNames
        self.nShapeModes = self.config['pcs_to_fit']
        if self.kneeCorr:
            self.LL.enable_knee_adduction_correction()
        else:
            self.LL.disable_knee_adduction_correction()
        if self.kneeDOF:
            self.LL.enable_knee_adduction_dof
        else:
            self.LL.disable_knee_adduction_dof()

    def updateLLModel(self):
        """update LL model using current transformations.
        Just shape model deformations
        """
        self.LL.update_all_models(self.T.shapeModeWeights,
                                  self.T.shapeModes,
                                  self.T.pelvisRigid,
                                  self.T.hipRot,
                                  self.T.kneeRot
                                  )

    def _preprocessLandmarks(self, l):
        return np.array(mocap_landmark_preprocess.preprocess_lower_limb(
                        self.markerRadius,
                        self.skinPad,
                        l[0], l[1], l[2], l[3], l[4], l[5], l[6]),
                        )

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
        if len(value)!=7:
            raise ValueError('7 input landmark names required for {}'.format(self._landmarkNames))
        else:
            self._targetLandmarkNames = value
            if self.inputLandmarks is not None:
                self._targetLandmarks = np.array([self.inputLandmarks[n] for n in self._targetLandmarkNames])

    @property
    def targetLandmarks(self):
        self._targetLandmarks = np.array([self.inputLandmarks[n] for n in self._targetLandmarkNames])
        self._targetLandmarks = self._preprocessLandmarks(self._targetLandmarks)
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

    @property
    def markerRadius(self):
        return float(self.config['marker_radius'])

    @markerRadius.setter
    def markerRadius(self, value):
        self.config['marker_radius'] = str(value)

    @property
    def skinPad(self):
        return float(self.config['skin_pad'])

    @skinPad.setter
    def skinPad(self, value):
        self.config['skin_pad'] = str(value)
    
    @nShapeModes.setter
    def nShapeModes(self, n):
        self.config['pcs_to_fit'] = str(n)
        n = int(n)
        self.T.nShapeModes = n
        self.T.shapeModes = np.arange(n, dtype=int)

    @property
    def kneeCorr(self):
        return self.config['knee_corr']=='True'

    @kneeCorr.setter
    def kneeCorr(self, value):
        self.T.kneeCorr = value
        if value:
            self.config['knee_corr'] = 'True'
            self.LL.enable_knee_adduction_correction()
        else:
            self.config['knee_corr'] = 'False'
            self.LL.disable_knee_adduction_correction()

    @property
    def kneeDOF(self):
        return self.config['knee_dof']=='True'

    @kneeDOF.setter
    def kneeDOF(self, value):
        self.T.kneeDOF = value
        if value:
            self.config['knee_dof'] = 'True'
            self.LL.enable_knee_adduction_dof()
        else:
            self.config['knee_dof'] = 'False'
            self.LL.disable_knee_adduction_dof()

    def register(self):
        self.updateFromConfig()
        mode = self.config['registration_mode']
        if mode=='shapemodel':
            print(self.T.shapeModelX)
            output = _registerShapeModel(self)
        elif mode=='uniformscale':
            print(self.T.uniformScalingX)
            output = _registerUniformScaling(self)
        elif mode=='perbonescaling':
            print(self.T.perBoneScalingX)
            output = _registerPerBoneScaling(self)
        return output

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
                    minimise_args=lldata.minArgs,
                    )
    lldata.landmarkRMSE = optLandmarkRMSE
    lldata.landmarkErrors = optLandmarkDist
    lldata.fitMDist =  fitInfo['mahalanobis_distance']
    lldata.T.shapeModelX = xFitted[-1]
    print('new X:'+str(lldata.T.shapeModelX))
    return xFitted, optLandmarkDist, optLandmarkRMSE, fitInfo

def _registerUniformScaling(lldata):
    raise NotImplementedError

def _registerPerBoneScaling(lldata):
    raise NotImplementedError