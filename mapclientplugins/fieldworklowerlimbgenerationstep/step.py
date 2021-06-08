'''
MAP Client Plugin Step
'''
import json

from mapclient.mountpoints.workflowstep import WorkflowStepMountPoint
from mapclientplugins.fieldworklowerlimbgenerationstep.configuredialog import ConfigureDialog

from mapclientplugins.fieldworklowerlimbgenerationstep import llstep
from mapclientplugins.fieldworklowerlimbgenerationstep.lowerlimbgenerationdialog import LowerLimbGenerationDialog

DEFAULT_MODEL_LANDMARKS = ('pelvis-LASIS', 'pelvis-RASIS', 'pelvis-Sacral',
                           'femur-MEC', 'femur-LEC', 'tibiafibula-MM',
                           'tibiafibula-LM',
                           )


class FieldworkLowerLimbGenerationStep(WorkflowStepMountPoint):
    '''
    Step for customising the left lower limb bones to motion capture markers.

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
    '''

    def __init__(self, location):
        super(FieldworkLowerLimbGenerationStep, self).__init__('Fieldwork Lower Limb Generation', location)
        self._configured = False  # A step cannot be executed until it has been configured.
        self._category = 'Registration'
        # Add any other initialisation code here:
        # Ports:
        self.addPort(('http://physiomeproject.org/workflow/1.0/rdf-schema#port',
                      'http://physiomeproject.org/workflow/1.0/rdf-schema#uses',
                      'http://physiomeproject.org/workflow/1.0/rdf-schema#landmarks'))
        self.addPort(('http://physiomeproject.org/workflow/1.0/rdf-schema#port',
                      'http://physiomeproject.org/workflow/1.0/rdf-schema#uses',
                      'ju#principalcomponents'))
        self.addPort(('http://physiomeproject.org/workflow/1.0/rdf-schema#port',
                      'http://physiomeproject.org/workflow/1.0/rdf-schema#uses',
                      'ju#fieldworkmodeldict'))
        self.addPort(('http://physiomeproject.org/workflow/1.0/rdf-schema#port',
                      'http://physiomeproject.org/workflow/1.0/rdf-schema#provides',
                      'ju#fieldworkmodeldict'))
        self.addPort(('http://physiomeproject.org/workflow/1.0/rdf-schema#port',
                      'http://physiomeproject.org/workflow/1.0/rdf-schema#provides',
                      'ju#lowerlimbtransform'))
        self._config = {}
        self._config['identifier'] = ''
        self._config['GUI'] = 'True'
        self._config['registration_mode'] = 'shapemodel'
        self._config['pcs_to_fit'] = '1'
        self._config['mweight'] = '0.1'
        self._config['knee_corr'] = 'False'
        self._config['knee_dof'] = 'False'
        self._config['marker_radius'] = '5.0'
        self._config['skin_pad'] = '5.0'
        self._config['side'] = 'left'
        self._config['landmarks'] = {}
        for l in DEFAULT_MODEL_LANDMARKS:
            self._config['landmarks'][l] = ''

        self._data = llstep.LLStepData(self._config)

    def execute(self):
        '''
        Add your code here that will kick off the execution of the step.
        Make sure you call the _doneExecution() method when finished.  This method
        may be connected up to a button in a widget for example.
        '''
        # Put your execute step code here before calling the '_doneExecution' method.
        self._data.loadData()
        self._data.updateFromConfig()
        print('LL estimation configs:')
        print
        self._data.config
        if self._config['GUI'] == 'True':
            # start gui
            self._widget = LowerLimbGenerationDialog(self._data, self._doneExecution)
            self._widget.setModal(True)
            self._setCurrentWidget(self._widget)
        else:
            self._data.register()
            self._doneExecution()

    def setPortData(self, index, dataIn):
        '''
        Add your code here that will set the appropriate objects for this step.
        The index is the index of the port in the port list.  If there is only one
        uses port for this step then the index can be ignored.
        '''
        if index == 0:
            self._data.inputLandmarks = dataIn  # http://physiomeproject.org/workflow/1.0/rdf-schema#landmarks
        elif index == 1:
            self._data.inputPCs = dataIn  # ju#principalcomponents
        else:
            self._data.inputModelDict = dataIn  # ju#fieldworkmodeldict

    def getPortData(self, index):
        '''
        Add your code here that will return the appropriate objects for this step.
        The index is the index of the port in the port list.  If there is only one
        provides port for this step then the index can be ignored.
        '''
        if index == 3:
            return self._data.outputModelDict
        else:
            return self._data.outputTransform

    def configure(self):
        '''
        This function will be called when the configure icon on the step is
        clicked.  It is appropriate to display a configuration dialog at this
        time.  If the conditions for the configuration of this step are complete
        then set:
            self._configured = True
        '''
        dlg = ConfigureDialog(self._main_window)
        dlg.identifierOccursCount = self._identifierOccursCount
        dlg.setConfig(self._config)
        dlg.validate()
        dlg.setModal(True)

        if dlg.exec_():
            self._config = dlg.getConfig()
            self._data.config = self._config
        
        self._configured = dlg.validate()
        self._configuredObserver()

    def getIdentifier(self):
        '''
        The identifier is a string that must be unique within a workflow.
        '''
        return self._config['identifier']

    def setIdentifier(self, identifier):
        '''
        The framework will set the identifier for this step when it is loaded.
        '''
        self._config['identifier'] = identifier

    def serialize(self):
        '''
        Add code to serialize this step to disk. Returns a json string for
        mapclient to serialise.
        '''
        self._fix_legacy_config()
        return json.dumps(self._config, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def deserialize(self, string):
        '''
        Add code to deserialize this step from disk. Parses a json string
        given by mapclient
        '''
        self._config.update(json.loads(string))
        self._fix_legacy_config()

        d = ConfigureDialog()
        d.identifierOccursCount = self._identifierOccursCount
        d.setConfig(self._config)
        self._configured = d.validate()

    def _fix_legacy_config(self):
        # legacy configs
        if self._config.get('landmarks') is None:
            self._config['landmarks'] = {}
            self._config['landmarks']['pelvis-LASIS'] = self._config['pelvis-LASIS']
            self._config['landmarks']['pelvis-RASIS'] = self._config['pelvis-RASIS']
            self._config['landmarks']['pelvis-Sacral'] = self._config['pelvis-Sacral']
            self._config['landmarks']['femur-MEC'] = self._config['femur-MEC']
            self._config['landmarks']['femur-LEC'] = self._config['femur-LEC']
            self._config['landmarks']['tibiafibula-MM'] = self._config['tibiafibula-MM']
            self._config['landmarks']['tibiafibula-LM'] = self._config['tibiafibula-LM']

            del self._config['pelvis-LASIS']
            del self._config['pelvis-RASIS']
            del self._config['pelvis-Sacral']
            del self._config['femur-MEC']
            del self._config['femur-LEC']
            del self._config['tibiafibula-MM']
            del self._config['tibiafibula-LM']
