# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'configuredialog.ui'
#
# Created: Tue Mar  8 12:15:09 2016
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(597, 606)
        self.gridLayout = QtGui.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.configGroupBox = QtGui.QGroupBox(Dialog)
        self.configGroupBox.setTitle("")
        self.configGroupBox.setObjectName("configGroupBox")
        self.formLayout = QtGui.QFormLayout(self.configGroupBox)
        self.formLayout.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setObjectName("formLayout")
        self.label0 = QtGui.QLabel(self.configGroupBox)
        self.label0.setObjectName("label0")
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.label0)
        self.lineEdit_id = QtGui.QLineEdit(self.configGroupBox)
        self.lineEdit_id.setObjectName("lineEdit_id")
        self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.lineEdit_id)
        self.label1 = QtGui.QLabel(self.configGroupBox)
        self.label1.setObjectName("label1")
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.label1)
        self.comboBox_regmode = QtGui.QComboBox(self.configGroupBox)
        self.comboBox_regmode.setObjectName("comboBox_regmode")
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.comboBox_regmode)
        self.label_14 = QtGui.QLabel(self.configGroupBox)
        self.label_14.setObjectName("label_14")
        self.formLayout.setWidget(2, QtGui.QFormLayout.LabelRole, self.label_14)
        self.comboBox_side = QtGui.QComboBox(self.configGroupBox)
        self.comboBox_side.setObjectName("comboBox_side")
        self.formLayout.setWidget(2, QtGui.QFormLayout.FieldRole, self.comboBox_side)
        self.label = QtGui.QLabel(self.configGroupBox)
        self.label.setObjectName("label")
        self.formLayout.setWidget(3, QtGui.QFormLayout.LabelRole, self.label)
        self.spinBox_pcsToFit = QtGui.QSpinBox(self.configGroupBox)
        self.spinBox_pcsToFit.setMinimum(1)
        self.spinBox_pcsToFit.setMaximum(99)
        self.spinBox_pcsToFit.setObjectName("spinBox_pcsToFit")
        self.formLayout.setWidget(3, QtGui.QFormLayout.FieldRole, self.spinBox_pcsToFit)
        self.label_2 = QtGui.QLabel(self.configGroupBox)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(4, QtGui.QFormLayout.LabelRole, self.label_2)
        self.doubleSpinBox_mWeight = QtGui.QDoubleSpinBox(self.configGroupBox)
        self.doubleSpinBox_mWeight.setSingleStep(0.1)
        self.doubleSpinBox_mWeight.setObjectName("doubleSpinBox_mWeight")
        self.formLayout.setWidget(4, QtGui.QFormLayout.FieldRole, self.doubleSpinBox_mWeight)
        self.label_3 = QtGui.QLabel(self.configGroupBox)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(5, QtGui.QFormLayout.LabelRole, self.label_3)
        self.label_12 = QtGui.QLabel(self.configGroupBox)
        self.label_12.setObjectName("label_12")
        self.formLayout.setWidget(6, QtGui.QFormLayout.LabelRole, self.label_12)
        self.doubleSpinBox_markerRadius = QtGui.QDoubleSpinBox(self.configGroupBox)
        self.doubleSpinBox_markerRadius.setObjectName("doubleSpinBox_markerRadius")
        self.formLayout.setWidget(6, QtGui.QFormLayout.FieldRole, self.doubleSpinBox_markerRadius)
        self.label_13 = QtGui.QLabel(self.configGroupBox)
        self.label_13.setObjectName("label_13")
        self.formLayout.setWidget(7, QtGui.QFormLayout.LabelRole, self.label_13)
        self.doubleSpinBox_skinPad = QtGui.QDoubleSpinBox(self.configGroupBox)
        self.doubleSpinBox_skinPad.setObjectName("doubleSpinBox_skinPad")
        self.formLayout.setWidget(7, QtGui.QFormLayout.FieldRole, self.doubleSpinBox_skinPad)
        self.label_10 = QtGui.QLabel(self.configGroupBox)
        self.label_10.setObjectName("label_10")
        self.formLayout.setWidget(8, QtGui.QFormLayout.LabelRole, self.label_10)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.checkBox_kneedof = QtGui.QCheckBox(self.configGroupBox)
        self.checkBox_kneedof.setObjectName("checkBox_kneedof")
        self.horizontalLayout.addWidget(self.checkBox_kneedof)
        self.checkBox_kneecorr = QtGui.QCheckBox(self.configGroupBox)
        self.checkBox_kneecorr.setObjectName("checkBox_kneecorr")
        self.horizontalLayout.addWidget(self.checkBox_kneecorr)
        self.formLayout.setLayout(8, QtGui.QFormLayout.FieldRole, self.horizontalLayout)
        self.label_11 = QtGui.QLabel(self.configGroupBox)
        self.label_11.setObjectName("label_11")
        self.formLayout.setWidget(9, QtGui.QFormLayout.LabelRole, self.label_11)
        self.checkBox_GUI = QtGui.QCheckBox(self.configGroupBox)
        self.checkBox_GUI.setText("")
        self.checkBox_GUI.setObjectName("checkBox_GUI")
        self.formLayout.setWidget(9, QtGui.QFormLayout.FieldRole, self.checkBox_GUI)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.tableWidgetLandmarks = QtGui.QTableWidget(self.configGroupBox)
        self.tableWidgetLandmarks.setObjectName("tableWidgetLandmarks")
        self.tableWidgetLandmarks.setColumnCount(2)
        self.tableWidgetLandmarks.setRowCount(0)
        item = QtGui.QTableWidgetItem()
        self.tableWidgetLandmarks.setHorizontalHeaderItem(0, item)
        item = QtGui.QTableWidgetItem()
        self.tableWidgetLandmarks.setHorizontalHeaderItem(1, item)
        self.tableWidgetLandmarks.horizontalHeader().setDefaultSectionSize(200)
        self.tableWidgetLandmarks.horizontalHeader().setMinimumSectionSize(200)
        self.verticalLayout.addWidget(self.tableWidgetLandmarks)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton_addLandmark = QtGui.QPushButton(self.configGroupBox)
        self.pushButton_addLandmark.setObjectName("pushButton_addLandmark")
        self.horizontalLayout_2.addWidget(self.pushButton_addLandmark)
        self.pushButton_removeLandmark = QtGui.QPushButton(self.configGroupBox)
        self.pushButton_removeLandmark.setObjectName("pushButton_removeLandmark")
        self.horizontalLayout_2.addWidget(self.pushButton_removeLandmark)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.formLayout.setLayout(5, QtGui.QFormLayout.FieldRole, self.verticalLayout)
        self.gridLayout.addWidget(self.configGroupBox, 0, 0, 1, 1)
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 1, 0, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        Dialog.setTabOrder(self.lineEdit_id, self.comboBox_regmode)
        Dialog.setTabOrder(self.comboBox_regmode, self.comboBox_side)
        Dialog.setTabOrder(self.comboBox_side, self.spinBox_pcsToFit)
        Dialog.setTabOrder(self.spinBox_pcsToFit, self.doubleSpinBox_mWeight)
        Dialog.setTabOrder(self.doubleSpinBox_mWeight, self.tableWidgetLandmarks)
        Dialog.setTabOrder(self.tableWidgetLandmarks, self.pushButton_addLandmark)
        Dialog.setTabOrder(self.pushButton_addLandmark, self.pushButton_removeLandmark)
        Dialog.setTabOrder(self.pushButton_removeLandmark, self.doubleSpinBox_markerRadius)
        Dialog.setTabOrder(self.doubleSpinBox_markerRadius, self.doubleSpinBox_skinPad)
        Dialog.setTabOrder(self.doubleSpinBox_skinPad, self.checkBox_kneedof)
        Dialog.setTabOrder(self.checkBox_kneedof, self.checkBox_kneecorr)
        Dialog.setTabOrder(self.checkBox_kneecorr, self.checkBox_GUI)
        Dialog.setTabOrder(self.checkBox_GUI, self.buttonBox)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Configure Lower Limb Registration Step", None, QtGui.QApplication.UnicodeUTF8))
        self.label0.setText(QtGui.QApplication.translate("Dialog", "identifier:  ", None, QtGui.QApplication.UnicodeUTF8))
        self.label1.setText(QtGui.QApplication.translate("Dialog", "Registration Mode:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_14.setText(QtGui.QApplication.translate("Dialog", "Side:", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Dialog", "PCs to Fit:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Dialog", "Mahalanobis Weight:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("Dialog", "Landmarks:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_12.setText(QtGui.QApplication.translate("Dialog", "Marker Radius:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_13.setText(QtGui.QApplication.translate("Dialog", "Skin Padding:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_10.setText(QtGui.QApplication.translate("Dialog", "Knee Options:", None, QtGui.QApplication.UnicodeUTF8))
        self.checkBox_kneedof.setText(QtGui.QApplication.translate("Dialog", "Abd. DOF", None, QtGui.QApplication.UnicodeUTF8))
        self.checkBox_kneecorr.setText(QtGui.QApplication.translate("Dialog", "Abd. Correction", None, QtGui.QApplication.UnicodeUTF8))
        self.label_11.setText(QtGui.QApplication.translate("Dialog", "GUI:", None, QtGui.QApplication.UnicodeUTF8))
        self.tableWidgetLandmarks.horizontalHeaderItem(0).setText(QtGui.QApplication.translate("Dialog", "Model Landmarks", None, QtGui.QApplication.UnicodeUTF8))
        self.tableWidgetLandmarks.horizontalHeaderItem(1).setText(QtGui.QApplication.translate("Dialog", "Target Landmarks", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton_addLandmark.setText(QtGui.QApplication.translate("Dialog", "Add Landmark", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton_removeLandmark.setText(QtGui.QApplication.translate("Dialog", "Remove Landmark", None, QtGui.QApplication.UnicodeUTF8))

