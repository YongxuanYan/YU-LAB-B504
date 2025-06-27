# coding:utf-8
import numpy as np
from PyQt5.QtCore import Qt, QEvent, QThread, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QIntValidator, QDoubleValidator
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFrame, QSpacerItem, QSizePolicy

from qfluentwidgets import (LineEdit, ScrollArea, PushButton, ToolButton, FluentIcon, MessageBox, CheckBox,
                            isDarkTheme, ToolTipFilter, BodyLabel, toggleTheme, ComboBox, FlowLayout, ProgressBar)
from ..common.style_sheet import StyleSheet
from ..common.icon import Icon
from ..var.globals import set_var, get_var
from ..functions.ButtonClickChange import buttonClickWithFloatingWindow
import os
import cv2
from scipy.interpolate import interp1d
import json
from ..functions.drr_display import update_display_drr
from ..functions.DRR_Generation import (getDRR, get_bone_only_DRR, get_bone_suppressed_DRR_constant,
                                        get_bone_enhanced_DRR, getLabel)


class SeparatorWidget(QWidget):
    """ Seperator widget """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setFixedSize(6, 16)

    def paintEvent(self, e):
        painter = QPainter(self)
        pen = QPen(1)
        pen.setCosmetic(True)
        c = QColor(255, 255, 255, 21) if isDarkTheme() else QColor(0, 0, 0, 15)
        pen.setColor(c)
        painter.setPen(pen)

        x = self.width() // 2
        painter.drawLine(x, 0, x, self.height())


class DRRGenerationThread(QThread):
    progress_signal = pyqtSignal(str)  # 用于发送进度消息到主线程

    def __init__(self, E0, muWater, muAir, pair, Geoinfo_save_path, resolution, tileSize, couchAngle, iso_x, iso_y, iso_z, sliceThickness, DRR_type, bitDepth, parent=None):
        super().__init__(parent)
        self.muWater = muWater
        self.muAir = muAir
        self.pair = pair
        self.E0 = E0
        self.Geoinfo_save_path = Geoinfo_save_path
        self.resolution = resolution
        self.tileSize = tileSize
        self.couchAngle = couchAngle
        self.iso_x = iso_x
        self.iso_y = iso_y
        self.iso_z = iso_z
        self.sliceThickness = sliceThickness
        self.DRR_type = DRR_type
        self.parent = parent
        self.bitDepth = bitDepth

    def run(self):
        try:
            if self.pair == 'imaging pair 1':
                completed, DRR, Label = self.parent.imaging_pair1_DRR_generation(self.E0, self.muWater, self.muAir,
                    self.Geoinfo_save_path, self.resolution, self.bitDepth, self.tileSize, self.couchAngle,
                    self.iso_x, self.iso_y, self.iso_z, self.sliceThickness, self.DRR_type)
                msg = f"DRR generation for imaging pair 1 completed! Automatically saved at {self.Geoinfo_save_path}\saved DRR" if completed else "Failed to generate DRR for imaging pair 1."
                if completed:
                    TempDRR = np.zeros((512, 512, 3))
                    TempDRR[:, :, 0] = cv2.resize(DRR, (512, 512))
                    TempDRR[:, :, 1] = cv2.resize(Label, (512, 512))
                    TempDRR[:, :, 2] = cv2.resize(0.85 * DRR + 0.15 * np.multiply(DRR, Label), (512, 512))
                    set_var("Left_DRR", TempDRR)
                    set_var("DRREXISTS_Left", 1)
                self.progress_signal.emit(msg)
            elif self.pair == 'imaging pair 2':
                completed, DRR, Label = self.parent.imaging_pair2_DRR_generation(self.E0, self.muWater, self.muAir,
                    self.Geoinfo_save_path, self.resolution, self.bitDepth, self.tileSize, self.couchAngle,
                    self.iso_x, self.iso_y, self.iso_z, self.sliceThickness, self.DRR_type)
                msg = f"DRR generation for imaging pair 2 completed! Automatically saved at {self.Geoinfo_save_path}\saved DRR" if completed else "Failed to generate DRR for imaging pair 2."
                if completed:
                    TempDRR = np.zeros((512, 512, 3))
                    TempDRR[:, :, 0] = cv2.resize(DRR, (512, 512))
                    TempDRR[:, :, 1] = cv2.resize(Label, (512, 512))
                    TempDRR[:, :, 2] = cv2.resize(0.85 * DRR + 0.15 * np.multiply(DRR, Label), (512, 512))
                    set_var("Right_DRR", TempDRR)
                    set_var("DRREXISTS_Right", 1)
                self.progress_signal.emit(msg)
            elif self.pair == 'both imaging pairs':
                completed1, DRR1, Label1 = self.parent.imaging_pair1_DRR_generation(self.E0, self.muWater, self.muAir,
                    self.Geoinfo_save_path, self.resolution, self.bitDepth, self.tileSize, self.couchAngle,
                    self.iso_x, self.iso_y, self.iso_z, self.sliceThickness, self.DRR_type)
                if completed1:
                    TempDRR = np.zeros((512, 512, 3))
                    TempDRR[:, :, 0] = cv2.resize(DRR1, (512, 512))
                    TempDRR[:, :, 1] = cv2.resize(Label1, (512, 512))
                    TempDRR[:, :, 2] = cv2.resize(0.85 * DRR1 + 0.15 * np.multiply(DRR1, Label1), (512, 512))
                    set_var("Left_DRR", TempDRR)
                    set_var("DRREXISTS_Left", 1)
                completed2, DRR2, Label2 = self.parent.imaging_pair2_DRR_generation(self.E0, self.muWater, self.muAir,
                    self.Geoinfo_save_path, self.resolution, self.bitDepth, self.tileSize, self.couchAngle,
                    self.iso_x, self.iso_y, self.iso_z, self.sliceThickness, self.DRR_type)
                if completed2:
                    TempDRR = np.zeros((512, 512, 3))
                    TempDRR[:, :, 0] = cv2.resize(DRR2, (512, 512))
                    TempDRR[:, :, 1] = cv2.resize(Label2, (512, 512))
                    TempDRR[:, :, 2] = cv2.resize(0.85 * DRR2 + 0.15 * np.multiply(DRR2, Label2), (512, 512))
                    set_var("Right_DRR", TempDRR)
                    set_var("DRREXISTS_Right", 1)
                msg = f"Both DRRs generation completed! Automatically saved at {self.Geoinfo_save_path}\saved DRR" if completed1 and completed2 else \
                      f"DRR generation partially failed! Automatically saved at {self.Geoinfo_save_path}\saved DRR" if completed1 or completed2 else \
                      "Both DRRs generation failed!"
                self.progress_signal.emit(msg)
        except Exception as e:
            self.progress_signal.emit(f"Error during DRR generation: {str(e)}")


def check_is_number(line_edit, isInt):
    try:
        if isInt:
            value = int(line_edit.text())
        else:
            value = float(line_edit.text())
        return value
    except ValueError:
        return None


class ToolBar(QWidget):
    """ Tool bar """
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.NoMoreAskingCheckBox = CheckBox('no more asking before generating DRR.', self)
        self.NoMoreAskingCheckBox.setFixedHeight(28)
        self.NoMoreAskingCheckBox.stateChanged.connect(
            lambda: set_var("NoMoreAsking", self.NoMoreAskingCheckBox.isChecked()))

        self.EnableMuEdittingCheckBox = CheckBox('enable μ edit.', self)
        self.EnableMuEdittingCheckBox.setFixedHeight(28)
        self.EnableMuEdittingCheckBox.stateChanged.connect(self.toggle_editing)

        self.LabelGenCheckBox = CheckBox('generate tumor label.', self)
        self.LabelGenCheckBox.setFixedHeight(28)

        self.themeButton = ToolButton(FluentIcon.CONSTRACT, self)
        self.generateDRRButton = PushButton(self.tr('generate DRR'), self, Icon.DRR)
        self.generateDRRButton.clicked.connect(self.on_generateDRRClicked)

        self.condition_label = BodyLabel(self.tr('DRR generation condition'), self)

        self.initial_I_label = BodyLabel(self.tr('KeV: '), self)
        self.initial_I_LineEdit = LineEdit(self)
        self.initial_I_LineEdit.setFixedSize(60, 28)
        self.initial_I_LineEdit.setToolTip('X-ray condition, 1~20000.')
        self.initial_I_LineEdit.setValidator(QIntValidator(1, 20000, self))
        self.initial_I_LineEdit.setText('60')
        self.initial_I_LineEdit.textChanged.connect(self.update_mu_values)

        self.mu_water_label = BodyLabel(self.tr('μ<sub>water<sub>'), self)
        self.mu_water_LineEdit = LineEdit(self)
        self.mu_water_LineEdit.setFixedSize(80, 28)
        self.mu_water_LineEdit.setToolTip('mass attenuation coefficient for water')
        self.mu_water_LineEdit.setValidator(QDoubleValidator(0, 1, 5))
        self.mu_water_LineEdit.setText('0.20590')
        self.mu_water_LineEdit.setReadOnly(True)

        self.mu_air_label = BodyLabel(self.tr('μ<sub>air<sub>'), self)
        self.mu_air_LineEdit = LineEdit(self)
        self.mu_air_LineEdit.setFixedSize(80, 28)
        self.mu_air_LineEdit.setToolTip('mass attenuation coefficient for air')
        self.mu_air_LineEdit.setValidator(QDoubleValidator(0, 1, 5))
        self.mu_air_LineEdit.setText('0.00024')
        self.mu_air_LineEdit.setReadOnly(True)

        self.tile_size_label = BodyLabel(self.tr('tile size'), self)
        self.tile_size_label.setToolTip('please use a small tile size to reduce memory usage.')
        self.tile_size_LineEdit = LineEdit(self)
        self.tile_size_LineEdit.setFixedSize(60, 28)
        self.tile_size_LineEdit.setText('16')
        self.tile_size_LineEdit.setToolTip('please make sure the tile size is divisible by the resolution.')
        self.tile_size_LineEdit.setValidator(QIntValidator(1, 9999, self))

        self.drr_resolution_label = BodyLabel(self.tr('DRR resolution: '), self)
        self.drr_resolution_LineEdit = LineEdit(self)
        self.drr_resolution_LineEdit.setFixedSize(60, 28)
        self.drr_resolution_LineEdit.setText('128')
        self.drr_resolution_LineEdit.setToolTip('please input 512 for resolution: 512*512')
        self.drr_resolution_LineEdit.setValidator(QIntValidator(1, 9999, self))

        self.drr_bitDepth_label = BodyLabel(self.tr('bit depth: '), self)
        self.drr_bitDepth_LineEdit = LineEdit(self)
        self.drr_bitDepth_LineEdit.setFixedSize(60, 28)
        self.drr_bitDepth_LineEdit.setText('8')
        self.drr_bitDepth_LineEdit.setToolTip('please inpuit N for 0~2^N')
        self.drr_bitDepth_LineEdit.setValidator(QIntValidator(0, 99, self))

        self.couch_angle_label = BodyLabel(self.tr('couch angle: '), self)
        self.couch_angle_LineEdit = LineEdit(self)
        self.couch_angle_LineEdit.setFixedSize(60, 28)
        self.couch_angle_LineEdit.setText('0')
        self.couch_angle_LineEdit.setToolTip('range: 0-360, float or integer.')
        self.couch_angle_LineEdit.setValidator(QDoubleValidator(0, 360, 2))

        self.imaging_pair_selection_label = BodyLabel(self.tr('imaging pair: '), self)
        self.imaging_pair_selection_comboBox = ComboBox()
        self.imaging_pair_selection_comboBox.setCurrentIndex(0)
        self.imaging_pair_selection_comboBox.setFixedSize(150, 32)

        self.DRR_type_label = BodyLabel(self.tr('DRR type: '), self)
        self.DRR_type_comboBox = ComboBox()
        self.DRR_type_comboBox.setCurrentIndex(0)
        self.DRR_type_comboBox.setFixedSize(100, 32)

        self.bone_threshold_label = BodyLabel(self.tr('bone threshold: '), self)
        self.bone_threshold_LineEdit = LineEdit(self)
        self.bone_threshold_LineEdit.setFixedSize(60, 28)
        self.bone_threshold_LineEdit.setToolTip('unit: Hu')
        self.bone_threshold_LineEdit.setText('350')
        self.bone_threshold_LineEdit.setValidator(QDoubleValidator(-1000, int(get_var("CT_MAX_HU")), 1))

        self.tumor_label_threshold_label = BodyLabel(self.tr('tumor label threshold: '), self)
        self.tumor_label_threshold_LineEdit = LineEdit(self)
        self.tumor_label_threshold_LineEdit.setFixedSize(60, 28)
        self.tumor_label_threshold_LineEdit.setToolTip('threshold projection, see this paper for more details: Markerless Lung Tumor Localization From Intraoperative Stereo Color Fluoroscopic Images for Radiotherapy')
        self.tumor_label_threshold_LineEdit.setText('10')
        self.tumor_label_threshold_LineEdit.setValidator(QIntValidator(0, 999))

        self.bone_suppress_method_label = BodyLabel(self.tr('bone suppress method: '), self)
        self.bone_suppress_method_combobox = ComboBox()
        self.bone_suppress_method_combobox.setCurrentIndex(0)
        self.bone_suppress_method_combobox.setFixedSize(150, 32)

        self.bone_replace_hu_label = BodyLabel(self.tr('bone replace hu:'), self)
        self.bone_replace_hu_LineEdit = LineEdit(self)
        self.bone_replace_hu_LineEdit.setFixedSize(60, 28)
        self.bone_replace_hu_LineEdit.setToolTip('unit: Hu')
        self.bone_replace_hu_LineEdit.setText('47.5')
        self.bone_replace_hu_LineEdit.setValidator(QDoubleValidator(-1000, int(get_var("CT_MAX_HU")), 1))

        self.bone_enhance_factor_label = BodyLabel(self.tr('bone enhance factor:'), self)
        self.bone_enhance_factor_LineEdit = LineEdit(self)
        self.bone_enhance_factor_LineEdit.setFixedSize(60, 28)
        self.bone_enhance_factor_LineEdit.setText('2')
        self.bone_enhance_factor_LineEdit.setToolTip('bone voxels CT value * bone enhance factor')
        self.bone_enhance_factor_LineEdit.setValidator(QDoubleValidator(-1000, int(get_var("CT_MAX_HU")), 1))

        self.deltaI_label = BodyLabel(self.tr('ΔKeV:'), self)
        self.deltaI_LineEdit = LineEdit(self)
        self.deltaI_LineEdit.setFixedSize(60, 28)
        self.deltaI_LineEdit.setText('60')
        self.deltaI_LineEdit.setToolTip('KeV1 - KeV2, can be negative')
        self.deltaI_LineEdit.setValidator(QIntValidator(-20000, 20000, self))

        self.sysMSGLabel = BodyLabel(self.tr(f'system message: generate DRR here.'), self)

        self.imaging_pair_1_display_label = BodyLabel(self.tr('DRR generated by imaging pair 1:'), self)
        self.imaging_pair_2_display_label = BodyLabel(self.tr('DRR generated by imaging pair 2:'), self)

        self.separator = SeparatorWidget(self)
        self.vBoxLayout = QVBoxLayout(self)
        self.buttonLayout = FlowLayout()
        self.labelTitleLayout = FlowLayout()
        self.labelLayout = FlowLayout()
        self.sysMSGLayout = FlowLayout()
        self.DRR_display_area_layout = QHBoxLayout(self)
        self.__initWidget()
        # 连接信号以动态更新控件显示状态
        self.DRR_type_comboBox.currentTextChanged.connect(self.update_control_visibility)
        self.bone_suppress_method_combobox.currentTextChanged.connect(self.update_control_visibility)
        self.LabelGenCheckBox.stateChanged.connect(self.update_control_visibility)

    def __initWidget(self):
        # Button layout
        self.setFixedHeight(300)
        self.vBoxLayout.addSpacing(0)
        self.vBoxLayout.addLayout(self.buttonLayout, 1)
        self.vBoxLayout.setAlignment(Qt.AlignTop)
        self.buttonLayout.addWidget(self.themeButton)
        self.buttonLayout.addWidget(self.separator)
        self.buttonLayout.addWidget(self.generateDRRButton)
        self.buttonLayout.addWidget(self.NoMoreAskingCheckBox)
        self.NoMoreAskingCheckBox.setChecked(True)
        self.buttonLayout.addWidget(self.EnableMuEdittingCheckBox)
        self.buttonLayout.addWidget(self.LabelGenCheckBox)
        self.LabelGenCheckBox.setChecked(True)
        self.themeButton.installEventFilter(ToolTipFilter(self.themeButton))
        self.themeButton.setToolTip(self.tr('toggle theme'))
        self.themeButton.clicked.connect(lambda: toggleTheme(True, False))

        # Label title layout
        self.vBoxLayout.addLayout(self.labelTitleLayout, 1)
        self.labelTitleLayout.addWidget(self.condition_label)

        # Label layout
        self.vBoxLayout.addLayout(self.labelLayout, 1)
        self.labelLayout.addWidget(self.initial_I_label)
        self.labelLayout.addWidget(self.initial_I_LineEdit)
        self.labelLayout.addWidget(self.mu_water_label)
        self.labelLayout.addWidget(self.mu_water_LineEdit)
        self.labelLayout.addWidget(self.mu_air_label)
        self.labelLayout.addWidget(self.mu_air_LineEdit)
        self.labelLayout.addWidget(self.tile_size_label)
        self.labelLayout.addWidget(self.tile_size_LineEdit)
        self.labelLayout.addWidget(self.drr_resolution_label)
        self.labelLayout.addWidget(self.drr_resolution_LineEdit)
        self.labelLayout.addWidget(self.drr_bitDepth_label)
        self.labelLayout.addWidget(self.drr_bitDepth_LineEdit)
        self.labelLayout.addWidget(self.couch_angle_label)
        self.labelLayout.addWidget(self.couch_angle_LineEdit)
        self.labelLayout.addWidget(self.imaging_pair_selection_label)
        self.labelLayout.addWidget(self.imaging_pair_selection_comboBox)
        self.labelLayout.addWidget(self.DRR_type_label)
        self.labelLayout.addWidget(self.DRR_type_comboBox)
        self.labelLayout.addWidget(self.bone_threshold_label)
        self.labelLayout.addWidget(self.bone_threshold_LineEdit)
        self.labelLayout.addWidget(self.tumor_label_threshold_label)
        self.labelLayout.addWidget(self.tumor_label_threshold_LineEdit)
        self.labelLayout.addWidget(self.bone_suppress_method_label)
        self.labelLayout.addWidget(self.bone_suppress_method_combobox)
        self.labelLayout.addWidget(self.bone_replace_hu_label)
        self.labelLayout.addWidget(self.bone_replace_hu_LineEdit)
        self.labelLayout.addWidget(self.bone_enhance_factor_label)
        self.labelLayout.addWidget(self.bone_enhance_factor_LineEdit)
        self.labelLayout.addWidget(self.deltaI_label)
        self.labelLayout.addWidget(self.deltaI_LineEdit)
        self.imaging_pair_selection_comboBox.addItem('imaging pair 1')
        self.imaging_pair_selection_comboBox.addItem('imaging pair 2')
        self.imaging_pair_selection_comboBox.addItem('both imaging pairs')
        self.DRR_type_comboBox.addItem('normal DRR')
        self.DRR_type_comboBox.addItem('bone only DRR')
        self.DRR_type_comboBox.addItem('bone suppressed DRR')
        self.DRR_type_comboBox.addItem('bone enhanced DRR')
        self.bone_suppress_method_combobox.addItem("constant")
        #self.bone_suppress_method_combobox.addItem("dual energy")

        self.vBoxLayout.addLayout(self.sysMSGLayout, 1)
        self.sysMSGLayout.addWidget(self.sysMSGLabel)

        self.vBoxLayout.addLayout(self.DRR_display_area_layout, 1)
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.imaging_pair_1_display_label, alignment=Qt.AlignCenter)
        # 创建右半部分布局
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.imaging_pair_2_display_label, alignment=Qt.AlignCenter)
        # 将左右布局分别添加到 DRR 显示区域布局中
        self.DRR_display_area_layout.addLayout(left_layout)
        self.DRR_display_area_layout.addLayout(right_layout)
        # 初始化控件可见性
        self.update_control_visibility()

    def update_control_visibility(self):
        """根据 DRR 类型和骨抑制方法动态更新控件可见性"""
        drr_type = self.DRR_type_comboBox.currentText()
        bone_suppress_method = self.bone_suppress_method_combobox.currentText()

        # 默认隐藏所有相关控件
        self.bone_threshold_label.setVisible(False)
        self.bone_threshold_LineEdit.setVisible(False)
        self.tumor_label_threshold_label.setVisible(False)
        self.tumor_label_threshold_LineEdit.setVisible(False)
        self.bone_suppress_method_label.setVisible(False)
        self.bone_suppress_method_combobox.setVisible(False)
        self.bone_replace_hu_label.setVisible(False)
        self.bone_replace_hu_LineEdit.setVisible(False)
        self.bone_enhance_factor_label.setVisible(False)
        self.bone_enhance_factor_LineEdit.setVisible(False)
        self.deltaI_label.setVisible(False)
        self.deltaI_LineEdit.setVisible(False)

        # 根据 DRR 类型显示相应的控件
        if drr_type == 'normal DRR':
            # a. normal DRR: 全部隐藏（已在默认设置中完成）
            pass
        elif drr_type == 'bone only DRR':
            # b. bone only DRR: 显示 bone threshold
            self.bone_threshold_label.setVisible(True)
            self.bone_threshold_LineEdit.setVisible(True)
        elif drr_type == 'bone suppressed DRR':
            # c. bone suppressed DRR: 显示 bone threshold 和 bone suppress method
            self.bone_threshold_label.setVisible(True)
            self.bone_threshold_LineEdit.setVisible(True)
            self.bone_suppress_method_label.setVisible(True)
            self.bone_suppress_method_combobox.setVisible(True)
            if bone_suppress_method == 'constant':
                # c1. constant: 显示 bone replace hu
                self.bone_replace_hu_label.setVisible(True)
                self.bone_replace_hu_LineEdit.setVisible(True)
            elif bone_suppress_method == 'dual energy':
                # c2. dual energy: 显示 ΔKeV
                self.deltaI_label.setVisible(True)
                self.deltaI_LineEdit.setVisible(True)
        elif drr_type == 'bone enhanced DRR':
            # d. bone enhanced DRR: 显示 bone threshold 和 bone enhance factor
            self.bone_threshold_label.setVisible(True)
            self.bone_threshold_LineEdit.setVisible(True)
            self.bone_enhance_factor_label.setVisible(True)
            self.bone_enhance_factor_LineEdit.setVisible(True)

        if self.LabelGenCheckBox.isChecked():
            self.tumor_label_threshold_label.setVisible(True)
            self.tumor_label_threshold_LineEdit.setVisible(True)

    def showConfirmationDialog(self, sliceThickness, resolution, tileSize, couchAngle, imaging_pair, iso_x, iso_y,
                               iso_z):
        """
        显示确认对话框
        """
        self.bone_threshold_LineEdit.text()
        dialog = MessageBox(
            title="Confirm DRR Generation",
            content=(
                f"are you sure you want to generate DRR with the following parameters?\n\n"
                f"CT slice thickness: {sliceThickness} mm\t\t\t DRR resolution: {resolution}\t\t\t tile size: {tileSize}\n"
                f"couch angle: {couchAngle}\t\t\t imaging pair: {imaging_pair}\t\t\t ISO center: (x={iso_x}, y={iso_y}, z={iso_z})\n"
                f"DRR type: {self.DRR_type_comboBox.text()}"
            ),
            parent=self.parent()
        )

        if dialog.exec():
            return True  # 确认生成
        else:
            return False  # 取消生成

    def on_generateDRRClicked(self):
        # 判断分辨率、图像块大小、治疗床角度、成像对的选择是否符合规范, CT数据是否存在
        ctexists = get_var("ctexist")
        if not ctexists:
            self.sys_msg("no CT data imported yet!")
            return
        E0 = check_is_number(self.initial_I_LineEdit, 1)
        if E0 is None:
            self.sys_msg("Invalid I<sub>0<sub>.")
            return
        bitDepth = check_is_number(self.drr_bitDepth_LineEdit, 1)
        if bitDepth is None:
            self.sys_msg("Invalid bit depth.")
            return
        muWater = check_is_number(self.mu_water_LineEdit, 0)
        if muWater is None:
            self.sys_msg("Invalid μ<sub>water<sub>.")
            return
        muAir = check_is_number(self.mu_air_LineEdit, 0)
        if muAir is None:
            self.sys_msg("Invalid μ<sub>air<sub>.")
            return
        resolution = check_is_number(self.drr_resolution_LineEdit, 1)
        if resolution is None:
            self.sys_msg("Invalid DRR resolution.")
            return
        if resolution < 1 or resolution > 2000:
            self.sys_msg("DRR resolution should be between 1-2000.")
            return
        tileSize = check_is_number(self.tile_size_LineEdit, 1)
        if tileSize is None:
            self.sys_msg("Invalid tile size.")
            return
        if tileSize < 1 or tileSize > resolution:
            self.sys_msg(f"Invalid tile size. Tile size range: 1~{resolution}")
            return
        if resolution % tileSize != 0:
            self.sys_msg("Please make sure DRR resolution is divisible by the tile size.")
            return
        couchAngle = check_is_number(self.couch_angle_LineEdit, 0)
        if couchAngle is None:
            self.sys_msg("Invalid couch angle.")
            return
        imaging_pair = self.imaging_pair_selection_comboBox.currentText()

        # 判断成像对的几何信息已被正确录入
        G1_OK = get_var("Imaging_pair_1_enabled")
        G2_OK = get_var("Imaging_pair_2_enabled")
        if not G1_OK and not G2_OK:
            self.sys_msg("Please go to setting page to confirm imaging geometry information first.")
            return
        elif not G1_OK and (imaging_pair == 'imaging pair 1' or imaging_pair == 'both imaging pairs'):
            self.sys_msg("Imaging pair 1 is not ready, please go to setting page to check and confirm again.")
            return
        elif not G2_OK and (imaging_pair == 'imaging pair 2' or imaging_pair == 'both imaging pairs'):
            self.sys_msg("Imaging pair 2 is not ready, please go to setting page to check and confirm again.")
            return

        # 获取 ISO 中心坐标
        iso_x = get_var("ISO_X")
        iso_y = get_var("ISO_Y")
        iso_z = get_var("ISO_Z")
        if iso_x is None:
            self.sys_msg("ISO center not determined yet!.")
            return
        sliceThickness = get_var("SliceThickness")

        NoMoreAsking = get_var("NoMoreAsking")
        # 如果 "no more asking" 未选中，显示确认对话框
        if not NoMoreAsking:
            confirmed = self.showConfirmationDialog(
                sliceThickness, resolution, tileSize, couchAngle, imaging_pair, iso_x, iso_y, iso_z,
            )
            if not confirmed:
                self.sys_msg("DRR generation canceled.")
                return  # 用户取消生成
        # 开始生成
        DRR_type = self.DRR_type_comboBox.text()
        Geoinfo_save_path = get_var("Geoinfo_save_path")

        # 启动多线程
        buttonClickWithFloatingWindow(self, 'Generating DRR', 'It may take few minutes', 'generate DRR', 1, self.generateDRRButton)
        """
        try:
            if imaging_pair == 'imaging pair 1':
                completed, DRR = self.imaging_pair1_DRR_generation(I0, muWater, muAir, Geoinfo_save_path, resolution, bitDepth, tileSize,
                                     couchAngle, iso_x, iso_y, iso_z, sliceThickness, DRR_type)
                msg = f"DRR generation for imaging pair 1 completed! Automatically saved at {Geoinfo_save_path}\saved DRR" if completed else "Failed to generate DRR for imaging pair 1."
                if completed:
                    TempDRR = np.zeros((512, 512, 3))
                    TempDRR[:, :, 0] = cv2.resize(DRR, (512, 512))
                    set_var("Left_DRR", TempDRR)
                    set_var("DRREXISTS_Left", 1)
            elif imaging_pair == 'imaging pair 2':
                completed, DRR = self.imaging_pair2_DRR_generation(I0, muWater, muAir, Geoinfo_save_path, resolution, bitDepth, tileSize,
                                     couchAngle, iso_x, iso_y, iso_z, sliceThickness, DRR_type)
                msg = f"DRR generation for imaging pair 2 completed! Automatically saved at {Geoinfo_save_path}\saved DRR" if completed else "Failed to generate DRR for imaging pair 2."
                if completed:
                    TempDRR = np.zeros((512, 512, 3))
                    TempDRR[:, :, 0] = cv2.resize(DRR, (512, 512))
                    set_var("Right_DRR", TempDRR)
                    set_var("DRREXISTS_Right", 1)
            elif imaging_pair == 'both imaging pairs':
                completed1, DRR1 = self.imaging_pair1_DRR_generation(I0, muWater, muAir, Geoinfo_save_path, resolution, bitDepth, tileSize,
                                     couchAngle, iso_x, iso_y, iso_z, sliceThickness, DRR_type)
                if completed1:
                    TempDRR = np.zeros((512, 512, 3))
                    TempDRR[:, :, 0] = cv2.resize(DRR1, (512, 512))
                    set_var("Left_DRR", TempDRR)
                    set_var("DRREXISTS_Left", 1)
                completed2, DRR2 = self.imaging_pair2_DRR_generation(I0, muWater, muAir, Geoinfo_save_path, resolution, bitDepth, tileSize,
                                     couchAngle, iso_x, iso_y, iso_z, sliceThickness, DRR_type)
                if completed2:
                    TempDRR = np.zeros((512, 512, 3))
                    TempDRR[:, :, 0] = cv2.resize(DRR2, (512, 512))
                    set_var("Right_DRR", TempDRR)
                    set_var("DRREXISTS_Right", 1)
                msg = f"Both DRRs generation completed! Automatically saved at {Geoinfo_save_path}\saved DRR" if completed1 and completed2 else \
                      f"DRR generation partially failed! Automatically saved at {Geoinfo_save_path}\saved DRR" if completed1 or completed2 else \
                      "Both DRRs generation failed!"
        except Exception as e:
            msg = f"Error during DRR generation: {str(e)}"
        self.sys_msg(msg)
        """
        self.drr_thread = DRRGenerationThread(E0, muWater, muAir,
            imaging_pair, Geoinfo_save_path, resolution, tileSize, couchAngle,
            iso_x, iso_y, iso_z, sliceThickness, DRR_type, bitDepth, parent=self
        )
        self.drr_thread.progress_signal.connect(self.sys_msg)  # 连接信号到 sys_msg 方法
        self.drr_thread.finished.connect(self.on_drr_generation_finished)
        self.drr_thread.start()  # 启动线程
        self.sys_msg("DRR generation started in the background...")
        self.generateDRRButton.setVisible(False)

    def on_drr_generation_finished(self):
        """当 DRR 生成任务完成时调用"""
        # 显示生成完成的浮动窗口
        buttonClickWithFloatingWindow(self, 'DRR generation completed!', 'you can check DRR now', 'generate DRR', 0, self.generateDRRButton)
        update_display_drr(self.parent())
        self.generateDRRButton.setVisible(True)

    def imaging_pair1_DRR_generation(self, E0, muWater, muAir, Geoinfo_save_path, resolution, bitDepth, tileSize,
                                     couchAngle, iso_x, iso_y, iso_z, sliceThickness, DRR_type):
        c = False
        labelexits = get_var('labelexits')
        DRR = np.zeros((resolution, resolution))
        Label = np.zeros((resolution, resolution))
        save_name = get_var("Imaging_pair_1_fileName")
        file_path = os.path.join(Geoinfo_save_path, save_name + '.json')

        if not os.path.exists(file_path):
            self.sys_msg(f'error, file {file_path} does not exist.')
            return c, DRR

        with open(file_path, 'r') as file:
            data = json.load(file)

        x = float(data.get("x", ""))
        y = float(data.get("y", ""))
        z = float(data.get("z", ""))
        IPEL = float(data.get("IPEL", ""))
        OID = float(data.get("OID", ""))

        if DRR_type == 'normal DRR':
            DRR = getDRR(x, y, z, IPEL, OID, resolution, tileSize, couchAngle, iso_x, iso_y,
                         iso_z, sliceThickness, save_name, Geoinfo_save_path, muWater, muAir, bitDepth)

        elif DRR_type == 'bone only DRR':
            bone_threshold = check_is_number(self.bone_threshold_LineEdit, 1)
            ct_max = get_var("CT_MAX_HU")
            if bone_threshold is None or -1000 > bone_threshold or bone_threshold > ct_max:
                self.sys_msg(f'error, bone threshold should be within -1000~{ct_max}, got {bone_threshold} instead.')
                return c, DRR
            DRR = get_bone_only_DRR(x, y, z, IPEL, OID, resolution, tileSize, couchAngle, iso_x, iso_y,
                                    iso_z, sliceThickness, save_name, Geoinfo_save_path, bone_threshold, muWater,
                                    muAir, bitDepth)

        elif DRR_type == 'bone suppressed DRR' and self.bone_suppress_method_combobox.text() == "constant":
            bone_threshold = check_is_number(self.bone_threshold_LineEdit, 1)
            ct_max = get_var("CT_MAX_HU")
            constant = check_is_number(self.bone_replace_hu_LineEdit, 0)

            if bone_threshold is None or -1000 > bone_threshold or bone_threshold > ct_max:
                self.sys_msg(f'error, bone threshold should be within -1000~{ct_max}, got {bone_threshold} instead.')
                return c, DRR

            if constant is None or -1000 > constant or constant > ct_max:
                self.sys_msg(f'error, bone replace hu should be within -1000~{ct_max}, got {constant} instead.')
                return c, DRR

            DRR = get_bone_suppressed_DRR_constant(x, y, z, IPEL, OID, resolution, tileSize, couchAngle, iso_x, iso_y,
                                                   iso_z, sliceThickness, save_name, Geoinfo_save_path, bone_threshold,
                                                   muWater, muAir, bitDepth, constant)

        elif DRR_type == 'bone enhanced DRR':
            bone_threshold = check_is_number(self.bone_threshold_LineEdit, 1)
            enhance_factor = check_is_number(self.bone_enhance_factor_LineEdit, 0)
            ct_max = get_var("CT_MAX_HU")
            if bone_threshold is None or -1000 > bone_threshold or bone_threshold > ct_max:
                self.sys_msg(f'error, bone threshold should be within -1000~{ct_max}, got {bone_threshold} instead.')
                return c, DRR
            DRR = get_bone_enhanced_DRR(x, y, z, IPEL, OID, resolution, tileSize, couchAngle, iso_x, iso_y,
                                                   iso_z, sliceThickness, save_name, Geoinfo_save_path, bone_threshold,
                                                   muWater, muAir, bitDepth, enhance_factor)
        if labelexits:
            LabelGen = self.LabelGenCheckBox.isChecked()
            if LabelGen:
                threshold = check_is_number(self.tumor_label_threshold_LineEdit, 1)
                Label = getLabel(x, y, z, IPEL, OID, resolution, tileSize, couchAngle, iso_x, iso_y,
                                 iso_z, sliceThickness, save_name, Geoinfo_save_path, threshold)
        c = True
        return c, DRR, Label

    '''
    elif DRR_type == 'bone suppressed DRR' and self.bone_suppress_method_combobox.text() == "dual energy":
        deltaI = check_is_number(self.deltaI_LineEdit, 1)

        if deltaI is None:
            self.sys_msg(f'error, invalid ΔKeV.')
            return c, DRR

        energy_values = np.array(list(interpolated_data["water"].keys()))
        if not (energy_values.min() <= E0 - deltaI <= energy_values.max()):
            self.sys_msg(f'out of range error, invalid ΔKeV.')
            return c, DRR

        DRR = get_bone_suppressed_DRR_dual_energy(x, y, z, IPEL, OID, resolution, tileSize, couchAngle, iso_x,
                                                  iso_y,
                                                  iso_z, sliceThickness, save_name, Geoinfo_save_path, muWater,
                                                  muAir, bitDepth, deltaI, interpolated_data)
    '''

    def imaging_pair2_DRR_generation(self, E0, muWater, muAir, Geoinfo_save_path, resolution, bitDepth, tileSize,
                                     couchAngle, iso_x, iso_y, iso_z, sliceThickness, DRR_type):
        c = False
        labelexits = get_var('labelexits')
        DRR = np.zeros((resolution, resolution))
        Label = np.zeros((resolution, resolution))
        save_name = get_var("Imaging_pair_2_fileName")
        file_path = os.path.join(Geoinfo_save_path, save_name + '.json')
        if not os.path.exists(file_path):
            self.sys_msg(f'error, file {file_path} does not exist.')
            return c, DRR
        with open(file_path, 'r') as file:
            data = json.load(file)
        x = float(data.get("x", ""))
        y = float(data.get("y", ""))
        z = float(data.get("z", ""))
        IPEL = float(data.get("IPEL", ""))
        OID = float(data.get("OID", ""))
        if DRR_type == 'normal DRR':
            DRR = getDRR(x, y, z, IPEL, OID, resolution, tileSize, couchAngle, iso_x, iso_y,
                         iso_z, sliceThickness, save_name, Geoinfo_save_path, muWater, muAir, bitDepth)
        elif DRR_type == 'bone only DRR':
            bone_threshold = check_is_number(self.bone_threshold_LineEdit, 1)
            ct_max = get_var("CT_MAX_HU")
            if bone_threshold is None or -1000 > bone_threshold or bone_threshold > ct_max:
                self.sys_msg(f'error, bone threshold should be within -1000~{ct_max}, got {bone_threshold} instead.')
                c = False
                return c, DRR
            DRR = get_bone_only_DRR(x, y, z, IPEL, OID, resolution, tileSize, couchAngle, iso_x, iso_y,
                                    iso_z, sliceThickness, save_name, Geoinfo_save_path, bone_threshold, muWater, muAir, bitDepth)
        elif DRR_type == 'bone suppressed DRR' and self.bone_suppress_method_combobox.text() == "constant":
            bone_threshold = check_is_number(self.bone_threshold_LineEdit, 1)
            ct_max = get_var("CT_MAX_HU")
            constant = check_is_number(self.bone_replace_hu_LineEdit, 0)
            if bone_threshold is None or -1000 > bone_threshold or bone_threshold > ct_max:
                self.sys_msg(f'error, bone threshold should be within -1000~{ct_max}, got {bone_threshold} instead.')
                c = False
                return c, DRR
            if constant is None or -1000 > constant or constant > ct_max:
                self.sys_msg(f'error, bone replace hu should be within -1000~{ct_max}, got {constant} instead.')
                c = False
                return c, DRR
            DRR = get_bone_suppressed_DRR_constant(x, y, z, IPEL, OID, resolution, tileSize, couchAngle, iso_x, iso_y,
                                    iso_z, sliceThickness, save_name, Geoinfo_save_path, bone_threshold, muWater,
                                    muAir, bitDepth, constant)
        elif DRR_type == 'bone enhanced DRR':
            bone_threshold = check_is_number(self.bone_threshold_LineEdit, 1)
            enhance_factor = check_is_number(self.bone_enhance_factor_LineEdit, 0)
            ct_max = get_var("CT_MAX_HU")
            if bone_threshold is None or -1000 > bone_threshold or bone_threshold > ct_max:
                self.sys_msg(f'error, bone threshold should be within -1000~{ct_max}, got {bone_threshold} instead.')
                return c, DRR
            DRR = get_bone_enhanced_DRR(x, y, z, IPEL, OID, resolution, tileSize, couchAngle, iso_x, iso_y,
                                                   iso_z, sliceThickness, save_name, Geoinfo_save_path, bone_threshold,
                                                   muWater, muAir, bitDepth, enhance_factor)
        if labelexits:
            LabelGen = self.LabelGenCheckBox.isChecked()
            if LabelGen:
                threshold = check_is_number(self.tumor_label_threshold_LineEdit, 1)
                Label = getLabel(x, y, z, IPEL, OID, resolution, tileSize, couchAngle, iso_x, iso_y,
                                 iso_z, sliceThickness, save_name, Geoinfo_save_path, threshold)
        c = True
        return c, DRR, Label

    def sys_msg(self, msg):
        # self.sysMSGLabel.setText(self.tr(f'system message: {msg}'))
        self.sysMSGLabel.setText(self.tr(f'<b>system message: {msg}</b>'))

    def toggle_editing(self, state):
        if state == Qt.Checked:
            self.mu_water_LineEdit.setReadOnly(False)
            self.mu_air_LineEdit.setReadOnly(False)
        else:
            self.mu_water_LineEdit.setReadOnly(True)
            self.mu_air_LineEdit.setReadOnly(True)

    def update_mu_values(self):
        try:
            # 获取 initial_I 的值
            initial_I = int(self.initial_I_LineEdit.text())

            # 查找 interpolated_data 对应的值
            mu_water = interpolated_data["water"].get(initial_I, None)
            mu_air = interpolated_data["air"].get(initial_I, None)

            if mu_water is not None:
                self.mu_water_LineEdit.setText(f"{mu_water:.5f}")
            else:
                self.mu_water_LineEdit.setText("N/A")  # 若无对应值则显示N/A

            if mu_air is not None:
                self.mu_air_LineEdit.setText(f"{mu_air:.5f}")
            else:
                self.mu_air_LineEdit.setText("N/A")  # 若无对应值则显示N/A
        except ValueError:
            # 当输入值无效时，清空对应字段
            self.mu_water_LineEdit.setText("")
            self.mu_air_LineEdit.setText("")


class Example2Card(QWidget):
    """ Example2 card """

    def __init__(self, container, left_widget: QWidget, right_widget: QWidget, parent1, stretch=0, parent0=None, ):
        super().__init__(parent=parent0)
        self.leftWidget = left_widget
        self.rightWidget = right_widget
        self.stretch = stretch
        self.container = container
        self.parent1 = parent1
        self.card = QFrame(self)

        self.vBoxLayout = QVBoxLayout(self)
        self.cardLayout = QHBoxLayout(self.card)

        # 使 left_widget 和 right_widget 接收鼠标事件
        self.leftWidget.setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.rightWidget.setAttribute(Qt.WA_AcceptTouchEvents, True)

        self.leftWidget.installEventFilter(self)
        self.rightWidget.installEventFilter(self)

        self.__initWidget()

    def __initWidget(self):
        self.__initLayout()
        self.card.setObjectName('card')

    def __initLayout(self):
        self.vBoxLayout.setSizeConstraint(QVBoxLayout.SetMinimumSize)
        self.cardLayout.setSizeConstraint(QHBoxLayout.SetMinimumSize)

        self.vBoxLayout.setSpacing(0)
        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.cardLayout.setContentsMargins(0, 0, 0, 0)
        self.vBoxLayout.addWidget(self.card, 0, Qt.AlignTop)
        self.vBoxLayout.setAlignment(Qt.AlignTop)

        self.cardLayout.setSpacing(0)
        self.cardLayout.setAlignment(Qt.AlignTop)
        self.cardLayout.addWidget(self.leftWidget)
        self.cardLayout.addWidget(self.rightWidget)

        if self.stretch == 0:
            self.cardLayout.addStretch(1)

        self.leftWidget.show()
        self.rightWidget.show()

    def eventFilter(self, source, event):
        """捕获鼠标滚轮事件"""
        drrexist_left = get_var('DRREXISTS_Left')
        drrexist_right = get_var('DRREXISTS_Right')
        if drrexist_left or drrexist_right:
            if event.type() == QEvent.Wheel:
                delta = event.angleDelta().y() // 120  # 每次滚动幅度
                if source == self.leftWidget or source == self.rightWidget:
                    current_slice = get_var("CurrentDRRSlice")
                    new_slice = max(1, current_slice + delta)  # 确保切片索引为正数
                    new_slice = min(3, new_slice)
                    set_var("CurrentDRRSlice", new_slice)
                    update_display_drr(self.container)  # 更新显示
        return super().eventFilter(source, event)


class DRRGalleryInterface(ScrollArea):
    """ Gallery interface """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.view = QWidget(self)
        self.toolBar = ToolBar(self)
        self.vBoxLayout = QVBoxLayout(self.view)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setViewportMargins(0, self.toolBar.height(), 0, 0)
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.vBoxLayout.setSpacing(5)
        self.vBoxLayout.setAlignment(Qt.AlignTop)
        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.view.setObjectName('view')
        StyleSheet.GALLERY_INTERFACE.apply(self)

    def add2HExampleCard(self, parent, left_widget, right_widget, stretch=0):
        card = Example2Card(parent, left_widget, right_widget, self.toolBar, stretch, parent0=self.view)
        self.vBoxLayout.addWidget(card, 0, Qt.AlignTop)
        return card

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.toolBar.resize(self.width(), self.toolBar.height())


# 原始数据（MeV）
data_mev = {
    "water": {
        "1.00000E-03": 4.078E+03,
        "1.50000E-03": 1.376E+03,
        "2.00000E-03": 6.173E+02,
        "3.00000E-03": 1.929E+02,
        "4.00000E-03": 8.278E+01,
        "5.00000E-03": 4.258E+01,
        "6.00000E-03": 2.464E+01,
        "8.00000E-03": 1.037E+01,
        "1.00000E-02": 5.329E+00,
        "1.50000E-02": 1.673E+00,
        "2.00000E-02": 8.096E-01,
        "3.00000E-02": 3.756E-01,
        "4.00000E-02": 2.683E-01,
        "5.00000E-02": 2.269E-01,
        "6.00000E-02": 2.059E-01,
        "8.00000E-02": 1.837E-01,
        "1.00000E-01": 1.707E-01,
        "1.50000E-01": 1.505E-01,
        "2.00000E-01": 1.370E-01,
        "3.00000E-01": 1.186E-01,
        "4.00000E-01": 1.061E-01,
        "5.00000E-01": 9.687E-02,
        "6.00000E-01": 8.956E-02,
        "8.00000E-01": 7.865E-02,
        "1.00000E+00": 7.072E-02,
        "1.25000E+00": 6.323E-02,
        "1.50000E+00": 5.754E-02,
        "2.00000E+00": 4.942E-02,
        "3.00000E+00": 3.969E-02,
        "4.00000E+00": 3.403E-02,
        "5.00000E+00": 3.031E-02,
        "6.00000E+00": 2.770E-02,
        "8.00000E+00": 2.429E-02,
        "1.00000E+01": 2.219E-02,
        "1.50000E+01": 1.941E-02,
        "2.00000E+01": 1.813E-02
    },
    "air": {
        "1.00000E-03": 3.606E+03,
        "1.50000E-03": 1.191E+03,
        "2.00000E-03": 5.279E+02,
        "3.00000E-03": 1.625E+02,
        "3.20290E-03": 1.340E+02,
        "4.00000E-03": 7.788E+01,
        "5.00000E-03": 4.027E+01,
        "6.00000E-03": 2.341E+01,
        "8.00000E-03": 9.921E+00,
        "1.00000E-02": 5.120E+00,
        "1.50000E-02": 1.614E+00,
        "2.00000E-02": 7.779E-01,
        "3.00000E-02": 3.538E-01,
        "4.00000E-02": 2.485E-01,
        "5.00000E-02": 2.080E-01,
        "6.00000E-02": 1.875E-01,
        "8.00000E-02": 1.662E-01,
        "1.00000E-01": 1.541E-01,
        "1.50000E-01": 1.356E-01,
        "2.00000E-01": 1.233E-01,
        "3.00000E-01": 1.067E-01,
        "4.00000E-01": 9.549E-02,
        "5.00000E-01": 8.712E-02,
        "6.00000E-01": 8.055E-02,
        "8.00000E-01": 7.074E-02,
        "1.00000E+00": 6.358E-02,
        "1.25000E+00": 5.687E-02,
        "1.50000E+00": 5.175E-02,
        "2.00000E+00": 4.447E-02,
        "3.00000E+00": 3.581E-02,
        "4.00000E+00": 3.079E-02,
        "5.00000E+00": 2.751E-02,
        "6.00000E+00": 2.522E-02,
        "8.00000E+00": 2.225E-02,
        "1.00000E+01": 2.045E-02,
        "1.50000E+01": 1.810E-02,
        "2.00000E+01": 1.705E-02
    }
}

data_kev = {key: {float(k) * 1e3: v for k, v in value.items()} for key, value in data_mev.items()}

# 进行插值
energies_kev = np.linspace(1, 20000, 20000)  # 1 keV 到 20000 keV
rho_air = 0.001293  # 空气密度 g/cm³
rho_water = 1.0  # 水密度 g/cm³

interpolated_data = {}
for material, values in data_kev.items():
    energies = np.array(list(values.keys()))
    mu_rho = np.array(list(values.values()))
    interpolation_function = interp1d(energies, mu_rho, kind='cubic', fill_value="extrapolate")
    interpolated_mu_rho = interpolation_function(energies_kev)
    # 转换为线衰减系数 (cm⁻¹)
    if material == "air":
        interpolated_mu = interpolated_mu_rho * rho_air
    else:  # material == "water"
        interpolated_mu = interpolated_mu_rho * rho_water
    interpolated_data[material] = dict(zip(energies_kev, interpolated_mu))
