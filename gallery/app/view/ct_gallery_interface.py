# coding:utf-8
from PyQt5.QtCore import Qt, QEvent, QThread, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFrame, QFileDialog
from qfluentwidgets import (ScrollArea, FlowLayout, PushButton, ToolButton, FluentIcon, LineEdit,
                            isDarkTheme, ToolTipFilter, BodyLabel, toggleTheme)
from ..common.style_sheet import StyleSheet
from ..common.icon import Icon
from ..var.globals import set_var, get_var
from ..functions.import_ct import LoadCT
from ..functions.ButtonClickChange import buttonClickWithFloatingWindow
from ..functions.update_ct_labels import update_ct_info, update_slice
from scipy.ndimage import zoom
from ..functions.contour_match_interpolation import contour_matching_interpolation
from ..functions.ct_display import update_display_ct
from .drr_gallery_interface import check_is_number
import time
import numpy as np
from ..functions.ct_display import convert_array_to_pixmap_window_WL


def draw_iso_center(x, y, z, m=2):
    pixelGrid = get_var("PixelsGrid")
    MAX = pixelGrid.max()
    sliceNum = get_var("SliceNum")
    if 1 <= z <= sliceNum:
        ct_iso_check = pixelGrid[:, :, z - 1]
        del pixelGrid

        ct_resolution = ct_iso_check.shape
        if 0 < x <= ct_resolution[0] and 0 < y <= ct_resolution[1]:
            # 画水平直线
            ct_iso_check[:, x - m // 2:x + m // 2] = MAX

            # 画垂直直线
            ct_iso_check[y - m // 2:y + m // 2, :] = MAX

            return ct_iso_check
        else:
            return None
    else:
        return None


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


class ToolBar(QWidget):
    """ Toolbar """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.themeButton = ToolButton(FluentIcon.CONSTRACT, self)
        self.themeButton.setFixedHeight(28)
        self.importCTButton = PushButton(self.tr('import CT'), self, Icon.IMPORT)
        self.importCTButton.setFixedHeight(28)
        self.importCTButton.clicked.connect(self.on_importCtClicked)

        self.window_width_label = BodyLabel(self.tr('window width: '), self)
        self.window_level_label = BodyLabel(self.tr('window level: '), self)
        self.window_width_LineEdit = LineEdit(self)
        self.window_width_LineEdit.setFixedSize(90, 28)
        self.window_width_LineEdit.setText(self.tr('1500'))
        self.window_width_LineEdit.setClearButtonEnabled(True)
        self.window_level_LineEdit = LineEdit(self)
        self.window_level_LineEdit.setFixedSize(90, 28)
        self.window_level_LineEdit.setText(self.tr('-500'))
        self.window_level_LineEdit.setClearButtonEnabled(True)
        self.window_WL_refresh_Btn = PushButton(self.tr('refresh'), self, Icon.REFRESH)
        self.window_WL_refresh_Btn.setFixedHeight(28)
        self.window_WL_refresh_Btn.clicked.connect(self.on_refreshWLClicked)

        self.interpolationBtn = PushButton(self.tr('interpolation'), self, Icon.LAYERS)
        self.interpolationBtn.setFixedHeight(28)
        self.interpolationBtn.clicked.connect(self.on_interpolationClicked)

        self.CTInfoLabel = BodyLabel(self.tr('CT information'), self)

        self.pixelSpacingLabel = BodyLabel(self.tr('pixel spacing: - (mm)     '), self)
        self.sliceThicknessLabel = BodyLabel(self.tr('slice thickness: - (mm)      '), self)
        self.labelDataExistLabel = BodyLabel(self.tr('tumor contour exists: No     '), self)
        self.patientNameLabel = BodyLabel(self.tr('patient name: -      '), self)
        self.currentSliceLabel = BodyLabel(self.tr('current slice: / '), self)
        self.jump2sliceBtn = PushButton(self.tr('jump'), self)
        self.jump2sliceBtn.clicked.connect(self.on_jumpSliceClicked)
        self.jumpSlice_LineEdit = LineEdit(self)
        self.jumpSlice_LineEdit.setFixedSize(64, 28)
        self.jumpSlice_LineEdit.returnPressed.connect(self.jump2sliceBtn.click)
        self.ISOCenterXLabel = BodyLabel(self.tr('ISO center x:'), self)
        self.ISOCenterX_LineEdit = LineEdit(self)
        self.ISOCenterX_LineEdit.setToolTip(self.tr('unit: pixel'))
        self.ISOCenterX_LineEdit.setFixedSize(64, 28)
        self.ISOCenterYLabel = BodyLabel(self.tr('y:'), self)
        self.ISOCenterY_LineEdit = LineEdit(self)
        self.ISOCenterY_LineEdit.setFixedSize(64, 28)
        self.ISOCenterY_LineEdit.setToolTip(self.tr('unit: pixel'))
        self.ISOCenterZLabel = BodyLabel(self.tr('z:'), self)
        self.ISOCenterZ_LineEdit = LineEdit(self)
        self.ISOCenterZ_LineEdit.setFixedSize(64, 28)
        self.ISOCenterZ_LineEdit.setToolTip(self.tr('unit: slice'))
        self.isoConfirmBtn = PushButton(self.tr('confirm ISO'), self)
        self.isoConfirmBtn.clicked.connect(self.on_checkISOClicked)

        self.sysMSGLabel = BodyLabel(self.tr(f'system message: import CT to satrt.'), self)
        self.separator = SeparatorWidget(self)

        self.vBoxLayout = QVBoxLayout(self)
        self.buttonLayout = FlowLayout()
        self.labelTitleLayout = FlowLayout()
        self.labelLayout = FlowLayout()
        self.sysMSGLayout = FlowLayout()
        self.__initWidget()

    def __initWidget(self):
        # Button layout
        self.setFixedHeight(200)
        self.vBoxLayout.addSpacing(0)
        self.vBoxLayout.addLayout(self.buttonLayout, 1)
        self.vBoxLayout.addLayout(self.labelTitleLayout, 1)
        self.vBoxLayout.addLayout(self.labelLayout, 1)
        self.vBoxLayout.addLayout(self.sysMSGLayout, 1)
        self.vBoxLayout.setAlignment(Qt.AlignTop)
        self.buttonLayout.addWidget(self.themeButton)
        self.buttonLayout.addWidget(self.importCTButton)
        self.buttonLayout.addWidget(self.separator)
        self.buttonLayout.addWidget(self.window_width_label)
        self.buttonLayout.addWidget(self.window_width_LineEdit)
        self.buttonLayout.addWidget(self.window_level_label)
        self.buttonLayout.addWidget(self.window_level_LineEdit)
        self.buttonLayout.addWidget(self.window_WL_refresh_Btn)
        self.buttonLayout.addWidget(self.interpolationBtn)
        self.buttonLayout.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.themeButton.installEventFilter(ToolTipFilter(self.themeButton))
        self.themeButton.setToolTip(self.tr('Toggle theme'))
        self.themeButton.clicked.connect(lambda: toggleTheme(True, False))

        # Label title layout
        self.labelTitleLayout.addWidget(self.CTInfoLabel)

        # Label layout
        self.labelLayout.addWidget(self.pixelSpacingLabel)
        self.labelLayout.addWidget(self.sliceThicknessLabel)
        self.labelLayout.addWidget(self.labelDataExistLabel)
        self.labelLayout.addWidget(self.patientNameLabel)
        self.labelLayout.addWidget(self.currentSliceLabel)
        self.labelLayout.addWidget(self.jumpSlice_LineEdit)
        self.labelLayout.addWidget(self.jump2sliceBtn)
        self.labelLayout.addWidget(self.ISOCenterXLabel)
        self.labelLayout.addWidget(self.ISOCenterX_LineEdit)
        self.labelLayout.addWidget(self.ISOCenterYLabel)
        self.labelLayout.addWidget(self.ISOCenterY_LineEdit)
        self.labelLayout.addWidget(self.ISOCenterZLabel)
        self.labelLayout.addWidget(self.ISOCenterZ_LineEdit)
        self.labelLayout.addWidget(self.isoConfirmBtn)

        # system message layout
        self.sysMSGLayout.addWidget(self.sysMSGLabel)

    def sys_msg(self, msg):
        self.sysMSGLabel.setText(self.tr(f'<b>system message: {msg}</b>'))

    def on_checkISOClicked(self):
        ctexists = get_var("ctexist")
        if not ctexists:
            self.sys_msg("not import CT yet.")
            return
        x = check_is_number(self.ISOCenterX_LineEdit, 1)
        y = check_is_number(self.ISOCenterY_LineEdit, 1)
        z = check_is_number(self.ISOCenterZ_LineEdit, 1)
        if all(parameter is not None for parameter in [x, y, z]):
            iso_check = draw_iso_center(x, y, z)
            if iso_check is not None:
                p = self.parent()
                iso_check = convert_array_to_pixmap_window_WL(iso_check)
                p.ct_updateLeftImage(iso_check)
                set_var("ISO_X", x)
                set_var("ISO_Y", y)
                set_var("ISO_Z", z)
                self.sys_msg("ISO center updated, please check if the ISO center is correct below.")
            else:
                self.sys_msg("please input valid ISO center first.")
        else:
            self.sys_msg("please input valid ISO center first.")

    def on_importCtClicked(self):
        """ Open file dialog to select CT folder """
        folder = QFileDialog.getExistingDirectory(self,
                                                  self.tr("Please select the folder where the CT data is stored."))
        if folder:
            self.sys_msg('importing CT data, please wait a moment')
            time.sleep(0.1)
            msg = LoadCT(folder)
            self.sys_msg(msg)
            set_var("CurrentSlice", 1)
            update_display_ct(self.parent())
            update_ct_info(self)
            update_slice(self)
        else:
            self.sys_msg('Folder selection canceled')

    def on_refreshWLClicked(self):
        MaxWindowWidth = get_var("MaxWindowWidth")
        MaxWindowLevel = get_var("MaxWindowLevel")
        MinWindowLevel = get_var("MinWindowLevel")
        try:
            ww = int(self.window_width_LineEdit.text())
            wl = int(self.window_level_LineEdit.text())
            if 0 < ww <= MaxWindowWidth and MinWindowLevel <= wl <= MaxWindowLevel:
                set_var("WindowWidth", int(self.window_width_LineEdit.text()))
                set_var("WindowLevel", int(self.window_level_LineEdit.text()))
                update_display_ct(self.parent())
                self.sys_msg('window width/level loaded.')
            else:
                self.sys_msg('invalid window width/level.')
        except ValueError:
            self.sys_msg("invalid window width/level.")

    def on_jumpSliceClicked(self):
        totalSliceNum = get_var("SliceNum")
        try:
            TargetSlice = int(self.jumpSlice_LineEdit.text())
            if 1 <= TargetSlice <= totalSliceNum:
                set_var("CurrentSlice", TargetSlice)
                update_display_ct(self.parent())
                update_slice(self)
                self.sys_msg('jumped to target slice.')
            else:
                self.sys_msg("trying to jump to invalid target slice.")
        except ValueError:
            self.sys_msg("invalid slice number.")

    def on_interpolationClicked(self):
        # 检查数据是否存在
        ctexists = get_var("ctexist")
        labelexists = get_var("labelexits")
        if not ctexists:
            self.sys_msg("not import CT yet.")
            return
        else:
            CTData = get_var("PixelsGrid")
        self.sys_msg("processing interpolation, please wait for a moment.")
        # 获取必要参数
        CurrentSliceThickness = get_var("SliceThickness")

        if CurrentSliceThickness == 1:
            self.sys_msg('current slice thickness is already 1mm.')
            return
        # 计算插值比例
        interpolation_factor = CurrentSliceThickness / 1.0  # 将层厚从 N mm 插值为 1 mm
        # 获取原始数据的尺寸信息
        original_shape = CTData.shape
        if len(original_shape) != 3:
            self.sys_msg("Invalid CT data, expected 3D array!")
            return

        # CT数据的双线性插值
        NewCTData = zoom(CTData, (1, 1, interpolation_factor), order=1)  # z轴插值
        set_var("PixelsGrid", NewCTData)

        # 更新层数
        NewSliceNum = NewCTData.shape[2]
        # 存储新数据
        set_var("SliceThickness", 1)
        set_var("SliceNum", NewSliceNum)

        # 肿瘤标签数据的插值
        if not labelexists:
            self.sys_msg("tumor contour data not found, proceeding with CT data only.")
        else:
            TumorLabelData = get_var("labeldata")
            NewTumorLabelData = contour_matching_interpolation(TumorLabelData, CurrentSliceThickness, 1)
            set_var("labeldata", NewTumorLabelData)
        # 更新显示
        update_display_ct(self.parent())
        update_ct_info(self)
        update_slice(self)
        self.sys_msg("interpolation completed!")


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
        ctexist = get_var('ctexist')
        lbctexist = get_var('labelexits')
        sliceNum = get_var("SliceNum")
        if ctexist or lbctexist:
            if event.type() == QEvent.Wheel:
                delta = event.angleDelta().y() // 120  # 每次滚动幅度
                if source == self.leftWidget or source == self.rightWidget:
                    current_slice = get_var("CurrentSlice")
                    new_slice = max(1, current_slice + delta)  # 确保切片索引为正数
                    new_slice = min(sliceNum, new_slice)
                    set_var("CurrentSlice", new_slice)
                    update_display_ct(self.container)  # 更新显示
                    update_slice(self.parent1)
        return super().eventFilter(source, event)


class CTGalleryInterface(ScrollArea):
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
