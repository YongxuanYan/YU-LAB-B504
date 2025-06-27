# coding:utf-8
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFrame, QMessageBox, QFileDialog
from qfluentwidgets import (ScrollArea, FlowLayout, PushButton, ToolButton, FluentIcon, LineEdit,
                            isDarkTheme, ToolTipFilter, BodyLabel, toggleTheme, ComboBox)
from ..common.style_sheet import StyleSheet
from ..common.icon import Icon
import os
import json
import re
from ..var.globals import set_var


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


def check_is_number(line_edit):
    try:
        value = float(line_edit.text())
        return value
    except ValueError:
        return None


class ToolBar(QWidget):
    """ Toolbar """

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.save_path_settings = QSettings("YU LAB-B504", "save path")  # 保存用户设置
        # 检查是否已有保存路径，如果没有则初始化为默认路径
        saved_path = self.save_path_settings.value("save path")
        if saved_path:
            self.save_path = saved_path
        else:
            # 获取当前用户的文档路径
            documents_path = os.path.join(os.path.expanduser("~"), "Documents")
            folder_path = os.path.join(documents_path, "YU LAB-B504")
            # 检查文件夹是否存在
            if not os.path.exists(folder_path):
                # 如果文件夹不存在，则创建它
                os.makedirs(folder_path)
            self.save_path = self.save_path_settings.value("save_path", f"{folder_path}")  # 默认保存路径
        # 确保 "saved DRR" 文件夹存在
        saved_drr_path = os.path.join(self.save_path, "saved DRR")
        if not os.path.exists(saved_drr_path):
            os.makedirs(saved_drr_path)
        set_var("Geoinfo_save_path", self.save_path)

        self.themeButton = ToolButton(FluentIcon.CONSTRACT, self)
        self.themeButton.setFixedHeight(28)

        self.save1Button = PushButton(self.tr('save'), self, Icon.SAVE)
        self.save1Button.setFixedHeight(28)
        self.save1Button.clicked.connect(self.on_save1Clicked)
        self.load1Button = PushButton(self.tr('load'), self, Icon.HD)
        self.load1Button.setFixedHeight(28)
        self.load1Button.clicked.connect(self.on_load1Clicked)
        self.confirmButton = PushButton(self.tr('confirm'), self)
        self.confirmButton.setFixedHeight(28)
        self.confirmButton.clicked.connect(self.on_confirmClicked)
        self.delete1Button = PushButton(self.tr('delete'), self, Icon.DEL)
        self.delete1Button.setFixedHeight(28)
        self.delete1Button.clicked.connect(self.on_delete1Clicked)
        self.clear_all_1_Btn = PushButton(self.tr('clear all'), self)
        self.clear_all_1_Btn.setFixedHeight(28)
        self.clear_all_1_Btn.clicked.connect(self.clear_all_1)
        self.comboBox1 = ComboBox()
        self.comboBox1.setCurrentIndex(0)
        self.comboBox1.setFixedSize(120, 28)

        self.saveName1_LineEdit = LineEdit(self)
        self.saveName1_LineEdit.setToolTip('please name the geometry set 1 before saving.')
        self.saveName1_LineEdit.setFixedSize(120, 28)
        self.imaging_pair1_label = BodyLabel(self.tr('X-ray tube location of imaging pair 1    (mm)   x:'), self)
        self.imaging_pair1_tube_y_label = BodyLabel(self.tr('y:'), self)
        self.imaging_pair1_tube_z_label = BodyLabel(self.tr('z:'), self)
        self.imaging_pair1_iso_imagingPlane_distance = BodyLabel(self.tr('OID:'), self)
        self.imaging_pair1_iso_imagingPlane_distance.setToolTip('distance between iso and imageing plane. (mm)')
        self.imaging_plane1_side_length_label = BodyLabel(self.tr('IPEL:'), self)
        self.imaging_plane1_side_length_label.setToolTip('imaging plane edge length (mm)')
        self.tube1_x_LineEdit = LineEdit(self)
        self.tube1_x_LineEdit.setFixedSize(90, 28)
        self.tube1_y_LineEdit = LineEdit(self)
        self.tube1_y_LineEdit.setFixedSize(90, 28)
        self.tube1_z_LineEdit = LineEdit(self)
        self.tube1_z_LineEdit.setFixedSize(90, 28)
        self.tube1_OID_LineEdit = LineEdit(self)
        self.tube1_OID_LineEdit.setFixedSize(90, 28)
        self.tube1_OID_LineEdit.setToolTip('distance between iso and center of imageing plane. ('
                                           'mm)')
        self.imaging_plane1_side_length_LineEdit = LineEdit(self)
        self.imaging_plane1_side_length_LineEdit.setFixedSize(90, 28)
        self.imaging_plane1_side_length_LineEdit.setToolTip('imaging plane edge length (mm)')

        self.save2Button = PushButton(self.tr('save'), self, Icon.SAVE)
        self.save2Button.setFixedHeight(28)
        self.save2Button.clicked.connect(self.on_save2Clicked)
        self.load2Button = PushButton(self.tr('load'), self, Icon.HD)
        self.load2Button.setFixedHeight(28)
        self.load2Button.clicked.connect(self.on_load2Clicked)
        self.delete2Button = PushButton(self.tr('delete'), self, Icon.DEL)
        self.delete2Button.setFixedHeight(28)
        self.delete2Button.clicked.connect(self.on_delete2Clicked)
        self.clear_all_2_Btn = PushButton(self.tr('clear all'), self)
        self.clear_all_2_Btn.setFixedHeight(28)
        self.clear_all_2_Btn.clicked.connect(self.clear_all_2)
        self.comboBox2 = ComboBox()
        self.comboBox2.setCurrentIndex(0)
        self.comboBox2.setFixedSize(120, 28)

        self.saveName2_LineEdit = LineEdit(self)
        self.saveName2_LineEdit.setToolTip('please name the geometry set 2 before saving.')
        self.saveName2_LineEdit.setFixedSize(120, 28)
        self.imaging_pair2_label = BodyLabel(self.tr('X-ray tube location of imaging pair 2    (mm)   x:'), self)
        self.imaging_pair2_tube_y_abel = BodyLabel(self.tr('y:'), self)
        self.imaging_pair2_tube_z_abel = BodyLabel(self.tr('z:'), self)
        self.imaging_pair2_iso_imagingPlane_distance = BodyLabel(self.tr('OID:'), self)
        self.imaging_pair2_iso_imagingPlane_distance.setToolTip('distance between iso and imageing plane. (mm)')
        self.imaging_plane2_side_length_label = BodyLabel(self.tr('IPEL:'), self)
        self.imaging_plane2_side_length_label.setToolTip('imaging plane edge length (mm)')
        self.tube2_x_LineEdit = LineEdit(self)
        self.tube2_x_LineEdit.setFixedSize(90, 28)
        self.tube2_y_LineEdit = LineEdit(self)
        self.tube2_y_LineEdit.setFixedSize(90, 28)
        self.tube2_z_LineEdit = LineEdit(self)
        self.tube2_z_LineEdit.setFixedSize(90, 28)
        self.tube2_OID_LineEdit = LineEdit(self)
        self.tube2_OID_LineEdit.setFixedSize(90, 28)
        self.tube2_OID_LineEdit.setToolTip('distance between iso and center of imageing plane. ('
                                           'mm)')
        self.imaging_plane2_side_length_LineEdit = LineEdit(self)
        self.imaging_plane2_side_length_LineEdit.setFixedSize(90, 28)
        self.imaging_plane2_side_length_LineEdit.setToolTip('imaging plane edge length (mm)')

        self.select_path_button = PushButton(self.tr('save path'), self, Icon.FOLDER)
        self.select_path_button.clicked.connect(self.on_select_path_clicked)
        self.select_path_button.setFixedHeight(28)
        self.path_label = BodyLabel(self.tr(f'{self.save_path}'), self)

        self.sysMSGLabel = BodyLabel(self.tr('Tips: you can set up 1 or 2 imaging pairs.'), self)
        self.separator = SeparatorWidget(self)

        self.vBoxLayout = QVBoxLayout(self)
        self.buttonLayout1 = FlowLayout()
        self.labelLayout1 = FlowLayout()
        self.buttonLayout2 = FlowLayout()
        self.labelLayout2 = FlowLayout()
        #self.visualLayout = FlowLayout()
        self.save_path_layout = FlowLayout()
        self.sysMSGLayout = FlowLayout()
        self.__initWidget()

    def __initWidget(self):
        # Button layout
        self.setFixedHeight(200)
        self.vBoxLayout.addSpacing(0)
        self.vBoxLayout.addLayout(self.buttonLayout1, 1)
        self.vBoxLayout.addLayout(self.labelLayout1, 1)
        self.vBoxLayout.addLayout(self.buttonLayout2, 1)
        self.vBoxLayout.addLayout(self.labelLayout2, 1)
        self.vBoxLayout.addLayout(self.save_path_layout, 1)
        self.vBoxLayout.addLayout(self.sysMSGLayout, 1)
        self.vBoxLayout.setAlignment(Qt.AlignTop)

        self.buttonLayout1.addWidget(self.saveName1_LineEdit)
        self.buttonLayout1.addWidget(self.save1Button)
        self.buttonLayout1.addWidget(self.comboBox1)
        self.buttonLayout1.addWidget(self.load1Button)
        self.buttonLayout1.addWidget(self.delete1Button)
        self.buttonLayout1.addWidget(self.clear_all_1_Btn)
        self.buttonLayout1.addWidget(self.themeButton)
        self.buttonLayout1.addWidget(self.confirmButton)
        self.buttonLayout1.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.labelLayout1.addWidget(self.imaging_pair1_label)
        self.labelLayout1.addWidget(self.tube1_x_LineEdit)
        self.labelLayout1.addWidget(self.imaging_pair1_tube_y_label)
        self.labelLayout1.addWidget(self.tube1_y_LineEdit)
        self.labelLayout1.addWidget(self.imaging_pair1_tube_z_label)
        self.labelLayout1.addWidget(self.tube1_z_LineEdit)
        self.labelLayout1.addWidget(self.imaging_pair1_iso_imagingPlane_distance)
        self.labelLayout1.addWidget(self.tube1_OID_LineEdit)
        self.labelLayout1.addWidget(self.imaging_plane1_side_length_label)
        self.labelLayout1.addWidget(self.imaging_plane1_side_length_LineEdit)

        self.buttonLayout2.addWidget(self.saveName2_LineEdit)
        self.buttonLayout2.addWidget(self.save2Button)
        self.buttonLayout2.addWidget(self.comboBox2)
        self.buttonLayout2.addWidget(self.load2Button)
        self.buttonLayout2.addWidget(self.delete2Button)
        self.buttonLayout2.addWidget(self.clear_all_2_Btn)
        self.buttonLayout2.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.labelLayout2.addWidget(self.imaging_pair2_label)
        self.labelLayout2.addWidget(self.tube2_x_LineEdit)
        self.labelLayout2.addWidget(self.imaging_pair2_tube_y_abel)
        self.labelLayout2.addWidget(self.tube2_y_LineEdit)
        self.labelLayout2.addWidget(self.imaging_pair2_tube_z_abel)
        self.labelLayout2.addWidget(self.tube2_z_LineEdit)
        self.labelLayout2.addWidget(self.imaging_pair2_iso_imagingPlane_distance)
        self.labelLayout2.addWidget(self.tube2_OID_LineEdit)
        self.labelLayout2.addWidget(self.imaging_plane2_side_length_label)
        self.labelLayout2.addWidget(self.imaging_plane2_side_length_LineEdit)

        self.themeButton.installEventFilter(ToolTipFilter(self.themeButton))
        self.themeButton.setToolTip(self.tr('Toggle theme'))
        self.themeButton.clicked.connect(lambda: toggleTheme(True, False))

        self.save_path_layout.addWidget(self.select_path_button)
        self.save_path_layout.addWidget(self.path_label)

        self.sysMSGLayout.addWidget(self.sysMSGLabel)

        self.update_comboBox(self.comboBox1)
        self.update_comboBox(self.comboBox2)

    def sys_msg(self, msg):
        self.sysMSGLabel.setText(self.tr(f'<b>system message: {msg}</b>'))

    def on_select_path_clicked(self):
        """打开文件资源管理器选择保存路径"""
        selected_path = QFileDialog.getExistingDirectory(self, "选择保存路径", self.save_path)
        if selected_path:
            self.save_path = selected_path
            self.path_label.setText(self.tr(self.save_path))
            self.save_path_settings.setValue("save_path", self.save_path)  # 保存路径到配置
            # 更新 ComboBox 内容
            self.update_comboBox(self.comboBox1)
            self.update_comboBox(self.comboBox2)
            self.sys_msg('new save path saved.')
        else:
            self.sys_msg('selecting canceled.')

    def update_comboBox(self, comboBox):
        """更新 ComboBox 内容，列出当前保存路径中的 JSON 文件"""
        comboBox.clear()
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        json_files = [f for f in os.listdir(self.save_path) if f.endswith(".json")]
        comboBox.addItems(json_file[:-5] for json_file in json_files)

    def on_save1Clicked(self):
        save_name = self.saveName1_LineEdit.text().strip()
        if save_name is None:
            self.sys_msg('warning, please provide a name for the geometry set 1.')
            return
        if not re.match(r'^[\w\-. ]+$', save_name):
            self.sys_msg('warning, the file name contains invalid characters. Please use only letters, numbers, '
                         'underscores, hyphens, dots, and spaces.')
            return

        file_path = os.path.join(self.save_path, f"{save_name}.json")
        x = self.tube1_x_LineEdit.text().strip()
        y = self.tube1_y_LineEdit.text().strip()
        z = self.tube1_z_LineEdit.text().strip()
        OID = self.tube1_OID_LineEdit.text().strip()
        IPEL = self.imaging_plane1_side_length_LineEdit.text().strip()

        if not all([x, y, z, OID, IPEL]):
            self.sys_msg('warning, please fill in all fields for geometry set 1.')
            return

        #根据坐标信息计算变换矩阵与平移向量
        try:
            x0 = float(x)
            y0 = float(y)
            z0 = float(z)
            OID0 = float(OID)
            IPEL0 = float(IPEL)
        except ValueError:
            self.sys_msg('error, invalid coordinates, please make sure they are numbers')
            return

        data = {"x": x, "y": y, "z": z, "OID": OID, "IPEL": IPEL}

        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

        self.update_comboBox(self.comboBox1)
        self.update_comboBox(self.comboBox2)
        self.sys_msg(f'geometry set 1 saved as {save_name}.json successfully!')

    def on_save2Clicked(self):
        save_name = self.saveName2_LineEdit.text().strip()
        if save_name is None:
            self.sys_msg('warning, please provide a name for the geometry set 2.')
            return
        if not re.match(r'^[\w\-. ]+$', save_name):
            self.sys_msg('warning, the file name contains invalid characters. Please use only letters, numbers, '
                         'underscores, hyphens, dots, and spaces.')
            return

        file_path = os.path.join(self.save_path, f"{save_name}.json")
        x = self.tube2_x_LineEdit.text().strip()
        y = self.tube2_y_LineEdit.text().strip()
        z = self.tube2_z_LineEdit.text().strip()
        OID = self.tube2_OID_LineEdit.text().strip()
        IPEL = self.imaging_plane2_side_length_LineEdit.text().strip()

        if not all([x, y, z, OID]):
            self.sys_msg('warning, please fill in all fields for geometry set 2.')
            return
            # 根据坐标信息计算变换矩阵与平移向量
        try:
            x0 = float(x)
            y0 = float(y)
            z0 = float(z)
            OID0 = float(OID)
            IPEL0 = float(IPEL)
        except ValueError:
            self.sys_msg('error, invalid coordinates, please make sure they are numbers')
            return

        data = {"x": x, "y": y, "z": z, "OID": OID, "IPEL": IPEL}

        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

        self.update_comboBox(self.comboBox1)
        self.update_comboBox(self.comboBox2)
        self.sys_msg(f'geometry set 2 saved as {save_name}.json successfully!')

    def on_load1Clicked(self):
        selected_file = self.comboBox1.currentText()
        if selected_file is None:
            self.sys_msg('warning, please select a file to load for geometry set 1.')
            return
        selected_file = selected_file + '.json'
        file_path = os.path.join(self.save_path, selected_file)
        if not os.path.exists(file_path):
            self.sys_msg(f'error, file {selected_file} does not exist.')
            return

        with open(file_path, 'r') as file:
            data = json.load(file)

        self.tube1_x_LineEdit.setText(data.get("x", ""))
        self.tube1_y_LineEdit.setText(data.get("y", ""))
        self.tube1_z_LineEdit.setText(data.get("z", ""))
        self.tube1_OID_LineEdit.setText(data.get("OID", ""))
        self.imaging_plane1_side_length_LineEdit.setText(data.get("IPEL", ""))
        self.saveName1_LineEdit.setText(selected_file[:-5])
        self.sys_msg(f"geometry set 1 loaded successfully from {selected_file}.")

    def on_load2Clicked(self):
        selected_file = self.comboBox2.currentText()
        if selected_file is None:
            self.sys_msg('warning, please select a file to load for geometry set 2.')
            return
        selected_file = selected_file + '.json'
        file_path = os.path.join(self.save_path, selected_file)
        if not os.path.exists(file_path):
            self.sys_msg(f'error, file {selected_file} does not exist.')
            return

        with open(file_path, 'r') as file:
            data = json.load(file)

        self.tube2_x_LineEdit.setText(data.get("x", ""))
        self.tube2_y_LineEdit.setText(data.get("y", ""))
        self.tube2_z_LineEdit.setText(data.get("z", ""))
        self.tube2_OID_LineEdit.setText(data.get("OID", ""))
        self.imaging_plane2_side_length_LineEdit.setText(data.get("IPEL", ""))
        self.saveName2_LineEdit.setText(selected_file[:-5])
        self.sys_msg(f"geometry set 2 loaded successfully from {selected_file}.")

    def on_delete1Clicked(self):
        selected_file = self.comboBox1.currentText()
        selected_file = selected_file + '.json'
        if selected_file is None:
            self.sys_msg(f'warning, please select a file to delete for geometry set 1.')
            return

        reply = QMessageBox.question(self, "confirm delete", f"are you sure you want to delete {selected_file}?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            file_path = os.path.join(self.save_path, selected_file)
            if os.path.exists(file_path):
                os.remove(file_path)
                self.update_comboBox(self.comboBox1)
                self.update_comboBox(self.comboBox2)
                self.sys_msg(f"{selected_file} has been deleted.")
            else:
                self.sys_msg(f"error, file {selected_file} does not exist.")

    def on_delete2Clicked(self):
        selected_file = self.comboBox2.currentText()
        selected_file = selected_file + '.json'
        if selected_file is None:
            self.sys_msg(f'warning, please select a file to delete for geometry set 2.')
            return

        reply = QMessageBox.question(self, "confirm delete", f"are you sure you want to delete {selected_file}?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            file_path = os.path.join(self.save_path, selected_file)
            if os.path.exists(file_path):
                os.remove(file_path)
                self.update_comboBox(self.comboBox1)
                self.update_comboBox(self.comboBox2)
                self.sys_msg(f"{selected_file} has been deleted.")
            else:
                self.sys_msg(f"error, file {selected_file} does not exist.")

    def clear_all_1(self):
        """
        清空相关输入框的内容
        """
        self.saveName1_LineEdit.clear()
        self.tube1_x_LineEdit.clear()
        self.tube1_y_LineEdit.clear()
        self.tube1_z_LineEdit.clear()
        self.tube1_OID_LineEdit.clear()
        self.imaging_plane1_side_length_LineEdit.clear()

    def clear_all_2(self):
        """
        清空相关输入框的内容
        """
        self.saveName2_LineEdit.clear()
        self.tube2_x_LineEdit.clear()
        self.tube2_y_LineEdit.clear()
        self.tube2_z_LineEdit.clear()
        self.tube2_OID_LineEdit.clear()
        self.imaging_plane2_side_length_LineEdit.clear()

    def on_confirmClicked(self):

        G1_OK = 0
        G2_OK = 0
        save_name1_invalid = 0
        save_name2_invalid = 0
        save_name1 = None
        save_name2 = None

        fields1 = [
            self.tube1_x_LineEdit.text(),
            self.tube1_y_LineEdit.text(),
            self.tube1_z_LineEdit.text(),
            self.tube1_OID_LineEdit.text(),
            self.imaging_plane1_side_length_LineEdit.text(),
            self.saveName1_LineEdit.text()
        ]

        fields2 = [
            self.tube2_x_LineEdit.text(),
            self.tube2_y_LineEdit.text(),
            self.tube2_z_LineEdit.text(),
            self.tube2_OID_LineEdit.text(),
            self.imaging_plane2_side_length_LineEdit.text(),
            self.saveName2_LineEdit.text()
        ]

        if all(field is not None for field in fields1):
            x1 = check_is_number(self.tube1_x_LineEdit)
            y1 = check_is_number(self.tube1_y_LineEdit)
            z1 = check_is_number(self.tube1_z_LineEdit)
            sid1 = check_is_number(self.tube1_OID_LineEdit)
            ipel1 = check_is_number(self.imaging_plane1_side_length_LineEdit)
            save_name1 = self.saveName1_LineEdit.text()
            if all(parameter is not None for parameter in [x1, y1, z1, sid1, ipel1]):
                if not re.match(r'^[\w\-. ]+$', save_name1):
                    save_name1_invalid = 1
                else:
                    G1_OK = 1

        if all(field is not None for field in fields2):
            x2 = check_is_number(self.tube2_x_LineEdit)
            y2 = check_is_number(self.tube2_y_LineEdit)
            z2 = check_is_number(self.tube2_z_LineEdit)
            sid2 = check_is_number(self.tube2_OID_LineEdit)
            ipel2 = check_is_number(self.imaging_plane2_side_length_LineEdit)
            save_name2 = self.saveName2_LineEdit.text()
            if all(parameter is not None for parameter in [x2, y2, z2, sid2, ipel2]):
                if not re.match(r'^[\w\-. ]+$', save_name2):
                    save_name2_invalid = 1
                else:
                    G2_OK = 1

        set_var("Imaging_pair_1_enabled", G1_OK)
        set_var("Imaging_pair_2_enabled", G2_OK)
        set_var("Geoinfo_save_path", self.save_path)
        if G1_OK and G2_OK:
            self.on_save1Clicked()
            self.on_save2Clicked()
            set_var("Imaging_pair_1_fileName", save_name1)
            set_var("Imaging_pair_2_fileName", save_name2)
            self.sys_msg(
                """imaging pair 1: enabled <span style="color:green;">√</span>, imaging pair 2: enabled <span 
                style="color:green;">√</span>.""")
        elif G1_OK:
            self.on_save1Clicked()
            set_var("Imaging_pair_1_fileName", save_name1)
            self.sys_msg(
                """imaging pair 1: enabled <span style="color:green;">√</span>, imaging pair 2: disabled <span 
                style="color:red;">×</span>.""")
        elif G2_OK:
            self.on_save2Clicked()
            set_var("Imaging_pair_1_fileName", save_name2)
            self.sys_msg(
                """imaging pair 1: disabled <span style="color:red;">×</span>, imaging pair 2: enabled <span 
                style="color:green;">√</span>.""")
        elif save_name1_invalid or save_name2_invalid:
            self.sys_msg(
                """the save name contains invalid characters, imaging pair 1: disabled 
                <span style="color:red;">×</span>, imaging pair 2: disabled <span style="color:red;">×</span>.""")
        else:
            self.sys_msg(
                """imaging pair 1: disabled <span style="color:red;">×</span>, imaging pair 2: disabled <span 
                style="color:red;">×</span>.""")


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


class StereoGeoGalleryInterface(ScrollArea):
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
