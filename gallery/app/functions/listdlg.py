from qfluentwidgets import ComboBox, Dialog
from PyQt5.QtWidgets import QVBoxLayout


class ContourSelectionDialog(Dialog):
    def __init__(self, contour_names, title, prompt, parent=None):
        super().__init__(title=title, content=prompt)
        self.setFixedSize(400, 250)

        # 下拉列表框
        self.combo_box = ComboBox(self)
        self.combo_box.setGeometry(20, 70, 360, 30)
        self.combo_box.addItems(contour_names)

        self.vBoxLayout = QVBoxLayout(self)
        self.vBoxLayout.addWidget(self.combo_box)

    def get_selected_index(self):
        """返回用户选择的索引（从 1 开始）"""
        return self.combo_box.currentIndex() + 1


def select_contour_from_dialog(contour_names, title, prompt):
    """
    弹出对话框供用户选择轮廓。
    如果用户选择了一个选项，则返回对应的索引（从 1 开始）。
    如果用户取消选择，则返回 None。
    Args:
        contour_names (list): 轮廓名称的列表。
    Returns:
        int or None: 用户选择的轮廓索引，或 None（取消选择）。
    """
    # 弹出对话框
    dialog = ContourSelectionDialog(contour_names, title, prompt)
    if dialog.exec() == Dialog.Accepted:
        return dialog.get_selected_index()
    else:
        return None
