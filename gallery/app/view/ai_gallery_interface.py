# coding:utf-8
import contextlib
import csv
import math
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import confusion_matrix
from PyQt5.QtCore import Qt, QEvent, QSettings
from PyQt5.QtGui import QPainter, QPen, QColor, QIntValidator, QPixmap, QImage
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFrame, QFileDialog
from ..functions.AI_display import update_inputs_display, update_outputs_display
from qfluentwidgets import (ScrollArea, PushButton, ToolButton, FluentIcon,
                            isDarkTheme, FlowLayout, ToolTipFilter, ComboBox, CheckBox, CaptionLabel, LineEdit,
                            StrongBodyLabel, BodyLabel, toggleTheme)
from ..common.style_sheet import StyleSheet
from ..common.icon import Icon
from ..functions.update_ai_info import update_ai_slice
import os
import time
import tensorflow as tf
from natsort import natsorted
import sys
import importlib.util
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from ..var.globals import set_var, get_var, del_var


# 添加评估指标计算函数
def calculate_metrics(pred, gt):
    """计算各种评估指标，自动检测任务类型"""
    # 将数据归一化到0-1范围
    pred = pred.astype(np.float32) / 255.0
    gt = gt.astype(np.float32) / 255.0

    results = {}

    # 检测是否为二值图像（分割任务）
    is_binary = False
    unique_vals = np.unique(gt)

    # 检查图像值范围
    if len(unique_vals) <= 2:
        # 检查值是否接近0和1
        if (np.min(gt) < 0.1 and np.max(gt) > 0.9 and
                np.min(pred) < 0.1 and np.max(pred) > 0.9):
            is_binary = True

    if is_binary:
        # 二值化处理 - 分割任务
        pred_bin = (pred > 0.5).astype(np.uint8)
        gt_bin = (gt > 0.5).astype(np.uint8)

        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(gt_bin.flatten(), pred_bin.flatten(), labels=[0, 1]).ravel()

        # 分割指标
        results["dice"] = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        results["iou"] = tp / (tp + fp + fn + 1e-8)
        results["precision"] = tp / (tp + fp + 1e-8)
        results["recall"] = tp / (tp + fn + 1e-8)
        results["accuracy"] = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        results["f1"] = 2 * (results["precision"] * results["recall"]) / (
                    results["precision"] + results["recall"] + 1e-8)
    else:
        # 非二值图像 - 回归/复原任务
        # 这些指标对分割任务没有意义
        results["dice"] = float('nan')
        results["iou"] = float('nan')
        results["precision"] = float('nan')
        results["recall"] = float('nan')
        results["accuracy"] = float('nan')
        results["f1"] = float('nan')

    # 所有任务类型都计算的指标
    results["ssim"] = ssim(gt, pred, data_range=1.0, channel_axis=-1 if len(gt.shape) > 2 else None)
    results["mse"] = np.mean((gt - pred) ** 2)
    results["nmse"] = results["mse"] / (np.mean(gt ** 2) + 1e-8)

    # 峰值信噪比 (PSNR) - 图像质量评估
    if results["mse"] == 0:
        results["psnr"] = float('inf')
    else:
        results["psnr"] = 20 * math.log10(1.0 / math.sqrt(results["mse"]))

    return results


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
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.save_path_settings = QSettings("YU LAB-B504", "save path")
        saved_path = self.save_path_settings.value("save path")
        if saved_path:
            self.save_path = saved_path
        else:
            documents_path = os.path.join(os.path.expanduser("~"), "Documents")
            folder_path = os.path.join(documents_path, "YU LAB-B504")
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            self.save_path = self.save_path_settings.value("save_path", f"{folder_path}")  # 默认保存路径
        set_var("Geoinfo_save_path", self.save_path)

        self.input_path_settings = QSettings("YU LAB-B504", "model inputs path")  # 保存用户设置
        inputs_path = self.input_path_settings.value("model inputs path")
        if inputs_path:
            self.inputs_path = inputs_path
        else:
            folder_path = os.path.join(self.save_path, "AI/Inputs")
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            self.inputs_path = self.input_path_settings.value("model inputs path", f"{folder_path}")  # 默认保存路径
        set_var("Model_inputs_path", self.inputs_path)
        saved_model_architecture_path = os.path.join(self.save_path, "AI/Architecture")
        if not os.path.exists(saved_model_architecture_path):
            os.makedirs(saved_model_architecture_path)
        set_var("Model_architecture_path", saved_model_architecture_path)
        saved_model_weights_path = os.path.join(self.save_path, "AI/Weights")
        if not os.path.exists(saved_model_weights_path):
            os.makedirs(saved_model_weights_path)
        set_var("Model_weights_path", saved_model_weights_path)
        set_var("Model_inputs_path", self.inputs_path)
        saved_model_outputs_path = os.path.join(self.save_path, "AI/Outputs")
        if not os.path.exists(saved_model_outputs_path):
            os.makedirs(saved_model_outputs_path)
        set_var("Model_outputs_path", saved_model_outputs_path)
        # 添加Ground Truth路径选择
        self.gt_path_settings = QSettings("YU LAB-B504", "ground truth path")
        gt_path = self.gt_path_settings.value("ground truth path")
        if gt_path:
            self.gt_path = gt_path
        else:
            folder_path = os.path.join(self.save_path, "AI/GroundTruth")
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            self.gt_path = self.gt_path_settings.value("ground truth path", f"{folder_path}")
        set_var("Ground_truth_path", self.gt_path)
        saved_evaluation_path = os.path.join(self.save_path, "AI/Evaluation")
        if not os.path.exists(saved_evaluation_path):
            os.makedirs(saved_evaluation_path)
        set_var("Model_evaluation_path", saved_evaluation_path)

        self.themeButton = ToolButton(FluentIcon.CONSTRACT, self)

        self.StartButton = PushButton(self.tr('▶ start'), self)
        self.StartButton.clicked.connect(self.on_StartClicked)

        self.ModelType_label = BodyLabel(self.tr('type: '))
        self.ModelTypeComboBox = ComboBox() # pytorch tensorflow
        self.ModelTypeComboBox.setCurrentIndex(0)
        self.ModelTypeComboBox.setFixedSize(120, 28)
        self.ModelTypeComboBox.addItem('pytorch')
        self.ModelTypeComboBox.addItem('tensorflow')

        self.architecture_label = BodyLabel(self.tr('architecture'), self)
        self.architecture_label.setToolTip('please select the architecture file with create_model function.')
        self.architectureComboBox = ComboBox()
        self.architectureComboBox.setCurrentIndex(0)
        self.architectureComboBox.setFixedSize(150, 28)

        self.weights_label = BodyLabel(self.tr('weights'), self)
        self.weights_label.setToolTip('please select the weights file.')
        self.WeightsComboBox = ComboBox()
        self.WeightsComboBox.setCurrentIndex(0)
        self.WeightsComboBox.setFixedSize(150, 28)

        self.ImgSize_label = BodyLabel(self.tr('image resize to: '))
        self.ImgSizeLength_LineEdit = LineEdit(self)
        self.ImgSizeLength_LineEdit.setFixedSize(60, 28)
        self.ImgSizeLength_LineEdit.setToolTip('it should fit the input size of your model.')
        self.ImgSizeLength_LineEdit.setText('256')
        self.ImgSizeLength_LineEdit.setValidator(QIntValidator(1, 9999, self))

        self.X_label = BodyLabel(self.tr('X'))
        self.ImgSizeWidth_LineEdit = LineEdit(self)
        self.ImgSizeWidth_LineEdit.setFixedSize(60, 28)
        self.ImgSizeWidth_LineEdit.setToolTip('it should fit the input size of your model.')
        self.ImgSizeWidth_LineEdit.setText('256')
        self.ImgSizeWidth_LineEdit.setValidator(QIntValidator(1, 9999, self))

        self.batchSize_label = BodyLabel(self.tr('batch size: '))
        self.batchSize_LineEdit = LineEdit(self)
        self.batchSize_LineEdit.setFixedSize(60, 28)
        self.batchSize_LineEdit.setToolTip('depends on your device memory size.')
        self.batchSize_LineEdit.setText('10')
        self.batchSize_LineEdit.setValidator(QIntValidator(1, 9999, self))

        self.select_path_button = PushButton(self.tr('input images\' path'), self, Icon.FOLDER)
        self.select_path_button.clicked.connect(self.on_select_path_clicked)
        self.path_label = BodyLabel(self.tr(f'{self.inputs_path}'), self)

        self.read_input_images_button = PushButton(self.tr("load inputs"), self, Icon.IMPORT)
        self.read_input_images_button.clicked.connect(self.on_read_input_images_clicked)

        # 添加评估按钮
        self.evaluateButton = PushButton(self.tr('evaluate'), self)
        self.evaluateButton.clicked.connect(self.on_EvaluateClicked)
        self.evaluateButton.setEnabled(False)

        # 添加Ground Truth路径按钮
        self.select_gt_path_button = PushButton(self.tr('ground truth path'), self, Icon.FOLDER)
        self.select_gt_path_button.clicked.connect(self.on_select_gt_path_clicked)
        self.gt_path_label = BodyLabel(self.tr(f'{self.gt_path}'), self)

        self.currentSliceLabel = BodyLabel(self.tr('current slice: / '), self)
        self.sysMSGLabel = BodyLabel(self.tr(f'system message: please store input images to {self.inputs_path}.'), self)

        self.input_display_label = BodyLabel(self.tr('input images:'), self)
        self.output_display_label = BodyLabel(self.tr('output images'), self)

        self.condition_label = BodyLabel(self.tr('load model and inputs'), self)

        self.separator = SeparatorWidget(self)
        self.vBoxLayout = QVBoxLayout(self)
        self.buttonLayout = FlowLayout()
        self.labelTitleLayout = QHBoxLayout()
        self.labelLayout = FlowLayout()
        self.sysMSGLayout = FlowLayout()
        self.input_output_display_area_layout = QHBoxLayout(self)
        self.__initWidget()
        # 连接信号以动态更新控件显示状态
        self.ModelTypeComboBox.currentTextChanged.connect(self.update_search_suffix)

    def __initWidget(self):

        # Button layout
        self.setFixedHeight(220)
        self.vBoxLayout.addSpacing(0)
        self.vBoxLayout.addLayout(self.buttonLayout, 1)
        self.vBoxLayout.setAlignment(Qt.AlignTop)
        self.buttonLayout.addWidget(self.themeButton)
        self.buttonLayout.addWidget(self.separator)
        self.buttonLayout.addWidget(self.ModelType_label)
        self.buttonLayout.addWidget(self.ModelTypeComboBox)
        self.buttonLayout.addWidget(self.StartButton)
        self.buttonLayout.addWidget(self.evaluateButton)

        self.themeButton.installEventFilter(ToolTipFilter(self.themeButton))
        self.themeButton.setToolTip(self.tr('toggle theme'))
        self.themeButton.clicked.connect(lambda: toggleTheme(True, False))

        # Label title layout
        self.vBoxLayout.addLayout(self.labelTitleLayout, 1)
        self.labelTitleLayout.addWidget(self.condition_label, 0, Qt.AlignLeft)

        # Label layout
        self.vBoxLayout.addLayout(self.labelLayout, 1)
        self.labelLayout.addWidget(self.architecture_label)
        self.labelLayout.addWidget(self.architectureComboBox)
        self.labelLayout.addWidget(self.weights_label)
        self.labelLayout.addWidget(self.WeightsComboBox)
        self.labelLayout.addWidget(self.ImgSize_label)
        self.labelLayout.addWidget(self.ImgSizeLength_LineEdit)
        self.labelLayout.addWidget(self.X_label)
        self.labelLayout.addWidget(self.ImgSizeWidth_LineEdit)
        self.labelLayout.addWidget(self.batchSize_label)
        self.labelLayout.addWidget(self.batchSize_LineEdit)
        self.labelLayout.addWidget(self.select_path_button)
        self.labelLayout.addWidget(self.path_label)
        self.labelLayout.addWidget(self.read_input_images_button)
        self.labelLayout.addWidget(self.currentSliceLabel)
        self.labelLayout.addWidget(self.select_gt_path_button)
        self.labelLayout.addWidget(self.gt_path_label)

        # 系统消息
        self.vBoxLayout.addLayout(self.sysMSGLayout, 1)
        self.sysMSGLayout.addWidget(self.sysMSGLabel)

        self.vBoxLayout.addLayout(self.input_output_display_area_layout, 1)
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.input_display_label, alignment=Qt.AlignCenter)
        # 创建右半部分布局
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.output_display_label, alignment=Qt.AlignCenter)
        self.input_output_display_area_layout.addLayout(left_layout)
        self.input_output_display_area_layout.addLayout(right_layout)


        # 初始化控件可见性
        self.update_search_suffix()
        self.update_ai_display()
        self.update_evaluate_button_state()
        update_ai_slice(self)

    def update_evaluate_button_state(self):
        """根据输出和GT路径是否存在并且图像数量相同更新评估按钮状态"""
        gt_path_exists = os.path.exists(self.gt_path) and os.listdir(self.gt_path)

        if gt_path_exists:
            outputs_count = len(os.listdir(get_var('Model_outputs_path')))
            gt_count = len(os.listdir(self.gt_path))
            # 检查图像数量是否相同
            self.evaluateButton.setEnabled(outputs_count == gt_count)
        else:
            self.evaluateButton.setEnabled(False)

    def on_StartClicked(self):
        self.StartButton.setEnabled(False)
        try:
            # 获取用户选择
            batch_size = int(self.batchSize_LineEdit.text())
            model_type = self.ModelTypeComboBox.currentText()
            architecture_name = self.architectureComboBox.currentText()
            weight_name = self.WeightsComboBox.currentText()
            # 构建路径
            architecture_path = os.path.join(get_var("Model_architecture_path"), architecture_name)
            weight_path = os.path.join(get_var("Model_weights_path"), weight_name)
            arch_dir = str(Path(architecture_path).parent)

            # 检查输入图像
            imgs = get_var("Model_inputs_images")
            if imgs is None or imgs.size == 0:
                self.sys_msg("please load inputs first!")
                self.StartButton.setEnabled(True)
                return

            # 自动检测设备 (PyTorch)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # 临时添加路径上下文
            with self.temporary_sys_path(arch_dir):
                # 模型加载和处理
                if model_type == 'pytorch':
                    out_np = self.process_pytorch(weight_path, imgs, device, batch_size)
                elif model_type == 'tensorflow':
                    out_np = self.process_tensorflow(weight_path, imgs, batch_size)

                if out_np is not None:
                    # 保存输出结果
                    set_var("Model_outputs_images", out_np)
                    set_var("Model_outputs_images_exists", True)
                    self.save_output_images(out_np)
                    self.sys_msg(f"{out_np.shape[0]} images processed successfully!")
                    self.update_evaluate_button_state()
                    self.update_ai_display()
        except Exception as e:
            self.sys_msg(f"Error: {str(e)}")
        finally:
            self.StartButton.setEnabled(True)

    def save_output_images(self, out_np):
        """保存输出图像到输出目录"""
        output_dir = get_var("Model_outputs_path")
        os.makedirs(output_dir, exist_ok=True)

        # 获取输入文件名列表
        input_filenames = get_var("Model_inputs_filenames")

        # 保存每张输出图像
        for i in range(out_np.shape[0]):
            img = Image.fromarray(out_np[i])
            img.save(os.path.join(output_dir, input_filenames[i]))

        self.sys_msg(f"{out_np.shape[0]} output images saved to {output_dir}")

    def on_EvaluateClicked(self):
        """执行评估流程"""
        self.evaluateButton.setEnabled(False)
        try:
            # 获取输出图像目录
            outputs_dir = get_var("Model_outputs_path")
            outputs_files = natsorted([f for f in os.listdir(outputs_dir)
                                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])

            # 获取Ground Truth图像
            gt_files = natsorted([f for f in os.listdir(self.gt_path)
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])

            if not outputs_files:
                self.sys_msg("No output images found for evaluation")
                self.evaluateButton.setEnabled(True)
                return

            if not gt_files:
                self.sys_msg("No ground truth images found for evaluation")
                self.evaluateButton.setEnabled(True)
                return

            # 确保文件数量匹配
            min_count = min(len(outputs_files), len(gt_files))
            if min_count == 0:
                self.sys_msg("No matching images found for evaluation")
                self.evaluateButton.setEnabled(True)
                return

            # 准备评估结果
            results = []
            all_metrics = {}

            # 逐对评估图像
            for i in range(min_count):
                # 加载输出图像
                output_path = os.path.join(outputs_dir, outputs_files[i])
                output_img = np.array(Image.open(output_path))

                # 加载Ground Truth图像
                gt_path = os.path.join(self.gt_path, gt_files[i])
                gt_img = np.array(Image.open(gt_path))

                # 确保图像尺寸匹配
                if output_img.shape != gt_img.shape:
                    gt_img = Image.open(gt_path).resize(output_img.shape[1::-1], Image.BILINEAR)
                    gt_img = np.array(gt_img)

                # 计算指标
                metrics = calculate_metrics(output_img, gt_img)
                results.append({
                    "image": outputs_files[i],
                    **metrics
                })

                # 累加指标用于计算平均值
                for key, value in metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = 0.0
                    all_metrics[key] += value

            # 计算平均指标
            avg_metrics = {f"avg_{key}": value / min_count for key, value in all_metrics.items()}

            # 保存评估结果
            self.save_evaluation_results(results, avg_metrics)

            self.sys_msg(f"Evaluation completed for {min_count} images")
            self.evaluateButton.setEnabled(True)

        except Exception as e:
            self.sys_msg(f"Evaluation failed: {str(e)}")
            self.evaluateButton.setEnabled(True)

    def save_evaluation_results(self, results, avg_metrics):
        """保存评估结果为CSV文件，并优先显示平均值"""
        # 生成文件名
        model_name = self.WeightsComboBox.currentText().split('.')[0]
        inputs_name = os.path.basename(self.inputs_path)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{inputs_name}_{model_name}_evaluation_{timestamp}.csv"

        # 使用新的评估路径
        evaluation_path = get_var("Model_evaluation_path")
        csv_path = os.path.join(evaluation_path, csv_filename)

        # 准备CSV列
        fieldnames = ["image", "dice", "iou", "precision", "recall", "accuracy",
                      "ssim", "mse", "nmse", "psnr", "f1"]

        # 写入CSV文件
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # 写入平均值为首行
            avg_row = {"image": "AVERAGE"}
            for k, v in avg_metrics.items():
                clean_k = k.replace("avg_", "")
                # 只保留有意义的平均值
                if not (isinstance(v, float) and math.isnan(v)) and clean_k in fieldnames:
                    avg_row[clean_k] = v
            writer.writerow(avg_row)

            # 写入每张图像的结果
            for result in results:
                row = {k: "" if isinstance(v, float) and math.isnan(v) else v
                       for k, v in result.items() if k in fieldnames}
                writer.writerow(row)

        self.sys_msg(f"Evaluation results saved to {csv_path}")

    def on_select_gt_path_clicked(self):
        """选择Ground Truth路径"""
        selected_path = QFileDialog.getExistingDirectory(
            self, "Select Ground Truth Directory", self.gt_path)

        if selected_path:
            self.gt_path = selected_path
            self.gt_path_label.setText(self.tr(self.gt_path))
            set_var("Ground_truth_path", self.gt_path)
            self.gt_path_settings.setValue("ground truth path", self.gt_path)
            self.sys_msg('Ground truth path updated')
            self.update_evaluate_button_state()

    # 上下文管理器方法
    @contextlib.contextmanager
    def temporary_sys_path(self, path):
        """临时添加系统路径的上下文管理器"""
        added = False
        if path not in sys.path:
            sys.path.insert(0, path)
            added = True
        try:
            yield
        finally:
            if added and sys.path[0] == path:
                sys.path.pop(0)

    # PyTorch处理
    def process_pytorch(self, weight_path, imgs, device, batch_size):
        """处理PyTorch模型推理"""
        # 加载模型
        try:
            model = torch.load(weight_path, map_location=device)
            model.eval()

            # 转换输入张量
            tensor_all = torch.from_numpy(imgs.transpose(0, 3, 1, 2)).float() / 255.0
            # 推理
            outputs = []
            with torch.no_grad():
                for i in range(0, len(tensor_all), batch_size):
                    batch = tensor_all[i:i + batch_size].to(device)  # 仅转移当前batch到GPU
                    outputs.append(model(batch).cpu())

            # 转换输出
            return (torch.cat(outputs).permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype(np.uint8)
        except Exception as e:
            self.sys_msg(f'processing failed: {e}')
            return None

    # TensorFlow处理
    def process_tensorflow(self, weight_path, imgs, batch_size):
        """处理TensorFlow模型推理"""
        try:
            # 加载模型
            if weight_path.endswith('.h5'):
                model = tf.keras.models.load_model(weight_path)
            else:
                model = tf.saved_model.load(weight_path)
            # 预处理输入
            dataset = tf.data.Dataset.from_tensor_slices(imgs / 255.0) \
                .batch(batch_size) \
                .prefetch(tf.data.AUTOTUNE)
            if isinstance(model, tf.keras.Model):
                outputs = model.predict(dataset, verbose=0)
            else:
                infer = model.signatures['serving_default']
                outputs = []
                for batch in dataset:
                    outputs.append(infer(tf.constant(batch))['output_0'])
                outputs = tf.concat(outputs, axis=0).numpy()

            return (outputs * 255).clip(0, 255).astype(np.uint8)
        except Exception as e:
            self.sys_msg(f'processing failed: {e}')
            return None

    def sys_msg(self, msg):
        self.sysMSGLabel.setText(self.tr(f'<b>system message: {msg}</b>'))

    def update_ai_display(self):
        inputs_exist = get_var('Model_inputs_images_exists')
        outputs_exist = get_var('Model_outputs_images_exists')
        sliceNum = get_var("TotalModelInputsNum")
        if inputs_exist or outputs_exist:
            current_slice = get_var("CurrentAiSlice")
            update_inputs_display(self.parent(), current_slice)
            update_outputs_display(self.parent(), current_slice)

    def on_read_input_images_clicked(self):
        del_var("Model_inputs_images")
        del_var("Model_outputs_images")
        folder = get_var("Model_inputs_path")
        exts = ('.bmp', '.cur', '.dcx', '.gif', '.ico', '.im', '.jpeg', '.jpg', '.msp',
                '.pcx', '.png', '.ppm', '.sgi', '.spider', '.tga', '.tiff', '.xbm', '.xpm')
        files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
        set_var("Model_inputs_filenames", files)  # 保存文件名列表
        if not files:
            self.sysMSGLabel.setText(self.tr('no image data found.'))
            return

        imgs = []
        skipped = 0
        first_mode = None  # 'L' or 'RGB'
        h = int(self.ImgSizeLength_LineEdit.text())
        w = int(self.ImgSizeWidth_LineEdit.text())

        for fn in natsorted(files):
            path = os.path.join(folder, fn)
            with Image.open(path) as img:
                mode = img.mode
                if first_mode is None:
                    if mode == 'L':
                        first_mode = 'L'
                        set_var("ColorInputs", False)
                    else:
                        first_mode = 'RGB'
                        set_var("ColorInputs", True)

                # 判断图像是否符合类型
                if (first_mode == 'L' and mode != 'L') or (first_mode == 'RGB' and mode == 'L'):
                    skipped += 1
                    continue
                need_convert = (mode != first_mode)
                need_resize = (img.width != w or img.height != h)
                if need_convert:
                    # 转换模式
                    if first_mode == 'L':
                        img = img.convert('L')
                    else:
                        img = img.convert('RGB')
                if need_resize:
                    img = img.resize((w, h), Image.BILINEAR)
                    img_np = np.array(img)
                else:
                    img_np = np.array(img)
                imgs.append(img_np)
        if not imgs:
            self.sysMSGLabel.setText(self.tr('no valid image data found.'))
            return
        arr = np.stack(imgs, axis=0)
        set_var("Model_inputs_images", arr)
        set_var("Model_inputs_images_exists", True)
        set_var("TotalModelInputsNum", arr.shape[0])
        set_var("CurrentSlice", 1)
        if skipped > 0:
            if first_mode == 'L':
                self.sys_msg(f"skipped {skipped} color images to ensure consistency, only grayscale images are loaded for this time.")
            else:
                self.sys_msg(f"skipped {skipped} grayscale images to ensure consistency, only color images are loaded for this time.")
        else:
            self.sys_msg(f"{arr.shape[0]} images loaded.")
        # 更新显示
        set_var("CurrentAiSlice", 1)
        self.update_ai_display()
        update_ai_slice(self)

    def update_search_suffix(self):
        self.architectureComboBox.setDisabled(False)
        weights_path = get_var("Model_weights_path")
        architecture_path = get_var("Model_architecture_path")
        saveType = self.ModelTypeComboBox.currentText()
        if saveType == 'pytorch':
            architecture_files = self.filter_files_by_extension(architecture_path, '.py')
            self.update_comboBox(self.architectureComboBox, architecture_files)
            weights_files_pt = self.filter_files_by_extension(weights_path, '.pt')
            weights_files_pth = self.filter_files_by_extension(weights_path, '.pth')
            weights_files = weights_files_pt + weights_files_pth
            self.update_comboBox(self.WeightsComboBox, weights_files)
        elif saveType == 'tensorflow':
            architecture_files = self.filter_files_by_extension(architecture_path, '.py')
            self.update_comboBox(self.architectureComboBox, architecture_files)
            weights_files = os.listdir(weights_path)
            self.update_comboBox(self.WeightsComboBox, weights_files)
        else:
            self.architectureComboBox.setDisabled(True)
            weights_files = self.filter_files_by_extension(weights_path, '.mat')
            self.update_comboBox(self.WeightsComboBox, weights_files)

    def filter_files_by_extension(self, directory, extension):
        filtered_files = []
        for filename in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, filename)) and filename.endswith(extension):
                filtered_files.append(filename)
        return filtered_files

    def update_comboBox(self, comboBox, list):
        comboBox.clear()
        files = [f for f in list]
        comboBox.addItems(file for file in files)

    def on_select_path_clicked(self):
        """打开文件资源管理器选择保存路径"""
        selected_path = QFileDialog.getExistingDirectory(self, "please select input direction", self.inputs_path)
        if selected_path:
            self.inputs_path = selected_path
            self.path_label.setText(self.tr(self.inputs_path))
            set_var("Model_inputs_path", self.inputs_path)
            self.input_path_settings.setValue("model inputs path", self.inputs_path)
            self.sys_msg('new save path saved.')
        else:
            self.sys_msg('selecting canceled.')



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

        self.vBoxLayout.setSpacing(12)
        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.cardLayout.setContentsMargins(0, 0, 0, 0)
        self.vBoxLayout.addWidget(self.card, 0, Qt.AlignTop)
        self.vBoxLayout.setAlignment(Qt.AlignTop)

        self.cardLayout.setSpacing(2)
        self.cardLayout.setAlignment(Qt.AlignTop)
        self.cardLayout.addWidget(self.leftWidget)
        self.cardLayout.addWidget(self.rightWidget)

        if self.stretch == 0:
            self.cardLayout.addStretch(1)

        self.leftWidget.show()
        self.rightWidget.show()

    def eventFilter(self, source, event):
        """捕获鼠标滚轮事件"""
        inputs_exist = get_var('Model_inputs_images_exists')
        outputs_exist = get_var('Model_outputs_images_exists')
        sliceNum = get_var("TotalModelInputsNum")
        if inputs_exist or outputs_exist:
            if event.type() == QEvent.Wheel:
                delta = event.angleDelta().y() // 120  # 每次滚动幅度
                if source == self.leftWidget or source == self.rightWidget:
                    current_slice = get_var("CurrentAiSlice")
                    new_slice = max(1, current_slice + delta)  # 确保切片索引为正数
                    new_slice = min(sliceNum, new_slice)
                    set_var("CurrentAiSlice", new_slice)
                    update_inputs_display(self.container, new_slice)
                    update_outputs_display(self.container, new_slice)
                    update_ai_slice(self.parent1)
        return super().eventFilter(source, event)


class AIGalleryInterface(ScrollArea):
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
