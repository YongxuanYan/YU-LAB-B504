import os
import json
import tempfile
import atexit
import numpy as np

# 创建一个临时文件夹，用于存储全局变量的 JSON 文件
TEMP_DIR = tempfile.TemporaryDirectory()  # 自动管理临时文件夹
GLOBAL_VARS_FILE = os.path.join(TEMP_DIR.name, "global_vars.json")

# 初始化全局变量
GLOBAL_VARS = {
    "ColorInputs": False,
    "CouchAngle": None,
    "CurrentSlice": 1,
    "CurrentDRRSlice": 1,
    "CurrentAiSlice": 1,
    "ctexist": 0,
    "CT_MAX_HU": 0,
    "DRREXISTS_Left": 0,
    "DRREXISTS_Right": 0,
    "DRR_Resolution": None,
    "DRRs": None,
    "Geoinfo_save_path": None,
    "I0": 1200,
    "ISO_X": None,
    "ISO_Y": None,
    "ISO_Z": None,
    "ImagingPair": None,
    "Imaging_pair_1_enabled": 0,
    "Imaging_pair_2_enabled": 0,
    "Imaging_pair_1_fileName": None,
    "Imaging_pair_2_fileName": None,
    "labeldata": None,
    "labelexits": 0,
    "Left_DRR": None,
    "LocationOfNotCTData": None,
    "MaxWindowLevel": 3071,
    "MaxWindowWidth": 4095,
    "MinWindowLevel": -1024,
    "Model_architecture_path": None,
    "Model_inputs_images": None,
    "Model_inputs_images_exists": False,
    "Model_inputs_path": None,
    "Model_loading_msg": '',
    "Model_outputs_images": None,
    "Model_outputs_images_exists": False,
    "Model_outputs_path": None,
    "Model_weights_path": None,
    "NoMoreAsking": False,
    "PDepth": None,
    "PHeight": None,
    "PWidth": None,
    "PatientName": None,
    "PixelSpacing": None,
    "PixelsGrid": None,
    "Right_DRR": None,
    "SliceLocation": None,
    "SliceNum": 0,
    "SliceThickness": None,
    "SysMSG": "Welcome to YU LAB-B504.",
    "TileSize": None,
    "TotalModelInputsNum": 0,
    "WindowLevel": -500,
    "WindowWidth": 1500,
}


# 初始化 JSON 文件
def initialize_global_vars():
    """初始化 JSON 文件，将全局变量写入文件"""
    with open(GLOBAL_VARS_FILE, 'w') as f:
        json.dump(GLOBAL_VARS, f, indent=4)


# 设置全局变量
def set_var(var_name, value):
    """设置全局变量的值"""
    if GLOBAL_VARS_FILE is None:
        raise RuntimeError("Global variables have not been initialized.")
    with open(GLOBAL_VARS_FILE, 'r') as f:
        global_vars = json.load(f)

    # 对于 ndarray 类型的数据，单独保存为 .npy 文件
    if isinstance(value, np.ndarray):
        npy_path = os.path.join(TEMP_DIR.name, f"{var_name}.npy")
        np.save(npy_path, value)
        value = {"type": "ndarray", "path": npy_path}

    if var_name in global_vars:
        global_vars[var_name] = value
        with open(GLOBAL_VARS_FILE, 'w') as f:
            json.dump(global_vars, f, indent=4)
    else:
        raise KeyError(f"Variable '{var_name}' is not defined in GLOBAL_VARS.")


def get_var(var_name):
    """获取全局变量的值"""
    if GLOBAL_VARS_FILE is None:
        raise RuntimeError("Global variables have not been initialized.")
    with open(GLOBAL_VARS_FILE, 'r') as f:
        global_vars = json.load(f)

    if var_name in global_vars:
        value = global_vars[var_name]
        # 如果是 ndarray 的路径，加载 .npy 文件
        if isinstance(value, dict) and value.get("type") == "ndarray":
            npy_path = value.get("path")
            if os.path.exists(npy_path):
                value = np.load(npy_path)
            else:
                raise FileNotFoundError(f"File {npy_path} does not exist.")
        return value
    else:
        raise KeyError(f"Variable '{var_name}' is not defined in GLOBAL_VARS.")


def del_var(var_name):
    npy_path = os.path.join(TEMP_DIR.name, f"{var_name}.npy")
    if os.path.exists(npy_path):
        os.remove(npy_path)


# 退出程序时删除临时文件夹
@atexit.register
def cleanup_temp_dir():
    """程序退出时删除临时文件夹"""
    TEMP_DIR.cleanup()
