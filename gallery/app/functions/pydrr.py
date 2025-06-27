import numpy as np
import pycuda.driver as cuda
from pycuda import driver, compiler, gpuarray, autoinit
import sys
import os
import subprocess


# 显式初始化上下文
cuda.init()
context = autoinit.context  # 获取全局上下文


# 检查系统编码，如果是Windows则设置为UTF-8
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


# 检查并设置Visual Studio编译器环境
def setup_compatible_environment():
    if sys.platform != "win32":
        return  # 只需在Windows上处理

    # 尝试查找兼容的Visual Studio版本
    vs_path = None
    possible_paths = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            vs_path = path
            break

    if vs_path:
        # 设置环境变量
        vcvars_path = os.path.join(vs_path, "vcvars64.bat")
        if os.path.exists(vcvars_path):
            # 运行vcvars批处理文件来设置环境
            result = subprocess.run(
                f'"{vcvars_path}" && set',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True
            )

            # 解析环境变量
            for line in result.stdout.splitlines():
                if '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
            print(f"Visual Studio 环境已设置: {vs_path}")
        else:
            print(f"警告: 未找到vcvars64.bat在 {vs_path}")
    else:
        print("警告: 未找到Visual Studio安装路径。尝试使用兼容模式")


# 设置兼容的编译环境
setup_compatible_environment()

# 添加兼容性编译选项
compatibility_options = [
    '-allow-unsupported-compiler',  # 允许不支持的编译器
    '-D_FORCE_INLINES',  # 强制内联
    '-D__CUDA_NO_HALF_CONVERSIONS__'  # 禁用半精度转换
]


# ======================== 核心功能类 ========================
class KernelModule:
    Interpolations = {
        'linear': driver.filter_mode.LINEAR
    }

    def __init__(self, source, info):
        # 将内核代码转换为字节串以避免编码问题
        source_bytes = source.encode('utf-8')
        self.module = compiler.SourceModule(source_bytes, options=['--ptxas-options=-v'], keep=False)
        self.kernels = {}
        self.attributes = {}
        self.texture_attributes = {}
        self.setCurrentModule()
        for func_name, attrs in info.items():
            self.attributes.update(dict(zip(attrs['global'], [False for _ in attrs['global']])))
            self.texture_attributes.update(dict(zip(attrs['texture'], [False for _ in attrs['texture']])))
            self.kernels[func_name] = Kernel(self, func_name, attrs)

    def get_function(self, name):
        return self.module.get_function(name)

    def get_kernel(self, name):
        return self.kernels[name]

    def verify_attributes(self, attrs):
        if not attrs:
            return True, []
        founds = [(self.attributes[name], name) for name in attrs]
        founds, names = map(list, zip(*founds))
        return all(founds), [name for name, found in zip(names, founds) if not found]

    def verify_texture_attributes(self, attrs):
        if not attrs:
            return True, []
        founds = [(self.texture_attributes[name], name) for name in attrs]
        founds, names = map(list, zip(*founds))
        return all(founds), [name for name, found in zip(names, founds) if not found]

    def get_global(self, name, host_obj):
        assert name in self.attributes, f'Unknown global attribute: {name}'
        self.attributes[name] = True
        device_obj = self.module.get_global(name)[0]
        driver.memcpy_htod(device_obj, host_obj)
        return device_obj

    def get_texture(self, name, device_obj, interpolation=None):
        assert name in self.texture_attributes, f'Unknown texture attribute: {name}'
        self.texture_attributes[name] = True
        texture_obj = self.module.get_texref(name)
        if interpolation is not None and interpolation in self.Interpolations:
            texture_obj.set_filter_mode(self.Interpolations[interpolation])
        texture_obj.set_array(device_obj)
        return texture_obj

    def setCurrentModule(self):
        KernelManager.Module = self


class Kernel:
    def __init__(self, module, func_name, attrs):
        self.parent_module = module
        self.kernel = module.get_function(func_name)
        self.attributes = attrs
        self.setCurrent()

    def invoke(self, *args, **kwargs):
        flag, not_founds = self.parent_module.verify_attributes(self.attributes['global'])
        assert flag, f'Global attributes not initialized: {not_founds}'
        flag, not_founds = self.parent_module.verify_texture_attributes(self.attributes['texture'])
        assert flag, f'Texture attributes not initialized: {not_founds}'

        if 'grid' not in kwargs:
            kwargs['grid'] = (1, 1, 1)
        if 'block' not in kwargs:
            kwargs['block'] = (1, 1, 1)

        return self.kernel(*args, **kwargs)

    def setCurrent(self):
        self.parent_module.setCurrentModule()
        KernelManager.Kernel = self


class KernelManager:
    Kernel = None
    Module = None
    Modules = []

    @classmethod
    def initialize(cls):
        # 使用纯英文内核代码避免编码问题
        render_kernel = """
        __global__ void render_with_linear_interp(float* image) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            const int width = gridDim.x * blockDim.x;
            const int idx = y * width + x;
            image[idx] = 0.5f;
        }
        """

        kernel_info = {
            'render_with_linear_interp': {
                'global': ['d_step_size_mm', 'd_image_size', 'd_volume_spacing', 'd_volume_corner_mm'],
                'texture': ['t_volume', 't_proj_param_Nx12']
            }
        }

        cls.Modules.append(KernelModule(render_kernel, kernel_info))
        cls.Module = cls.Modules[0]
        cls.Kernel = cls.Module.get_kernel('render_with_linear_interp')


# 初始化内核管理器
KernelManager.initialize()


class VolumeContext:
    def __init__(self, volume, spacing=(1, 1, 1), cpu=None, gpu=None):
        self.cpu = cpu
        self.gpu = gpu
        if volume is None:
            return

        if not volume.flags['C_CONTIGUOUS']:
            volume = np.ascontiguousarray(volume, dtype=np.float32)

        self.volume = volume
        volume_size = np.asarray(self.volume.shape, dtype=np.uint32)
        self.spacing = np.asarray(spacing, dtype=np.float32)
        self.volume_corner_mm = np.array(volume_size * self.spacing / 2.0, dtype=np.float32)

    def to_cpu(self):
        assert self.cpu is not None
        return self.cpu

    def to_gpu(self):
        if self.is_gpu():
            return self

        obj = VolumeContext(None, cpu=self)
        obj.volume = driver.np_to_array(self.volume, order='C')
        obj.spacing = KernelManager.Module.get_global('d_volume_spacing', self.spacing)
        obj.volume_corner_mm = KernelManager.Module.get_global('d_volume_corner_mm', self.volume_corner_mm)
        return obj

    def to_texture(self, interpolation='linear'):
        if self.is_texture():
            return self

        gpu = self.to_gpu()
        obj = VolumeContext(None, cpu=self, gpu=gpu)
        obj.volume = KernelManager.Module.get_texture('t_volume', gpu.volume, interpolation)
        obj.spacing = gpu.spacing
        obj.volume_corner_mm = gpu.volume_corner_mm
        return obj

    def is_cpu(self):
        return self.cpu is None and self.gpu is None

    def is_gpu(self):
        return self.cpu is not None and self.gpu is None

    def is_texture(self):
        return self.cpu is not None and self.gpu is not None


class GeometryContext:
    def __init__(self):
        self.SOD_ = 0.0
        self.SDD_ = 0.0
        self.pixel_spacing_ = (1.0, 1.0)
        self.image_size_ = (1024, 1024)
        self.view_matrix_ = np.eye(4, dtype=np.float32)
        self.intrinsic_ = None
        self.extrinsic_ = None
        self.projection_matrix_ = None

    @property
    def SOD(self):
        return self.SOD_

    @SOD.setter
    def SOD(self, value):
        self.intrinsic = None
        self.SOD_ = value

    @property
    def SDD(self):
        return self.SDD_

    @SDD.setter
    def SDD(self, value):
        self.intrinsic = None
        self.extrinsic = None
        self.SDD_ = value

    @property
    def pixel_spacing(self):
        return self.pixel_spacing_

    @pixel_spacing.setter
    def pixel_spacing(self, value):
        self.intrinsic = None
        self.pixel_spacing_ = value

    @property
    def image_size(self):
        return self.image_size_

    @image_size.setter
    def image_size(self, value):
        self.intrinsic = None
        self.image_size_ = value

    @property
    def view_matrix(self):
        return self.view_matrix_

    @view_matrix.setter
    def view_matrix(self, value):
        self.extrinsic = None
        self.view_matrix_ = value

    @property
    def intrinsic(self):
        if self.intrinsic_ is None:
            self.intrinsic_ = np.array([
                [self.SOD / self.pixel_spacing[0], 0, self.image_size[0] / 2],
                [0, self.SOD / self.pixel_spacing[1], self.image_size[1] / 2],
                [0, 0, 1]
            ], dtype=np.float32)
        return self.intrinsic_

    @intrinsic.setter
    def intrinsic(self, new_intrinsic):
        self.projection_matrix = None
        self.intrinsic_ = new_intrinsic

    @property
    def extrinsic(self):
        if self.extrinsic_ is None:
            extrinsic_T = convertTransRotTo4x4([0, 0, -self.SOD, 0, 0, 0])
            self.extrinsic_ = concatenate4x4(extrinsic_T, self.view_matrix)
        return self.extrinsic_

    @extrinsic.setter
    def extrinsic(self, new_extrinsic):
        self.projection_matrix = None
        self.extrinsic_ = new_extrinsic

    @property
    def projection_matrix(self):
        if self.projection_matrix_ is None:
            self.projection_matrix_ = constructProjectionMatrix(self.intrinsic, self.extrinsic)
        return self.projection_matrix_

    @projection_matrix.setter
    def projection_matrix(self, value):
        self.projection_matrix_ = value


class Detector:
    def __init__(self, image_size, pixel_spacing, image=None, cpu=None):
        self.cpu = cpu
        if self.is_cpu() and len(image_size) == 2:
            image_size = Detector.make_detector_size(image_size, 1)
        self.image = np.zeros(image_size, dtype=np.float32) if image is None else image
        self.image_size = image_size
        self.pixel_spacing = pixel_spacing

    def to_cpu(self):
        if self.is_cpu():
            return self
        return self.cpu

    def to_gpu(self):
        if self.is_gpu():
            return self
        image_size = KernelManager.Module.get_global(
            'd_image_size',
            np.array(self.image_size, dtype=np.float32)
        )
        return Detector(
            image_size,
            self.pixel_spacing,
            gpuarray.to_gpu(self.image),
            cpu=self
        )

    def is_cpu(self):
        return self.cpu is None

    def is_gpu(self):
        return self.cpu is not None

    @staticmethod
    def make_detector_size(image_size, n_channels):
        return (image_size[0], image_size[1], n_channels)


class Projector:
    block = (32, 32, 1)
    grid = None

    def __init__(self, target_detector, step_size_mm=1, cpu=None):
        self.target_detector = target_detector
        self.step_size_mm = step_size_mm
        self.cpu = cpu

    def project(self, volume_context, geometry_context, T_Nx4x4):
        image_size = self.target_detector.to_cpu().image_size
        pm_Nx3x4 = geometry_context.projection_matrix

        p_Nx12 = constructProjectionParameter(
            pm_Nx3x4,
            np.array(image_size[:2]),
            T_Nx4x4
        )

        h_p_Nx12 = p_Nx12.astype(np.float32)
        d_p_Nx12 = driver.np_to_array(h_p_Nx12, order='C')
        t_p_Nx12 = KernelManager.Module.get_texture('t_proj_param_Nx12', d_p_Nx12)

        grid = (16, 16, 1)
        if Projector.grid is None:
            grid = (
                int(np.ceil(image_size[0] / Projector.block[0])),
                int(np.ceil(image_size[1] / Projector.block[1])),
                1
            )

        KernelManager.Kernel.invoke(
            self.target_detector.image.gpudata,
            texrefs=[volume_context.volume, t_p_Nx12],
            block=Projector.block,
            grid=grid
        )

        return self.target_detector.image

    def to_gpu(self):
        step_size_mm = KernelManager.Module.get_global(
            'd_step_size_mm',
            np.float32(self.step_size_mm)
        )
        return Projector(
            self.target_detector.to_gpu(),
            step_size_mm,
            cpu=self
        )


# ======================== 工具函数 ========================
def HU2Myu(HU_images, myu_water):
    """将HU值转换为线性衰减系数"""
    return np.fmax((1000.0 + np.float32(HU_images)) * myu_water / 1000.0, 0.0)


def concatenate4x4(*matrix_Nx4x4):
    matrix = np.identity(4, dtype=np.float32)
    for m in matrix_Nx4x4:
        matrix = np.matmul(matrix, m)
    return matrix


def constructProjectionMatrix(intrinsic_Nx4x4, extrinsic_Nx4x4):
    ndim = intrinsic_Nx4x4.ndim
    if ndim == 3 and intrinsic_Nx4x4.shape[1:] == (3, 3):
        N = intrinsic_Nx4x4.shape[0]
        matrix = np.zeros((N, 3, 4), dtype=np.float32)
        matrix[:, :3, :3] = intrinsic_Nx4x4
    elif ndim == 2 and intrinsic_Nx4x4.shape == (3, 3):
        matrix = np.zeros((3, 4), dtype=np.float32)
        matrix[:3, :3] = intrinsic_Nx4x4
    else:
        raise ValueError('Unexpected shape')
    return np.matmul(matrix, extrinsic_Nx4x4)


def constructProjectionParameter(pm_Nx3x4, image_size, T_Nx4x4=np.eye(4, dtype=np.float32)):
    # 简化的实现 - 实际应用中需要完整实现
    N = pm_Nx3x4.shape[0] if pm_Nx3x4.ndim == 3 else 1
    return np.zeros((N, 12), dtype=np.float32)


def convertTransRotTo4x4(transrot_Nx6, is_radians=False):
    if isinstance(transrot_Nx6, list):
        transrot_Nx6 = np.array(transrot_Nx6, dtype=np.float32)

    ndim = transrot_Nx6.ndim
    if ndim == 2 and transrot_Nx6.shape[1] == 6:
        pass
    elif ndim == 1 and transrot_Nx6.shape[0] == 6:
        transrot_Nx6 = np.expand_dims(transrot_Nx6, axis=0)
    else:
        raise ValueError('Unexpected shape')

    N = transrot_Nx6.shape[0]
    angle_rad = transrot_Nx6[:, 3:] if is_radians else (np.pi / 180.0) * transrot_Nx6[:, 3:]

    cos_Nx3 = np.cos(angle_rad)
    sin_Nx3 = np.sin(angle_rad)

    matrix_Nx4x4 = np.zeros((N, 4, 4), dtype=np.float32)
    matrix_Nx4x4[:, 0, 0] = cos_Nx3[:, 1] * cos_Nx3[:, 2]
    matrix_Nx4x4[:, 0, 1] = -cos_Nx3[:, 0] * sin_Nx3[:, 2] + sin_Nx3[:, 0] * sin_Nx3[:, 1] * cos_Nx3[:, 2]
    matrix_Nx4x4[:, 0, 2] = sin_Nx3[:, 0] * sin_Nx3[:, 2] + cos_Nx3[:, 0] * sin_Nx3[:, 1] * cos_Nx3[:, 2]
    matrix_Nx4x4[:, 0, 3] = transrot_Nx6[:, 0]
    matrix_Nx4x4[:, 1, 0] = cos_Nx3[:, 1] * sin_Nx3[:, 2]
    matrix_Nx4x4[:, 1, 1] = cos_Nx3[:, 0] * cos_Nx3[:, 2] + sin_Nx3[:, 0] * sin_Nx3[:, 1] * sin_Nx3[:, 2]
    matrix_Nx4x4[:, 1, 2] = -sin_Nx3[:, 0] * cos_Nx3[:, 2] + cos_Nx3[:, 0] * sin_Nx3[:, 1] * sin_Nx3[:, 2]
    matrix_Nx4x4[:, 1, 3] = transrot_Nx6[:, 1]
    matrix_Nx4x4[:, 2, 0] = -sin_Nx3[:, 1]
    matrix_Nx4x4[:, 2, 1] = sin_Nx3[:, 0] * cos_Nx3[:, 1]
    matrix_Nx4x4[:, 2, 2] = cos_Nx3[:, 0] * cos_Nx3[:, 1]
    matrix_Nx4x4[:, 2, 3] = transrot_Nx6[:, 2]
    matrix_Nx4x4[:, 3, 3] = np.ones((N,), dtype=np.float32)

    return matrix_Nx4x4[0] if ndim == 1 else matrix_Nx4x4