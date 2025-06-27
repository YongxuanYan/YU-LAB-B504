from PyQt5.QtGui import QPixmap
from .stereo_geo_gallery_interface import StereoGeoGalleryInterface
from ..common.config import cfg
from qfluentwidgets import ImageLabel


class StereoGeoInterface(StereoGeoGalleryInterface):
    """ StereoGeo interface """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setMaximumSize(1014, 712)
        self.setObjectName('StereoGeoInterface')
        self.left_label = ImageLabel()
        self.left_label.setFixedSize(512, 512)
        self.right_label = ImageLabel()
        self.right_label.setFixedSize(512, 512)
        cfg.blurRadius.valueChanged.connect(self.onBlurRadiusChanged)
        self.stereoGeo_updateLeftImage(QPixmap(':/gallery/images/GlobalCoordinate.png'))
        self.stereoGeo_updateRightImage(QPixmap(':/gallery/images/OID.png'))
        self.add2HExampleCard(
            self,
            self.left_label,
            self.right_label,
            stretch=1
        )

    def onBlurRadiusChanged(self, radius: int):
        self.right_label.blurRadius = radius
        self.left_label.blurRadius = radius

    def stereoGeo_updateLeftImage(self, left_image):
        self.left_label.setPixmap(left_image)

    def stereoGeo_updateRightImage(self, right_image):
        self.right_label.setPixmap(right_image)