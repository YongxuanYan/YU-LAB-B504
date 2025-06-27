from qfluentwidgets import ImageLabel
from .ct_gallery_interface import CTGalleryInterface
from ..common.config import cfg


class CTInterface(CTGalleryInterface):
    """ CT interface """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        #self.setFixedSize(1060, 665)
        self.setMaximumSize(1014, 712)
        self.setObjectName('CTInterface')
        self.left_label = ImageLabel()
        self.left_label.setFixedSize(512, 512)
        self.right_label = ImageLabel()
        self.right_label.setFixedSize(512, 512)
        cfg.blurRadius.valueChanged.connect(self.onBlurRadiusChanged)

        self.add2HExampleCard(
            self,
            self.left_label,
            self.right_label,
            stretch=1
        )

    def onBlurRadiusChanged(self, radius: int):
        self.right_label.blurRadius = radius
        self.left_label.blurRadius = radius

    def ct_updateLeftImage(self, left_image):
        self.left_label.setPixmap(left_image)

    def ct_updateRightImage(self, right_image):
        self.right_label.setPixmap(right_image)