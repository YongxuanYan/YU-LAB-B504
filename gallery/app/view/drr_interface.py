from qfluentwidgets import ImageLabel
from .drr_gallery_interface import DRRGalleryInterface
from ..common.config import cfg


class DRRInterface(DRRGalleryInterface):
    """ DRR interface """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setMaximumSize(1014, 812)
        self.setObjectName('DRRInterface')
        self.left_label = ImageLabel()
        self.left_label.setFixedSize(512, 512)
        self.right_label = ImageLabel()
        self.right_label.setFixedSize(512, 512)

        self.add2HExampleCard(
            self,
            self.left_label,
            self.right_label,
            stretch=1
        )

    def drr_updateLeftImage(self, left_image):
        self.left_label.setPixmap(left_image)

    def drr_updateRightImage(self, right_image):
        self.right_label.setPixmap(right_image)

