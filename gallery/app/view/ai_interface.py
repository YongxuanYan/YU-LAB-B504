from qfluentwidgets import ImageLabel
from .ai_gallery_interface import AIGalleryInterface
from ..common.config import cfg


class AIInterface(AIGalleryInterface):
    """ CT interface """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setMaximumSize(1024, 712)
        self.setObjectName('AIInterface')
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

    def updateLeftImage(self, left_image):
        self.left_label.setPixmap(left_image)

    def updateRightImage(self, right_image):
        self.right_label.setPixmap(right_image)