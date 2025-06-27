# coding:utf-8
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor, QBrush, QPainterPath, QLinearGradient
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from typing import Union
from qfluentwidgets import ScrollArea, isDarkTheme
from qfluentwidgets import FluentIcon as FIF
from ..common.config import cfg, HELP_URL, REPO_URL, EXAMPLE_URL, FEEDBACK_URL
from ..common.icon import Icon, FluentIconBase
from ..components.link_card import LinkCardView
from ..components.sample_card import SampleCardView
from ..common.style_sheet import StyleSheet


class BannerWidget(QWidget):
    """ Banner widget """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setFixedHeight(336)

        self.vBoxLayout = QVBoxLayout(self)
        self.galleryLabel = QLabel('YU-LAB-B504', self)
        self.banner = QPixmap(':/gallery/images/Lab.png')
        self.linkCardView = LinkCardView(self)

        self.galleryLabel.setObjectName('galleryLabel')

        self.vBoxLayout.setSpacing(0)
        self.vBoxLayout.setContentsMargins(0, 20, 0, 0)
        self.vBoxLayout.addWidget(self.galleryLabel)
        self.vBoxLayout.addWidget(self.linkCardView, 1, Qt.AlignBottom)
        self.vBoxLayout.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.linkCardView.addCard(
            Icon.BOOK,
            self.tr('Getting started'),
            self.tr('You can check how to get started using this APP by reading this document.'),
            HELP_URL
        )

        self.linkCardView.addCard(
            Icon.GITHUB,
            self.tr('Contact/Feedback'),
            self.tr('Please contact us via d002wcu@yamaguchi-u.ac.jp'),
            FEEDBACK_URL
        )

    def paintEvent(self, e):
        super().paintEvent(e)
        painter = QPainter(self)
        painter.setRenderHints(
            QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)

        path = QPainterPath()
        path.setFillRule(Qt.WindingFill)
        w, h = self.width(), self.height()
        path.addRoundedRect(QRectF(0, 0, w, h), 10, 10)
        path.addRect(QRectF(0, h-20, 20, 20))
        path.addRect(QRectF(w-20, 0, 20, 20))
        path.addRect(QRectF(w-20, h-20, 20, 20))
        path = path.simplified()

        # init linear gradient effect
        gradient = QLinearGradient(0, 0, 0, h)

        # draw background color
        if not isDarkTheme():
            gradient.setColorAt(0, QColor(207, 216, 228, 255))
            gradient.setColorAt(1, QColor(207, 216, 228, 0))
        else:
            gradient.setColorAt(0, QColor(0, 0, 0, 255))
            gradient.setColorAt(1, QColor(0, 0, 0, 0))
            
        painter.fillPath(path, QBrush(gradient))

        # draw banner image
        pixmap = self.banner.scaled(
            self.size(), transformMode=Qt.SmoothTransformation)
        painter.fillPath(path, QBrush(pixmap))



class HomeInterface(ScrollArea):
    """ Home interface """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.banner = BannerWidget(self)
        self.view = QWidget(self)
        self.vBoxLayout = QVBoxLayout(self.view)

        self.__initWidget()
        self.loadSamples()


    def __initWidget(self):
        self.view.setObjectName('view')
        self.setObjectName('homeInterface')
        StyleSheet.HOME_INTERFACE.apply(self)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidget(self.view)
        self.setWidgetResizable(True)

        self.vBoxLayout.setContentsMargins(0, 0, 0, 36)
        self.vBoxLayout.setSpacing(40)
        self.vBoxLayout.addWidget(self.banner)
        self.vBoxLayout.setAlignment(Qt.AlignTop)

    def loadSamples(self):
        """ load samples """
        basicInputView = SampleCardView(
            self.tr("Navigation"), self.view)
        basicInputView.addSampleCard(
            Icon.CT,
            title="CT",
            content=self.tr(
                "You can import and view CT data here, if tumor contour data is included, binary tumor CT will be automatically generated."),
            routeKey="CTInterface",
            index=0
        )
        basicInputView.addSampleCard(
            Icon.DRR,
            title="DRR",
            content=self.tr("Generate DRR based on imported CT, or import and view fluoroscopic images here."),
            routeKey="DRRInterface",
            index=0
        )
        basicInputView.addSampleCard(
            Icon.AI,
            title="AI",
            content=self.tr(
                "You can do deep learning based image processing and analysis here. Import a model and feed DRR or fluoroscopic images to it to get reports."),
            routeKey="AIInterface",
            index=0
        )

        basicInputView.addSampleCard(
            FIF.SETTING,
            title="Setting",
            content=self.tr(
                "Set up, save, load or delete geometry parameters of stereo X-ray imaging systems."),
            routeKey="StereoGeoInterface",
            index=0
        )
        self.vBoxLayout.addWidget(basicInputView)
