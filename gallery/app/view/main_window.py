# coding: utf-8
from PyQt5.QtCore import QSettings, QUrl, QSize, QTimer, QPoint
from PyQt5.QtGui import QIcon, QDesktopServices
from PyQt5.QtWidgets import QApplication
from qfluentwidgets import (NavigationItemPosition, FluentWindow,
                            SplashScreen, SystemThemeListener, isDarkTheme)
from qfluentwidgets import FluentIcon as FIF
from .ct_gallery_interface import CTGalleryInterface
from .ai_gallery_interface import AIGalleryInterface
from .drr_gallery_interface import DRRGalleryInterface
from .home_interface import HomeInterface
from .ct_interface import CTInterface
from .ai_interface import AIInterface
from .drr_interface import DRRInterface
from  .stereo_geo_interface import StereoGeoInterface
from ..common.config import ZH_SUPPORT_URL, EN_SUPPORT_URL, cfg
from ..common.icon import Icon
from ..common.signal_bus import signalBus
from ..common.translator import Translator
from ..common import resource
from ..var.globals import get_var
from ..var.globals import initialize_global_vars


class MainWindow(FluentWindow):
    def __init__(self):
        super().__init__()
        # global_vars.json 用于存储全局变量，如CT数据，LBCT数据等
        initialize_global_vars()
        self.settings = QSettings("YU-LAB-B504", "MainWindowSizePosition")  # 设置组和应用名称
        self.initWindow()
        # create system theme listener
        self.themeListener = SystemThemeListener(self)
        # create sub interface
        self.homeInterface = HomeInterface(self)
        self.ctInterface = CTInterface(self)
        self.aiInterface = AIInterface(self)
        self.drrInterface = DRRInterface(self)
        self.stereoGeoInterface = StereoGeoInterface(self)
        #self.basicInputInterface = BasicInputInterface(self)
        # enable acrylic effect
        self.navigationInterface.setAcrylicEnabled(True)
        self.connectSignalToSlot()
        # add items to navigation interface
        self.initNavigation()
        self.splashScreen.finish()
        # start theme listener
        self.themeListener.start()

    def connectSignalToSlot(self):
        signalBus.micaEnableChanged.connect(self.setMicaEffectEnabled)
        signalBus.switchToSampleCard.connect(self.switchToSample)
        signalBus.supportSignal.connect(self.onSupport)

    def initNavigation(self):
        # add navigation items
        t = Translator()
        self.addSubInterface(self.homeInterface, FIF.HOME, 'Home')
        #加入分割线
        self.navigationInterface.addSeparator()
        pos = NavigationItemPosition.SCROLL
        self.addSubInterface(self.ctInterface, Icon.CT, 'CT')
        self.addSubInterface(self.drrInterface, Icon.DRR, 'DRR')
        self.addSubInterface(self.aiInterface, Icon.AI, 'AI')
        self.addSubInterface(self.stereoGeoInterface, FIF.SETTING, 'Setting')
    def initWindow(self):
        self.load_window_settings()
        self.setWindowIcon(QIcon(':/gallery/images/logo.png'))
        self.setWindowTitle('YU-LAB-B504')
        self.setMicaEffectEnabled(cfg.get(cfg.micaEnabled))
        # create splash screen
        self.splashScreen = SplashScreen(self.windowIcon(), self)
        self.splashScreen.setIconSize(QSize(600, 600))
        self.splashScreen.raise_()
        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w//2 - self.width()//2, h//2 - self.height()//2)
        self.show()
        QApplication.processEvents()

    def onSupport(self):
        language = cfg.get(cfg.language).value
        if language.name() == "zh_CN":
            QDesktopServices.openUrl(QUrl(ZH_SUPPORT_URL))
        else:
            QDesktopServices.openUrl(QUrl(EN_SUPPORT_URL))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, 'splashScreen'):
            self.splashScreen.resize(self.size())

    def closeEvent(self, e):
        self.themeListener.terminate()
        self.themeListener.deleteLater()
        self.save_settings()
        super().closeEvent(e)

    def save_settings(self):
        """保存窗口大小和位置"""
        self.settings.setValue("size", self.size())  # 保存窗口尺寸
        self.settings.setValue("pos", self.pos())  # 保存窗口位置
        save_path = get_var("Geoinfo_save_path")
        self.settings.setValue("save path",save_path)
        model_input_path = get_var("Model_inputs_path")
        self.settings.setValue("model inputs path", model_input_path)


    def load_window_settings(self):
        """加载窗口大小和位置"""
        size = self.settings.value("size", QSize(1070, 812))  # 默认大小 800x600
        pos = self.settings.value("pos", QPoint(0, 0))  # 默认位置 (100, 100)
        self.resize(size)
        self.move(pos)

    def _onThemeChangedFinished(self):
        super()._onThemeChangedFinished()
        # retry
        if self.isMicaEffectEnabled():
            QTimer.singleShot(100, lambda: self.windowEffect.setMicaEffect(self.winId(), isDarkTheme()))

    def switchToSample(self, routeKey):
        """ switch to sample """
        if routeKey == 'CTInterface':
            self.stackedWidget.setCurrentWidget(self.findChildren(CTGalleryInterface)[0], False)
        elif routeKey == 'AIInterface':
            self.stackedWidget.setCurrentWidget(self.findChildren(AIGalleryInterface)[0], False)
        elif routeKey == 'DRRInterface':
            self.stackedWidget.setCurrentWidget(self.findChildren(DRRGalleryInterface)[0], False)
        elif routeKey == 'StereoGeoInterface':
            self.stackedWidget.setCurrentWidget(self.findChildren(StereoGeoInterface)[0], False)


