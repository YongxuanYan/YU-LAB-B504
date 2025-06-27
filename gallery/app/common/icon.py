# coding: utf-8
from enum import Enum

from qfluentwidgets import FluentIconBase, getIconColor, Theme


class Icon(FluentIconBase, Enum):

    GRID = "Grid"
    MENU = "Menu"
    TEXT = "Text"
    PRICE = "Price"
    EMOJI_TAB_SYMBOLS = "EmojiTabSymbols"
    CT = "box"
    AI = "cpu"
    IMPORT = "download"
    DRR = "eye"
    GITHUB = "github"
    LAYERS = "layers"
    LOADER = "loader"
    LOGIN = "log-in"
    LOGOUT = "log-out"
    PAUSE = "pause"
    PLAY = "play"
    REFRESH = "refresh-ccw"
    USER = 'user'
    MAIL = 'mail'
    USERCK = 'user-check'
    BOOK = "book-open"
    SAVE = "save"
    HD = "hard-drive"
    DEL = 'trash'
    FOLDER = 'folder'

    def path(self, theme=Theme.AUTO):
        return f":/gallery/images/icons/{self.value}_{getIconColor(theme)}.svg"
