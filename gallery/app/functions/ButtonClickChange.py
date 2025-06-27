from qfluentwidgets import StateToolTip


def buttonClickWithFloatingWindow(C, WT, WMSG, CN, FT, BTN):
    """C: container, WT: window title, WMSG: windowMSG, CN: container_name, FT: First time click"""
    if FT:
        C.stateTooltip = StateToolTip(
            C.tr(WT), C.tr(WMSG), C.window())
        BTN.setText(C.tr(CN))
        C.stateTooltip.move(C.stateTooltip.getSuitablePos())
        C.stateTooltip.show()
    else:
        C.stateTooltip.setContent(
            C.tr(WT) + ' 😆 ' + WMSG)
        BTN.setText(C.tr(CN))
        C.stateTooltip.setState(True)
        C.stateTooltip = None
