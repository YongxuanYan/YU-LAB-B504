from ..var.globals import get_var


def update_ai_slice(C):
    totalSliceNum = get_var("TotalModelInputsNum")
    if totalSliceNum > 0:
        currentSlice = get_var("CurrentAiSlice")
        C.currentSliceLabel.setText(C.tr(f'current slice: {str(currentSlice)} / {str(totalSliceNum)}'))
