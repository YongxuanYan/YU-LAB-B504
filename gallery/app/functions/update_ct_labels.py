from ..var.globals import get_var


def update_ct_info(C):
    """C: container"""
    PixelSpacing = get_var('PixelSpacing')
    if PixelSpacing is not None:
        C.pixelSpacingLabel.setText(C.tr(f'pixel spacing: {str(PixelSpacing)} (mm)     '))
        SliceThickness = get_var('SliceThickness')
        C.sliceThicknessLabel.setText(C.tr(f'slice thickness: {str(SliceThickness)} (mm)      '))
        labelexits = get_var('labelexits')
        lbexist = 'Yes' if labelexits else 'No'
        C.labelDataExistLabel.setText(C.tr(f'tumor contour exists: {lbexist}     '))
        PN = get_var('PatientName')
        C.patientNameLabel.setText(C.tr(f'patient name: {PN}      '))


def update_slice(C):
    totalSliceNum = get_var("SliceNum")
    if totalSliceNum > 0:
        currentSlice = get_var("CurrentSlice")
        C.currentSliceLabel.setText(C.tr(f'current slice: {str(currentSlice)} / {str(totalSliceNum)}'))



