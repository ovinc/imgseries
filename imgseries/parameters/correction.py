"""Classes to store correction parameters: flicker, shaking, etc.

These parameters are directly applied to images upon loading and are thus
impacting further analysis of the images, similarly to transform parameters.
"""


# Local imports
from ..process import divide
from .parameters_base import CorrectionParameter


class Flicker(CorrectionParameter):
    """Class to store flicker correction on image series"""

    parameter_name = 'flicker'

    def apply(self, img, num):
        """Flicker correction by dividing image by factor"""
        return divide(
            img=img,
            value=self.img_series.flicker.data['correction']['ratio'].loc[num]
        )


class Shaking(CorrectionParameter):
    """Class to store flicker correction transform"""

    parameter_name = 'shaking'

    def apply(self, img, num):
        """NOT IMPLEMENTED YET // TODO"""
        return img


All_Corrections = (
    Flicker,
    Shaking,
)

CORRECTIONS = {
    Correction.parameter_name: Correction for Correction in All_Corrections
}
