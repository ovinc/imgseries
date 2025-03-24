"""Classes to store correction parameters: flicker, shaking, etc.

These parameters are directly applied to images upon loading and are thus
impacting further analysis of the images, similarly to transform parameters.
"""


# Local imports
from ..process import divide
from .parameters_base import CorrectionParameter


class Flicker(CorrectionParameter):
    """Class to store flicker correction on image series"""

    name = 'flicker'

    def apply(self, data, num):
        """Flicker correction by dividing image by factor"""
        return divide(
            img=data,
            value=self.img_series.flicker.data['correction']['ratio'].loc[num]
        )


class Shaking(CorrectionParameter):
    """Class to store flicker correction transform"""

    name = 'shaking'

    def apply(self, data, num):
        """NOT IMPLEMENTED YET // TODO"""
        return data


All_Corrections = (
    Flicker,
    Shaking,
)

CORRECTIONS = {
    Correction.name: Correction for Correction in All_Corrections
}
