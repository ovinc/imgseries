"""Classes to store correction parameters: flicker, shaking, etc.

These parameters are directly applied to images upon loading and are thus
impacting further analysis of the images, similarly to transform parameters.
"""


# Local imports
from .parameters_base import CorrectionParameter


class Flicker(CorrectionParameter):
    """Class to store flicker correction on image series"""

    parameter_type = 'flicker'


class Shaking(CorrectionParameter):
    """Class to store flicker correction transform"""

    parameter_type = 'shaking'


all_corrections = (
    Flicker,
    Shaking,
)

Corrections = {
    correction.parameter_type: correction for correction in all_corrections
}
