from . import pde
from . import geometry
from . import testproblem

import dolfin as dl
dl.set_log_active(False)

from . import _version
__version__ = _version.get_versions()['version']

#TODO: This logic need to be removed after _get_identity_geometries 
# is removed from cuqipy 
from cuqi.geometry import _identity_geometries
_identity_geometries.append(geometry.FEniCSContinuous)
