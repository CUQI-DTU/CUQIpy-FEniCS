from . import pde
from . import geometry
from . import testproblem

import dolfin as dl
dl.set_log_active(False)

from . import _version
__version__ = _version.get_versions()['version']
