"""
my_cut_extractor package

This package contains modules for extracting cut-planes from simulation solution files.

Modules included:
    - config: Contains default configuration settings.
    - io_utils: Contains file/folder utilities and restart logic.
    - cut_mapper: Contains functions to map the cut selection to geometric parameters.
    - init: Contains initialization routines (argument parsing, sync_print, etc.).
    - main: Contains the main routine to execute the extraction.
"""

from .config import *
from .io_utils import *
from .cut_mapper_utils import *
from .initialize_arguments import parse_args, init
from .cutplane_extract import CutplaneExtract, main