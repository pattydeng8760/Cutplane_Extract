"""
Configuration module for extracting cut-planes from simulation data.

This module contains default configuration settings such as:
  - Mesh paths (MESH_PATH and MESH_FILE)
  - Solution directory (SOL_DIR)
  - Default sizing parameters (TIP_GAP, SPAN, AOA)
  
These defaults can be overridden by command-line arguments.
"""

import os

# Default paths (update these paths as needed)
MESH_PATH = '/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_10AOA_LES/MESH_ZONE_Apr24/'
MESH_FILE = 'Bombardier_10AOA_Combine_Apr24.mesh.h5'
SOL_DIR = '/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_10AOA_LES/RUN_ZONE_Apr24/SOLUT/'

# Default sizing parameters for the cut (can be overridden later)
TIP_GAP = -0.1034
SPAN = -0.2286
AOA = 10

def get_default_mesh_filename() -> str:
    """
    Return the full default path to the mesh file.

    Returns:
        str: The full path constructed from MESH_PATH and MESH_FILE.
    """
    return os.path.join(MESH_PATH, MESH_FILE)
