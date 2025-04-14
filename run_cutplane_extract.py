"""
python_wrapper to run CutplaneExtract

This script demonstrates how to call the CutplaneExtract class from the
cutplane_extract_core package with custom arguments using an AttrDict.
"""

from argparse import Namespace
from cutplane_extract_core.cutplane_extract import CutplaneExtract

def run_extraction(args):
    # If your CutplaneExtract class expects an argparse.Namespace,
    # you can convert your AttrDict to a Namespace:
    args_namespace = Namespace(**args)
    extractor = CutplaneExtract(args_namespace)
    extractor.run()
    
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
if __name__ == '__main__':
    # Create an AttrDict with your arguments.
    args = AttrDict()
    args.update({
        "cut_selection": "PIV1",                         # The cut location
        "output": "Temp",                                   # Output directory
        "nstart": 6,                                         # Starting folder index
        "mstart": 0,                                         # Files to skip initially
        "max_file": 100,                                     # Maximum number of files to extract
        "VGT": False,                                # Whether to extract the velocity gradient tensor
        "restart": True,                                    # Whether to restart extraction from the last file
        "cut_style": "plane",                                # Style of the cut (plane, cylinder, or sphere)
        "mesh_path": "/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_10AOA_LES/MESH_ZONE_Nov24/",  # Mesh directory
        "mesh_file": "Bombardier_10AOA_Combine_Nov24.mesh.h5",                                     # Mesh file name
        "sol_dir": "/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_10AOA_LES/RUN_ZONE_Nov24/SOLUT/",
        "AoA": 10,                                          # Angle of attack (if needed in your code)
        "treatment": "iso",                                  # Treatment type (iso or cut)
        "isovar": "Q",                                       # Isosurface variable
        "isovalue": 2e5,                                    # Isosurface value
    })
    
    run_extraction(args)