"""
initialize_arguments.py for cutplane_extractor, a module for extracting cut-planes from simulation solution files.
This module handles command-line argument parsing, output directory creation, and logging setup.
"""
import argparse
import os
import sys
import builtins

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments using argparse.
    
    Returns:
        Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Extract cut-planes from simulation solution files."
    )
    
    parser.add_argument("-nstart", type=int, default=0, help="Starting folder index (default: 0).")
    parser.add_argument("-mstart", type=int, default=0, help="Number of files to skip initially (default: 0).")
    parser.add_argument("-max_file", type=int, required=True, help="Maximum number of files to extract.")
    parser.add_argument("-AoA", type=int, default=10, help="Angle of attack degrees (default: 10).")
    parser.add_argument("-VGT", action="store_true", help="Extract the velocity gradient tensor.")
    parser.add_argument("-restart", action="store_true", help="Restart extraction from last file in output directory.")
    parser.add_argument("-cut_style", required=True, choices=['plane', 'cylinder', 'sphere'],
                        help="Style of the cut: plane, cylinder, or sphere.")
    parser.add_argument("--cut_selection", required=True, help="Cut selection location. Must be in the form of <%>_TE for spanwise cut or <mm>_tip for streamwise cut.")
    parser.add_argument("--output", required=False, default="Temp", help="Output directory path.")
    # Arguments for mesh locations and solution directory
    parser.add_argument("--mesh_path", default='/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_10AOA_LES/MESH_ZONE_Apr24/',
                        help="Path to the mesh directory.")
    parser.add_argument("--mesh_file", default='Bombardier_10AOA_Combine_Apr24.mesh.h5',
                        help="Name of the mesh file.")
    parser.add_argument("--sol_dir", default='/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_10AOA_LES/RUN_ZONE_Apr24/SOLUT/',
                        help="Path to the solution directory.")
    
    return parser.parse_args()

        
def print_redirect(text):
    builtins.print(text)
    os.fsync(sys.stdout.fileno())
    
def init() -> argparse.Namespace:
    """
    Handles initialization by parsing command-line arguments,
    
    Returns:
        Namespace: Parsed command-line arguments.
    """
    args = parse_args()
    return args
