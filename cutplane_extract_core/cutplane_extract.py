#!/usr/bin/env python3
"""
Main module to extract cut-planes from simulation solution files using argparse.

This module uses the antares API (Reader, Treatment, Writer) to read the mesh and
solution files, apply geometric cuts, compute gradients if needed, and save the output.
The extraction process is encapsulated in the Extractor class.
"""

import os
import sys
import time
import numpy as np
import h5py
import re
import builtins
# Import antares API components (assumed to be provided)
from antares import Reader, Treatment, Writer

# Import custom modules and initialization routines
from .config import TIP_GAP, SPAN, AOA
from .io_utils import select_folder, sort_files, compute_restart_parameters
from .cut_mapper_utils import map_cut
from .initialize_arguments import init, print_redirect


class CutplaneExtract:
    """
    A class to encapsulate the extraction of cut-planes from simulation solution files.
    """

    def __init__(self, args):
        """
        Initialize the Extractor instance.

        Args:
            args (Namespace): Parsed command-line arguments.
        """
        # Redirect stdout to a log file in the output directory.
        self.args = args                                # Store the command-line arguments
        if args.output == "Temp":
            default_output = f"Cut_{args.cut_selection}"
        if args.extract_VGT:
            default_output += "_VGT"
        self.args.output = default_output
        os.makedirs(self.args.output, exist_ok=True)    # Create the output directory if it doesn't exist
        log_filename = os.path.join(f'log_{os.path.basename(self.args.output)}.txt')    # Log file name
        sys.stdout = open(log_filename, "w", buffering=1)                               # Open log file for writing
        
        self.print = print_redirect
        self.print(f'\n{"Starting Cutplane Extract":=^100}\n')
        self.mesh_fileName = os.path.join(self.args.mesh_path, self.args.mesh_file)
        self.print(f'\n{"Initializing Cutplane Extract":.^60}\n')
        self.print('----> Directory Settings:')
        self.print(f'   The solution directory is: {self.args.sol_dir}')
        self.print(f'   The output directory is: {self.args.output}') 
        self.print(f'   The mesh directory is: {self.mesh_fileName}')
        self.print(f'   The cut style is: {self.args.cut_style}')
        self.print(f'   The angle of attack is: {self.args.AoA} degrees')
        self.print('----> Extraction Settings:')
        self.print(f'   The cut selection is: {self.args.cut_selection}')  
        self.print(f'   The starting folder index is: {self.args.nstart}')
        self.print(f'   The number of files to skip initially is: {self.args.mstart}')
        self.print(f'   The maximum number of files to extract is: {self.args.max_file}')
        self.print(f'   The restart option is: {self.args.restart}')
        self.print(f'   The velocity gradient tensor extraction option is: {self.args.extract_VGT}')
        self.print(f'\n{"Cutplane Extract Initialized":.^60}\n')

        # Build the full mesh file name from command-line arguments
        self.base_mesh = None

    def load_mesh(self):
        """
        Loads the mesh file using the antares Reader.

        Returns:
            The base mesh object.
        """
        self.print(f'\n{"Loading Mesh":.^60}\n')  
        self.print('----> Mesh Properties:')  
        r = Reader('hdf_avbp')
        r['filename'] = self.mesh_fileName
        r['shared'] = True  # Use the same mesh for all solutions
        base_mesh = r.read()
        base_mesh.show()
        self.print(f'\n{"Mesh Loaded":.^60}\n')  
        # Returns the base mesh object
        return base_mesh

    def check_velocity_gradient_tensor(self, solfile: str) -> bool:
        """
        Check if the velocity gradient tensor exists in the solution file.

        Args:
            solfile (str): Full path to the solution file.

        Returns:
            bool: True if the tensor exists, otherwise False.
        """
        velocity_gradient_components = [
            'du_dx', 'du_dy', 'du_dz',
            'dv_dx', 'dv_dy', 'dv_dz',
            'dw_dx', 'dw_dy', 'dw_dz'
        ]
        with h5py.File(solfile, 'r') as h5_file:
            key = 'Additionals'
            if key in h5_file:
                additionals_group = h5_file[key]
                for component in velocity_gradient_components:
                    if component not in additionals_group:
                        self.print(f"Component '{component}' does not exist!")
                        return False
                return True
            else:
                self.print("The key 'Additionals' does not exist in the file.")
                return False

    def process_solution_file(self, sol_file: str, count: int) -> int:
        """
        Process a single solution file: read, compute derived variables,
        apply cut treatment, and save output.

        Args:
            sol_file (str): The full path to the solution file.
            count (int): The global file count.

        Returns:
            int: The updated global file count.
        """
        self.print(f"       Reading full solution file: {sol_file}")
        r = Reader('hdf_avbp')
        r['base'] = self.base_mesh
        r['filename'] = sol_file
        comp = r.read()

        self.print("        Computing non-conservative variables")
        comp.compute('u=rhou/rho', location='node')
        comp.compute('v=rhov/rho', location='node')
        comp.compute('w=rhow/rho', location='node')
        comp.compute('Q=Q1+Q2', location='node')

        if self.args.extract_VGT:
            if self.check_velocity_gradient_tensor(sol_file):
                comp.rename_variables(
                    ['du_dx', 'du_dy', 'du_dz',
                     'dv_dx', 'dv_dy', 'dv_dz',
                     'dw_dx', 'dw_dy', 'dw_dz'],
                    ['grad_u_x', 'grad_u_y', 'grad_u_z',
                     'grad_v_x', 'grad_v_y', 'grad_v_z',
                     'grad_w_x', 'grad_w_y', 'grad_w_z'],
                    location=None
                )
                self.print("        Velocity gradient tensor exists; no gradient calculation required.")
            else:
                self.print("        Calculating gradient for velocity components")
                treatment_grad = Treatment('gradient')
                treatment_grad['base'] = comp
                treatment_grad['variables'] = ('u', 'v', 'w')
                comp = treatment_grad.execute()
                comp.cell_to_node()

        # Apply the cut or clip treatment based on the cut style.
        self.print(f"       Applying cut at: {self.args.cut_selection} with style: {self.args.cut_style}")
        if self.args.cut_style == 'plane':
            treatment_cut = Treatment('cut')
            treatment_cut['base'] = comp
            treatment_cut['type'] = 'plane'
            treatment_cut['base'].attrs['Time'] = float(count)
        else:
            treatment_cut = Treatment('clip')
            treatment_cut['base'] = comp
            treatment_cut['type'] = self.args.cut_style
            treatment_cut['axis'] = 'x'
            treatment_cut['radius'] = 0.1

        origin, normal = map_cut(self.args.cut_selection, self.args.cut_style, TIP_GAP, SPAN, AOA)
        treatment_cut['origin'] = origin
        treatment_cut['normal'] = normal
        # Execute the treatment
        comp = treatment_cut.execute()
        # Saving the cut
        self.print("        Saving solution")
        writer = Writer('hdf_antares')
        if self.args.extract_VGT:
            writer['base'] = comp[:, :, ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'dilatation',
                                          'pressure', 'vort_x', 'grad_u_x', 'grad_u_y', 'grad_u_z',
                                          'grad_v_x', 'grad_v_y', 'grad_v_z',
                                          'grad_w_x', 'grad_w_y', 'grad_w_z']]
            filename = os.path.join(self.args.output, f'B_10AOA_VGT_{self.args.cut_selection}_{count:03d}')
        else:
            writer['base'] = comp[:, :, ['x', 'y', 'z', 'u', 'v', 'w', 'dilatation', 'pressure', 'Q', 'rho', 'vort_x']]
            filename = os.path.join(self.args.output, f'B_{int(self.args.AoA)}AOA_{self.args.cut_selection}_{count:03d}')
        writer['filename'] = filename
        writer.dump()
        self.print(f"       Exported file: {filename}")

        return count + 1

    def process_solution_directories(self) -> None:
        """
        Iterates over solution directories and processes each file based on the provided arguments.
        """
        # Use the solution directory from command-line arguments
        sol_dirs, arr_dir, arr = select_folder(self.args.sol_dir)
        i_start, j_start, i_end, j_end, count, source_list = compute_restart_parameters(
            self.args.output, arr_dir, arr,
            self.args.nstart, self.args.restart, self.args.max_file, self.args.mstart
        )
        self.print(f'\n{"Iterating Solution Directories":.^60}\n')  
        for i in range(i_start, min(i_end + 1, len(sol_dirs))):
            current_dir = os.path.join(arr_dir, arr[i])
            self.print(f"\nProcessing directory: {current_dir}")
            files = sort_files(current_dir)
            j_start_current = j_start if i == i_start else 0
            j_end_current = j_end if i == i_end else len(files)
            for j in range(j_start_current, j_end_current):
                iteration_start = time.time()
                self.print(f"\n    Iteration: {count}")
                sol_file = os.path.join(self.args.sol_dir, arr[i], f"{files[j]}.h5")
                # Process the solution file
                count = self.process_solution_file(sol_file, count)
                elapsed = time.time() - iteration_start
                self.print(f"   Iteration time: {elapsed:1.0f} s")
        self.print(f'\n{"Complete Iterating Solution Directories":.^60}\n')  
    
    def timer(func):
        """ Decorator to time the function func to track the time taken for the function to run"""
        def inner(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            elapsed = end - start
            print('The total compute time is: {0:1.0f} s'.format(elapsed))
            return elapsed
        return inner
    
    @timer
    def run(self) -> None:
        """
        Runs the extraction process: loads the mesh and processes all solution files.
        """
        # Load the mesh and store it in the instance
        self.base_mesh = self.load_mesh()
        # Process all solution directories and files
        self.process_solution_directories()
        self.print(f'\n{"Cutplane Extract Complete":=^100}\n')


def main() -> None:
    """
    Main routine: initialize and run the Extractor.
    """
    args = init()  # Initialize (parse arguments, setup logging, etc.)
    extractor = CutplaneExtract(args)
    extractor.run()


if __name__ == '__main__':
    main()
