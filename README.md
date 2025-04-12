# Cutplane Extractor

Cutplane Extractor is a Python package designed to extract cut-planes (or clip regions) from simulation solution files. It leverages the Antares API (Reader, Treatment, Writer) to read mesh and solution data, apply geometric cuts based on user-defined parameters, compute additional derived variables (such as gradients), and export the resulting data for postprocessing.

## Features

- **Flexible Cut Selection:**  
  Supports different cut locations (e.g., "midspan", "25mm_tip", "PIV1", etc.) and cut styles (plane, cylinder, or sphere).

- **Velocity Gradient Extraction:**  
  Optionally compute or retrieve the velocity gradient tensor from simulation files.

- **Restart Capability:**  
  Restart extraction from a specific file, enabling parallel execution or recovery from partial runs.

- **Command-Line and Programmatic Interfaces:**  
  Run the extraction process via a command-line interface or import and use the functionality programmatically in your own scripts.

- **Customizable Output Directory:**  
  The output directory can be specified via a command-line argument; if not provided, it defaults to a name based on the cut selection (and appends `_VGT` if velocity gradient extraction is enabled).

## Directory Structure


> **Note:** In this project the core functionality is encapsulated in the `cutplane_extract_core` package. The defaults for mesh path, solution directory, and other parameters are now specified via command-line arguments (with built-in defaults) rather than in the configuration file to avoid confusion.

## Installation

### Option 1: Install via setup.py

1. Clone or download the repository.
2. In the repository root, run:

   ```bash
   pip install --user .
   ```
### Option 2: Install python path
If you prefer not to install the package, add the repository’s root directory to your PYTHONPATH:
   ```bash
export PYTHONPATH="/path/to/Cutplane_Extract:$PYTHONPATH"
   ```
## Usage
Command-Line Interface
After installation or setting the PYTHONPATH, you can run the extraction using the provided wrapper or the bash script
