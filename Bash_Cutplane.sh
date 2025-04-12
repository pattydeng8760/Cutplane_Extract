#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=0-08:59
#SBATCH --job-name=Cut_Plane_25mm_tip
#SBATCH --mail-user=patrickgc.deng@mail.utoronto.ca
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-plavoie

source /project/m/moreaust/Env/avbpNpy_env.sh
use_py_tools
export PYTHONPATH="/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_10AOA_LES/Isosurface/Cutplane_Extract:$PYTHONPATH"

echo "Starting direct run of CutplaneExtract class with shell arguments..."

python -m cutplane_extract_core.cutplane_extract \
  --cut_selection "25mm_tip" \
  --output "Temp" \
  --nstart 6 \
  --mstart 0 \
  --max_file 1000 \
  --cut_style "plane" \
  --mesh_path "/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_10AOA_LES/MESH_ZONE_Nov24/" \
  --mesh_file "Bombardier_10AOA_Combine_Nov24.mesh.h5" \
  --sol_dir "/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_10AOA_LES/RUN_ZONE_Nov24/SOLUT/" \
  --AoA 10

echo "Direct run of CutplaneExtract completed."
