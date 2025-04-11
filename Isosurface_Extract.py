####################################################################################
# Extracting the isosurface of the Q criterion for each time step as a movie
####################################################################################
# Importing the required modules
from antares import *
import numpy as np
import os
import vortexDetection as vd   
import matplotlib.pyplot as plt
import sys
import builtins
import glob
import time

# The following lines syncs the print function and sync the flushed file also on the operating system side
# The inputs
output = str(sys.argv[1])
nstart = int(sys.argv[2])
isovalue = "{:1.1e}".format(float(sys.argv[3]))
isovar = "Q"
restart = bool(sys.argv[4])

sys.stdout = open(os.path.join('log_'+output+'_'+str(isovalue)+'.txt'), "w", buffering=1)
def print(text):
    builtins.print(text)
    os.fsync(sys.stdout)


# loading the path and directory information
text = 'Loading File Directory'
print(f'\n{text:.^80}\n')  
# The mesh file
meshpath = '/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_10AOA_LES/MESH_ZONE_Nov24/'
meshfile = 'Bombardier_10AOA_Combine_Nov24.mesh.h5'
mesh_fileName = os.path.join(meshpath,meshfile)
# The solution path
sol_dirName = '/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_10AOA_LES/RUN_ZONE_Nov24/SOLUT/'
# The output path
projpath = ''
output_path = os.path.join(projpath,output+'_'+str(isovalue))
os.makedirs(output_path, exist_ok = True)
isovalue = float(isovalue)

# Iterating through the top level solution folders
arr_dir = os.path.join(sol_dirName)
arr = os.listdir(arr_dir)
arr.sort()
arr = list(arr)
sol_dir = np.array([])
for i in range(0,len(arr)):
    filename = arr[i]
    parts = filename.split('_')
    sol_dir_part = parts[len(parts)-1].split('.')[0]
    sol_dir = np.append(sol_dir, sol_dir_part)
sol_dir = np.unique(sol_dir)

# Constants
tip_gap = -0.1034
span = -0.2286
# Midspan
z_mid_span = (tip_gap + span/2)

# Function to take out the unique file names in the folder
def sort_files(rand):
    rand_sub = os.listdir(rand)
    rand_arr = np.array([]) 
    for i in range(0,np.shape(rand_sub)[0]):
        file_split = os.path.splitext(rand_sub[i])[0]
        rand_arr = np.append(rand_arr,file_split)
    rand_arr = [*set(rand_arr)] # removing the duplicates
    rand_arr = [ rand_arr for rand_arr in rand_arr if 'B_10AOA_Zone.sol_collection' not in rand_arr]
    rand_arr = [ rand_arr for rand_arr in rand_arr if 'last_solution' not in rand_arr]
    rand_arr.sort()
    return rand_arr


def restart_compute(output_path):
    # loading the counts of all the files in the solution directory
    count = 0
    source_list = np.zeros([1,3])
    for i in range(nstart,np.shape(sol_dir)[0]):
        dir = os.path.join(arr_dir,arr[i])
        text = 'Processing the directory: '; print(f'\n{text}{dir}') 
        files = sort_files(dir)
        for j in range(0,np.shape(files)[0]):
            if count == 0:
                source_list = [[i,j,count]]
            else:
                source_list = np.append(source_list,[[i,j,count]],axis=0)

            count+=1
    # Finding the latest re-start file from the source directory
    source_files = sorted(glob.glob('{0:s}/*.h5'.format(output_path)))
    filename = source_files[-1]
    dummy = filename.split('_')
    part = dummy[len(dummy)-1].split('.')[0]
    part = int(part)
    indexes = np.where(source_list[:,2]==part)
    i_start = int(source_list[indexes,0])
    j_start = int(source_list[indexes,1])
    return i_start, j_start




text = 'Reading Mesh'
print(f'\n{text:.^80}\n')  
# Reading the mesh
r = Reader('hdf_avbp')
r['filename'] = mesh_fileName
r['shared'] = True # Same mesh for all sol
b = r.read() # b is the Base object of the Antares API

text = 'Iterating Solution'
print(f'\n{text:.^80}\n') 
# looping over all main solver directories
count = 0
#converted_count = "% s" % count
for i in range(nstart,np.shape(sol_dir)[0]):
    dir = os.path.join(arr_dir,arr[i])
    text = 'Processing the directory: '; print(f'\n{text}{dir}') 
    files = sort_files(dir)
    for j in range(0,np.shape(files)[0]):
        text = 'Iteration: '; print(f'\n{text}{count}') 
        text = '    Reading full solution file '; print(f'{text}:{files[j]}') 
        sol_file = os.path.join(sol_dirName,arr[i],files[j]+'.h5') # The full solution file
        r = Reader('hdf_avbp')
        r['base'] = b
        r['filename'] = sol_file
        calc_t = time.time()
        comp_q = r.read()
        comp_q.compute(f'u=rhou/rho',location='node')
        comp_q.compute(f'v=rhov/rho',location='node')
        comp_q.compute(f'w=rhow/rho',location='node')
        
        if isovar == "L2":
            text = '    Computing Gradient'; print(f'{text}')
            treatment = Treatment('gradient')
            treatment['base'] = comp_q
            treatment['variables']=('u','v','w')
            comp_q = treatment.execute()
            comp_q.cell_to_node()
            text = '    Computing L2'; print(f'{text}')
            # Assigning your velocity variables
            vd.velocity_names('u','v','w')
            # Assigning your velocity gradients
            vd.velocity_gradient_names('grad_u_x','grad_u_y','grad_u_z','grad_v_x','grad_v_y','grad_v_z','grad_w_x','grad_w_y','grad_w_z')
            # Computing invariants
            comp_q=vd.vorti(comp_q)
            comp_q=vd.lambda_2(comp_q, eigVect=False)
            if isovalue > 0:
                isovalue = -1*isovalue
            text = '    Gradient and L2 Computation Complete'; print(f'{text}')
        else:
            text = '    Computing IsoQ'; print(f'{text}') 
            comp_q.compute(f'Q=Q1+Q2',location='node')
        text = '    Performing Isosurface'; print(f'{text}') 
        treatment = Treatment('isosurface')
        treatment['base'] = comp_q
        treatment['variable'] = isovar
        treatment['base'] .attrs['Time'] = float(count)
        treatment['value'] = isovalue
        isosurf = treatment.execute()
        text = 'Saving solution'; print(f'{text}') 
        w = Writer('hdf_antares')
        w['filename'] = os.path.join(output_path,'B_10AOA_{0:s}_anim_{1:03d}'.format(isovar,count)) 
        w['base'] = isosurf[:,:,['x','y','z','u']]
        w.dump()
        print('    The file '+w['filename'] +' is exported')
        elapsed = time.time() - calc_t
        print('    The iteration calcualtion time is: {0:1.0f} s'.format(elapsed))
        count += 1
        
text = 'Solution Iteration Complete'
print(f'\n{text:.^80}\n') 
