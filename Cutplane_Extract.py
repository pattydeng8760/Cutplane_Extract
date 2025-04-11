####################################################################################
# Extracting cuts or clips at desired locations for post processing
# Author Patrick Deng
####################################################################################

# Importing the required modules
from antares import *
import numpy as np
from sys import argv
import os
import matplotlib.pyplot as plt
import sys
import glob 
import builtins
import re
import h5py
import time

# The input parameters
cut_selection = argv[1]          # The cut location
output = argv[2]                 # The desired output path
nstart = int(argv[3])            # The starting folder count based in SOLUT
mstart = int(argv[4])            # The number of files to skip initially (to enable launching multiple files at the same time), this parameter is not applied in restart
max_file = int(argv[5])          # The maximum number of files to extract
extract_VGT = eval(sys.argv[6])  # True or False to extract the velocity gradient tensor
assert isinstance(extract_VGT , bool)
restart =  eval(sys.argv[7])     # True or False to apply restart, if True, the code will restart from the last file in the output directory
assert isinstance(restart , bool)
cut_style = argv[8].lower()        # The style of the cut, either 'plane' or 'cylinder'
assert cut_style in ['plane', 'cylinder','sphere']

# The mesh file path and name
meshpath = '/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_10AOA_LES/MESH_ZONE_Apr24/'
#meshpath = '/project/p/plavoie/denggua1/BBDB_10AOA/MESH_ZONE_Apr24/'
meshfile = 'Bombardier_10AOA_Combine_Apr24.mesh.h5'
mesh_fileName = os.path.join(meshpath,meshfile)
# The solution path
sol_dirName = '/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_10AOA_LES/RUN_ZONE_Apr24/SOLUT/'
#sol_dirName = '/project/p/plavoie/denggua1/BBDB_10AOA/RUN_ZONE_Apr24/SOLUT/'
# The desired output path == projpath
projpath = ''
output_path = os.path.join(projpath,output)
os.makedirs(output_path, exist_ok = True)
# Default sizing paramaters for the cut
tip_gap = -0.1034
span = -0.2286
AoA = 10

def print(text):
    """ Function to print the text and flush the output"""
    builtins.print(text)
    os.fsync(sys.stdout)
    
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
    
def select_folder(sol_dirName):
    """ Function to select the folder based on the solution directory"""
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
    return sol_dir, arr_dir, arr

# The cut location class
class constants:
    """ Constants for the cut location"""
    def __init__(self, tip_gap, span):
        # Midspan
        self.z_mid_span = (tip_gap + span/2)
        # 2 inch from tip
        self.z_2inch_tip = (tip_gap - 0.0508)
        # 1 inch from tip
        self.z_1inch_tip = (tip_gap - 0.0254)
        # 0.25 inch from tip
        self.z_025inch_tip = (tip_gap - 0.00635)
        self.z_5mm_tip= (tip_gap - 0.005)
        self.z_25mm_tip= (tip_gap - 0.025)
        self.z_tip_gap= tip_gap

# Function to take out the unique file names in the folder
def sort_files(rand):
    """ Function to sort the files in the folder based on the file name and extract unique sequential file names and remove the duplicates"""
    rand_sub = os.listdir(rand)
    rand_arr = np.array([]) 
    for i in range(0,np.shape(rand_sub)[0]):
        file_split = os.path.splitext(rand_sub[i])[0]
        rand_arr = np.append(rand_arr,file_split)
    rand_arr = [*set(rand_arr)] # removing the duplicates
    rand_arr = [ rand_arr for rand_arr in rand_arr if 'sol_collection' not in rand_arr]     # Removing the sol_collection files
    rand_arr = [ rand_arr for rand_arr in rand_arr if 'last_solution' not in rand_arr]      # Removing the last_solution files
    rand_arr.sort() # sorting 
    return rand_arr

# Mappign the cut selection location
def map_cut(cut_selection:str,cut_style:str,tip_gap:float,span:float,AoA:int):
    """Function to map the cut selection to the location of the cut plane given the tip gap and span constnats
    Args:
        cut_selection (str): The location of the cut plane, defined either explicitly or with a %_TE location designating the distance from the trailing edge
        cut_style (str): The style of the cut, either 'plane' or 'cylinder'
        tip_gap (float): the tip gap size
        span (float): the span size
        AoA (int): the angle of attack
    Returns:
        origin (list): The origin of the cut plane
        normal (list): The normal of the cut plane
    """
    z_loc = constants(tip_gap,span)
    if cut_style == 'plane':
        if cut_selection.find("midspan") != -1:
            origin = [1.225,0.,z_loc.z_mid_span]
            normal = [0.,0.,1.]
        elif cut_selection.find("2inch_tip") != -1:
            origin = [1.225,0.,z_loc.z_2inch_tip]
            normal = [0.,0.,1.]
        elif cut_selection.find("1inch_tip") != -1:  
            origin= [1.225,0.,z_loc.z_1inch_tip]
            normal = [0.,0.,1.]
        elif cut_selection.find("025inch_tip") != -1:  
            origin = [1.225,0.,z_loc.z_025inch_tip]
            normal = [0.,0.,1.]
        elif cut_selection.find("25mm_tip") != -1:  
            origin= [1.225,0.,z_loc.z_25mm_tip]
            normal = [0.,0.,1.]
        elif cut_selection.find("5mm_tip") != -1:  
            origin= [1.225,0.,z_loc.z_5mm_tip]
            normal = [0.,0.,1.]
        elif cut_selection.find('PIV1') != -1:
            x,y,z = 1.42222035, 0, z_loc.z_mid_span
            origin = [x,y,z]
            normal  = [1,0,0]
        elif cut_selection.find('PIV2') != -1:
            x,y,z = 1.48172998, 0, z_loc.z_mid_span
            origin = [x,y,z]
            normal  = [1,0,0]
        elif cut_selection.find('PIV3') != -1:
            x,y,z = 1.5641908, 0, z_loc.z_mid_span
            origin= [x,y,z]
            normal = [1,0,0]
        elif cut_selection.find("TE") != -1:
            Loc = float(re.findall(r"\d+", cut_selection)[0])/100
            PIV = 1.25 + np.array(Loc)*0.3048*np.cos(AoA*np.pi/180)
            origin =  [PIV,0.,z_loc.z_mid_span]
            normal = [1.,0.,0.]
    elif cut_style == 'cylinder' or cut_style == 'sphere':
        origin = [1.42222035,0.,z_loc.z_tip_gap]
        normal = [1.,0.,0.]
    print('The selected cut is of style: {0}'.format(cut_style))
    print('The selected cut origin is: {0}'.format(origin))
    print('The selected cut normal is: {0}'.format(normal))
    return origin,normal

# Applying restart in extracting the cutplane
# Computing the restart parameters based on maximum files and restart position
def restart_compute_old(output_path,sol_dir,arr_dir,arr,n_start:int,restart:bool,max_file:int):
    """ The legacy function to compute the restart parameters based on maximum files and restart position
    """
    # loading the counts of all the files in the solution directory
    count = 0
    source_list = np.zeros([1,3])
    for i in range(nstart,np.shape(sol_dir)[0]):
        dir = os.path.join(arr_dir,arr[i])
        files = sort_files(dir)
        for j in range(0,np.shape(files)[0]):
            if count == 0:
                source_list = [[i,j,count]]
            else:
                source_list = np.append(source_list,[[i,j,count]],axis=0)
            count+=1
    # Finding the latest re-start file from the source directory
    source_files = sorted(glob.glob('{0:s}/*.h5'.format(output_path)))
    count_start = len(source_files)
    # Case of restart 
    if restart == True and count_start != 0:
        # check for a maximum file
        filename = source_files[-1]
        dummy = filename.split('_')
        part = dummy[len(dummy)-1].split('.')[0]
        part = int(part)
        indexes = np.where(source_list[:,2]==part)
        i_start = int(source_list[indexes,0])
        j_start = int(source_list[indexes,1])
        file_end =  int(np.min((source_list[indexes,2]+max_file,int(np.max(source_list[:,2])))))
        i_end = int(source_list[np.argwhere(source_list[:,2] == file_end),0])
        j_end = int(source_list[np.argwhere(source_list[:,2] == file_end),1])
        print('Restart applied, starting at iteration {0:1.0f}'.format(count_start))
        print('Extraction will start at folder {0:d} ending at folder {1:d}'.format(i_start,i_end))
    # case of no restart
    else:
        i_start,j_start,count_start = n_start,0,0
        indexes = 0
        file_end =  int(np.min((source_list[indexes,2]+max_file,int(np.max(source_list[:,2])))))
        i_end = int(source_list[np.argwhere(source_list[:,2] == file_end),0])
        j_end = int(source_list[np.argwhere(source_list[:,2] == file_end),1])
        print('Restart not applied, total of {0:1.0f} files'.format(count))
        print('Extraction will start at folder {0:d} ending at folder {1:d}'.format(i_start,i_end))
    print('The maximum interation will be {0:d} extracting {1:d} files'.format(file_end,int(file_end-count_start)))
    # mapping the array to track the MPI count
    return i_start,j_start,i_end, j_end, count_start, source_list 

# Existing code modified to include mstart parameter
def restart_compute(output_path, arr_dir:list, arr:list, n_start:int, restart: bool, max_file: int):
    """Computing the restart parameters based on maximum files and restart position
    Args:
        output_path: the file output path
        arr_dir: the solution file path ./SOLUT
        arr: the array of subdirectories in the sol_dir path
        n_start (int): the starting folder count based in SOLUT
        restart (bool): True or False to apply restart
        max_file (int): The maximum number of files to extract

    Returns:
        i_start (int): the main folder start index
        j_start (int): the file start index
        i_end (int): the main folder end index
        j_end (int): the file end index
        count_start (int): the starting count
        source_list (np.array): the list of files with main and sub folder indices
    """
    # Initialize the comprehensive source list from n_start
    source_list = []
    total_files = 0
    # The starting point for the restart based on the iteration from file
    m_start = int(re.findall(r'\d+', sorted(glob.glob('./'+output_path+'/*.h5'))[-1])[-2])
    # Populate the source list with file information and a global count
    for i, folder_name in enumerate(arr[n_start:], start=n_start):
        dir_path = os.path.join(arr_dir, folder_name)
        files = sort_files(dir_path)
        for j, file_name in enumerate(files):
            source_list.append([i, j, total_files])
            total_files += 1

    # Convert source_list to a numpy array for easier indexing
    source_list = np.array(source_list)
    start_files = len(glob.glob('./'+output_path+'/*.h5'))
    # Adjust the start point based on m_start
    if m_start > 0 and m_start < total_files:
        i_start, j_start, count_start = source_list[m_start][:3]
    else:
        i_start, j_start, count_start = n_start, 0, 0  # Defaults if m_start is out of bounds
    

    # Adjust for the ending index based on max_file constraint
    if count_start + max_file < total_files:
        file_end = count_start + max_file - start_files+1
    else:
        file_end = total_files - 1  # Limit to the total number of files

    # Find the ending folder and file indices
    if file_end < len(source_list):
        i_end, j_end, _ = source_list[file_end]
    else:
        i_end, j_end = source_list[-1][0], source_list[-1][1]
    print('Restart applied, starting at iteration {0:1.0f}'.format(count_start))
    print(f'Starting from folder {i_start} and file index {j_start}, total starting count is {count_start}.')
    print(f'Ending at folder {i_end} and file index {j_end}, with a maximum file count of {file_end - count_start}.')
    print(f'The total file count after extraction is {file_end-max_file}.')
    return i_start, j_start, i_end, j_end, count_start, source_list

# Existing code modified to include mstart parameter
def restart_compute_mstart(output_path, arr_dir, arr, n_start, restart: bool, max_file: int, m_start: int):
    """Computing the restart parameters based on maximum files and restart position, 
    with m_start as an additional parameter to skip files
    Args:
        output_path: the file output path
        arr_dir: the solution file path ./SOLUT
        arr: the array of subdirectories in the sol_dir path
        n_start (int): the starting folder count based in SOLUT
        restart (bool): True or False to apply restart
        max_file (int): The maximum number of files to extract
        m_start (int): The number of files to skip initially (to enable launching multiple files at the same time)

    Returns:
        i_start (int): the main folder start index
        j_start (int): the file start index
        i_end (int): the main folder end index
        j_end (int): the file end index
        count_start (int): the starting count
        source_list (np.array): the list of files with main and sub folder indices
    """
    # Initialize the comprehensive source list from n_start
    source_list = []
    total_files = 0

    # Populate the source list with file information and a global count
    for i, folder_name in enumerate(arr[n_start:], start=n_start):
        dir_path = os.path.join(arr_dir, folder_name)
        files = sort_files(dir_path)
        for j, file_name in enumerate(files):
            source_list.append([i, j, total_files])
            total_files += 1

    # Convert source_list to a numpy array for easier indexing
    source_list = np.array(source_list)

    # Adjust the start point based on m_start
    if m_start > 0 and m_start < total_files:
        i_start, j_start, count_start = source_list[m_start][:3]
    else:
        i_start, j_start, count_start = n_start, 0, 0  # Defaults if m_start is out of bounds
    
    # Adjust for the ending index based on max_file constraint
    if count_start + max_file < total_files:
        file_end = count_start + max_file
    else:
        file_end = total_files - 1  # Limit to the total number of files

    # Find the ending folder and file indices
    if file_end < len(source_list):
        i_end, j_end, _ = source_list[file_end]
    else:
        i_end, j_end = source_list[-1][0], source_list[-1][1]

    print(f'Starting from folder {i_start} and file index {j_start}, total starting count is {count_start}.')
    print(f'Ending at folder {i_end} and file index {j_end}, with a maximum extracted file count of {file_end - count_start}.')
    print(f'The total file count after extraction is {file_end-max_file}.')
    return i_start, j_start, i_end, j_end, count_start, source_list

def check_velocity_gradient_tensor(solfile):
    """ check if the velocity gradient tensor exists in the file
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
            # Check if all components exist under 'additionals'
            for component in velocity_gradient_components:
                if component not in additionals_group:
                    print(f"Component '{component}' does not exist!.")
                    return 0
            return 1
        else:
            print("The key 'Additionals' does not exist in the file.")
            return 0

@timer
def main(mesh_fileName:str, sol_dirName:str, output_path:str, cut_selection:str, cut_style:str,\
    restart:bool,extract_VGT:bool,n_start:int,mstart:int,tip_gap:float,span:float,AoA:int,max_file:int):
    """ The main function to extract the cut plane from the solution files
    Args:
        mesh_fileName (str): The path of the mesh file
        sol_dirName (str): The path of the solution directory
        output_path (str): The path of the output directory
        cut_selection (str): Cut selection location
        cut_style (str): The style of the cut, either 'plane' or 'cylinder
        extract_VGT (bool): True or False to extract the velocity gradient tensor
        n_start (int): The starting folder count based in SOLUT
        mstart (int): The number of files to skip initially (to enable launching multiple files at the same time), this parameter is not applied in restart
        tip_gap (float): The tip gap size
        span (float): The span size
        AoA (int): The angle of attack
        max_file (int): Maximum number of files to extract
    """
    # The output path
    print(f'\nThe cut selection is: {cut_selection}\n')  
    print(f'\nThe output directory is: {output_path}\n')  
    text = 'Loading File Directory';print(f'\n{text:.^80}\n')  
    # The main compute file
    text = 'Reading Mesh'
    print(f'\n{text:.^80}\n')  
    # Reading the mesh
    r = Reader('hdf_avbp')
    r['filename'] = mesh_fileName
    r['shared'] = True # Same mesh for all sol
    b = r.read() # b is the Base object of the Antares API
    b.show()
    text = 'Iterating Solution'
    print(f'\n{text:.^80}\n') 
    # looping over all main solver directories
    sol_dir, arr_dir,arr = select_folder(sol_dirName)
    # Applying restart
    if restart: 
        i_start, j_start, i_end,j_end,count,source_list = restart_compute(output_path,arr_dir, arr, n_start,restart,max_file)
    else: 
        i_start, j_start, i_end,j_end,count,source_list = restart_compute_mstart(output_path,arr_dir, arr, n_start,restart,max_file,mstart)
    for i in range(i_start,np.min((i_end+1,np.shape(sol_dir)[0]))):
        dir = os.path.join(arr_dir,arr[i])
        text = 'Processing the directory: '; print(f'\n{text}{dir}') 
        files = sort_files(dir)
        j_startt = j_start if i == i_start else 0
        j_endd = j_end if i == i_end else np.shape(files)[0]
        for j in range(j_startt,j_endd):
            calc_t = time.time()
            text = 'Iteration: '; print(f'\n{text}{count}') 
            text = 'Reading full solution file '; print(f'{text}:{files[j]}') 
            sol_file = os.path.join(sol_dirName,arr[i],files[j]+'.h5') # The full solution file
            r = Reader('hdf_avbp')
            r['base'] = b
            r['filename'] = sol_file
            comp = r.read()
            text = 'Computing cut'; print(f'{text}') 
            comp.compute(f'u=rhou/rho',location='node')
            comp.compute(f'v=rhov/rho',location='node')
            comp.compute(f'w=rhow/rho',location='node')
            comp.compute(f'Q=Q1+Q2',location='node')
            # Extracting the velocity gradient tensor (VGT)
            if extract_VGT == True:
                # Check if the velocity gradient tensor exists in the file
                VGT_status = check_velocity_gradient_tensor(sol_file)
                if VGT_status == 1:
                    comp.rename_variables(['du_dx', 'du_dy', 'du_dz','dv_dx', 'dv_dy', 'dv_dz','dw_dx', 'dw_dy', 'dw_dz'],\
                        ['grad_u_x','grad_u_y','grad_u_z','grad_v_x','grad_v_y','grad_v_z','grad_w_x','grad_w_y','grad_w_z'], location=None)
                    print('The velocity gradient tensor exists in the file, gradient calculation not required.')
                # If the velocity gradient tensor does not exists, compute the gradient
                if VGT_status == 0:
                    text = 'Calculating Gradient'; print(f'{text}') 
                    treatment = Treatment('gradient')
                    treatment['base'] = comp
                    treatment['variables']=('u','v','w')
                    comp = treatment.execute()
                    comp.cell_to_node()
            # Computing the cut
            print('Applying cut at: {0:s}, with Treatment: {1:s}'.format(cut_selection,cut_style)) 
            # Applying the cut locations based on user locations
            if cut_style == 'plane':
                t = Treatment('cut')
                t['base'] = comp
                t['type'] = 'plane'
                t['base'].attrs['Time'] = float(count)
            elif cut_style == 'cylinder' or cut_style == 'sphere':
                t = Treatment('clip')
                t['base'] = comp
                t['type'] = cut_style
                t['axis'] = 'x'
                t['radius'] = 0.1
            origin, normal = map_cut(cut_selection,cut_style,tip_gap,span,AoA)
            t['origin'] = origin
            t['normal'] = normal
            # Applying the cut
            comp = t.execute() # This a new object with only mid sections
            text = 'Saving solution'; print(f'{text}') 
            # Extracting the velocity gradient tensor (VGT)
            if extract_VGT == True:
                w = Writer('hdf_antares') # This is another format (stll hdf5)
                w['base'] = comp[:,:,['x','y','z','u','v','w','rho','dilatation','pressure','vort_x','grad_u_x','grad_u_y','grad_u_z','grad_v_x','grad_v_y','grad_v_z','grad_w_x','grad_w_y','grad_w_z']]
                w['filename'] = os.path.join(output_path,'B_10AOA_VGT_{:s}_{:03d}'.format(cut_selection,count)) 
                w.dump()
                print('The file '+w['filename'] +' is exported')
                count += 1
            else:
                w = Writer('hdf_antares') # This is another format (stll hdf5)
                w['base'] = comp[:,:,['x','y','z','u','v','w','dilatation','pressure','Q','rho','vort_x']]
                w['filename'] = os.path.join(output_path,'B_10AOA_{:s}_{:03d}'.format(cut_selection,count)) 
                w.dump()
                print('The file '+w['filename'] +' is exported')
                count += 1
            del comp
            elapsed = time.time() - calc_t
            print('The iteration calcualtion time is: {0:1.0f} s'.format(elapsed))
            
    text = 'Solution Iteration Complete'
    print(f'\n{text:.^80}\n')


if __name__ == '__main__':
    # The following lines syncs the print function and sync the flushed file also on the operating system side
    sys.stdout = open(os.path.join('log_'+output+'.txt'), "w", buffering=1)
    main(mesh_fileName, sol_dirName, output_path, cut_selection, cut_style, restart, extract_VGT, nstart, mstart, tip_gap, span, AoA, max_file)
    