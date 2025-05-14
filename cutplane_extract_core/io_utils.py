"""
I/O Utilities for simulation postprocessing.

This module contains functions related to directory listing, sorting file names,
and computing restart index parameters for simulation data extraction.
"""

import os
import re
import glob
import numpy as np
from typing import List, Tuple, Any

def select_folder(sol_dir: str) -> Tuple[np.ndarray, str, List[str]]:
    """
    Selects and sorts the solution folders.
    
    Args:
        sol_dir (str): Path to the solution directory.
        
    Returns:
        sol_dirs (np.ndarray): Unique sorted folder identifiers.
        arr_dir (str): The directory listing path.
        arr (List[str]): List of folder names.
    """
    arr_dir = sol_dir
    arr = sorted(os.listdir(arr_dir))
    sol_dirs = []
    for folder in arr:
        parts = folder.split('_')
        sol_dir_part = parts[-1].split('.')[0]
        sol_dirs.append(sol_dir_part)
    sol_dirs = np.unique(sol_dirs)
    return sol_dirs, arr_dir, arr

def sort_files(directory: str) -> List[str]:
    """
    Sorts files in a directory based on their names, removing duplicates and unwanted files.
    
    Args:
        directory (str): Directory path to list files from.
    
    Returns:
        List[str]: Sorted list of unique file basenames (without extension).
    """
    files = os.listdir(directory)
    unique_files = set()
    for fname in files:
        base, _ = os.path.splitext(fname)
        unique_files.add(base)
    # Remove unwanted files (like sol_collection and last_solution)
    filtered = [f for f in unique_files if 'sol_collection' not in f and 'last_solution' not in f]
    filtered.sort()
    return filtered

def build_source_list(arr_dir: str, arr: List[str], n_start: int) -> Tuple[np.ndarray, int]:
    """
    Build a comprehensive list of source file indices from the solution folders.
    
    Args:
        arr_dir (str): The directory containing solution subdirectories.
        arr (List[str]): Sorted list of subdirectory names.
        n_start (int): The starting folder index.
    
    Returns:
        source_list (np.ndarray): Array of [folder_index, file_index, global_count].
        total_files (int): Total number of files found.
    """
    source_list = []
    total_files = 0
    for i, folder_name in enumerate(arr[n_start:], start=n_start):
        dir_path = os.path.join(arr_dir, folder_name)
        files = sort_files(dir_path)
        for j, _ in enumerate(files):
            source_list.append([i, j, total_files])
            total_files += 1
    return np.array(source_list), total_files

def compute_restart_parameters(output_path: str, arr_dir: str, arr: List[str],
                               n_start: int, restart: bool, max_file: int,
                               m_start: int) -> Tuple[int, int, int, int, int, np.ndarray]:
    """
    Compute the restart parameters based on previously extracted files, 
    the maximum file count, and m_start to allow parallel launching.
    
    Args:
        output_path (str): Output directory path.
        arr_dir (str): Parent directory containing solution folders.
        arr (List[str]): List of solution folder names.
        n_start (int): Starting folder index.
        restart (bool): Flag indicating whether to restart.
        max_file (int): Maximum number of files to extract.
        m_start (int): Number of files to skip initially.
    
    Returns:
        Tuple containing:
            i_start (int): Starting folder index.
            j_start (int): Starting file index within the folder.
            i_end (int): Ending folder index.
            j_end (int): Ending file index.
            count_start (int): Global starting file count.
            source_list (np.ndarray): Comprehensive source list.
    """
    source_list, total_files = build_source_list(arr_dir, arr, n_start)
    # Count existing h5 files in output directory
    output_files = sorted(glob.glob(os.path.join(output_path, '*.h5')))
    start_files = len(output_files)
    
    if restart and start_files > 0:
        # If restarting, assume the last file indicates the start index via its filename
        last_file = output_files[-1]
        base_file = os.path.splitext(os.path.basename(last_file))[0]  
        parts = re.findall(r'\d+', base_file)   
        m_start_from_file = int(parts[-1]) if parts else 0
        print(f"Restarting from file index {m_start_from_file} based on existing files.")
    else:
        m_start_from_file = m_start
        print(f'Starting from file index {m_start_from_file} as no existing files were found or restart is not set.')
    
    if m_start_from_file > 0 and m_start_from_file < total_files:
        i_start, j_start, count_start = source_list[m_start_from_file]
    else:
        i_start, j_start, count_start = n_start, 0, 0

    # Compute the ending file index (global count)
    if count_start + max_file < total_files:
        file_end = count_start + max_file
    else:
        file_end = total_files - 1
    
    if file_end < len(source_list):
        i_end, j_end, _ = source_list[file_end]
    else:
        i_end, j_end = source_list[-1][0], source_list[-1][1]
    
    print(f"Restart starting at folder {i_start}, file index {j_start} (global count: {count_start}).")
    print(f"Will end at folder {i_end}, file index {j_end} with {file_end - count_start} files extracted.")
    return int(i_start), int(j_start), int(i_end), int(j_end), int(count_start), source_list