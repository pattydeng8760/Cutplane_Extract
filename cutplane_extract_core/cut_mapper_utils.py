"""
Cut mapper module

This module contains a class for cut location constants and a function to map
the cut selection to a specific geometric location.
"""

import re
import numpy as np
from typing import List, Tuple

class CutConstants:
    """
    Class containing constants computed from tip gap and span.
    """
    def __init__(self, tip_gap: float, span: float):
        self.z_mid_span = tip_gap + span / 2
        self.z_2inch_tip = tip_gap - 0.0508
        self.z_1inch_tip = tip_gap - 0.0254
        self.z_025inch_tip = tip_gap - 0.00635
        self.z_5mm_tip = tip_gap - 0.005
        self.z_25mm_tip = tip_gap - 0.025
        self.z_tip_gap = tip_gap

def map_cut(cut_selection: str, cut_style: str, tip_gap: float,
            span: float, AoA: int) -> Tuple[List[float], List[float]]:
    """
    Map the cut selection string to the cut plane (or volume) parameters.
    
    Args:
        cut_selection (str): The location of the cut.
        cut_style (str): The style of the cut ('plane', 'cylinder', or 'sphere').
        tip_gap (float): The tip gap size.
        span (float): The span size.
        AoA (int): The angle of attack.
    
    Returns:
        origin (List[float]): The origin of the cut.
        normal (List[float]): The normal of the cut.
    """
    consts = CutConstants(tip_gap, span)
    if cut_style == 'plane':
        if "midspan" in cut_selection:
            origin = [1.225, 0.0, consts.z_mid_span]
            normal = [0.0, 0.0, 1.0]
        elif "2inch_tip" in cut_selection:
            origin = [1.225, 0.0, consts.z_2inch_tip]
            normal = [0.0, 0.0, 1.0]
        elif "1inch_tip" in cut_selection:
            origin = [1.225, 0.0, consts.z_1inch_tip]
            normal = [0.0, 0.0, 1.0]
        elif "025inch_tip" in cut_selection:
            origin = [1.225, 0.0, consts.z_025inch_tip]
            normal = [0.0, 0.0, 1.0]
        elif "25mm_tip" in cut_selection:
            origin = [1.225, 0.0, consts.z_25mm_tip]
            normal = [0.0, 0.0, 1.0]
        elif "5mm_tip" in cut_selection:
            origin = [1.225, 0.0, consts.z_5mm_tip]
            normal = [0.0, 0.0, 1.0]
        elif "PIV1" in cut_selection:
            origin = [1.42222035, 0.0, consts.z_mid_span]
            normal = [1.0, 0.0, 0.0]
        elif "PIV2" in cut_selection:
            origin = [1.48172998, 0.0, consts.z_mid_span]
            normal = [1.0, 0.0, 0.0]
        elif "PIV3" in cut_selection:
            origin = [1.5641908, 0.0, consts.z_mid_span]
            normal = [1.0, 0.0, 0.0]
        elif "TE" in cut_selection:
            loc_val = float(re.findall(r"\d+", cut_selection)[0]) / 100
            PIV = 1.25 + loc_val * 0.3048 * np.cos(AoA * np.pi / 180)
            origin = [PIV, 0.0, consts.z_mid_span]
            normal = [1.0, 0.0, 0.0]
        else:
            raise ValueError("Unrecognized cut_selection for plane cut.")
    elif cut_style in ['cylinder', 'sphere']:
        origin = [1.42222035, 0.0, consts.z_tip_gap]
        normal = [1.0, 0.0, 0.0]
    else:
        raise ValueError("Unsupported cut style provided.")
    
    print('        The selected cut is of style: {0},  origin: {1},  normal: {2}'.format(cut_style,origin,normal))
    return origin, normal
