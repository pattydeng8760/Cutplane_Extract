"""
Functions used to compute the Q-criterion and Lambda_ci criterion from the velocity gradient tensor A_ij
Written by Hao Wu, 2017

Change from P. Kholodov (14/07/2020):
Reversed vorti1 and vorti3 because before it was vorti1 for z-vorticity and vorti3 for x-vorticity.

Change from P.Deng (13/11/2023):
Reversed for syntax update to be compatable with python 3.10 and antares 2.1.0

Change from P.Deng (12/04/2025):
Appendend the functionality to cutplane_extract module to extract the isosurfaces of the Q-criterion and Lambda_ci criterion.
"""

from antares import *
from antares.core.PrintUtility import progress_bar
import numpy as np

gradU=None
gradV=None
gradW=None
Vel=None

def velocity_gradient_names(dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz):
  """
  Define the names of the velocity gradient vectors in your antares base
  Call velocity_gradient_names with the 9 strings defining the three
  components of gradU then gradV ane finally gradW. These strings
  are the variable names in your antares base
  """
  global gradU
  global gradV
  global gradW
  gradU=(dudx,dudy,dudz)
  gradV=(dvdx,dvdy,dvdz)
  gradW=(dwdx,dwdy,dwdz)

def velocity_names(u,v,w):
  """
  Define the names of the velocity vector in your antares base
  Call velocity_gradient_names with the 3 strings defining the three
  components of velocity U,V,W. These strings   are the variable names in your antares base
  """
  global Vel
  Vel=(u,v,w)

def check_names():
  global Vel
  global gradU
  global gradV
  global gradW
  if Vel is None:
    raise RuntimeError("Velocity have not been set. Use velocity_names function")
  if gradU is None or gradV is None or gradW is None:
    raise RuntimeError("Velocity gradient have not been set. Use velocity_gradient_names function")


def trJ(J):
    """Compute the trace operator of J (i.e. incompressibility).

    Keyword argument:
    J -- Antares base wich defines the velocity gradient tensor.
    """
    check_names()
    global gradU
    global gradV
    global gradW
    J.compute('trJ = {0:s} + {1:s} + {2:s}'.format(gradU[0],gradV[1],gradW[2]))
    return J

def vorti(J):
    """Compute the vorticity.

    Keyword argument:
    J -- Antares base wich defines the velocity gradient tensor.
    """
    check_names()
    global gradU
    global gradV
    global gradW
    # x-vorticity
    J.compute('vorti1 =  ({0:s} - {1:s})'.format(gradW[1],gradV[2]))
    # y-vorticity
    J.compute('vorti2 = ({0:s} - {1:s})'.format(gradU[2],gradW[0]))
    # z-vorticity
    J.compute('vorti3 =  ({0:s} - {1:s})'.format(gradV[0],gradU[1]))
    # Vorticity magnitude
    J.compute('vorti = (vorti1 ** 2 + vorti2 ** 2 + vorti3 ** 2) ** 0.5')

    return J

def sheari(J):
    """Compute the shear.

    Keyword argument:
    J -- Antares base wich defines the velocity gradient tensor.
    """
    check_names()
    global gradU
    global gradV
    global gradW
    J.compute('shear1 = ({0:s} + {1:s})'.format(gradV[0],gradU[1]))
    J.compute('shear2 =({0:s} + {1:s})'.format(gradU[2],gradW[0]))
    J.compute('shear3 = ({0:s} + {1:s})'.format(gradW[1],gradV[2]))

    return J

def helicity(J):
    """Compute the helicity

    Keyword argument:
    J -- Antares base which defines the velocity gradient tensor.
    """
    check_names()
    global Vel
    J.compute('helicity = vorti1 * {0:s} + vorti2 * {1:s} + vorti3 * {2:s}'.format(Vel[0],Vel[1],Vel[2]))

    return J

def Q(J):
    """Compute the Q criterion.

    Keyword argument:
    J -- Antares base wich defines the velocity gradient tensor.
    """
    check_names()
    global gradU
    global gradV
    global gradW

    if 'vorti1' not in J[0][0].keys('nodes'):
      J=vorti(J)
    if 'shear1' not in J[0][0].keys('nodes'):
      J=sheari(J)

    #J.compute('Q = 0.25 * (vorti1 ** 2 + vorti2 ** 2 + vorti3 ** 2 - \
    #            2 * ({0:s} ** 2 + {1:s} ** 2 + {2:s} ** 2) - \
    #            (shear1 ** 2 + shear2 ** 2 + shear3 ** 2))'.format(gradU[0],gradV[1],gradW[2]))
    J.compute('Q = 0.25 * (vorti1 ** 2 + vorti2 ** 2 + vorti3 ** 2 - \
                2 * ({0:s} ** 2 + {1:s} ** 2 + {2:s} ** 2) - \
               (shear1 ** 2 + shear2 ** 2 + shear3 ** 2))'.format(gradU[0],gradV[1],gradW[2]))

    return J

def Qstar(J):
    """Compute the Q* criterion.

    Keyword argument:
    J -- Antares base wich defines the velocity gradient tensor.
    """
    check_names()
    if 'Q' not in J[0][0].keys('nodes'):
      J=Q(J)
    if 'trJ' not in J[0][0].keys('nodes'):
      J=trJ(J)
    J.compute('Qstar = Q + 0.5 * trJ ** 2')

    return J

def Delta(J):
    """Compute the Delta criterion.

    Keyword argument:
    J -- Antares base wich defines the velocity gradient tensor.
    """
    check_names()
    global gradU
    global gradV
    global gradW
    J.compute('detJ = {0:s} * {4:s} * {8:s} - \
                    {0:s} * {5:s} * {7:s} - \
                    {1:s} * {3:s} * {8:s} + \
                    {1:s} * {5:s} * {6:s} + \
                    {2:s} * {3:s} * {7:s} - \
                    {2:s} * {4:s} * {6:s}'.format(gradU[0],gradU[1],gradU[2],gradV[0],gradV[1],gradV[2],gradW[0],gradW[1],gradW[2]))

    if 'Q' not in J[0][0].keys('nodes'):
      J=Q(J)
    J.compute('Delta = (Q/3) ** 3 + (detJ/2) ** 2')

    return J

def lambda_2(J, eigVect=False):
    """Compute the lambda_2 criterion based on the velocity gradient tensor.

    Keyword arguments:
    J       -- Antares base defining the velocity gradient tensor.

    eigVect -- Boolean indicating whether or not the computation of eigenvectors
            is done.
            Its default value is 'False' in order to get faster computations.

    The lambda_2 criterion is based on a particular decomposition of the velocity
    gradient tensor. This tensor can be expressed as the sum of its symetric
    part (S) and its antisymetric part (O) as :
        J = O + S with,
        O = ([[0 -vorti1 vorti2], [vorti1 0 -vorti3], [-vorti2 vorti3 0]])
        S = ([[J11 shear1 shear2], [shear1 J22 shear3], [shear2 shear3 J33]])
    The computation of the shear and vorticity elements is detailed in the code.

    Then the eigenvalues of O^2 + S^2 are computed and lambda_2 is defined as the
    second eigenvalue of this matrix. It can be shown from the Navier-Stokes
    equations, and by neglecting the viscosity and the unstationnary shear effects,
    that where lambda_2 is negative there is a pressure minimum corresponding to
    a vortex core. The pressure minimum (therefore the vortex) is defined in the
    plane of the two first eigenvectors of O^2 + S^2.

    For more details on the criterion see :
        J. Jeong, and F. Hussain. On the identification of a vortex. Journal of
        Fluid Mechanics, 285 :69-94, (1995).
    """
    check_names()
    global gradU
    global gradV
    global gradW

    if eigVect:
        print('\n >>> Compute variable l2 and its eigenvectors')
    else:
        print('\n >>> Compute variable l2')

    if 'vorti1' not in J[0][0].keys('nodes'):
      J=vorti(J)
    if 'shear1' not in J[0][0].keys('nodes'):
      J=sheari(J)

    for zone in progress_bar(J.keys()):        # loop over each zone of J
        for inst in J[zone].keys():            # loop over each instant
            # Definition of new variables in order to reduce the notation.
            vorti1 = J[zone][inst]['vorti1']
            vorti2 = J[zone][inst]['vorti2']
            vorti3 = J[zone][inst]['vorti3']
            shear1 = J[zone][inst]['shear1']
            shear2 = J[zone][inst]['shear2']
            shear3 = J[zone][inst]['shear3']
            J11    = J[zone][inst][gradU[0]]
            J22    = J[zone][inst][gradV[1]]
            J33    = J[zone][inst][gradW[2]]

            # Check if variables are multidimensional structured arrays or
            # 1D non-structured arrays.
            # The computations are done differently for both cases.
            if len(J[0][0][0].shape) != 1 :
                # Definition of the matrices O and S with 3D structured datas.
                # The transpose operation is used to reorganise each 2D array of
                # 3D structured datas as :
                #     Matrix = Matrix(x,y,z,i,j), where - (x,y,z) are the datas indices
                #                                       - (i,j)   are the array indices
                # Keep in mind that O and S are 2D arrays defined at each coordinates
                # (x,y,z) of the flow.
                O = np.array([[np.zeros_like(vorti1), -vorti1, vorti2],
                    [vorti1, np.zeros_like(vorti1), -vorti3],
                    [-vorti2, vorti3, np.zeros_like(vorti1)]]).transpose(2, 3, 4, 0, 1)
                O2 = np.matmul(O, O)    # square of the antisymetric array (x,y,z,i,j)

                S = np.array([[J11, shear1, shear2],
                    [shear1, J22, shear3],
                    [shear2, shear3, J33]]).transpose(2, 3, 4, 0, 1)
                S2 = np.matmul(S, S)    # square of the symetric array (x,y,z,i,j)

                # Computation of lambda_2 and (if eigVect==True) the eigenvectors.
                # Note that eigh and eigvalsh of the module np.linalg are used
                # because O2 + S2 is real symetric.
                if eigVect:
                    L, V = np.linalg.eigh(O2 + S2)
                    # Only the second eigenvalue (i.e. lambda_2 ) is saved.
                    J[zone][inst]['l2'] = L[:, :, :, 1]

                    # Only the first and second eigenvectors are saved because they
                    # define the plane in which a pressure minimum occurs.
                    J[zone][inst]['l2_V1x'] = V[:, :, :, 0, 0]
                    J[zone][inst]['l2_V1y'] = V[:, :, :, 1, 0]
                    J[zone][inst]['l2_V1z'] = V[:, :, :, 2, 0]
                    J[zone][inst]['l2_V2x'] = V[:, :, :, 0, 1]
                    J[zone][inst]['l2_V2y'] = V[:, :, :, 1, 1]
                    J[zone][inst]['l2_V2z'] = V[:, :, :, 2, 1]
                else:
                    # Only the eigenvalues are computed then only lambda_2 is saved.
                    J[zone][inst]['l2'] = np.linalg.eigvalsh(O2 + S2)[:, :, :, 1]

            else:
                # Definition of the matrices O and S with 1D non-structured datas.
                # The transpose operation is used to reorganise each 2D array of
                # 1D non-structured datas as :
                #     Matrix = Matrix(x,i,j), where - (x,)  is the datas index
                #                                   - (i,j) are the array indices
                # Keep in mind that O and S are 2D arrays defined at each x-coordinate
                # of the flow.
                O = np.array([[np.zeros_like(vorti1), -vorti1, vorti2],
                    [vorti1, np.zeros_like(vorti1), -vorti3],
                    [-vorti2, vorti3, np.zeros_like(vorti1)]]).transpose(2, 0, 1)
                O2 = np.matmul(O, O)    # square of the antisymetric array (x,i,j)

                S = np.array([[J11, shear1, shear2],
                    [shear1, J22, shear3],
                    [shear2, shear3, J33]]).transpose(2, 0, 1)
                S2 = np.matmul(S, S)    # square of the symetric array (x,i,j)

                # Computation of lambda_2 and (if eigVect==True) the eigenvectors.
                # Note that eigh and eigvalsh of the module np.linalg are used
                # because O2 + S2 is real symetric.
                if eigVect:
                    L, V = np.linalg.eigh(O2 + S2)
                    # Only the second eigenvalue (i.e. lambda_2 ) is saved.
                    J[zone][inst]['l2'] = L[:, 1]

                    # Only the first and second eigenvectors are saved because they
                    # define the plane in which a pressure minimum occurs.
                    J[zone][inst]['l2_V1x'] = V[:, 0, 0]
                    J[zone][inst]['l2_V1y'] = V[:, 1, 0]
                    J[zone][inst]['l2_V1z'] = V[:, 2, 0]
                    J[zone][inst]['l2_V2x'] = V[:, 0, 1]
                    J[zone][inst]['l2_V2y'] = V[:, 1, 1]
                    J[zone][inst]['l2_V2z'] = V[:, 2, 1]
                else:
                    # Only the eigenvalues are computed then only lambda_2 is saved.
                    J[zone][inst]['l2'] = np.linalg.eigvalsh(O2 + S2)[:, 1]

    return J

def lambda_ci(J, eigVect=False):
    """Compute the lambda_ci criterion based on the velocity gradient tensor.

    Keyword arguments:
    J       -- Antares base defining the velocity gradient tensor.

    eigVect -- Boolean indicating whether or not the computation of eigenvectors
            is done.
            Its default value is 'False' in order to get faster computations.

    The lambda_ci criterion is based on a particular decomposition of the velocity
    gradient tensor. This tensor can be expressed in its eigenvector basis as :
        J = ([[l_cr + i*l_ci 0 0], [0 l_cr - i*l_ci 0], [0 0 l_r]]),
        where l denotes the greek letter lambda.

    It can be shown that lci quantifies the strength of the local swirl motion.
    Thus it is used as a vortex detection criterion if its value is sufficiently
    high which, in theory, just needs to be greater than zero.

    For more details on the criterion see :
        J. Zhou, R. J. Adrian, S. Balachandar, and T. M. Kendall. Mechanisms for
        generating coherent packets of hairpin vortices in channel flow. Journal
        of Fluid Mechanics, 387 :353-396, (1999).
    """
    check_names()
    global gradU
    global gradV
    global gradW

    if eigVect:
        print('\n >>> Compute variable lci and its eigenvector')
    else:
        print('\n >>> Compute variable lci')

    for zone in progress_bar(J.keys()):        # loop over each zone of J
        for inst in J[zone].keys():            # loop over each instant
            # Definition of new variables in order to reduce the notation.
            J11 = J[zone][inst][gradU[0]]
            J12 = J[zone][inst][gradU[1]]
            J13 = J[zone][inst][gradU[2]]
            J21 = J[zone][inst][gradV[0]]
            J22 = J[zone][inst][gradV[1]]
            J23 = J[zone][inst][gradV[2]]
            J31 = J[zone][inst][gradW[0]]
            J32 = J[zone][inst][gradW[1]]
            J33 = J[zone][inst][gradW[2]]

            # Check if variables are multidimensional structured arrays or
            # 1D non-structured arrays.
            # The computations are done differently for both cases.
            if len(J[0][0][0].shape) != 1 :
                # Definition of the velocity gradient tensor with 3D structured datas.
                # The transpose operation is used to reorganise the 2D array of
                # 3D structured datas as :
                #     Matrix = Matrix(x,y,z,i,j), where - (x,y,z) are the datas indices
                #                                       - (i,j)   are the array indices
                # Keep in mind that the velocity gradient is a 2D array defined at each
                # coordinate (x,y,z) of the flow.
                J_temp = np.array([[J11, J12, J13],
                            [J21, J22, J23],
                            [J31, J32, J33]]).transpose(2, 3, 4, 0, 1)

                # Computation of lambda_ci and (if eigVect==True) the eigenvectors.
                if eigVect:
                    L, V = np.linalg.eig(J_temp)
                    # Imaginary part of the first eigenvalue (l1 = l_cr + i*l_ci).
                    J[zone][inst]['lci'] = L[:, :, :, 0].imag
                    # Third eigenvector which is normal to the plane of the vortex.
                    # This eigenvector is real but is considered a complex number with
                    # a null imaginary part. In order to be consistent with the type of
                    # datas used in the base J we need to take only the real part.
                    J[zone][inst]['lci_Vx'] = V[:, :, :, 0, 2].real
                    J[zone][inst]['lci_Vy'] = V[:, :, :, 1, 2].real
                    J[zone][inst]['lci_Vz'] = V[:, :, :, 2, 2].real
                else:
                    # Only the eigenvalues are computed and only l_ci is saved.
                    J[zone][inst]['lci'] = (np.linalg.eigvals(J_temp)[:, :, :, 0]).imag

            else:
                # Definition of the velocity gradient tensor with 1D non-structured datas.
                # The transpose operation is used to reorganise the 2D array of
                # 1D non-structured datas as :
                #     Matrix = Matrix(x,i,j), where - (x,)  is the datas index
                #                                   - (i,j) are the array indices
                # Keep in mind that the velocity gradient is a 2D array defined at each
                # x-coordinate of the flow.
                J_temp = np.array([[J11, J12, J13],
                            [J21, J22, J23],
                            [J31, J32, J33]]).transpose(2, 0, 1)

                # Computation of lambda_ci and (if eigVect==True) the eigenvectors.
                if eigVect:
                    L, V = np.linalg.eig(J_temp)
                    # Imaginary part of the first eigenvalue (l1 = l_cr + i*l_ci).
                    J[zone][inst]['lci'] = L[:, 0].imag
                    # Third eigenvector which is normal to the plane of the vortex.
                    # This eigenvector is real but is considered a complex number with
                    # a null imaginary part. In order to be consistent with the type of
                    # datas used in the base J we need to take only the real part.
                    J[zone][inst]['lci_Vx'] = V[:, 0, 2].real
                    J[zone][inst]['lci_Vy'] = V[:, 1, 2].real
                    J[zone][inst]['lci_Vz'] = V[:, 2, 2].real
                else:
                    # Only the eigenvalues are computed and only l_ci is saved.
                    J[zone][inst]['lci'] = (np.linalg.eigvals(J_temp)[:, 0]).imag

    return J