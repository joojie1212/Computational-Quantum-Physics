import numpy as np
import numpy.linalg as LA
import scipy.sparse.linalg as LAs
import Sub as Sub
import math, copy



def GetMpo_Obc(Dp=2, g=0.428):
    """
    This is a Function to generate needed MPO matrix 
    Mi =
    s0 -sz gsx gsz  0   0  -sx
    0   0   0   0   0   0   sz
    0   0   0   0   sz  0   0
    0   0   0   0   0   Sz  0
    0   0   0   0   0   0   sz
    0   0   0   0   0   0   sx
    0   0   0   0   0   0   s0
    BE CARE: The spin operators here do not have 1/2 factor

    Parameters:
        Dp : physical dimension
        g : three-body interaction strength
    Returns:
        Mpo : the mpo matrix shaped (Dmpo, Dp, Dmpo, Dp)
    """
    S0, Sp, Sm, Sz, Sx, Sy = Sub.SpinOper(Dp)
    
    Dmpo = 7
    Mpo = np.zeros((Dmpo, Dp, Dmpo, Dp))
    
    Mpo[0, :, 0, :] = S0
    Mpo[0, :, 1, :] = -2*Sz
    Mpo[0, :, 2, :] = 2*g * Sx
    Mpo[0, :, 3, :] = 2*g * Sz
    Mpo[0, :, 6, :] = -2*Sx
    Mpo[1, :, 6, :] = 2*Sz
    Mpo[2, :, 4, :] = 2*Sz
    Mpo[3, :, 5, :] = 2*Sz
    Mpo[4, :, 6, :] = 2*Sz
    Mpo[5, :, 6, :] = 2*Sx
    Mpo[6, :, 6, :] = S0
    
    return Mpo


def InitMps(Ns, Dp, Ds):
    """
    This is a fuction to initialize random MPS in canonical form
    All the matrices here are in right canonical form
    
    Parameters: 
        Ns : number of sites
        Dp : physical dimension
        Ds : maximum bond dimension
    Returns:
        T : list of MPS matrices with length Ns
    """
    T = [None] * Ns
    for i in range(Ns):
        Dl = min(Dp ** i, Dp ** (Ns - i), Ds)
        Dr = min(Dp ** (i + 1), Dp ** (Ns - 1 - i), Ds)
        T[i] = np.random.rand(Dl, Dp, Dr)
    
    U = np.eye(np.shape(T[-1])[-1])
    for i in range(Ns - 1, 0, -1):
        U, T[i] = Sub.Mps_LQP(T[i], U)
    
    return T


def InitH(Mpo, T):
    """
    This is a function to initialize left and right environment tensors for MPS optimization
    Parameters:
        Mpo : the mpo matrix shaped (Dmpo, Dp, Dmpo, Dp)
        T : list of MPS matrices with length Ns
    Returns:
        HL, HR : lists of left and right environment tensors with length Ns, HL will not be calculated here
    """
    Ns = len(T)
    Dmpo = np.shape(Mpo)[0]
    
    HL = [None] * Ns
    HR = [None] * Ns
    
    HL[0] = np.zeros((1, Dmpo, 1))
    HL[0][0, 0, 0] = 1.0
    HR[-1] = np.zeros((1, Dmpo, 1))
    HR[-1][0, -1, 0] = 1.0
    
    for i in range(Ns - 1, 0, -1):
        HR[i - 1] = Sub.NCon([HR[i], T[i], Mpo, np.conj(T[i])],
                             [[1, 3, 5], [-1, 2, 1], [-2, 2, 3, 4], [-3, 4, 5]])
    
    return HL, HR


def OptTtwoSite(Mpo, HL, HR, T1, T2, direction):
    """
    This is the central step to optimize two neighboring MPS tensors
    Parameters:
        Mpo : the mpo matrix shaped (Dmpo, Dp, Dmpo, Dp)
        HL : left environment tensor
        HR : right environment tensor
        T1, T2 : two neighboring MPS tensors to be optimized
        direction : 0 for left to right, 1 for right to left
    Returns:
        T1_new, T2_new : optimized MPS tensors
    """
    DT1 = np.shape(T1)
    DT2 = np.shape(T2)    
    DT_twosite = (DT1[0], DT1[1], DT2[1], DT2[2])
    
    A = Sub.NCon([HL, Mpo, Mpo, HR],
                 [[-1, 1, -5],
                  [1, -6, 2, -2],
                  [2, -7, 3, -3],
                  [-8, 3, -4]])
    A = Sub.Group(A, [[0, 1, 2, 3], [4, 5, 6, 7]])
    Eig, V = LAs.eigsh(A, k=1, which='SA')

    A = np.reshape(V, (DT1[0] * DT1[1], DT2[1] * DT2[2]))
    U, S, V, Dc = Sub.SplitSvd_Lapack(A, DT1[2], 0)
    
    if direction == 0:
        V = np.diag(S) @ V
    elif direction == 1:
        U = U @ np.diag(S)
    else:
        print("direction error!")
    
    T1_new = np.reshape(U, (DT1[0], DT1[1], Dc))
    T2_new = np.reshape(V, (Dc, DT2[1], DT2[2]))
    
    return T1_new, T2_new, Eig


def OptT(Mpo, HL, HR, T,tol=1.0e-7):
    """
    This is routine to optimize the whole MPS by sweeping through all sites from lift to right and right to left
    Parameters:
        Mpo : the mpo matrix shaped (Dmpo, Dp, Dmpo, Dp)
        HL : left environment tensor list 
        HR : right environment tensor list
        T : MPS tensors
        tol :  permissible error
    Returns:
        T : optimized MPS tensors
        Eng1 / float(Ns) : energy per site

    """
    Ns = len(T)
    Eng0 = np.zeros(Ns)
    Eng1 = np.zeros(Ns)
    
    for r in range(100):
        #print(r)
        
        for i in range(Ns - 2):
            T[i], T[i + 1], Eng1[i] = OptTtwoSite(Mpo, HL[i], HR[i + 1], T[i], T[i + 1], 0)
            HL[i + 1] = Sub.NCon([HL[i], np.conj(T[i]), Mpo, T[i]],
                                  [[1, 3, 5], [1, 2, -1], [3, 4, -2, 2], [5, 4, -3]])
        
        for i in range(Ns - 1, 1, -1):
            T[i - 1], T[i], Eng1[i] = OptTtwoSite(Mpo, HL[i - 1], HR[i], T[i - 1], T[i], 1)
            HR[i - 1] = Sub.NCon([HR[i], T[i], Mpo, np.conj(T[i])],
                                  [[1, 3, 5], [-1, 2, 1], [-2, 2, 3, 4], [-3, 4, 5]])
        
        #print(Eng1)
        if abs(Eng1[1] - Eng0[1]) < tol:
            break
        Eng0 = copy.copy(Eng1)
    
    #print(Eng1 / float(Ns))
    
    return T, Eng1 / float(Ns)

def SingleSiteOperator(T, Op):
    """
    This is a function to calculate the expectation value of single-site operator Op on each site
    Parameters:
        T : MPS tensors
        Op : single-site operator
    Returns:
        Exp_val : expectation values on each site
    """
    Ns = len(T)
    Exp_val = np.zeros(Ns)
    
    for i in range(Ns):
        if i==0:
            TL=np.eye(np.shape(T[0])[0], dtype=complex)
        if i==Ns-1:
            TR=np.eye(np.shape(T[-1])[-1], dtype=complex)
        for j in range(Ns):
            if j<i:
                if j==0:
                    TL=Sub.NCon([T[j], np.conj(T[j])],
                             [[1, 2, -2], [1, 2, -1]])
                else:
                    TL=Sub.NCon([TL, T[j], np.conj(T[j])],
                             [[1, 2], [2,3,-2], [1, 3, -1]])
            if j>i:
                if j==i+1:
                    TR=Sub.NCon([T[j], np.conj(T[j])],
                             [[-2, 1, -4], [-1, 1, -3]])
                else:
                    TR=Sub.NCon([TR,T[j], np.conj(T[j])],
                             [[-1, -2, 1,2], [2, 3, -4], [1, 3, -3]])
                if j==Ns-1:
                    TR=Sub.NCon([TR],
                             [[-1,-2,1,1]])
        exp_val=Sub.NCon([TL,np.conj(T[i]), Op, T[i],TR],
                             [[1,2],[1,3,4],[3,5],[2,5,6],[4,6]])/Sub.NCon([TL,np.conj(T[i]),T[i],TR],
                             [[1,2],[1,3,4],[2,3,5],[4,5]])
        Exp_val[i] = exp_val.real  
    return Exp_val
if __name__ == "__main__":
    Ns = 10
    Dp = 2
    Ds = 4
    g=0.428
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    print(f"-------------------------------------")
    print(f"(3) g={g}, Ns={Ns}, Ds={Ds}")
    Mpo = GetMpo_Obc(Dp, g)
    T = InitMps(Ns, Dp, Ds)
    HL, HR = InitH(Mpo, T)
    T,E_site = OptT(Mpo, HL, HR, T)
    Sx=SingleSiteOperator(T, sx)
    Sz=SingleSiteOperator(T, sz)
    print(f"(3) energy per site:{E_site}")
    print(f"(3) energy avarge:{np.mean(E_site)}")
    print(f"(3) sigma_x per site:{Sx}")
    print(f"(3) sigma_z per site:{Sz}")
    print(f"-------------------------------------")
    Ds = 6
    print(f"-------------------------------------")
    print(f"(3) g={g}, Ns={Ns}, Ds={Ds}")
    Mpo = GetMpo_Obc(Dp, g)
    T = InitMps(Ns, Dp, Ds)
    HL, HR = InitH(Mpo, T)
    T,E_site = OptT(Mpo, HL, HR, T)
    Sx=SingleSiteOperator(T, sx)
    Sz=SingleSiteOperator(T, sz)
    print(f"(3) energy per site:{E_site}")
    print(f"(3) energy avarge:{np.mean(E_site)}")
    print(f"(3) sigma_x per site:{Sx}" )
    print(f"(3) sigma_z per site:{Sz}" )
    print(f"-------------------------------------")
