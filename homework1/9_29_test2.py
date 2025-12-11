import numpy as np
from scipy.linalg import eigh
from scipy import sparse
import matplotlib.pyplot as plt
# we are concerning  single_particle basis, 
# if there are two particles, the system will become very difficult.

def BuildHamiltonian(F,N):
    """
    Construct the Hamiltonian matrix for a 1D tight-binding model (dense matrix).

    Parameters
    --------
    F : float
        Linear potential coefficient in the Hamiltonian 
    N : int
        Dimension of the system, i.e., the number of lattice sites.

    Returns
    --------
    Hamr : ndarray
        N x N dense matrix representing the Hamiltonian.
    """
    HI, HJ, HV = [], [], []
    for i0 in range(N):
        #off-diagonal term
        if(i0>0):
            HI.append(i0-1)
            HJ.append(i0)
            HV.append(-1)
        if(i0<N-1):
            HI.append(i0+1)
            HJ.append(i0)
            HV.append(-1)
        #diagonal term 
        HI.append(i0)
        HJ.append(i0)
        HV.append(F*i0)
    
    Hamr = sparse.coo_matrix((HV, (HI, HJ)), shape=(N, N)).tocsc()
    return Hamr.toarray()
def Build_Ini_state(k0,alpha,N0,N):
    """
    straight forward constructing initial state 
    """
    Ini_state = np.zeros(N, dtype=complex)
    for i in range(N):
        Ini_state[i] = np.exp(-(alpha*(i - N0))**2 / 2 + 1j * k0 * i)
    norm = np.linalg.norm(Ini_state)   
    Ini_state /= norm
    return Ini_state
def Time_evo(t,Ini,eigvals, eigvecs):
    """
    Time evolution of systum.

    Parameters
    --------
    t : float 
        time of the systum
    Ini : np.array
        initial state 
    eigvals : np.array
        eigenvalues of the Hamiltonian
    eigvecs : np.array
        matrix of eigenvalues of the Hamiltonian
    Returns
    --------
    Final : np.array
        array of state at time t 
    """
    eigvecs_hermi = eigvecs.conj().T          
    Ini_eigbase   = eigvecs_hermi @ Ini        
    Final_eigbase = np.exp(-1j * eigvals * t) * Ini_eigbase
    Final = eigvecs @ Final_eigbase
    return Final
if __name__=="__main__":
    N=100
    F=0.1
    Pi=3.14159265
    k0=Pi/2
    alpha=0.15
    N0=51
    print(f"----N={N}, F={F}, k0=pi/2, alpha={alpha}, N0={N0}----\n")
    Hamr=BuildHamiltonian(F,N)
    #calculate eigen values 
    eigvals, eigvecs = eigh(Hamr)
    print("(1) lowest 10 eigenvalue: ")
    for i in range(10):
        print(f"{eigvals[i]}")
    print("\n")
    

    Ini=Build_Ini_state(k0,alpha,N0,N)
    state_time_42=Time_evo(42,Ini,eigvals, eigvecs)
    print(f"(2) norm for t=42 J=10:{np.abs(state_time_42[9])**2} J=20:{np.abs(state_time_42[19])**2} J=30:{np.abs(state_time_42[29])**2} J=40:{np.abs(state_time_42[39])**2} J=50:{np.abs(state_time_42[49])**2}")



    t_max=200
    t_list = np.linspace(0, t_max, 1000)  

    prob_matrix = np.zeros((N, len(t_list)))  


    for idx, t in enumerate(t_list):
        psi_t = Time_evo(t, Ini, eigvals, eigvecs)
        prob_matrix[:, idx] = np.abs(psi_t)**2     


    plt.figure(figsize=(8,6))

    plt.imshow(prob_matrix, 
           origin='lower',         
           aspect='auto', 
           extent=[t_list[0], t_list[-1], 0, N-1],
           cmap='viridis')      
    plt.colorbar(label=r'$|\psi(t)|^2$')
    plt.xlabel('Time t')
    plt.ylabel('Grid index j')
    plt.title('Time evolution of probability amplitude')
    plt.savefig('timeevo.png', bbox_inches='tight')
    plt.show()

    
      


    