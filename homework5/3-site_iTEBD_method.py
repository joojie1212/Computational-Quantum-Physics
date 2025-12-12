import numpy as np
from scipy.linalg import expm

# Pauli matrices
sx = np.array([[0,1],[1,0]],dtype=complex)
sz = np.array([[1,0],[0,-1]],dtype=complex)
id2 = np.eye(2,dtype=complex)

def three_site_hamiltonian(g):
    """
    Constructs the 3-site Hamiltonian:
    H = - (sigma1^x + sigma1^z sigma2^z) + g (sigma1^x sigma2^z sigma3^z + sigma1^z sigma2^z sigma3^x)
    We only need the Hamiltonian between any three site in our simulation
    """
    # Single-site term: 
    term1 = - np.kron(sx, np.kron(id2,id2))
    
    # Two-site zz interaction:
    term2 = - np.kron(sz, np.kron(sz,id2))
    
    # Three-site interactions:
    term3 = g * (np.kron(sx, np.kron(sz,sz)) + np.kron(sz, np.kron(sz,sx)))
    
    # Total Hamiltonian
    H = term1 + term2 + term3
    return H


def evolution_operator(H, dt):
    """
    Constructs the imaginary time-evolve operator
    """
    return expm(-dt*H)  

def random_mps(D):
    """
    Constructs Initial state randomly

    Parameters:
        D : the bind demension 

    Returns: 
        ABC : three matrix shape as(2,D,D)
    """

    A = np.random.rand(2,D,D) + 1j*np.random.rand(2,D,D)
    B = np.random.rand(2,D,D) + 1j*np.random.rand(2,D,D)
    C = np.random.rand(2,D,D) + 1j*np.random.rand(2,D,D)
    A /= np.linalg.norm(A)
    B /= np.linalg.norm(B)
    C /= np.linalg.norm(C)
    return A,B,C

def contract_three(A,B,C,Lambda1,Lambda2):
    """
    Contract 3-site MPS with bond tensors

    Parameters:

        A,B,C are MPS matrixs defined in function random_mps
        Lambda1,Lambda2 are arrays dimension D, regard as schmidt coefficient

    Returns: 
        temp : large matrix to be evolve shaped (2,2,2,D,D)
    """
    temp = np.tensordot(A, np.diag(Lambda1), axes=(2,0))    
    temp = np.tensordot(temp, B, axes=(2,1))
    temp = np.tensordot(temp, np.diag(Lambda2), axes=(3,0))
    temp = np.tensordot(temp, C, axes=(3,1)) 
    return temp 
def svd_3site_update_canonical(theta, D):
    """
    SVD split for 3-site tensor with canonical form.
    Returns tensors A,B,C and bond Lambdas (Lambda1,Lambda2)
    All bond Lambdas are normalized Schmidt coefficients.
    """
    d1, d2, d3, Dl, Dr = theta.shape

    # --- First SVD: (d1*Dl) | (d2*d3*Dr)
    theta_reshaped = theta.reshape(d1*Dl, d2*d3*Dr)
    U, S1, Vh = np.linalg.svd(theta_reshaped, full_matrices=False)
    U = U[:, :D]
    S1 = S1[:D]
    Vh = Vh[:D, :]
    Lambda1 = S1 / np.linalg.norm(S1)  # normalize Schmidt coefficients

    A = U.reshape(d1, Dl, -1)

    # --- Multiply S1 into Vh ---
    V = np.dot(np.diag(S1), Vh)

    # --- Second SVD: (D1*d2) | (d3*Dr)
    V_reshaped = V.reshape(A.shape[2]*d2, d3*Dr)
    U2, S2, Vh2 = np.linalg.svd(V_reshaped, full_matrices=False)
    U2 = U2[:, :D]
    S2 = S2[:D]
    Vh2 = Vh2[:D, :]
    Lambda2 = S2 / np.linalg.norm(S2)

    B = U2.reshape(d2, A.shape[2], -1)
    C = Vh2.reshape(d3, len(S2), Dr)

    # --- Canonical normalization ---
    # Left-normalize A
    for s in range(d1):
        A[s] /= np.linalg.norm(A[s])
    # Left-normalize B
    for s in range(d2):
        B[s] /= np.linalg.norm(B[s])
    # Left-normalize C
    for s in range(d3):
        C[s] /= np.linalg.norm(C[s])

    return A, Lambda1, B, Lambda2, C
def svd_3site_update(theta, D):
    """
    Perform two successive SVDs to split a 3-site tensor into A, B, C 
    and bond Lambdas for 3-site iTEBD. This routine of SVD decomposition of arbitrary bond dimension. 
    The truncate process may not be needed, but we left it here for robustness.
    
    Parameters:
        theta: shape (d1, d2, d3, Dl, Dr)
        D: maximum bond dimension


    """
    d1, d2, d3, Dl, Dr = theta.shape

    # merge first site + left bond
    theta_reshaped = theta.reshape(d1 * Dl, d2 * d3 * Dr)
    U, S1, Vh = np.linalg.svd(theta_reshaped, full_matrices=False)

    # truncate
    U = U[:, :D]
    S1 = S1[:D]
    Vh = Vh[:D, :]

    # reshape U -> A (d1, Dl, D1)
    A = U.reshape(d1, Dl, -1)  
    Lambda1 = S1

    # multiply S1 into Vh for right block
    V = np.dot(np.diag(S1), Vh)  # shape (D1, d2*d3*Dr)

    #second SVD to split B and C
    V_reshaped = V.reshape(-1, d3 * Dr)  # merge (D1*d2) | (d3*Dr)
    U2, S2, Vh2 = np.linalg.svd(V_reshaped, full_matrices=False)

    # truncate
    U2 = U2[:, :D]
    S2 = S2[:D]
    Vh2 = Vh2[:D, :]

    # reshape 
    B = U2.reshape(d2, len(S1), len(S2))
    Lambda2 = S2

    # reshape
    C = Vh2.reshape(d3, len(S2), Dr)

    return A, Lambda1, B, Lambda2, C

def iTEBD_3site(g=0.428, D=6, dt=10**-3, maxstep=10**6, tol=1e-4):
    """
    3-site iTEBD with adaptive stopping criterion:
    stops when relative change of system energy < tol
    """
    # initialize
    #A,B,C = random_mps(D)
    A = np.zeros((2, D, D))
    A[0, 0, 0] = 1.0

    B = np.zeros((2, D, D))
    B[0, 0, 0] = 1.0

    C = np.zeros((2, D, D))
    C[0, 0, 0] = 1.0
    Lambda1 = np.ones(D)
    Lambda2 = np.ones(D)
    Lambda3 = np.ones(D)
    
    H = three_site_hamiltonian(g)
    U = evolution_operator(H, dt)
    U = U.reshape(2,2,2, 2,2,2)

    E_prev = None

    for step in range(maxstep):
        # --- first 3-site update ---
        theta1 = contract_three(A,B,C,Lambda1,Lambda2)
        theta1 = np.tensordot(U, theta1, axes=([3,4,5],[0,2,3]))
        A,Lambda1,B,Lambda2,C = svd_3site_update_canonical(theta1, D)

        # --- second 3-site update ---
        theta2 = contract_three(B,C,A,Lambda2,Lambda3)
        theta2 = np.tensordot(U, theta2, axes=([3,4,5],[0,2,3]))
        B,Lambda2,C,Lambda3,A = svd_3site_update_canonical(theta2, D)

        # --- third 3-site update ---
        theta3 = contract_three(C,A,B,Lambda3,Lambda1)
        theta3 = np.tensordot(U, theta3, axes=([3,4,5],[0,2,3]))
        C,Lambda3,A,Lambda1,B = svd_3site_update_canonical(theta3, D)
        Lambda3 /= np.linalg.norm(Lambda3)
        # compute average site energy for convergence check
        """
        debug
        print("A.shape =", A.shape)
        print("B.shape =", B.shape)
        print("C.shape =", C.shape)
        print("Lambda1.shape =", Lambda1.shape)
        print("Lambda2.shape =", Lambda2.shape)
        print("Lambda3.shape =", Lambda3.shape)
        """
        psi = np.einsum(
    "s a b, b, t b c, c, u c a -> s t u",
    A, Lambda1, B, Lambda2, C
)
        psi = psi.reshape(8)

        E_curr = np.real(np.vdot(psi, H @ psi))
        E_curr /= 3.0

        # check relative change
        if E_prev is not None:
            rel_change = np.abs(E_curr - E_prev) /( np.abs(E_prev)+1.e-10)
            if rel_change < tol:
                break
        E_prev = E_curr
    print(f" step {step}, tol={tol}, change={rel_change}")
    return A,B,C,Lambda1,Lambda2,Lambda3,E_curr

def site_expectation(tensor, Lambda_left, Lambda_right, op):
    """
    tensor: (d, Dl, Dr)
    Lambda_left/right: (Dl,) / (Dr,)
    op: (d,d)
    """
    theta = np.tensordot(np.diag(Lambda_left), tensor, axes=(1,1))
    theta = np.tensordot(theta, np.diag(Lambda_right), axes=(2,0))  # shape: (d, Dl*Dr)
    theta = theta.reshape(tensor.shape[0], -1)  # merge bonds
    expec = np.vdot(theta.conj().T, op @ theta)  # <O>
    return np.real(expec)

def bond_entropy(Lambda):
    # normalized Schmidt coefficients
    p = Lambda**2 / np.sum(Lambda**2)
    S = -np.sum(p * np.log(p + 1e-16)) 
    return S

def compute_observables(A, B, C, Lambda1, Lambda2, Lambda3):
    """
    Compute local expectation values and entanglement.

    Inputs:
        A,B,C : MPS tensors (shape: d,D,D)
        Lambda1,Lambda2,Lambda3 : bond vectors

    Returns:
        result : dict with keys:
            'sx' : array of <sigma_x> for sites [A,B,C]
            'sz' : array of <sigma_z> for sites [A,B,C]
            'ent_entropy' : array of entanglement entropy [S1,S2,S3]
            'ent_spectrum' : list of arrays, each bond Schmidt values squared
    """
    # local <sx> and <sz>
    sx_vals = [
        site_expectation(A, Lambda3, Lambda1, sx),
        site_expectation(B, Lambda1, Lambda2, sx),
        site_expectation(C, Lambda2, Lambda3, sx)
    ]
    sz_vals = [
        site_expectation(A, Lambda3, Lambda1, sz),
        site_expectation(B, Lambda1, Lambda2, sz),
        site_expectation(C, Lambda2, Lambda3, sz)
    ]
    
    # entanglement entropy from bond Lambda
    ent_entropy = [bond_entropy(Lambda1), bond_entropy(Lambda2), bond_entropy(Lambda3)]
    
    # entanglement spectrum (Schmidt values squared)
    ent_spectrum = [-np.log(Lambda1),
                    -np.log(Lambda2),
                    -np.log(Lambda3)]
    
    result = {
        'sx': sx_vals,
        'sz': sz_vals,
        'ent_entropy': ent_entropy,
        'ent_spectrum': ent_spectrum
    }
    
    return result



if __name__ == "__main__":
    g=1
    D=6
    A,B,C,Lambda1,Lambda2,Lambda3,E_curr= iTEBD_3site(g,D)
    result=compute_observables(A, B, C, Lambda1, Lambda2, Lambda3)
    
    sx_list = result['sx']             
    sz_list = result['sz']              
    ent_entropy = result['ent_entropy']     
    ent_spectrum = result['ent_spectrum']  
    
    print(f"Ds={D}, g={g}")
    print(f"(1)ground_state_energy_per_site: {E_curr:.12f}")
    print(f"sigmax_per_site: {sx_list}")
    print(f"sigmaz_per_site: {sz_list}")
    print(f"Entanglement_entropy_per_site: {ent_entropy}")
    print(f"Entanglement_spectrum_energy_per_site: {ent_spectrum}")