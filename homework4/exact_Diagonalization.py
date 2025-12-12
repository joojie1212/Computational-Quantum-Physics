import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Pauli matrices (with out 1/2 factor)
sx = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
sz = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
id2 = sp.identity(2, format='csr', dtype=complex)

#----operators construction functions----
def kronN(op_list):
    """Sparse Kronecker product of list of operators (op_list[0] kron op_list[1] kron ...)."""
    res = op_list[0]
    for op in op_list[1:]:
        res = sp.kron(res, op, format='csr')
    return res

def single_site_operator(N, site, op):
    """Return operator acting with `op` on `site` (0-based) in an N-site chain."""
    ops = []
    for i in range(N):
        ops.append(op if i == site else id2)
    return kronN(ops)

def two_site_operator(N, i, j, op_i, op_j):
    """Return operator acting with op_i on site i and op_j on site j (i<j)."""
    ops = []
    for s in range(N):
        if s == i:
            ops.append(op_i)
        elif s == j:
            ops.append(op_j)
        else:
            ops.append(id2)
    return kronN(ops)

def three_site_operator(N, i, j, k, op_i, op_j, op_k):
    """Return operator acting with op_i on i, op_j on j, op_k on k (i<j<k)."""
    ops = []
    for s in range(N):
        if s == i:
            ops.append(op_i)
        elif s == j:
            ops.append(op_j)
        elif s == k:
            ops.append(op_k)
        else:
            ops.append(id2)
    return kronN(ops)

def build_hamiltonian(N, g=0.428):
    """Construct H as a sparse matrix (csr) for given N, J, g with open boundary conditions."""
    dim = 2**N
    H = sp.csr_matrix((dim, dim), dtype=complex)

    # - sigma^x_j 
    for j in range(N):
        H += -single_site_operator(N, j, sx)

    # - sigma^z_j sigma^z_{j+1} terms (two-site), j=0..N-2
    for j in range(N-1):
        H += -two_site_operator(N, j, j+1, sz, sz)

    # g * (sigma^x_j sigma^z_{j+1} sigma^z_{j+2}) and g*(sigma^z_j sigma^z_{j+1} sigma^x_{j+2})
    # three-site terms, j=0..N-3
    if N >= 3 :
        for j in range(N-2):
            H += g * three_site_operator(N, j, j+1, j+2, sx, sz, sz)
            H += g * three_site_operator(N, j, j+1, j+2, sz, sz, sx)

    return H

def ground_state(H, k=1, which='SA', return_evec=True):
    """
    Function to compute ground state using sparse eigensolver.
    we only find the lowest k eigenvalues/vectors.

    Parameters:
        H : sparse matrix, Hamiltonian operator.

    Returns:
        (energies, vectors) or (energies, None).
    """

    vals, vecs = spla.eigsh(H, k=k, which=which, return_eigenvectors=return_evec)

    idx = np.argsort(vals)
    vals = vals[idx]
    if return_evec:
        vecs = vecs[:, idx]
        return vals, vecs
    else:
        return vals, None

def expectation(op, state):
    """A function to calculate the expectation value for any operator op by directly calculating <state|op|state>"""
    return (state.conj().T @ (op.dot(state))).item()

if __name__ == "__main__":
  
    N = 10
    g = 0.428      

    print(f"N={N}, g={g}")
    H = build_hamiltonian(N, g)
    energies, vecs = ground_state(H)
    E0 = energies[0].real
    psi0 = vecs[:, 0]

    print(f"Ground state energy E0 = {E0}")
    print(f"Energy per site = {E0/N}")

    sigmax_vals = np.zeros(N)  
    sigmaz_vals = np.zeros(N)  

    for j in range(N):
        op_j = single_site_operator(N, j, 0.5*sx) 
        sigmax_vals[j] = expectation(op_j, psi0).real  
    for j in range(N):
        op_j = single_site_operator(N, j, 0.5*sz) 
        sigmaz_vals[j] = expectation(op_j, psi0).real  
    
    print(f"sigma_x per site: {sigmax_vals}")
    print(f"sigma_z per site: {sigmaz_vals}")
