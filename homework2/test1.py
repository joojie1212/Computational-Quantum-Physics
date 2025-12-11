import numpy as np
from scipy.linalg import eigh, qr


# --- parameters ---
N = 8
g = 1.0
J = 1.0
h = 1.0

# Pauli matrices
sx = np.array([[0,1],[1,0]], dtype=complex)
sy = np.array([[0,-1j],[1j,0]], dtype=complex)
sz = np.array([[1,0],[0,-1]], dtype=complex)
id2 = np.eye(2, dtype=complex)

# Kronecker product builder for single-site operator at site j (0-based)
def single_site_op(op, site, N):
    """Return full 2^N x 2^N matrix for operator 'op' acting on 'site'."""
    res = 1
    for i in range(N):
        res = np.kron(res, op if i == site else id2)
    return res

# More general: product of single-site ops given as dict {site: matrix}
def multi_site_op(op_dict, N):
    res = 1
    for i in range(N):
        res = np.kron(res, op_dict[i] if i in op_dict else id2)
    return res

# Build Hamiltonian
dim = 2**N
H = np.zeros((dim, dim), dtype=complex)

for j in range(N):
    jm1 = (j-1) % N
    jp1 = (j+1) % N
    # three-spin term g * sz_{j-1} * sy_j * sz_{j+1}
    term1 = multi_site_op({jm1: sz, j: sy, jp1: sz}, N)
    # zz coupling J * sz_j * sz_{j+1}
    term2 = multi_site_op({j: sz, jp1: sz}, N)
    # hx term h * sx_j
    term3 = multi_site_op({j: sx}, N)
    H += -( g * term1 + J * term2 + h * term3 )   # note overall minus sign

# Build translation operator T (shifts every site by +1 -> right shift of bits)
# We build T in the computational basis of spins (|s_{N-1} ... s_0>)
dim = 2**N
T = np.zeros((dim, dim), dtype=complex)
for state in range(dim):
    # represent as bitstring length N (site 0 is least significant bit)
    bits = [(state >> i) & 1 for i in range(N)]
    # rotate right by 1: newbits[i] = bits[(i-1)%N]
    newbits = [bits[(i-1) % N] for i in range(N)]
    # convert newbits back to integer (least significant bit index 0)
    newstate = sum((newbits[i] << i) for i in range(N))
    T[newstate, state] = 1.0

# Confirm [H, T] ~ 0 (numerical)
#comm_norm = np.linalg.norm(H @ T - T @ H)
#print("||[H,T]|| =", comm_norm)

# For each momentum sector k, build projector Pk and reduce H
mom_vals = []
E_k_list = []
k_list = []
for m in range(N):
    k = 2.0 * np.pi * m / N
    # build projector P_k = (1/N) sum_n e^{-i k n} T^n
    Pk = np.zeros((dim, dim), dtype=complex)
    Tn = np.eye(dim, dtype=complex)
    for n in range(N):
        Pk += np.exp(-1j * k * n) * Tn
        Tn = T @ Tn
    Pk /= N

    # find an orthonormal basis for image(Pk) by applying Pk to standard basis and QR
    # stack columns Pk @ e_i for i=0..dim-1 then do QR on nonzero columns
    cols = []
    for i in range(dim):
        v = Pk[:, i]
        if np.linalg.norm(v) > 1e-10:
            cols.append(v)
    if len(cols) == 0:
        # empty sector (shouldn't happen)
        continue
    M = np.column_stack(cols)
    Q, R = qr(M, mode='economic')  # Q columns are orthonormal basis for sector
    # small matrix representation of H in this subspace
    Hsmall = Q.conj().T @ (H @ Q)
    # Hermitian ensure
    Hsmall = (Hsmall + Hsmall.conj().T) / 2.0
    evals, evecs_small = eigh(Hsmall)
    E_k_list.append(np.sort(evals.real))
    k_list.append(k)
    mom_vals.append(m)

    local_min_E = evals[0].real
    if(m==0): 
        global_min_E = local_min_E

        psi_small = evecs_small[:, 0]
        #full basis： |psi_full> = Q @ |psi_small>
        psi_full = Q @ psi_small

        psi_full = psi_full / np.linalg.norm(psi_full)
        global_psi_full = psi_full
        global_min_sector = m
        global_min_k = k
    if local_min_E < global_min_E - 1e-12:  
        global_min_E = local_min_E

        psi_small = evecs_small[:, 0]

        psi_full = Q @ psi_small

        psi_full = psi_full / np.linalg.norm(psi_full)
        global_psi_full = psi_full
        global_min_sector = m
        global_min_k = k


# Print results
for m,k,evals in zip(mom_vals, k_list, E_k_list):
    print(f"(1)_Momentum_Sector_ki={m} : {evals}")




E0 = global_min_E
e0_per_site = E0 / N

print(f"_Ground_state_energy_per_site= {E0}")


# 计算 <sigma^z_i> 和 <sigma^x_i> 在基态下（每个格点）
sz_expect_sites = np.zeros(N, dtype=float)
sx_expect_sites = np.zeros(N, dtype=float)
for j in range(N):
    Sz_j = single_site_op(sz, j, N)
    Sx_j = single_site_op(sx, j, N)
    sz_exp = np.vdot(global_psi_full, Sz_j @ global_psi_full).real
    sx_exp = np.vdot(global_psi_full, Sx_j @ global_psi_full).real
    sz_expect_sites[j] = sz_exp
    sx_expect_sites[j] = sx_exp

sz_mean = sz_expect_sites.mean()
sx_mean = sx_expect_sites.mean()

print(f"_Ground_state_sigmax_per_site= {sx_mean}")
print(f"_Ground_state_sigmaz_per_site= {sz_mean}")



