import numpy as np
from scipy.linalg import expm

# Pauli matrices
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
id2 = np.eye(2, dtype=complex)

def three_site_hamiltonian(g):
    term1 = -np.kron(sx, np.kron(id2, id2))
    term2 = -np.kron(sz, np.kron(sz, id2))
    term3 = g * (np.kron(sx, np.kron(sz, sz)) + np.kron(sz, np.kron(sz, sx)))
    return term1 + term2 + term3

def evolution_operator(H, dt):
    return expm(-dt * H).reshape(2, 2, 2, 2, 2, 2)

def contract_three(A, B, C, Lambda1, Lambda2):
    temp = np.tensordot(np.diag(Lambda1), A, axes=(1, 1))
    temp = np.tensordot(temp, B, axes=(2, 1))
    temp = np.tensordot(temp, np.diag(Lambda2), axes=(3, 0))
    temp = np.tensordot(temp, C, axes=(3, 1))
    return temp  # shape: (Dl, d1, d2, d3, Dr)

def svd_3site_update_canonical(theta, D):
    Dl, d1, d2, d3, Dr = theta.shape
    theta = np.transpose(theta, (1, 0, 2, 3, 4))  # (d1, Dl, d2, d3, Dr)
    theta1 = theta.reshape(d1 * Dl, d2 * d3 * Dr)
    U, S1, Vh = np.linalg.svd(theta1, full_matrices=False)
    U = U[:, :D]
    S1 = S1[:D] / np.linalg.norm(S1)
    Vh = Vh[:D, :]
    A = U.reshape(d1, Dl, -1)
    Lambda1 = S1
    V = np.dot(np.diag(S1), Vh)
    V = V.reshape(-1, d3 * Dr)
    U2, S2, Vh2 = np.linalg.svd(V, full_matrices=False)
    U2 = U2[:, :D]
    S2 = S2[:D] / np.linalg.norm(S2)
    Vh2 = Vh2[:D, :]
    B = U2.reshape(d2, A.shape[2], -1)
    Lambda2 = S2
    C = Vh2.reshape(d3, len(S2), Dr)
    return A, Lambda1, B, Lambda2, C

def iTEBD_3site(g=0.428, D=6, dt_init=1e-2, maxstep=50000, tol=1e-6):
    A = np.zeros((2, D, D), dtype=complex)
    B = np.zeros((2, D, D), dtype=complex)
    C = np.zeros((2, D, D), dtype=complex)
    A[0, 0, 0] = B[0, 0, 0] = C[0, 0, 0] = 1.0
    Lambda1 = np.ones(D)
    Lambda2 = np.ones(D)
    Lambda3 = np.ones(D)
    dt = dt_init
    H = three_site_hamiltonian(g)
    U = evolution_operator(H, dt)
    E_prev = None
    for step in range(maxstep):
        for (X, Y, Z, L1, L2, label) in [
            (A, B, C, Lambda1, Lambda2, "ABC"),
            (B, C, A, Lambda2, Lambda3, "BCA"),
            (C, A, B, Lambda3, Lambda1, "CAB")]:

            theta = contract_three(X, Y, Z, L1, L2)
            theta = np.tensordot(U, theta, axes=([3, 4, 5], [1, 2, 3]))
            theta = np.transpose(theta, (3, 0, 1, 2, 4))
            X, L1_new, Y, L2_new, Z = svd_3site_update_canonical(theta, D)

            if label == "ABC":
                A, Lambda1, B, Lambda2, C = X, L1_new, Y, L2_new, Z
            elif label == "BCA":
                B, Lambda2, C, Lambda3, A = X, L1_new, Y, L2_new, Z
            elif label == "CAB":
                C, Lambda3, A, Lambda1, B = X, L1_new, Y, L2_new, Z

        psi = np.einsum("s a b, b, t b c, c, u c a, a -> s t u", A, Lambda1, B, Lambda2, C, Lambda3).reshape(8)
        E_curr = np.real(np.vdot(psi, H @ psi)) / 3

        if E_prev is not None:
            rel_change = np.abs(E_curr - E_prev) / (np.abs(E_prev) + 1e-10)
            if rel_change < tol:
                print(f"Converged at step {step}, dt={dt:.2e}, energy={E_curr:.12f}, ΔE={rel_change:.2e}")
                break
            if step % 1000 == 0 and step > 0:
                dt *= 0.5
                U = evolution_operator(H, dt)
        E_prev = E_curr
    return A, B, C, Lambda1, Lambda2, Lambda3, E_curr

def bond_entropy(Lambda):
    p = Lambda ** 2 / np.sum(Lambda ** 2)
    return -np.sum(p * np.log(p + 1e-16))

def site_expectation(tensor, Lambda_left, Lambda_right, op):
    theta = np.tensordot(np.diag(Lambda_left), tensor, axes=(1,1))
    theta = np.tensordot(theta, np.diag(Lambda_right), axes=(2,0))
    theta = theta.reshape(tensor.shape[0], -1)
    return np.real(np.vdot(theta.conj().T, op @ theta))

def compute_observables(A, B, C, Lambda1, Lambda2, Lambda3):
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
    ent_entropy = [bond_entropy(Lambda1), bond_entropy(Lambda2), bond_entropy(Lambda3)]
    ent_spectrum = [-np.log(Lambda1 + 1e-16), -np.log(Lambda2 + 1e-16), -np.log(Lambda3 + 1e-16)]
    return {'sx': sx_vals, 'sz': sz_vals, 'ent_entropy': ent_entropy, 'ent_spectrum': ent_spectrum}

if __name__ == "__main__":
    g = 0.428
    D = 6
    A, B, C, Lambda1, Lambda2, Lambda3, E_curr = iTEBD_3site(g, D)
    result = compute_observables(A, B, C, Lambda1, Lambda2, Lambda3)
    print(f"D={D}, g={g}")
    print(f"(1) Ground state energy per site: {E_curr:.12f}")
    print(f"(2) <σx> per site: {result['sx']}")
    print(f"(3) <σz> per site: {result['sz']}")
    print(f"(4) Entanglement entropy per bond: {result['ent_entropy']}")
    print(f"(5) Entanglement spectrum energy per bond: {result['ent_spectrum']}")

