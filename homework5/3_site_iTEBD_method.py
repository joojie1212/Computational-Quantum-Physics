import numpy as np
import numpy.linalg as LA
from scipy.linalg import expm
from scipy import linalg
import scipy.sparse.linalg as LAs

import Sub as Sub


# ============================================================
#  Three-site Hamiltonian
# ============================================================

def three_site_hamiltonian(Dp, g):
    """
    Construct the 3-site Hamiltonian tensor H_{s1 s2 s3, s1' s2' s3'}.

    The Hamiltonian is:
        H = - (2 S1^x + 2 S1^z S2^z)
            + 8 g ( S1^x S2^z S3^z + S1^z S2^z S3^x )

    This Hamiltonian is used as the local operator in the 3-site iTEBD update.

    Parameters
    ----------
    Dp : int
        Physical dimension of each site.
    g : float
        Coupling constant.

    Returns
    -------
    H : ndarray, shape (Dp, Dp, Dp, Dp, Dp, Dp)
        Three-site Hamiltonian tensor.
    """
    S0, Sp, Sm, Sz, Sx, Sy = Sub.SpinOper(Dp)

    H = (
        -np.kron(2 * Sx, np.kron(S0, S0))
        -np.kron(2 * Sz, np.kron(2 * Sz, S0))
        + 8 * g * (
            np.kron(Sx, np.kron(Sz, Sz))
            + np.kron(Sz, np.kron(Sz, Sx))
        )
    )

    H = np.reshape(H, [Dp, Dp, Dp, Dp, Dp, Dp])
    return H


# ============================================================
#  Imaginary-time evolution operator
# ============================================================

def evolution_operator(H, dt):
    """
    Construct the imaginary-time evolution operator exp(-dt * H).

    Parameters
    ----------
    H : ndarray
        Three-site Hamiltonian tensor.
    dt : float
        Imaginary time step.

    Returns
    -------
    UH : ndarray
        Imaginary-time evolution operator with the same tensor shape as H.
    """
    Dp = H.shape[0]

    if LA.norm(H) < 1.0e-12:
        UH = np.reshape(np.eye(Dp ** 3), [Dp, Dp, Dp, Dp, Dp, Dp])
    else:
        A = np.reshape(H, [Dp ** 3, Dp ** 3])
        V, S, Dc = Sub.SplitEigh(A, Dp ** 3)

        W = np.diag(np.exp(-dt * S))
        A = V @ W @ np.conj(V).T
        UH = np.reshape(A, [Dp, Dp, Dp, Dp, Dp, Dp])

    return UH


# ============================================================
#  Initialization
# ============================================================

def InitTG(Dp, Ds):
    """
    Initialize the MPS tensors T and Schmidt values G.

    Parameters
    ----------
    Dp : int
        Physical dimension.
    Ds : int
        Bond dimension.

    Returns
    -------
    T : list of ndarray
        Site tensors T[i] with shape (Ds, Dp, Ds).
    G : list of ndarray
        Schmidt coefficients on each bond.
    """
    T = [np.random.rand(Ds, Dp, Ds) for _ in range(3)]
    G = [np.random.rand(Ds) for _ in range(3)]
    return T, G


# ============================================================
#  Three-site evolution and SVD
# ============================================================

def Evo_Bond_Threesite(Tl, Tm, Tr, Gl, Gm, Gn, Gr, Ds, UH):
    """
    Perform a single 3-site imaginary-time evolution step and SVD truncation.

    This updates three site tensors and two Schmidt spectra.

    Parameters
    ----------
    Tl, Tm, Tr : ndarray
        Left, middle, right MPS tensors.
    Gl, Gm, Gn, Gr : ndarray
        Schmidt values on surrounding bonds.
    Ds : int
        Maximum bond dimension.
    UH : ndarray
        Imaginary-time evolution operator.

    Returns
    -------
    Tl, Tm, Tr : ndarray
        Updated site tensors.
    Gm, Gn : ndarray
        Updated Schmidt spectra.
    """
    A = Sub.NCon(
        [np.diag(Gl), Tl, np.diag(Gm), Tm, np.diag(Gn), Tr, np.diag(Gr), UH],
        [[-1, 1], [1, 7, 2], [2, 3], [3, 8, 4],
         [4, 5], [5, 9, 6], [6, -5], [-2, -3, -4, 7, 8, 9]]
    )

    DA = A.shape
    A = Sub.Group(A, [[0, 1], [2, 3, 4]])

    U, Gm, V, Dc = Sub.SplitSvd_Lapack(A, Ds, 0, prec=1.0e-12)
    Gm /= np.linalg.norm(Gm)

    U = np.reshape(U, [DA[0], DA[1], Dc])
    Tl = np.tensordot(np.diag(1.0 / Gl), U, (1, 0))

    V = np.reshape(V, [Dc, DA[2], DA[3], DA[4]])
    B = Sub.NCon([np.diag(Gm), V], [[-1, 1], [1, -2, -3, -4]])

    DB = B.shape
    B = Sub.Group(B, [[0, 1], [2, 3]])

    U, Gn, V, Dc = Sub.SplitSvd_Lapack(B, Ds, 0, prec=1.0e-12)
    Gn /= np.linalg.norm(Gn)

    U = np.reshape(U, [DB[0], DB[1], Dc])
    Tm = np.tensordot(np.diag(1.0 / Gm), U, (1, 0))

    V = np.reshape(V, [Dc, DB[2], DB[3]])
    Tr = np.tensordot(V, np.diag(1.0 / Gr), (2, 0))

    return Tl, Tm, Tr, Gm, Gn


# ============================================================
#  iTEBD evolution
# ============================================================

def Evo(Ds, Ham, T, G, Tau, Iter, Prec, Label):
    """
    Perform full imaginary-time evolution using decreasing time steps.

    Parameters
    ----------
    Ds : int
        Bond dimension.
    Ham : ndarray
        Three-site Hamiltonian.
    T, G : list
        Initial MPS tensors and Schmidt values.
    Tau : list of float
        Imaginary time steps.
    Iter : int
        Maximum number of iterations per time step.
    Prec : list of float
        Convergence threshold for each dt.
    Label : str
        Output log label.

    Returns
    -------
    T, G : list
        Converged MPS tensors and Schmidt values.
    """
    Dp = Ham.shape[0]
    File_Log = open('Log' + Label + '.dat', 'w')
    r0 = 0

    for idt, dt in enumerate(Tau):
        UH = evolution_operator(Ham, dt)
        G0 = np.ones(3)

        for r in range(Iter):
            for ib in range(2, -1, -1):
                T[ib], T[(ib + 1) % 3], T[(ib + 2) % 3], G[ib], G[(ib + 1) % 3] = \
                    Evo_Bond_Threesite(
                        T[ib], T[(ib + 1) % 3], T[(ib + 2) % 3],
                        G[(ib + 2) % 3], G[ib], G[(ib + 1) % 3], G[(ib + 2) % 3],
                        Ds, UH
                    )

            Err = sum(abs(G[i][0] - G0[i]) for i in range(3))

            if r % 100 == 1:
                File_Log.write(f"{r + r0}\t{Err:.15e}\n")
                File_Log.flush()

            if Err < Prec[idt]:
                r0 += r
                break

            for i in range(3):
                G0[i] = G[i][0]

    File_Log.close()
    return T, G


# ============================================================
#  Observables
# ============================================================

def Cal_Site(Op, T, Gl, Gr):
    """
    Compute single-site expectation value <Op>.

    Parameters
    ----------
    Op : ndarray
        Single-site operator.
    T : ndarray
        MPS site tensor.
    Gl, Gr : ndarray
        Left and right Schmidt values.

    Returns
    -------
    Val : complex
        Expectation value.
    """
    A = Sub.NCon([np.diag(Gl), T, np.diag(Gr)], [[-1, 1], [1, -2, 2], [2, -3]])
    Val = Sub.NCon([A, Op, np.conj(A)], [[3, 1, 4], [2, 1], [3, 2, 4]])
    return Val


def Cal_Energy(Ham, T, G):
    """
    Compute the ground-state energy per site.

    Parameters
    ----------
    Ham : ndarray
        Three-site Hamiltonian.
    T, G : list
        MPS tensors and Schmidt values.

    Returns
    -------
    Eng : float
        Energy per site.
    """
    Eng = np.zeros(3)
    Nom = np.zeros(3)

    for ib in range(3):
        A = Sub.NCon(
            [np.diag(G[(ib + 2) % 3]), T[ib], np.diag(G[ib]),
             T[(ib + 1) % 3], np.diag(G[(ib + 1) % 3]),
             T[(ib + 2) % 3], np.diag(G[(ib + 2) % 3])],
            [[-1, 1], [1, -2, 2], [2, 3],
             [3, -3, 4], [4, 5], [5, -4, 6], [6, -5]]
        )

        Nom[ib] = Sub.NCon([np.conj(A), A], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        Eng[ib] = Sub.NCon(
            [np.conj(A), Ham, A],
            [[1, 2, 3, 4, 5], [2, 3, 4, 6, 7, 8], [1, 6, 7, 8, 5]]
        ) / Nom[ib]

    return np.mean(Eng)


def compute_entanglement(G):
    """
    Compute entanglement entropy and spectrum for each bond.

    Parameters
    ----------
    G : list of ndarray
        Schmidt values on each bond.

    Returns
    -------
    entanglement_spectra : list of ndarray
        Entanglement spectra (-log lambda).
    entanglement_entropy : list of float
        von Neumann entanglement entropy.
    """
    entanglement_spectra = []
    entanglement_entropy = []

    for lam in G:
        lam = lam / np.linalg.norm(lam)
        spectrum = lam ** 2

        entanglement_spectra.append(-np.log(lam + 1e-16))
        entanglement_entropy.append(-np.sum(spectrum * np.log(spectrum + 1e-16)))

    return entanglement_spectra, entanglement_entropy

if __name__ == "__main__":
    sx = np.array([[0,1/2],[1/2,0]],dtype=complex)
    sz = np.array([[1/2,0],[0,-1/2]],dtype=complex)
    id2 = np.eye(2,dtype=complex)

    Ds=6
    Dp=2
    g=0.428
    Ham=three_site_hamiltonian(Dp,g)
    T,G=InitTG(Dp,Ds)
    Tau = [0.1,0.01,0.001]
    Iter = 100000
    Prec = [1.0e-10]*3
    T,G=Evo(Ds,Ham,T,G,Tau,Iter,Prec,"test")
    Eng=Cal_Energy(Ham,T,G)


    sx_site=np.array([Cal_Site(sx, T[0], G[2], G[0]),Cal_Site(sx, T[1], G[0], G[1]),Cal_Site(sx, T[2], G[1], G[2])])
    sz_site=np.array([Cal_Site(sz, T[0], G[2], G[0]),Cal_Site(sz, T[1], G[0], G[1]),Cal_Site(sz, T[2], G[1], G[2])])
    en_s,en_ent=compute_entanglement(G)

    print(f"Ds={Ds}, g={g}")
    print(f"(1) ground_energy_per_site: {Eng}")
    print(f"(1) sigma_x per site: {sx_site}")
    print(f"(1) sigma_z per site: {sz_site}")
    print(f"(2) entanglement entropy per site: {en_ent}")
    print(f"(2) entanglement spectra per site: {en_s}")