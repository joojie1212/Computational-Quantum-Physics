import numpy as np
from numba import njit
from tqdm import tqdm
import os

V1=np.array([1.0,0.0], dtype=np.float64)
V2=np.array([0.5,np.sqrt(3)/2], dtype=np.float64)
V3=np.array([-0.5,np.sqrt(3)/2], dtype=np.float64)
V4=np.array([-1.0,0.0], dtype=np.float64)
V5=np.array([-0.5,-np.sqrt(3)/2], dtype=np.float64)
V6=np.array([0.5,-np.sqrt(3)/2], dtype=np.float64)

def init_spins(L, seed=None):
    """
    Initialize the spin configuration.

    Parameters
    ----------
    L : int
        Linear system size. The lattice contains L x L unit cells.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    spins : ndarray of shape (L, L, 3), dtype int8
        Spin variables on each bond.
        Each spin takes values +1 or -1.
        The third index labels the three bond directions.
    """
    rng = np.random.default_rng(seed)
    spins = rng.choice([-1, 1], size=(L, L, 3))
    return spins.astype(np.int8)

@njit
def charge_A(spins, x, y, L):
    """
    Compute the magnetic charge on an A-type vertex.

    Parameters
    ----------
    spins : ndarray
        Spin configuration of shape (L, L, 3).
    x, y : int
        Lattice coordinates of the vertex.
    L : int
        System size (periodic boundary conditions assumed).

    Returns
    -------
    qA : int
        Magnetic charge at the A vertex.
    """
    return (
        spins[x % L, y % L, 0]
        + spins[x % L, y % L, 1]
        + spins[x % L, y % L, 2]
    )

@njit
def charge_B(spins, x, y, L):
    """
    Compute the magnetic charge on a B-type vertex.

    Parameters
    ----------
    spins : ndarray
        Spin configuration of shape (L, L, 3).
    x, y : int
        Lattice coordinates of the vertex.
    L : int
        System size (periodic boundary conditions assumed).

    Returns
    -------
    qB : int
        Magnetic charge at the B vertex.
    """
    return -(
        spins[x % L, y % L, 0]
        + spins[x % L, (y + 1) % L, 1]
        + spins[(x + 1) % L, (y + 1) % L, 2]
    )

@njit
def direction_A(spins, x, y, L):
    """
    Compute the polarization (direction) vector of an A vertex.

    The direction is determined by the local spin configuration.

    Parameters
    ----------
    spins : ndarray
        Spin configuration of shape (L, L, 3).
    x, y : int
        Lattice coordinates of the A vertex.
    L : int
        System size (periodic boundary conditions assumed).

    Returns
    -------
    v : ndarray of shape (2,)
        Two-dimensional direction vector of the vertex.
    """
    v = np.zeros(2, dtype=np.float64)

    s0 = spins[x % L, y % L, 0]
    s1 = spins[x % L, y % L, 1]
    s2 = spins[x % L, y % L, 2]

    if s0 == 1 and s1 == -1 and s2 == -1:
        v[0], v[1] = 0.0, 1.0

    elif s0 == 1 and s1 == 1 and s2 == -1:
        v[0], v[1] = -np.sqrt(3) / 2, 0.5

    elif s0 == -1 and s1 == 1 and s2 == -1:
        v[0], v[1] = -np.sqrt(3) / 2, -0.5

    elif s0 == -1 and s1 == 1 and s2 == 1:
        v[0], v[1] = 0.0, -1.0

    elif s0 == -1 and s1 == -1 and s2 == 1:
        v[0], v[1] = np.sqrt(3) / 2, -0.5

    elif s0 == 1 and s1 == -1 and s2 == 1:
        v[0], v[1] = np.sqrt(3) / 2, 0.5

    return v

@njit
def affected_vertices(x, y, d, L):
    """
    Return the two vertices affected by flipping a given bond.

    Parameters
    ----------
    x, y : int
        Coordinates of the bond.
    d : int
        Bond direction index (0, 1, or 2).
    L : int
        System size.

    Returns
    -------
    v1, v2 : tuple of int
        Coordinates (x, y) of the two affected vertices.
    """
    if d == 0:
        return (x, y), (x, y)
    elif d == 1:
        return (x, y), (x, (y - 1) % L)
    else:  # d == 2
        return (x, y), ((x - 1) % L, (y - 1) % L)

@njit
def total_energy(spins, L, J=1.0, A=1.0, B=1.0):
    """
    Compute the total energy of the system.

    The Hamiltonian contains:
    - A charge term proportional to q^2
    - A nearest-neighbor vertex interaction term

    Parameters
    ----------
    spins : ndarray
        Spin configuration of shape (L, L, 3).
    L : int
        System size.
    J : float
        Magnetic charge coupling constant.
    A : float
        Strength of the isotropic vertex interaction.
    B : float
        Strength of the anisotropic interaction.

    Returns
    -------
    E : float
        Total energy of the configuration.
    """
    E = 0.0

    for x in range(L):
        for y in range(L):

            # Charge contribution
            qA = charge_A(spins, x, y, L)
            qB = charge_B(spins, x, y, L)
            E += 0.5 * J * (qA**2 + qB**2)

            # Directional interaction
            dA = direction_A(spins, x, y, L)
            dA1 = direction_A(spins, x + 1, y, L)
            dA2 = direction_A(spins, x - 1, y, L)
            dA3 = direction_A(spins, x, y + 1, L)
            dA4 = direction_A(spins, x, y - 1, L)
            dA5 = direction_A(spins, x - 1, y - 1, L)
            dA6 = direction_A(spins, x + 1, y + 1, L)

            E += -0.5 * A * (
                dA @ (dA1 + dA2 + dA3 + dA4 + dA5 + dA6)
                - 3 * B * (dA @ V1) * (dA1 @ V1)
                - 3 * B * (dA @ V2) * (dA4 @ V2)
                - 3 * B * (dA @ V3) * (dA5 @ V3)
                - 3 * B * (dA @ V4) * (dA2 @ V4)
                - 3 * B * (dA @ V5) * (dA3 @ V5)
                - 3 * B * (dA @ V6) * (dA6 @ V6)
            )

    return E

@njit
def delta_energy_flip(spins, x, y, d, L, J=1.0, A=1.0, B=1.0):
    """
    Compute the energy change caused by flipping a single bond.

    Only local quantities are recomputed, making this suitable
    for Monte Carlo updates.

    Parameters
    ----------
    spins : ndarray
        Spin configuration of shape (L, L, 3).
    x, y : int
        Coordinates of the bond.
    d : int
        Bond direction index.
    L : int
        System size.
    J, A, B : float
        Hamiltonian parameters.

    Returns
    -------
    dE : float
        Energy difference E_after - E_before.
    """
    vert_A, vert_B = affected_vertices(x, y, d, L)

    # Before flip
    qa_before = direction_A(spins, vert_A[0], vert_A[1], L)
    qA_before = charge_A(spins, vert_A[0], vert_A[1], L)
    qB_before = charge_B(spins, vert_B[0], vert_B[1], L)
    q_before = qA_before**2 + qB_before**2

    # Perform flip
    spins[x, y, d] *= -1

    # After flip
    qa_after = direction_A(spins, vert_A[0], vert_A[1], L)
    qA_after = charge_A(spins, vert_A[0], vert_A[1], L)
    qB_after = charge_B(spins, vert_B[0], vert_B[1], L)
    q_after = qA_after**2 + qB_after**2

    # Restore spin
    spins[x, y, d] *= -1

    # Neighbor directions
    dA1 = direction_A(spins, x + 1, y, L)
    dA2 = direction_A(spins, x - 1, y, L)
    dA3 = direction_A(spins, x, y + 1, L)
    dA4 = direction_A(spins, x, y - 1, L)
    dA5 = direction_A(spins, x - 1, y - 1, L)
    dA6 = direction_A(spins, x + 1, y + 1, L)

    # Energy difference
    term1 = (qa_after - qa_before) @ (dA1 + dA2 + dA3 + dA4 + dA5 + dA6)
    term2 = -3 * B * ((qa_after - qa_before) @ V1) * (dA1 @ V1)
    term3 = -3 * B * ((qa_after - qa_before) @ V2) * (dA4 @ V2)
    term4 = -3 * B * ((qa_after - qa_before) @ V3) * (dA5 @ V3)
    term5 = -3 * B * ((qa_after - qa_before) @ V4) * (dA2 @ V4)
    term6 = -3 * B * ((qa_after - qa_before) @ V5) * (dA3 @ V5)
    term7 = -3 * B * ((qa_after - qa_before) @ V6) * (dA6 @ V6)

    dE = (
        0.5 * J * (q_after - q_before)
        - A * (term1 + term2 + term3 + term4 + term5 + term6 + term7)
    )

    return dE

@njit
def metropolis_step(spins, L, beta, J=1.0, A=1.0, B=1.0):
    """
    Perform a single Metropolis Monte Carlo update.

    A random bond is selected and flipped according to the
    Metropolis acceptance rule.

    Parameters
    ----------
    spins : ndarray
        Spin configuration of shape (L, L, 3).
    L : int
        System size.
    beta : float
        Inverse temperature (1 / T).
    J, A, B : float
        Hamiltonian parameters.

    Returns
    -------
    accepted : bool
        True if the spin flip is accepted, False otherwise.
    """
    x = np.random.randint(L)
    y = np.random.randint(L)
    d = np.random.randint(3)

    dE = delta_energy_flip(spins, x, y, d, L, J, A, B)

    if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
        spins[x, y, d] *= -1
        return True

    return False

@njit
def metropolis_step_return_energy(spins, E, L, beta, J=1.0, A=1.0, B=1.0):
    """
    Perform a single Metropolis update while tracking the total energy.

    This version avoids recomputing the full energy by incrementally
    updating it using the local energy difference.

    Parameters
    ----------
    spins : ndarray
        Spin configuration of shape (L, L, 3).
    E : float
        Current total energy.
    L : int
        System size.
    beta : float
        Inverse temperature (1 / T).
    J, A, B : float
        Hamiltonian parameters.

    Returns
    -------
    E_new : float
        Updated total energy after the Monte Carlo step.
    """
    x = np.random.randint(L)
    y = np.random.randint(L)
    d = np.random.randint(3)

    dE = delta_energy_flip(spins, x, y, d, L, J, A, B)

    if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
        spins[x, y, d] *= -1
        E += dE

    return E

@njit
def monopole_count(spins, L):
    """
    Count the number of magnetic monopoles in the system.

    A monopole is defined as a vertex with |q| = 3.

    Parameters
    ----------
    spins : ndarray
        Spin configuration of shape (L, L, 3).
    L : int
        System size.

    Returns
    -------
    count : int
        Total number of monopoles (A and B vertices combined).
    """
    count = 0
    for x in range(L):
        for y in range(L):
            if abs(charge_A(spins, x, y, L)) == 3:
                count += 1
            if abs(charge_B(spins, x, y, L)) == 3:
                count += 1
    return count

@njit
def total_magnetization(spins, L):
    """
    Compute the total magnetization of the system.

    The magnetization is calculated in the laboratory (x, y) frame
    using the projection of the three bond spins.

    Parameters
    ----------
    spins : ndarray
        Spin configuration of shape (L, L, 3).
    L : int
        System size.

    Returns
    -------
    Mx, My : float
        Normalized magnetization components.
    """
    Mx = 0.0
    My = 0.0

    for x in range(L):
        for y in range(L):
            My += spins[x, y, 0] - 0.5 * (spins[x, y, 1] + spins[x, y, 2])
            Mx += (np.sqrt(3) / 2) * (spins[x, y, 1] - spins[x, y, 2])

    N = 3 * L * L
    return Mx / N, My / N


@njit
def get_monopole_positions(spins, L, Nmono):
    """
    Extract positions, charges, and types of all monopoles.

    Parameters
    ----------
    spins : ndarray
        Spin configuration of shape (L, L, 3).
    L : int
        System size.
    Nmono : int
        Total number of monopoles.

    Returns
    -------
    positions : ndarray of shape (Nmono, 2)
        Lattice coordinates (x, y) of monopoles.
    charges : ndarray of shape (Nmono,)
        Magnetic charges of monopoles.
    types : ndarray of shape (Nmono,)
        Vertex type: 0 for A-vertex, 1 for B-vertex.
    """
    positions = np.empty((Nmono, 2), dtype=np.int32)
    charges = np.empty(Nmono, dtype=np.int8)
    types = np.empty(Nmono, dtype=np.int8)

    idx = 0
    for x in range(L):
        for y in range(L):

            qA = charge_A(spins, x, y, L)
            if abs(qA) == 3:
                positions[idx] = x, y
                charges[idx] = qA
                types[idx] = 0
                idx += 1

            qB = charge_B(spins, x, y, L)
            if abs(qB) == 3:
                positions[idx] = x, y
                charges[idx] = qB
                types[idx] = 1
                idx += 1

    return positions, charges, types


@njit
def monopole_pair_correlation(spins, L, Nmono, dr=1, r_max=None):
    """
    Compute the monopole pair correlation function C_{+-}(r).

    Only opposite-charge monopole pairs are included.
    Periodic boundary conditions are assumed.

    Parameters
    ----------
    spins : ndarray
        Spin configuration of shape (L, L, 3).
    L : int
        System size.
    Nmono : int
        Number of monopoles.
    dr : float
        Radial bin width.
    r_max : float or None
        Maximum distance considered.

    Returns
    -------
    r_centers : ndarray
        Centers of distance bins.
    C_r : ndarray
        Monopole pair correlation function.
    """
    pos, q, charge_type = get_monopole_positions(spins, L, Nmono)
    N = len(q)

    if r_max is None:
        r_max = 3 * L / 4

    n_bins = int(r_max / dr)
    C_r = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)

    if N > 0:
        for i in range(N):
            for j in range(i + 1, N):

                xi, yi = pos[i]
                xj, yj = pos[j]

                # Periodic wrapping
                xi -= L * int(np.round(xi / L))
                yi -= L * int(np.round(yi / L))
                xj -= L * int(np.round(xj / L))
                yj -= L * int(np.round(yj / L))

                # Real-space coordinates
                cix = np.sqrt(3) * xi - np.sqrt(3) / 2 * yi
                ciy = -3 / 2 * yi - charge_type[i]
                cjx = np.sqrt(3) * xj - np.sqrt(3) / 2 * yj
                cjy = -3 / 2 * yj - charge_type[j]

                r = np.sqrt((cjx - cix) ** 2 + (cjy - ciy) ** 2)
                bin_id = int(r / dr)

                if bin_id < n_bins and q[i] * q[j] < 0:
                    C_r[bin_id] += -q[i] * q[j]
                    counts[bin_id] += 1

        valid = counts > 0
        C_r[valid] /= counts[valid]

    r_centers = (np.arange(n_bins) + 0.5) * dr
    return r_centers, C_r

def write_monopole_correlation(filename, r, C_r, T):
    """
    Write monopole pair correlation function to file.

    Parameters
    ----------
    filename : str
        Output file name.
    r : ndarray
        Distance bin centers.
    C_r : ndarray
        Correlation function values.
    T : float
        Temperature.
    """
    with open(filename, "w") as f:
        f.write(f"T={T}\n")
        f.write("# r  C(r)\n")
        for ri, ci in zip(r, C_r):
            f.write(f"{ri:.6f} {ci:.6e}\n")

def read_monopole_correlation(filename):
    """
    Read a monopole pair correlation function from file.

    Parameters
    ----------
    filename : str
        File produced by write_monopole_correlation.

    Returns
    -------
    T : float
        Temperature.
    r : ndarray
        Distance bin centers.
    C_r : ndarray
        Correlation function.
    """
    r_list, C_list = [], []
    T = None

    with open(filename, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i == 0 and line.startswith("T="):
                T = float(line.split("=")[1])
                continue
            if not line or line.startswith("#"):
                continue
            ri, ci = map(float, line.split())
            r_list.append(ri)
            C_list.append(ci)

    return T, np.array(r_list), np.array(C_list)

def measurements(results, spins, L, T, beta, J, A, B, n_mc, folder,
                 measure_frequent=10, use_tqdm=True,
                 flag_mag=True, flag_correlation=True):
    """
    Perform Monte Carlo measurements and collect observables.

    Parameters
    ----------
    results : list
        List to store measurement dictionaries.
    spins : ndarray
        Spin configuration.
    L : int
        System size.
    T : float
        Temperature.
    beta : float
        Inverse temperature.
    J, A, B : float
        Hamiltonian parameters.
    n_mc : int
        Number of Monte Carlo steps.
    folder : str
        Output folder name.
    measure_frequent : int
        Measurement interval.
    use_tqdm : bool
        Whether to display a progress bar.
    flag_mag : bool
        Measure magnetization if True.
    flag_correlation : bool
        Measure monopole correlations if True.

    Returns
    -------
    results : list
        Updated results list.
    """
    E_list, Mx_list, My_list, rho_list, C_r_list = [], [], [], [], []

    N = 3 * L * L
    E = total_energy(spins, L, J, A, B)

    iterator = range(n_mc)
    if use_tqdm:
        iterator = tqdm(iterator, desc="Measurement", leave=False)

    for step in iterator:
        E = metropolis_step_return_energy(spins, E, L, beta, J, A, B)

        if step % measure_frequent == 0:
            E_list.append(E)

            if flag_mag:
                Mx, My = total_magnetization(spins, L)
                Mx_list.append(Mx)
                My_list.append(My)

            Nmono = monopole_count(spins, L)
            rho_list.append(Nmono / (L * L))

            if flag_correlation and Nmono > 0:
                r_centers, C_r = monopole_pair_correlation(spins, L, Nmono)
                if len(C_r) > 0:
                    C_r_list.append(C_r)

    # Thermodynamic averages
    E_mean = np.mean(E_list) / (L * L)
    Cv = beta**2 / N * (np.mean(np.array(E_list)**2) - np.mean(E_list)**2)

    Mx_mean = np.mean(Mx_list) / N if Mx_list else 0.0
    My_mean = np.mean(My_list) / N if My_list else 0.0
    rho_mean = np.mean(rho_list) if rho_list else 0.0

    chi_x = beta / N * (np.mean(np.array(Mx_list)**2) - np.mean(Mx_list)**2) if Mx_list else 0.0
    chi_y = beta / N * (np.mean(np.array(My_list)**2) - np.mean(My_list)**2) if My_list else 0.0

    # Correlation output
    if flag_correlation:
        save_path = f"output/correlation/{folder}"
        os.makedirs(save_path, exist_ok=True)
        T_int = int(T * 1000)
        filename = os.path.join(save_path, f"correlation_T{T_int}.txt")

        if C_r_list:
            C_r_mean = np.mean(np.array(C_r_list), axis=0)
        else:
            C_r_mean = np.zeros(int(np.sqrt(3) * L))
            r_centers = (np.arange(len(C_r_mean)) + 0.5)

        write_monopole_correlation(filename, r_centers, C_r_mean, T)

    results.append({
        "T": T,
        "E": E_mean,
        "Cv": Cv,
        "Mx": Mx_mean,
        "My": My_mean,
        "rho_m": rho_mean,
        "chi_x": chi_x,
        "chi_y": chi_y
    })

    return results

def write_snapshot(filename, spins):
    """
    Write a spin configuration snapshot to file.

    The snapshot is stored in a plain text format with one line per bond:
        x  y  d  s

    Parameters
    ----------
    filename : str
        Output file name.
    spins : ndarray
        Spin configuration of shape (L, L, 3).
    """
    with open(filename, "w") as f:
        f.write("# x y d s\n")
        L = spins.shape[0]
        for x in range(L):
            for y in range(L):
                for d in range(3):
                    f.write(f"{x} {y} {d} {spins[x, y, d]}\n")

def read_snapshot(filename):
    """
    Read a spin configuration snapshot from file.

    The snapshot is returned as a dictionary indexed by (x, y, d).

    Parameters
    ----------
    filename : str
        Snapshot file produced by write_snapshot.

    Returns
    -------
    spins_dict : dict
        Dictionary with keys (x, y, d) and spin values (+1 or -1).
    """
    spins_dict = {}

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            x, y, d, s = map(int, line.split())
            spins_dict[(x, y, d)] = s

    return spins_dict

def write_results(filename, results):
    """
    Write thermodynamic measurement results to file.

    Parameters
    ----------
    filename : str
        Output file name.
    results : list of dict
        List of measurement dictionaries produced by `measurements`.
    """
    with open(filename, "w") as f:
        f.write("# T E Cv Mx My rho_m chi_x chi_y\n")
        for res in results:
            f.write(
                f"{res['T']} {res['E']} {res['Cv']} "
                f"{res['Mx']} {res['My']} {res['rho_m']} "
                f"{res['chi_x']} {res['chi_y']}\n"
            )

def read_results(filename):
    """
    Read thermodynamic results from file.

    Parameters
    ----------
    filename : str
        File produced by write_results.

    Returns
    -------
    results : list of dict
        List of measurement dictionaries.
    """
    results = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            T, E, Cv, Mx, My, rho, chi_x, chi_y = map(float, line.split())
            results.append({
                "T": T,
                "E": E,
                "Cv": Cv,
                "Mx": Mx,
                "My": My,
                "rho_m": rho,
                "chi_x": chi_x,
                "chi_y": chi_y
            })

    return results

def run_mc_sweep(T_list, folder="example", L=8,
                 n_therm=5000, n_mc=10000,
                 measure_frequent=10,
                 J=1.0, A=1.0, B=1.0,
                 seed=None,
                 flag_measurements=True,
                 flag_mag=True,
                 flag_correlation=True):
    """
    Perform a full Monte Carlo temperature sweep.

    For each temperature:
      1. Thermalize the system
      2. Measure observables
      3. Save a snapshot of the final configuration

    Parameters
    ----------
    T_list : array-like
        List of temperatures.
    folder : str
        Output folder name.
    L : int
        Linear system size.
    n_therm : int
        Number of thermalization steps.
    n_mc : int
        Number of Monte Carlo steps for measurements.
    measure_frequent : int
        Measurement interval.
    J, A, B : float
        Hamiltonian parameters.
    seed : int or None
        Random seed for spin initialization.
    flag_measurements : bool
        Enable or disable measurements.
    flag_mag : bool
        Measure magnetization if True.
    flag_correlation : bool
        Measure monopole correlations if True.

    Returns
    -------
    results : list of dict
        Thermodynamic results for all temperatures.
    """
    results = []
    spins = init_spins(L, seed)

    save_path = f"output/snapshot/{folder}"
    os.makedirs(save_path, exist_ok=True)

    for idx, T in enumerate(T_list):
        beta = 1.0 / T
        print(f"\n=== Running T = {T:.3f}, L = {L} ({idx+1}/{len(T_list)}) ===")

        # Thermalization
        for _ in tqdm(range(n_therm), desc="Thermalization", leave=False):
            metropolis_step(spins, L, beta, J, A, B)

        # Measurements
        if flag_measurements:
            results = measurements(
                results,
                spins,
                L,
                T,
                beta,
                J, A, B,
                n_mc,
                folder=folder,
                measure_frequent=measure_frequent,
                use_tqdm=True,
                flag_mag=flag_mag,
                flag_correlation=flag_correlation
            )

        # Save snapshot
        T_int = int(T * 1000)
        snapshot_filename = os.path.join(
            save_path, f"snapshot_T{T_int}.txt"
        )
        write_snapshot(snapshot_filename, spins)

        print(f" T = {T:.3f} finished")

    return results

if __name__ == "__main__":
    T_low = np.logspace(-0.5, -2, 10)  
    T_mid = np.logspace(0.5, -0.5, 10)  
    T_high = np.logspace(2, 0.5, 20)    
    T_list = np.concatenate([T_high, T_mid,T_low])
  

    

    results =run_mc_sweep(T_list,folder="L40_0_01_100withint",L=40, A=0.4,n_therm=100000, n_mc=200000,flag_correlation=True,measure_frequent=100)

    write_results("output/result/results_vs_T_L40_0_01_100withint.txt", results)
    
    """
    results =run_mc_sweep(T_list,folder="L10_0_01_100withint",L=10, A=0.4,n_therm=100000, n_mc=200000,flag_correlation=True,measure_frequent=100)

    write_results("output/result/results_vs_T_L10_0_01_100withint.txt", results)

    results =run_mc_sweep(T_list,folder="L20_0_01_100withint",L=20, A=0.4,n_therm=100000, n_mc=200000,flag_correlation=True,measure_frequent=100)

    write_results("output/result/results_vs_T_L20_0_01_100withint.txt", results)
    
    results =run_mc_sweep(T_mid,folder="L80_0_01_100withint",L=80, A=0.4,n_therm=100000, n_mc=200000,flag_correlation=True,flag_mag=False,measure_frequent=100)

    write_results("output/result/results_vs_T_L80_0_01_100withint.txt", results)
    
    results =run_mc_sweep(T_list,folder="L10_0_01_100withoutint",L=10, A=0,n_therm=100000, n_mc=200000,flag_correlation=True,measure_frequent=100)

    write_results("output/result/results_vs_T_L10_0_01_100withoutint.txt", results)
    results =run_mc_sweep(T_list,folder="L20_0_01_100withoutint",L=20, A=0,n_therm=100000, n_mc=200000,flag_correlation=True,measure_frequent=100)

    write_results("output/result/results_vs_T_L20_0_01_100withoutint.txt", results)
    results =run_mc_sweep(T_list,folder="L40_0_01_100withoutint",L=40, A=0,n_therm=100000, n_mc=200000,flag_correlation=True,measure_frequent=100)

    write_results("output/result/results_vs_T_L40_0_01_100withoutint.txt", results)
    
    results =run_mc_sweep(T_mid,folder="L80_0_01_100withoutint",L=80, A=0,n_therm=100000, n_mc=200000,flag_correlation=True,flag_mag=False,measure_frequent=100)

    write_results("output/result/results_vs_T_L80_0_01_100withoutint.txt", results)
    
    results =run_mc_sweep(T_list,folder="L10_0_01_100order",L=10, A=0.4,B=0,n_therm=100000, n_mc=200000,flag_correlation=True,measure_frequent=100)

    write_results("output/result/results_vs_T_L10_0_01_100order.txt", results)
    results =run_mc_sweep(T_list,folder="L20_0_01_100order",L=20, A=0.4,B=0,n_therm=100000, n_mc=200000,flag_correlation=True,measure_frequent=100)

    write_results("output/result/results_vs_T_L20_0_01_100order.txt", results)
    results =run_mc_sweep(T_list,folder="L40_0_01_100order",L=40, A=0.4,B=0,n_therm=100000, n_mc=200000,flag_correlation=True,measure_frequent=100)

    write_results("output/result/results_vs_T_L40_0_01_100order.txt", results)

    
    results =run_mc_sweep(T_mid,folder="L80_0_01_100order",L=80, A=0.4,B=0,n_therm=100000, n_mc=200000,flag_correlation=True,flag_mag=False,measure_frequent=100)

    write_results("output/result/results_vs_T_L80_0_01_100order.txt", results)
    """
    

    
