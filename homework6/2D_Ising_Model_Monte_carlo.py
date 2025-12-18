from numba import njit
import numpy as np
import matplotlib.pyplot as plt


def init_lattice(L):
    """
    Initialize a 2D Ising lattice with random spins.

    Each lattice site is assigned a spin value of +1 or -1
    with equal probability.

    Parameters
    ----------
    L : int
        Linear size of the lattice. The total number of spins is L * L.

    Returns
    -------
    spin : ndarray of shape (L, L)
        Initial spin configuration with values +1 or -1.
    """
    return np.random.choice([-1, 1], size=(L, L))

@njit
def metropolis_sweep(spin, L, beta, J, H):
    """
    Perform one Metropolis Monte Carlo sweep.

    A single sweep consists of L*L single-spin update attempts.
    At each attempt, a random lattice site is selected and its
    spin is flipped with the Metropolis acceptance probability.

    Periodic boundary conditions are applied.

    Parameters
    ----------
    spin : ndarray of shape (L, L)
        Current spin configuration (modified in place).
    L : int
        Linear size of the lattice.
    beta : float
        Inverse temperature, beta = 1 / T.
    J : float
        Nearest-neighbor coupling constant.
    H : float
        External magnetic field.

    Returns
    -------
    None
        The spin array is updated in place.
    """
    for _ in range(L * L):
        i = np.random.randint(L)
        j = np.random.randint(L)

        s = spin[i, j]
        nb = (
            spin[(i + 1) % L, j] +
            spin[(i - 1) % L, j] +
            spin[i, (j + 1) % L] +
            spin[i, (j - 1) % L]
        )

        dE = 2 * s * (J * nb + H)

        if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
            spin[i, j] = -s



@njit
def total_energy_and_magnetization(spin, L, J, H):
    """
    Compute the total energy and magnetization of a 2D Ising lattice.

    The Hamiltonian is given by:
        H = -J * sum_<i,j> (s_i s_j) - H * sum_i s_i

    Each nearest-neighbor bond is counted once.
    Periodic boundary conditions are assumed.

    Parameters
    ----------
    spin : ndarray of shape (L, L)
        Spin configuration.
    L : int
        Linear size of the lattice.
    J : float
        Nearest-neighbor coupling constant.
    H : float
        External magnetic field.

    Returns
    -------
    E : float
        Total energy of the configuration.
    M : int
        Total magnetization of the configuration.
    """
    E = 0.0
    M = 0

    for i in range(L):
        for j in range(L):
            s = spin[i, j]
            E += -J * s * (spin[(i + 1) % L, j] + spin[i, (j + 1) % L])
            M += s

    E += -H * M
    return E, M



def adaptive_N(T, N_min, N_max, T0=1.8):
    """
    Determine an adaptive number of Monte Carlo sweeps.

    The number of sweeps increases at lower temperatures to
    compensate for critical slowing down, and is reduced at
    higher temperatures where equilibration is faster.

    Parameters
    ----------
    T : float
        Temperature.
    N_min : int
        Minimum number of sweeps (used at high temperatures).
    N_max : int
        Maximum number of sweeps (used at low temperatures).
    T0 : float, optional
        Reference temperature below which the sweep number
        saturates (default: 1.8).

    Returns
    -------
    N : int
        Adaptive number of Monte Carlo sweeps.
    """
    T_eff = max(T, T0)
    N = N_min + (N_max - N_min) * (T0 / T_eff) ** 2
    return int(min(max(N, N_min), N_max))



def cal_physical_quan(L, J, H, T_list, N_eq, N_mc):
    """
    Compute thermodynamic observables of the 2D Ising model
    using the Metropolis Monte Carlo algorithm.

    For each temperature in T_list, the system is first equilibrated,
    then sampled to estimate ensemble averages.

    Observables computed:
        - Energy per spin
        - Absolute magnetization per spin
        - Specific heat per spin
        - Magnetic susceptibility per spin

    Parameters
    ----------
    L : int
        Linear size of the lattice.
    J : float
        Nearest-neighbor coupling constant.
    H : float
        External magnetic field.
    T_list : array-like
        List or array of temperatures.
    N_eq : int
        Base number of equilibration sweeps.
    N_mc : int
        Base number of measurement sweeps.

    Returns
    -------
    E_avg : list of float
        Average energy per spin for each temperature.
    M_avg : list of float
        Average absolute magnetization per spin for each temperature.
    Cv : list of float
        Specific heat per spin for each temperature.
    Chi : list of float
        Magnetic susceptibility per spin for each temperature.
    """
    E_avg, M_avg, Cv, Chi = [], [], [], []
    N = L * L

    print(f"\n[INFO] Start simulation for L = {L}")

    for idx, T in enumerate(T_list):
        beta = 1.0 / T
        spin = init_lattice(L)

        # Adaptive sweep numbers
        N_eq_T = adaptive_N(T, N_eq, 10 * N_eq)
        N_mc_T = adaptive_N(T, N_mc, 10 * N_mc)

        print(
            f"  [L={L}] T {idx+1:02d}/{len(T_list)} = {T:.4f} "
            f"(N_eq={N_eq_T}, N_mc={N_mc_T})"
        )

        # Equilibration
        for _ in range(N_eq_T):
            metropolis_sweep(spin, L, beta, J, H)

        # Measurement
        E_samples = np.empty(N_mc_T)
        M_samples = np.empty(N_mc_T)

        for n in range(N_mc_T):
            metropolis_sweep(spin, L, beta, J, H)
            E, M = total_energy_and_magnetization(spin, L, J, H)
            E_samples[n] = E
            M_samples[n] = M

        # Ensemble averages
        E_mean = np.mean(E_samples) / N
        M_mean = np.mean(np.abs(M_samples)) / N
        E2_mean = np.mean(E_samples**2)
        M2_mean = np.mean(M_samples**2)

        Cv_T = beta**2 * (E2_mean - np.mean(E_samples)**2) / N
        Chi_T = beta * (M2_mean - np.mean(M_samples)**2) / N

        E_avg.append(E_mean)
        M_avg.append(M_mean)
        Cv.append(Cv_T)
        Chi.append(Chi_T)

    print(f"[INFO] Finished simulation for L = {L}")
    return E_avg, M_avg, Cv, Chi





if __name__ == "__main__":

    print("==========================================")
    print(" 2D Ising Model Monte Carlo Simulation")
    print("==========================================")

    # Parameters
    J = 1.0
    H = 0.0
    Tc = 2.269 * J

    # Temperature grid (dense near Tc)
    T_low = np.linspace(1.5, Tc - 0.2, 15)
    T_mid = np.linspace(Tc - 0.2, Tc + 0.2, 25)
    T_high = np.linspace(Tc + 0.2, 3.0, 10)
    T_list = np.concatenate([T_low, T_mid, T_high])

    N_eq = 50000
    N_mc = 50000
    L_list = [10, 20, 30]

    results = {}

    # Run simulations
    for L in L_list:
        print("\n------------------------------------------")
        print(f"[MAIN] Running simulations for L = {L}")
        print("------------------------------------------")

        E_avg, M_avg, Cv, Chi = cal_physical_quan(
            L=L,
            J=J,
            H=H,
            T_list=T_list,
            N_eq=N_eq,
            N_mc=N_mc
        )

        results[L] = {
            "E": E_avg,
            "M": M_avg,
            "Cv": Cv,
            "Chi": Chi
        }

    print("\n[MAIN] All simulations completed successfully.")
# -----------------------------
    # Observables info for plotting
    # -----------------------------
    observables = {
        "E": {
            "ylabel": "Average Energy per Spin",
            "title": "Energy vs Temperature",
            "filename": "Eng_vs_T"
        },
        "M": {
            "ylabel": "Average Magnetization per Spin",
            "title": "Magnetization vs Temperature",
            "filename": "Mag_vs_T"
        },
        "Cv": {
            "ylabel": "Specific Heat per Spin",
            "title": "Specific Heat vs Temperature",
            "filename": "Cv_vs_T"
        },
        "Chi": {
            "ylabel": "Magnetic Susceptibility per Spin",
            "title": "Susceptibility vs Temperature",
            "filename": "Chi_vs_T"
        }
    }

    # -----------------------------
    # Plot all observables vs T
    # -----------------------------
    for key, info in observables.items():
        plt.figure()
        for L in L_list:
            plt.plot(
                T_list,
                results[L][key],
                marker='o',
                label=f"L={L}"
            )

        plt.xlabel("Temperature T")
        plt.ylabel(info["ylabel"])
        plt.title(f"2D Ising Model: {info['title']}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"images/{info['filename']}.png",
                    dpi=300, bbox_inches="tight")
        plt.close()

    # -----------------------------
    # Finite-size scaling of magnetization
    # -----------------------------
    plt.figure()
    for L in L_list:
        M_avg = np.array(results[L]["M"])
        x = np.abs(T_list - Tc) * L
        y = M_avg * L**(1/8)
        plt.plot(x, y, 'o', label=f"L={L}")

    # Reference slopes (log-log)
    x_ref = np.linspace(1e-2, 20, 200)
    plt.plot(x_ref, x_ref**(1/8), 'k--', label=r"slope $+1/8$")
    plt.plot(x_ref, x_ref**(-7/8), 'k-.', label=r"slope $-7/8$")

    # Plot settings
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$|T - T_c| \, L$")
    plt.ylabel(r"$M \, L^{1/8}$")
    plt.title("Finite-Size Scaling of Magnetization (2D Ising)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(f"images/fit",dpi=300, bbox_inches="tight")
    plt.close()