import numpy as np
import simple_dmrg_02_finite_system as fidmrg
from scipy.sparse import kron, identity
from scipy.sparse.linalg import eigsh  
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def cft_entropy(L, c, c0, N):
    """
    CFT entanglement entropy formula.

    S(L) = (c/6) * ln[ (N/pi) * sin(pi L / N) ] + c0
    """
    return (c / 6.0) * np.log((N / np.pi) * np.sin(np.pi * L / N)) + c0
def fit_SL_cft(SL_dict, N):
    """
    Fit entanglement entropy data to CFT formula.

    Parameters
    ----------
    SL_dict : dict
        keys = L, values = S(L)
    N : int
        Total system size

    Returns
    -------
    c : float
        Central charge
    c0 : float
        Non-universal constant
    popt : array
        Raw fit parameters [c, c0]
    pcov : 2x2 array
        Covariance matrix
    """
    L = np.array(sorted(SL_dict.keys()))
    S = np.array([SL_dict[l] for l in L])


    p0 = (1.0, 0.1)  

    popt, pcov = curve_fit(
        lambda L, c, c0: cft_entropy(L, c, c0, N),
        L, S, p0=p0
    )

    c, c0 = popt
    return c, c0, popt, pcov,L,S



if __name__ == "__main__":

    energy1, SL_dict1=fidmrg.finite_system_algorithm(L=40, m_warmup=10, m_sweep_list=[10],g=0.5)
    SL_dict1 = dict(sorted(SL_dict1.items(), key=lambda x: x[0]))
    print("(1) m=10, g=0.5")
    print(f"ground state energy:{energy1}")
    print(f"entanglement entropy: {SL_dict1}")
    energy2, SL_dict2=fidmrg.finite_system_algorithm(L=40, m_warmup=10, m_sweep_list=[10],g=1.0)
    SL_dict2 = dict(sorted(SL_dict2.items(), key=lambda x: x[0]))
    print("(1) m=10, g=1.0")
    print(f"ground state energy:{energy2}")
    print(f"entanglement entropy: {SL_dict2}")
    energy3, SL_dict3=fidmrg.finite_system_algorithm(L=40, m_warmup=10, m_sweep_list=[10],g=1.5)
    SL_dict3 = dict(sorted(SL_dict3.items(), key=lambda x: x[0]))
    print("(1) m=10, g=1.5")
    print(f"ground state energy:{energy3}")
    print(f"entanglement entropy: {SL_dict3}")
    energy4, SL_dict4=fidmrg.finite_system_algorithm(L=40, m_warmup=10, m_sweep_list=[10],g=1.0)
    SL_dict4 = dict(sorted(SL_dict4.items(), key=lambda x: x[0]))
    print("(2) m=10, g=1.0")
    print(f"ground state energy:{energy4}")
    print(f"entanglement entropy: {SL_dict4}")
    energy5, SL_dict5=fidmrg.finite_system_algorithm(L=40, m_warmup=10, m_sweep_list=[20],g=1.0)
    SL_dict5 = dict(sorted(SL_dict5.items(), key=lambda x: x[0]))
    print("(2) m=20, g=1.0")
    print(f"ground state energy:{energy5}")
    print(f"entanglement entropy: {SL_dict5}")
    energy6, SL_dict6=fidmrg.finite_system_algorithm(L=40, m_warmup=10, m_sweep_list=[30],g=1.0)
    SL_dict6 = dict(sorted(SL_dict6.items(), key=lambda x: x[0]))
    print("(2) m=30, g=1.0")
    print(f"ground state energy:{energy6}")
    print(f"entanglement entropy: {SL_dict6}")

    plt.figure()

    L_list = sorted(SL_dict1.keys())
    SL_list = [SL_dict1[L] for L in L_list]
    plt.plot(L_list, SL_list,label="m=10, g=0.5")
    L_list = sorted(SL_dict2.keys())
    SL_list = [SL_dict2[L] for L in L_list]
    plt.plot(L_list, SL_list,label="m=10, g=1")
    L_list = sorted(SL_dict3.keys())
    SL_list = [SL_dict3[L] for L in L_list]
    plt.plot(L_list, SL_list,label="m=10, g=1.5")
    plt.xlabel("L")
    plt.ylabel("S")
    plt.title("Entanglement entropy S vs system size L")
    plt.grid(True)
    plt.legend() 
    plt.savefig("images/figure1", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()

    L_list = sorted(SL_dict4.keys())
    SL_list = [SL_dict4[L] for L in L_list]
    plt.plot(L_list, SL_list,label="m=10, g=1")
    L_list = sorted(SL_dict5.keys())
    SL_list = [SL_dict5[L] for L in L_list]
    plt.plot(L_list, SL_list,label="m=20, g=1")
    L_list = sorted(SL_dict6.keys())
    SL_list = [SL_dict6[L] for L in L_list]
    plt.plot(L_list, SL_list,label="m=30, g=1")
    plt.xlabel("L")
    plt.ylabel("S")
    plt.title("Entanglement entropy S vs system size L")
    plt.grid(True)
    plt.legend() 
    plt.savefig("images/figure2", dpi=300, bbox_inches="tight")
    plt.close()

    c, c0, popt, pcov, L, S = fit_SL_cft(SL_dict5, N=40)
    print(f"c={c}")

    L_fit = np.linspace(min(L), max(L), 200)
    S_fit = cft_entropy(L_fit, c, c0, 40)
    

    plt.figure()
    plt.plot(L, S, 'o', label="DMRG data")           # 原始散点
    plt.plot(L_fit, S_fit, '-', label=f"fit, c={c:.3f}")  # 拟合曲线
    plt.xlabel("L")
    plt.ylabel("S(L)")
    plt.title("Entanglement entropy vs system size")
    plt.legend()
    plt.grid(True)
    plt.savefig("images/fit_figure", dpi=300, bbox_inches="tight")
    plt.close()
    