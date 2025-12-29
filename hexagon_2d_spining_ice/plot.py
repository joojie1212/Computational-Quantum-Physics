import numpy as np
import matplotlib.pyplot as plt
import simulation as si
from scipy.optimize import curve_fit
import os

def plot_spin_ice(spins, L, title=None, figsize=(8,8), savefile=None):
    """
    Honeycomb spin ice visualization:
    - thin gray bonds
    - arrows on bond centers
    - colored monopole charges on vertices

    Parameters
    ----------
    spins : dict or ndarray
        ndarray of shape (L, L, 3)
    L : int
        Linear system size
    title : str
        Plot title
    figsize : tuple
        Figure size
    savefile : str
        File path to save the figure
    """
    if isinstance(spins, dict):
        def get_spin(x,y,d):
            return spins[(x,y,d)]
    else:
        spins_array = spins
        def get_spin(x,y,d):
            return spins_array[x,y,d]

    coords_A, charges_A = [], []
    coords_B, charges_B = [], []

    for x in range(L):
        for y in range(L):
            cx_A = np.sqrt(3)*x - np.sqrt(3)/2*y
            cy_A = -3/2*y
            coords_A.append([cx_A, cy_A])
            qA = get_spin(x,y,0) + get_spin(x,y,1) + get_spin(x,y,2)
            charges_A.append(qA)

            cx_B = cx_A
            cy_B = cy_A - 1
            coords_B.append([cx_B, cy_B])
            qB = -(get_spin(x,y,0) + get_spin(x,(y+1)%L,1) + get_spin((x+1)%L,(y+1)%L,2))
            charges_B.append(qB)

    coords_A = np.array(coords_A)
    charges_A = np.array(charges_A)
    coords_B = np.array(coords_B)
    charges_B = np.array(charges_B)

    bond_lines = []
    directions = [(0,-1), (np.sqrt(3)/2,1/2), (-np.sqrt(3)/2,1/2)]
    for x in range(L):
        for y in range(L):
            cx = np.sqrt(3)*x - np.sqrt(3)/2*y
            cy = -3/2*y
            for dx, dy in directions:
                bond_lines.append([[cx, cx+dx],[cy, cy+dy]])

    arrow_x, arrow_y = [], []
    arrow_dx, arrow_dy = [], []

    for x in range(L):
        for y in range(L):
            for d in range(3):
                s = get_spin(x,y,d)
                cx = np.sqrt(3)*x - np.sqrt(3)/2*y
                cy = -3/2*y

                if d == 0:
                    dx, dy = 0, -1
                elif d == 1:
                    dx, dy = np.sqrt(3)/2, 1/2
                elif d == 2:
                    dx, dy = -np.sqrt(3)/2, 1/2

                mx = cx + 0.5*dx
                my = cy + 0.5*dy

                arrow_x.append(mx)
                arrow_y.append(my)
                arrow_dx.append(-np.sign(s)*0.3*dx)
                arrow_dy.append(-np.sign(s)*0.3*dy)

    plt.figure(figsize=figsize)

    # lattice bonds
    for line in bond_lines:
        plt.plot(line[0], line[1], color='gray', linewidth=0.6, alpha=0.7, zorder=0)

    # arrows
    plt.quiver(arrow_x, arrow_y, arrow_dx, arrow_dy,
               angles='xy', scale_units='xy', scale=1,
               color='k', width=0.004, zorder=1)

    def draw_charges(coords, charges):
        # |Q| = 1
        plt.scatter(coords[charges==1,0], coords[charges==1,1],
                    s=150, c='#ff9999', edgecolors='k', zorder=3)
        plt.scatter(coords[charges==-1,0], coords[charges==-1,1],
                    s=150, c='#9999ff', edgecolors='k', zorder=3)
        # |Q| = 3
        plt.scatter(coords[charges==3,0], coords[charges==3,1],
                    s=400, c="#cc5720", edgecolors='k', zorder=3)
        plt.scatter(coords[charges==-3,0], coords[charges==-3,1],
                    s=400, c="#252abb", edgecolors='k', zorder=3)

    draw_charges(coords_A, charges_A)
    draw_charges(coords_B, charges_B)

    plt.axis('equal')
    plt.axis('off')
    if title: plt.title(title)
    if savefile: plt.savefig(savefile, dpi=300, bbox_inches='tight')
    plt.show()

def plot_multiple_results(
    results_list,
    x_key,
    y_key,
    labels=None,
    xlabel=None,
    ylabel=None,
    title=None,
    marker='o',
    linestyle='-',
    markersize=4,
    savepath=None
):
    """
    Plot multiple result sets on the same figure.

    Parameters
    ----------
    results_list : list
        A list of `results` objects (each is a list of dict),
        or a single `results` object.
    x_key : str
        Key used for the x-axis (e.g. "T").
    y_key : str
        Key used for the y-axis (e.g. "rho_m", "Cv").
    labels : list of str or None
        Labels for each curve.
    xlabel, ylabel : str or None
        Axis labels (defaults to key names if None).
    title : str or None
        Figure title.
    marker : str
        Marker style.
    linestyle : str
        Line style.
    markersize : float
        Marker size.
    savepath : str or None
        If provided, save the figure to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    # Wrap single results into a list
    if not isinstance(results_list[0], list):
        results_list = [results_list]

    fig, ax = plt.subplots()

    for i, results in enumerate(results_list):
        x = np.array([r[x_key] for r in results])
        y = np.array([r[y_key] for r in results])

        # Sort by x
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]

        label = labels[i] if labels is not None else None
        ax.plot(
            x, y,
            marker=marker,
            linestyle=linestyle,
            markersize=markersize,
            label=label
        )

    ax.set_xscale("log")
    ax.set_xlabel(xlabel if xlabel else x_key, size=14)
    ax.set_ylabel(ylabel if ylabel else y_key, size=14)

    if title:
        ax.set_title(title, size=16)

    ax.grid(alpha=0.3)

    if labels is not None:
        ax.legend()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_from_results(
    results,
    x_key,
    y_key,
    xlabel=None,
    ylabel=None,
    title=None,
    marker='o',
    linestyle='-',
    markersize=4,
    savepath=None,
    show=True
):
    """
    Plot a single observable from Monte Carlo results.

    Parameters
    ----------
    results : list of dict
        Output from `measurements`.
    x_key : str
        Key for x-axis (e.g. "T").
    y_key : str
        Key for y-axis (e.g. "E", "rho_m").
    xlabel, ylabel : str or None
        Axis labels.
    title : str or None
        Figure title.
    marker : str
        Marker style.
    linestyle : str
        Line style.
    markersize : float
        Marker size.
    savepath : str or None
        Save figure if provided.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    None
    """
    x = np.array([r[x_key] for r in results])
    y = np.array([r[y_key] for r in results])

    # Sort by x
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    plt.figure()
    plt.plot(x, y, marker=marker, linestyle=linestyle, markersize=markersize)
    plt.xscale("log")

    plt.xlabel(xlabel if xlabel else x_key, size=24)
    plt.ylabel(ylabel if ylabel else y_key, size=24)

    if title:
        plt.title(title, size=26)

    plt.grid(alpha=0.3)

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()


def exponential_fit(r, sigma, A):
    """
    Exponential decay function.

    Parameters
    ----------
    r : ndarray
        Distance.
    sigma : float
        Decay constant (string tension).
    A : float
        Amplitude.

    Returns
    -------
    ndarray
        A * exp(-sigma * r)
    """
    return A * np.exp(-sigma * r)


def compute_string_tension_from_folder(folder_path, r_min=3.0, r_max=None):
    """
    Compute string tension from monopole correlation functions.

    Each correlation file in the folder is fitted to an exponential
    decay C(r) ~ exp(-sigma r) in a specified distance range.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing correlation files.
    r_min : float
        Minimum distance used for fitting.
    r_max : float or None
        Maximum distance used for fitting (defaults to last data point).

    Returns
    -------
    T_sorted : ndarray
        Temperatures.
    sigma_sorted : ndarray
        Extracted string tensions.
    """
    T_list = []
    sigma_list = []

    # Collect all correlation files
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])

    for file in files:
        filepath = os.path.join(folder_path, file)
        T, r, C_r = si.read_monopole_correlation(filepath)

        # Define fitting range
        r_max_fit = r[-1] if r_max is None else r_max
        mask = (r >= r_min) & (r <= r_max_fit) & (C_r > 0)

        if np.sum(mask) < 3:
            continue  # Not enough points to fit

        r_fit = r[mask]
        C_fit = C_r[mask]

        # Exponential fit
        try:
            popt, _ = curve_fit(
                exponential_fit,
                r_fit,
                C_fit,
                p0=(0.1, 1.0)
            )
            sigma_list.append(popt[0])
            T_list.append(T)
        except Exception as e:
            print(f"Fitting failed for {file}: {e}")

    # Sort by temperature
    T_list = np.array(T_list)
    sigma_list = np.array(sigma_list)
    idx = np.argsort(T_list)

    return T_list[idx], sigma_list[idx]
def bin_average(r, C_r, nbins=30):
    """
    Perform equal-width bin averaging of the correlation function C(r).

    Parameters
    ----------
    r : array-like
        Radial distances
    C_r : array-like
        Correlation function values
    nbins : int
        Number of bins

    Returns
    -------
    r_bin : ndarray
        Bin-center radii
    C_bin : ndarray
        Bin-averaged correlation values
    """

    r = np.asarray(r)
    C_r = np.asarray(C_r)

    # Define equal-width bins over the full r range
    bins = np.linspace(r.min(), r.max(), nbins + 1)

    # Bin centers
    r_bin = 0.5 * (bins[:-1] + bins[1:])
    C_bin = np.zeros(nbins)

    # Compute mean C(r) in each bin
    for i in range(nbins):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        if np.any(mask):
            C_bin[i] = C_r[mask].mean()
        else:
            C_bin[i] = np.nan

    return r_bin, C_bin

def plot_string_tension_vs_T(
    T_list,
    sigma_list,
    xlabel="T",
    ylabel="String tension sigma",
    title="sigma vs T",
    savepath=None
):
    """
    Plot the string tension sigma as a function of temperature T.

    Parameters
    ----------
    T_list : array-like
        Temperature values
    sigma_list : array-like
        Corresponding string tension values
    xlabel, ylabel : str
        Axis labels
    title : str
        Plot title
    savepath : str, optional
        If provided, the figure will be saved
    """

    plt.figure()
    plt.plot(T_list, sigma_list, marker='o', linestyle='-')

    plt.xlabel(xlabel, size=24)
    plt.ylabel(ylabel, size=24)

    if title:
        plt.title(title, size=26)

    plt.grid(alpha=0.3)

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')

    plt.show()


def plot_string_tension_from_folders(
    folder_paths,
    labels=None,
    r_min=3.0,
    r_max=None,
    xlabel="Temperature T",
    ylabel=r"String tension $\sigma$",
    title=None,
    marker='o',
    linestyle='-',
    markersize=4,
    savepath=None,
):
    """
    Compute and plot the string tension sigma(T) from multiple folders.

    For each folder:
        1. Read monopole correlation data C(r)
        2. Perform an exponential fit in a specified r-range
        3. Extract the string tension sigma
        4. Plot sigma as a function of temperature T

    Parameters
    ----------
    folder_paths : str or list of str
        Path(s) to folders containing correlation data files (.txt)
    labels : list of str, optional
        Legend labels for each curve
    r_min, r_max : float
        Radial range used for the exponential fit
        If r_max is None, the maximum available r is used
    xlabel, ylabel : str
        Axis labels
    title : str, optional
        Plot title
    marker, linestyle, markersize :
        Matplotlib plotting style parameters
    savepath : str, optional
        If provided, the figure will be saved to this path
    show : bool
        Whether to display the figure (kept for interface consistency)

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes objects
    """

    # If a single folder path is provided, wrap it into a list
    if isinstance(folder_paths, str):
        folder_paths = [folder_paths]

    fig, ax = plt.subplots()

    # Loop over all folders
    for i, folder in enumerate(folder_paths):

        # Compute string tension sigma(T) from the current folder
        T, sigma = compute_string_tension_from_folder(
            folder, r_min=r_min, r_max=r_max
        )

        # Skip folders with no valid data
        if len(T) == 0:
            print(f"[Warning] No valid data in {folder}")
            continue

        # Set legend label
        label = labels[i] if labels is not None else folder

        # Plot sigma(T)
        ax.plot(
            T,
            sigma,
            marker=marker,
            linestyle=linestyle,
            markersize=markersize,
            label=label,
        )

    # Temperature is typically shown on a logarithmic scale
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    if title:
        ax.set_title(title, fontsize=16)

    ax.grid(alpha=0.3)

    if labels is not None:
        ax.legend()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig, ax

if __name__=='__main__':
    spins=si.read_snapshot("output/snapshot/L10_0_01_100withoutint/snapshot_T13538.txt")
    plot_spin_ice(spins, L=10, title="Spin Ice Charges")


    
    results_L10int = si.read_results("output/result/results_vs_T_L10_0_01_100withint.txt")
    results_L20int = si.read_results("output/result/results_vs_T_L20_0_01_100withint.txt")
    results_L40int = si.read_results("output/result/results_vs_T_L40_0_01_100withint.txt")
    results_L10order = si.read_results("output/result/results_vs_T_L10_0_01_100order.txt")
    results_L20order = si.read_results("output/result/results_vs_T_L20_0_01_100order.txt")
    results_L40order = si.read_results("output/result/results_vs_T_L40_0_01_100order.txt")
    results_L10noint = si.read_results("output/result/results_vs_T_L10_0_01_100withoutint.txt")
    results_L20noint = si.read_results("output/result/results_vs_T_L20_0_01_100withoutint.txt")
    results_L40noint = si.read_results("output/result/results_vs_T_L40_0_01_100withoutint.txt")
 
    


    all_results = [results_L10int, results_L20int, results_L40int,results_L10order, results_L20order, results_L40order,results_L10noint, results_L20noint, results_L40noint]
    labels = ["EM interaction L=10", "EM interaction L=20", "EM interaction L=40", "exchange interaction L=10", "exchange interaction L=20", "exchange interaction L=40", "no interaction L=10", "no interaction L=20", "no interaction L=40"]


    fig, ax = plot_multiple_results(
        all_results,
        x_key="T",
        y_key="rho_m",  
        labels=labels,
        xlabel="Temperature T",
        ylabel="Monopole density",
        title="Monopole density vs Temperature",
        marker='o',
        linestyle='-',
        markersize=4,
        savepath="images/Density_vs_T.png"
    )
    plt.show()
    """
    """
    folders = [
    "output/correlation/L80_0_01_100withint",
    "output/correlation/L80_0_01_100order",
    "output/correlation/L80_0_01_100withoutint",
    ]

    labels = [
    "EM interaction",
    "exchange interaction",
    "no interaction",
    ]

    plot_string_tension_from_folders(
        folders,
        labels=labels,
        r_min=3.0,
        r_max=8.0,
        title="String tension vs Temperature",
        savepath="images/sigma_vs_T.png"
    )
    plt.show()
    
    plt.figure()
    T, r, C_r = si.read_monopole_correlation(
    "output/correlation/L80_0_01_100order/correlation_T926.txt"
    )

    r_b, C_b = bin_average(r, C_r, nbins=25)


    plt.plot(r_b, C_b, marker='o', linestyle='-', label=f"T={T:.2f},exchange interaction")
    T, r, C_r = si.read_monopole_correlation(
    "output/correlation/L80_0_01_100withint/correlation_T926.txt"
    )

    r_b, C_b = bin_average(r, C_r, nbins=25)
    plt.plot(r_b, C_b, marker='o', linestyle='-', label=f"T={T:.2f},EM interaction")
    
    T, r, C_r = si.read_monopole_correlation(
    "output/correlation/L80_0_01_100withoutint/correlation_T926.txt"
    )

    r_b, C_b = bin_average(r, C_r, nbins=25)
    plt.plot(r_b, C_b, marker='o', linestyle='-', label=f"T={T:.2f},no interaction")
    
    
    plt.xlabel("r", size=14)
    plt.ylabel("Cr", size=14)
    plt.title("Cr vs r", size=16)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig("images/C_r_vs_r.png", dpi=300, bbox_inches='tight')
    plt.show()
    