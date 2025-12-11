import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def ReadBit(state, pos):
    return (state >> pos) & 1

def FlipBit(state, pos):
    return state ^ (1 << pos)

def CountOnes(state, pos):
    """
    special function to deal with (-) in fermion
    demoning python 3.8+
    """
    mask = (1 << (pos)) - 1
    return (state & mask).bit_count()

def BuildHamiltonian(t,U,V,mu):
    """
    """
    ns=12#0-5 denote spin-up; 6-11 denote spin down  
    nl=2**ns #dimension of hilbert space

    list1 = [[0,2],[1,3],[2,4],[4,5],[1,2],[3,4]]   
    list1.extend([[x+6, y+6] for (x,y) in list1])
    list2=[[0,1],[2,3],[1,4],[3,5]]
    HI, HJ, HV = [], [], []

    for i0 in range(nl):
        for (Pos0, Pos1) in list1:
            if ReadBit(i0, Pos0) != ReadBit(i0, Pos1):
                sign1=(-1)**CountOnes(i0,Pos0)
                i1 = FlipBit(i0, Pos0)
                sign2=(-1)**CountOnes(i1,Pos0)#actually this two sign can be calculated in one step...
                i1 = FlipBit(i1, Pos1)
                HI.append(i1)
                HJ.append(i0)
                HV.append(sign1*sign2*(-t))

        HI.append(i0)
        HJ.append(i0)
        hv=-mu* CountOnes(i0,ns)
        for Pos0 in range(ns//2):
            hv+=U*ReadBit(i0, Pos0)*ReadBit(i0,Pos0+6)
        for (Pos0,Pos1) in list2:
            hv-=V*(ReadBit(i0, Pos0)+ReadBit(i0,Pos0+6))*(ReadBit(i0, Pos1)+ReadBit(i0,Pos1+6))
            
        HV.append(hv)
    
    Hamr = sparse.coo_matrix((HV, (HI, HJ)), shape=(nl, nl)).tocsc()
    return Hamr

def BuildHamiltonian_subspace(t, U, V, mu, N_up, N_dn):
    ns = 12  

    basis = []
    index_map = {}
    for i0 in range(2**ns):
        Nup = sum(ReadBit(i0, pos) for pos in range(6))
        Ndn = sum(ReadBit(i0, pos) for pos in range(6, 12))
        if (Nup, Ndn) == (N_up, N_dn):
            index_map[i0] = len(basis)
            basis.append(i0)

    dim = len(basis)   


    list1 = [[0,2],[1,3],[2,4],[4,5],[1,2],[3,4]]
    list1.extend([[x+6, y+6] for (x,y) in list1]) 
    list2 = [[0,1],[2,3],[1,4],[3,5]]

    HI, HJ, HV = [], [], []


    for i0 in basis:
        j0 = index_map[i0]  
        #i0 is index in whole space, while j0 is index in subspace 

        for (Pos0, Pos1) in list1:
            if ReadBit(i0, Pos0) != ReadBit(i0, Pos1):
                sign1 = (-1)**CountOnes(i0, Pos0)
                i1 = FlipBit(i0, Pos0)
                sign2 = (-1)**CountOnes(i1, Pos0)
                i1 = FlipBit(i1, Pos1)
                if i1 in index_map:  #always true
                    HI.append(index_map[i1])
                    HJ.append(j0)
                    HV.append(sign1 * sign2 * (-t))

        
        HI.append(j0)
        HJ.append(j0)
        hv = -mu * (N_up + N_dn) 
        for Pos0 in range(ns//2):
            hv += U * ReadBit(i0, Pos0) * ReadBit(i0, Pos0+6)
        for (Pos0, Pos1) in list2:
            hv -= V * (ReadBit(i0, Pos0)+ReadBit(i0,Pos0+6)) * (ReadBit(i0, Pos1)+ReadBit(i0,Pos1+6))
        HV.append(hv)


    Hamr = sparse.coo_matrix((HV, (HI, HJ)), shape=(dim, dim)).tocsc()
    return Hamr, basis

def lowest_k_in_subspace(Ham_sparse, k, return_vecs=False):
    dim = Ham_sparse.shape[0]
    if dim == 0:
        return np.array([]), (np.zeros((0,0)) if return_vecs else None)


    if k >= dim:
        H_dense = Ham_sparse.toarray()
        w, v = np.linalg.eigh(H_dense)
        idx = np.argsort(w)[:k]
        if return_vecs:
            return w[idx], v[:, idx]
        else:
            return w[idx], None


    try:
        w, v = eigsh(Ham_sparse, k=k, which='SA', tol=1e-10, maxiter=5000)

        order = np.argsort(w)
        w = w[order]
        v = v[:, order]
        if return_vecs:
            return w, v
        else:
            return w, None
    except Exception as e:

        H_dense = Ham_sparse.toarray()
        w, v = np.linalg.eigh(H_dense)
        idx = np.argsort(w)[:k]
        if return_vecs:
            return w[idx], v[:, idx]
        else:
            return w[idx], None

def global_lowest_n(t, U, V, mu, n, ns=12, return_eigenvecs=False):
    """
 
    """
    results = [] 

    max_up = ns // 2
    max_dn = ns // 2


    top_k_per_subspace = n

    for N_up in range(max_up + 1):
        for N_dn in range(max_dn + 1):
            Ham, basis = BuildHamiltonian_subspace(t, U, V, mu, N_up, N_dn)
            dim = Ham.shape[0]
            if dim == 0:
                continue
            k = min(top_k_per_subspace, dim)

            w, _ = lowest_k_in_subspace(Ham, k, return_vecs=False)
            for idx_local, energy in enumerate(w):
                results.append((energy, N_up, N_dn, idx_local))  

    results.sort(key=lambda x: x[0])
    top_results = results[:n]
    if (not return_eigenvecs):
        return top_results
    else:

        detailed = []
        for energy, N_up, N_dn, idx_local in top_results:
            Ham, basis = BuildHamiltonian_subspace(t, U, V, mu, N_up, N_dn)

            dim = Ham.shape[0]
            k_req = idx_local + 1
            if k_req >= dim:
                w_all, v_all = np.linalg.eigh(Ham.toarray())
                order = np.argsort(w_all)
                vec = v_all[:, order[idx_local]]
            else:
                w_sub, v_sub = eigsh(Ham, k=k_req, which='SA')
                order = np.argsort(w_sub)
                vec = v_sub[:, order[idx_local]]

            detailed.append((energy, N_up, N_dn, basis, vec))

        return detailed
if __name__ == "__main__":
    t=1.0
    U=8.0
    V=0.4
    mu=4.0
    print(f"----t={t},U={U},V={V},mu={mu}----")
    hamr,basis = BuildHamiltonian_subspace(t,U,V,mu,2,4)
    lowest_6_E,v=lowest_k_in_subspace(hamr,6)

    print(f"(1) lowest 6 eigenvalues in up2down4 subspace: {lowest_6_E}\n")
    data=global_lowest_n(t, U, V, mu, 20, ns=12, return_eigenvecs=False)
    
    ener=[]
    for entry in data:
        energy,nup,ndn,basis=entry
        ener.append(energy)
    print(f"(2) lowest 20 eigenvalues of whole system: {ener}\n")
    data_lowest=global_lowest_n(t, U, V, mu,1, ns=12, return_eigenvecs=True)
    ener,nup,ndn,basis,vec=data_lowest[0]
    density_spin_up=[0,0,0,0,0,0]
    density_spin_down=[0,0,0,0,0,0]

    for i in range(len(vec)):
        for site in range(6):
            density_spin_up[site] += ReadBit(basis[i], site) * vec[i]**2
            density_spin_down[site] += ReadBit(basis[i], site+6) * vec[i]**2
    print(f"(3) density_spin_up {density_spin_up}\n")
    print(f"(3) density_spin_down {density_spin_down}\n")