import numpy as np

from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from RydbergIsingChain import RydbergIsingChain

dmrg_params = {
    'mixer': True,  # setting this to True helps to escape local minima
    'trunc_params': {
        'chi_max': 200,
        'svd_min': 1.e-10
    },
    'max_E_err': 1.e-10,
}

def compute_Es(dmrg_params, Omegas, Delta, L):
    E_densities = []
    for Omega in Omegas:
        model_params = {
            "Omega": Omega,
            "Delta": Delta,
            "bc_MPS": "infinite",
            "L": L,
        }
        M = RydbergIsingChain(model_params)
        psi = MPS.from_lat_product_state(M.lat, [["up"]])
        eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
        E, psi = eng.run()
        E_densities.append(E)
        print("E = {E:.13f}".format(E=E))
    return E_densities