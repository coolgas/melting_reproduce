import tenpy
import numpy as np

from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from RydbergIsingChain import RydbergIsingChain

tenpy.tools.misc.setup_logging(to_stdout="INFO")

L = 6

dmrg_params = {
    'mixer': True,  # setting this to True helps to escape local minima
    'trunc_params': {
        'chi_max': 200,
        'svd_min': 1.e-10
    },
    'max_E_err': 1.e-10,
}

model_params = {
    "Omega": 0.5,
    "Delta": 0.5,
    "bc_MPS": "infinite",
    "L": L,
}
states = dmrg_params['orthogonal_to'] = []
Es = []
M = RydbergIsingChain(model_params)
psi0 = MPS.from_lat_product_state(M.lat, [["up"]])
# for i in range(3):
#     psi = psi0.copy()
#     eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
#     E, psi = eng.run()
#     states.append(psi)
#     Es.append(E)
# print(Es)
eng = dmrg.TwoSiteDMRGEngine(psi0, M, dmrg_params)
E, psi = eng.run()
print("E = {E:.13f}".format(E=E))


