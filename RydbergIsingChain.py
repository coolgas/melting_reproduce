from tenpy.networks.site import SpinHalfSite
from tenpy.models.model import CouplingMPOModel
from tenpy.models.lattice import Chain
from tenpy.tools.fit import fit_with_sum_of_exp
from scipy.special import zeta

class RydbergIsingChain(CouplingMPOModel):
    r"""Customized chain of Rydberg atoms with power-law decaying interactions.

    .. math ::
        H = \frac{\Omega}{2}\sum_{i}\sigma_x^i - \Delta\sum_{i}n_i + \sum_{i<j}\frac{n_i n_j}{|i-j|^6}.
    """
    default_lattice = Chain
    force_default_lattice = True

    def init_sites(self, model_params):
        conserve = model_params.get("conserve", None)
        sort_charge = model_params.get('sort_charge', None)
        site = SpinHalfSite(conserve=conserve, sort_charge=sort_charge)
        return site

    def init_terms(self, model_params):
        Omega = model_params.get("Omega", 0.)
        Delta = model_params.get("Delta", 1.)
        V = model_params.get("V", 1.)

        # below it's about initializing the interaction terms
        f = lambda x: 1/x**6
        n_exp = model_params.get("n_exp", 5)
        fit_range = model_params.get("fit_range", 20)
        lam, pref = fit_with_sum_of_exp(f, n_exp, fit_range)

        for pr, la in zip(pref, lam):
            self.add_exponentially_decaying_coupling(pr/4, la, 'Sigmaz', 'Sigmaz')

        # here we deal with the onsite interactions
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(Omega/2, u, "Sigmax")
            self.add_onsite((zeta(6)-Delta)/2, u, "Sigmaz")

        





