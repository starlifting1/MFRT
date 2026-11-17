"""Physical rescaling of saved trace-v2 realizations (native kpc, km/s, 1e10 Msun).

The transformations keep N, the dimensionless DF, and dimensionless softening
fixed. They do not create an independent realization or change its enhancement.
"""
from dataclasses import dataclass
import math

G_NATIVE = 43007.1
KPC_PER_KMS_MYR = 3.0856775814913673e16 / (365.25 * 86400 * 1e6)
OLD_R0 = 75.0
OLD_MASS = 137.0
OLD_V0 = math.sqrt(G_NATIVE * OLD_MASS / OLD_R0)
FIXED_TOTAL_MASS_MSUN = 1.e7
FIXED_RM_PC = 2000.
FIT_MIN_OVER_RM = 10. / FIXED_RM_PC
FIT_MAX_OVER_RM = 1000. / FIXED_RM_PC
VELOCITY_FIT_MIN_OVER_V0 = (9*math.pi/(2*10000))**(1/3)/2
VELOCITY_FIT_MAX_OVER_V0 = .25


@dataclass(frozen=True)
class PhysicalScale:
    n: int
    rm_pc: float
    total_mass_msun: float = FIXED_TOTAL_MASS_MSUN

    def __post_init__(self):
        if self.n < 2 or self.rm_pc <= 0 or self.total_mass_msun <= 0:
            raise ValueError('Positive physical mass, radius and N >= 2 required')

    @classmethod
    def for_n(cls, n):
        return cls(n, FIXED_RM_PC)

    @property
    def particle_mass_msun(self): return self.total_mass_msun / self.n

    @property
    def r0_kpc(self): return 1.5 * self.rm_pc / 1000.

    @property
    def mass_native(self): return self.total_mass_msun / 1e10

    @property
    def v0(self): return math.sqrt(G_NATIVE * self.mass_native / self.r0_kpc)

    @property
    def crossing_myr(self): return self.r0_kpc / self.v0 * KPC_PER_KMS_MYR

    def factors(self, metadata):
        a = self.r0_kpc / metadata['R0']
        b = self.v0 / metadata['v0']
        mu = self.mass_native / metadata['mass']
        return dict(position=a, velocity=b, mass=mu,
                    diffusion_native=mu**2/(a**3*b),
                    diffusion_per_myr=mu**2/(a**3*b)/KPC_PER_KMS_MYR,
                    time_native=a**3*b**3/mu**2)

    def row(self):
        return dict(N=self.n, N_sample=self.n, particle_mass_msun=self.particle_mass_msun,
                    total_mass_msun=self.total_mass_msun,
                    Rm_pc=self.rm_pc, R0_pc=self.r0_kpc*1000, v0_kms=self.v0,
                    t_cross_Myr=self.crossing_myr,
                    epsilon_pc=self.r0_kpc*1000/(.2*self.n),
                    ln_Lambda=math.log(.2*self.n),
                    fit_min_pc=FIT_MIN_OVER_RM*self.rm_pc,
                    fit_max_pc=FIT_MAX_OVER_RM*self.rm_pc)

    def revision_options(self, space):
        """Physical flags for new runs, including an N-independent fit window."""
        if space not in ('pos','vel'):raise ValueError('space must be pos or vel')
        scale=self.rm_pc/1000 if space=='pos' else self.v0
        lo=FIT_MIN_OVER_RM if space=='pos' else VELOCITY_FIT_MIN_OVER_V0
        hi=FIT_MAX_OVER_RM if space=='pos' else VELOCITY_FIT_MAX_OVER_V0
        return {'n':self.n,'R0':self.r0_kpc,'mass':self.mass_native,
                'count-min':scale*lo,'count-max':scale*hi,'count-radii':12,
                'eps-factor':1,'veps-factor':0}


def scaled_tail_parameters(parameters, velocity_factor):
    p = dict(parameters)
    for key in ('sigma', 'vc', 'v1'):
        p[key] *= velocity_factor
    p['k'] /= velocity_factor
    return p
