import jax
import jax.numpy as jnp
import numpy as np

def realFun(x):
    return jnp.real(x)

def imagFun(x):
    return 0.5 * ( x - jnp.conj(x) )

class TDVP:

    def __init__(self, sampler, snrTol=2, makeReal='imag', rhsPrefactor=1.j):

        self.sampler = sampler
        self.snrTol = snrTol
        self.rhsPrefactor = rhsPrefactor

        self.makeReal = realFun
        if makeReal == 'imag':
            self.makeReal = imagFun


    def get_tdvp_equation(self, Eloc, gradients):
        
        Eloc -= jnp.mean(Eloc)
        gradients -= jnp.mean(gradients, axis=0)
        
        def eoFun(carry, xs):
            return carry, xs[0] * xs[1]
        _, EOdata = jax.lax.scan(eoFun, [None], (Eloc, jnp.conjugate(gradients)))
        EOdata = self.makeReal( -self.rhsPrefactor * EOdata )

        F = jnp.mean(EOdata, axis=0)
        S = self.makeReal( jnp.matmul(jnp.conj(jnp.transpose(gradients)), gradients) )

        return S, F, EOdata


    def get_sr_equation(self, Eloc, gradients):

        return get_tdvp_equation(Eloc, gradients, rhsPrefactor=1.)
    
    
    def transform_to_eigenbasis(self, S, F, EOdata):
        
        self.ev, self.V = jnp.linalg.eigh(S)
        self.VtF = jnp.dot(jnp.transpose(self.V),F)

        EOdata = jnp.matmul(EOdata, jnp.conj(self.V))
        self.rhoVar = jnp.var(EOdata, axis=0)

        print(self.rhoVar)

    def solve(self, Eloc, gradients):

        S, F, Fdata = self.get_tdvp_equation(Eloc, gradients)

        self.transform_to_eigenbasis(S,F,Fdata)

        return jnp.real( jnp.dot( self.V, (1./self.ev * self.VtF) ) )


    def __call__(self, netParameters, t, rhsArgs):
        
        # Get sample
        sampleConfigs, sampleLogPsi =  self.sampler.sample(rhsArgs['psi'], rhsArgs['numSamples'])

        # Evaluate local energy
        sampleOffdConfigs, matEls = rhsArgs['hamiltonian'].get_s_primes(sampleConfigs)
        sampleLogPsiOffd = rhsArgs['psi'](sampleOffdConfigs)
        Eloc = rhsArgs['hamiltonian'].get_O_loc(sampleLogPsi,sampleLogPsiOffd)

        # Evaluate gradients
        sampleGradients = rhsArgs['psi'].gradients(sampleConfigs)

        return self.solve(Eloc, sampleGradients)


