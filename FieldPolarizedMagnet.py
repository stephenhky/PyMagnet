import numpy as np
import theano
import theano.tensor as T

# good reference: http://www.marekrei.com/blog/theano-tutorial/

# defining
r, u = T.dscalars('r', 'u')
m, H = T.dvectors('m', 'H')
helmoltz_energy = 0.5*r*T.dot(m, m) + 0.25*u*T.dot(m, m)*T.dot(m, m) - T.dot(H, m)   # Landau model

# numerical variables
grad_m = T.grad(helmoltz_energy, m)
alpha = T.dscalar('alpha')
init_m = T.sqrt(T.abs_(r)/u)

# Helhholtz free energy
energy_density = theano.function(inputs=[r, u, H, m], outputs=helmoltz_energy)
grad_energy_wrt_m = theano.function(inputs=[r, u, H, m], outputs=grad_m)

