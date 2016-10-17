import numpy as np
import theano
import theano.tensor as T

# defining
r, u = T.dscalars('r', 'u')
m, H = T.dvectors('m', 'H')
H = 0.5*r*T.dot(m, m) + 0.25*u*T.dot(m, m)*T.dot(m, m) - T.dot(H, m)   # Landau model

# numerical variables
grad_m = T.grad(H, m)
alpha = T.dscalar('alpha')
init_m = T.sqrt(T.abs_(r)/u)

# Helhholtz free energy
energy_density = theano.function(inputs=[r, u, H, m], outputs=H)

# gradient descent
gresults, gupdates = theano.scan(lambda oldm: oldm - alpha*grad_m,
                                 outputs_info={

                                     T.ones_like(init_m)
                                 },
                                 non_sequences=m)