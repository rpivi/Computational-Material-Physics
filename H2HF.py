import sys
import numpy as np
import scipy.linalg as la
from scipy.special import erf
import matplotlib.pyplot as plt

eps=1e-8
pi=np.pi
max_iter=400
alpha = np.array([0.121949, 0.444529, 1.962079, 13.00773])

class H2:
    def __init__(self, alpha, RA, RB):
        self.n = 2*len(alpha)
        self.w = np.zeros(self.n)
        self.C = np.zeros(self.n)
        self.E = 0.
        self.__alpha = np.concatenate((alpha, alpha))
        R = np.array([RA, RA, RA, RA, RB, RB, RB, RB])
        normR = np.array([[np.linalg.norm(Rp-Rq)**2 for Rq in R] for Rp in R])
        #auxiliary matrices
        a = np.array([self.__alpha for i in range(self.n)])
        self.__atot = a + a.T
        self.__ar = np.tensordot(self.__alpha, self.__alpha, axes=0)/self.__atot
        self.__K = np.exp(-normR*self.__ar)
        a = np.array([(R.T*self.__alpha).T for i in range(self.n)])
        self.__avR = a + np.transpose(a, (1,0,2))
        for b in self.__avR.T: b /= self.__atot
        # overlap matrix
        self.__S = self.__K*(pi/self.__atot)**1.5
        # kinetic matrix
        self.__T = self.__ar*(3. - 2.*self.__ar*normR)*self.__S
        # nuclear matrix
        self.__h = self.__T + self.__V(RA) + self.__V(RB)
        # two electrons matrix
        Q = np.zeros((self.n, self.n, self.n, self.n))
        for p in range(self.n):
            for q in range(p+1):
                for r in range(p):
                    for s in range(r+1):
                        self.__Qprqs(Q,p,r,q,s)
                r=p
                for s in range(q+1):
                    self.__Qprqs(Q,p,r,q,s)
        self.__G = 2.*Q - np.transpose(Q, (0,1,3,2))
        # nuclear repulsion energy
        self.__E_nuc = 1./np.linalg.norm(RA-RB)
        self.it = 1

    def __call__(self, accuracy, Pi=None, alpha=1., wprev=None, max_iter=900):
        if self.it > max_iter: sys.exit('Maximum number of iteration has been '
                'reached -> Abort.')
        if Pi is None: Pi = np.zeros((self.n, self.n))
        if wprev is None: wprev = np.zeros(self.n)
        G = np.tensordot(Pi, self.__G, axes=([0,1], [1,3]))
        F = self.__h + .5*G
        self.w, v = la.eigh(F, b=self.__S)
        P = alpha*2.*np.tensordot(v.T[0], v.T[0], axes=0) + (1-alpha)*Pi
        for wk, wkp in zip(self.w, wprev):
            if abs(wk-wkp) > accuracy:
                self.it += 1
                return self(accuracy, P, alpha, self.w, max_iter)
        self.C = v.T[0]
        self.E = np.tensordot(P, self.__h+.25*G, axes=([0,1],[0,1])) + self.__E_nuc
        return self.E, self.C, self.it

    def __V(self, R):
        V = np.zeros((self.n, self.n))
        for p in range(self.n):
            for q in range(p+1):
                V[p,q] = self.__Vpq(R, p, q)
                V[q,p] = V[p,q]
        return V

    def __Vpq(self, R, p, q):
        eqR = np.linalg.norm(self.__avR[p,q]-R)
        if eqR > eps: return -self.__S[p,q]/eqR*erf(eqR*self.__atot[p,q]**.5)
        else: return -2.*pi/self.__atot[p,q]*self.__K[p,q]

    def __Qprqs(self, Q, p, r, q, s):
        rr = np.linalg.norm(self.__avR[p,q]-self.__avR[r,s])
        a = np.sqrt((self.__alpha[p] + self.__alpha[q])*(self.__alpha[r] +
            self.__alpha[s])/(self.__alpha[p] + self.__alpha[q] +
                self.__alpha[r] + self.__alpha[s]))
        if rr > eps: qq = self.__S[p,q]*self.__S[r,s]/rr*erf(rr*a)
        else: qq = 2.*self.__S[p,q]*self.__S[r,s]*a/pi**.5
        # exploit full S_4's subgroup that contains Q symmetries
        Q[p,r,q,s] = qq
        Q[q,r,p,s] = qq
        Q[r,p,s,q] = qq
        Q[r,q,s,p] = qq
        Q[p,s,q,r] = qq
        Q[q,s,p,r] = qq
        Q[s,p,r,q] = qq
        Q[s,q,r,p] = qq


print('##################################################')
print('#            molecule\'s bond length')
print('##################################################\n')
RA = np.array([0., 0., 0.]) 
RBx = np.arange(0.3, 8., 0.001)
E = np.zeros(len(RBx))
text = ''
for i, RBxi in enumerate(RBx):
    RB = np.array([RBxi, 0., 0.])
    h2i = H2(alpha, RA, RB)
    E[i] = h2i(1e-6)[0]
    text += str(RBxi) + ' ' + str(h2i.E) + '\n'
# store data on a file
with open('bonding_energy.txt', 'w+') as f:
    f.write(text)
del text
# plot
plt.plot(RBx, E)
plt.xlabel('R ($a_0$)')
plt.ylabel('B.E. (Hartree)')
plt.savefig('bonding_energy.pdf')
# equilibrium bonding length
i = np.argmin(E)
print('Equilibrium bonding energy: ', E[i], ' (Hartree)')
print('Equilibrium bond length: ', RBx[i], ' (a0)')

print('\n##################################################')
print('# ground state eigenvalue and eigenvector of the')
print('# Fock operator in the equilibrium configuration')
print('##################################################\n')
RB = np.array([RBx[i], 0., 0.])
h2 = H2(alpha, RA, RB)
h2(1e-6)
print('eigenvalue: ', h2.w[0])
print('eigenvector: ', h2.C)
print('number of iteration: ', h2.it, '\n')
print('##################################################')
