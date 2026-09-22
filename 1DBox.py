import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

V0 = 10
L = 1
a = 0.5

def phi(k,x):
    if k % 2 == 1:
        return np.cos(k*np.pi*x/(2*L))
    if k % 2 == 0:
        return np.sin(k*np.pi*x/(2*L))

def H_kpk(kp,k):
    f = lambda x: phi(kp,x)*phi(k,x)
    return V0*quad(f,-a,a)[0]
    
def get_Hmat():
    H = np.empty([Nmax,Nmax])
    for k in range(Nmax):
        H[k,k]=((k+1)*np.pi)**2/8+H_kpk(k+1,k+1)
        for kp in range(k):
            H[kp,k]=H_kpk(kp+1,k+1)
            H[k,kp]=H[kp,k]
    return H

def PSI(C,x):
    out = 0
    for k, c in enumerate(C,start=1):
        out += c*phi(k,x)
    return out
        
def plot_PSI(C):
    x = np.linspace(-L,L)
    y = PSI(C[0],x)
    plt.plot(x,y,label = '$N='+str(Nmax)+'$')
    

for Nmax in [1,3,30]:
    E, C = np.linalg.eigh(get_Hmat())
    plot_PSI(C.T)
    print(E)
    
plt.legend()
plt.title('$V_0='+str(V0)+',$ $a='+str(a)+',$ $L='+str(L)+'$')
plt.xlim([-L,L])
plt.xlabel('$x$')
plt.ylabel('$\\Psi(x)$')
plt.show()
