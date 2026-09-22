import numpy as np
from numpy import pi
import scipy as sp
from scipy import linalg
from scipy import integrate
import matplotlib.pyplot as plt
#%% Construction of H, S and Q tensors                     
#α=[0.298073, 1.242567, 5.782948, 38.474970]  # α (suggested α values)
α=[0.01 * (10000)**(i/10) for i in range(10)] # α runs from 0.01 to 100 (geometric progression) 
N=len(α)
F=np.zeros((N, N)) 
H=np.zeros((N, N))
S=np.zeros((N, N)) 
Q=np.zeros((N, N, N, N)) #just creates F, H, S, Q tensors with random (zero) values
for p in range(N):
        for q in range(N):
                H[p][q]=3*α[p]*α[q]*pi**1.5/(α[p]+α[q])**2.5-4*pi/(α[p]+α[q])
                S[p][q]=(pi/(α[p]+α[q]))**1.5
                for r in range(N):
                        for s in range(N):
                            Q[p][r][q][s]=2*pi**2.5/((α[p]+α[q])*(α[r]+α[s])*(α[p]+α[q]+α[r]+α[s])**0.5)
#%% Construction of a random C
C=list(range(N))
def norm(vec, mat):
    scalarProduct=0
    for p in range(len(vec)):
            for q in range(len(vec)):
                scalarProduct+=vec[p]*mat[p][q]*vec[q]
    return np.sqrt(scalarProduct) # norm is square root of the scalar product of vec with itself
def normalize(vec, mat):
    return vec/norm(vec, mat)
C=normalize(C, S)
#%% Starting of the infinite loop (there is a break statement inside it)
E, previousE = 0, 1 #some random number for E and previousE
count=0
while True:
    count+=1
#%% Updating F tensor
    for p in range(N):
            for q in range(N):
                F[p][q]=H[p][q]
                for r in range(N):
                    for s in range(N):
                        F[p][q]+=Q[p][r][q][s]*C[r]*C[s]
#%% Solving the generalized eigenvalue problem                            
    alpha, beta, vl, eigenvectors, work, info = sp.linalg.lapack.zggev(F, S) # zggev is the LAPACK function to perform the generalized eigenvalue problem
    eigenvalues=(alpha/beta).real #the eigenvalues are actually real, but i casted them explicitely real to avoid tedious warnings
    L=list(range(N))
    def sortFunction(index):
            return eigenvalues[index]
    L.sort(key=sortFunction) # sort a index array by the corrisponding eigenvalue[index] value. L[0] will be the lowest eigenvalue index
    C=normalize(eigenvectors[:, L[0]].real, S) #the eigenvectors are actually real, but i casted them to real to avoid tedious warnings
#%% Computing the energy mean value (different from the eigenvalue)
    E=0
    for p in range(N):
            for q in range(N):
                E+=2*C[p]*C[q]*H[p][q]
                for r in range(N):
                    for s in range(N):
                        E+=Q[p][r][q][s]*C[p]*C[q]*C[r]*C[s]
    if np.absolute(E-previousE)<10**(-6):
        break # stop the infinite loop when target precision (10**(-n)) is reached
    previousE=E
def toeV(E):
    return 27.211386245988*E
print("Energy of 1s orbital ground state : %.6f (%.6f eV)  ...  expected: %.6f (%.6f eV)  [%d ITERATIONS]" % (E, toeV(E), -2.90338574, toeV(-2.90338574), count))
#%% Plotting the graph
def eigenfunction(x):
    y=0
    for i in range(N):
        y+=C[i]*np.e**(-α[i]*x**2) # C[i] is the array of the coefficients (eigenvector)
    return y
def radialProbability(x):
    return x**2*eigenfunction(x)**2
xvalues = np.linspace(0, 5, 1000)
yvalues = []
I, err=sp.integrate.quad(radialProbability, 0, np.inf)# I is the normalization constant (quad is the function to compute a definite integral)
for x in xvalues:
    y=radialProbability(x)/I
    yvalues.append(y)
plt.plot(xvalues, yvalues)
bottom, top = plt.ylim()
plt.xlim(0, 5)
plt.ylim(0, top)
plt.xlabel('Radius ($a_0$)')
plt.ylabel('$r^2[R(r)]^2$ (ad.)')
plt.show()
