#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import rc

# Parameters
L = 2
Npoints = 1000
Estep = 0.05
accuracy = 0.0001
Nlevels = 5

# phi function
def phi(psi, k, E, dx):
	return psi[k] * (1 + dx*dx*E/6)

# Integrate wavefunciton
def integrate(psi, dx):
	res = 0.0
	for i in range(1, len(psi)):
		res += (psi[i-1] + psi[i]) * dx / 2

# Generate wavefunction
def numerov(E, dx):
	# Apply the formula
	psi = [0, 1]
	for k in range(2, Npoints):
		psi.append(0)
		psi[k] = (2*phi(psi, k-1, E, dx) - phi(psi, k-2, E, dx) - 2*dx*dx*E*psi[k-1]) / (1 + dx*dx*E/6)

	# Normalize the resulting wavefunction
	norm = np.sqrt(integrate([y*y for y in psi], dx))
	return np.array(psi) / norm

# Bisection algorithm, find the zero of f(x) between x10 and x20
def bisect(f, x10, x20):
	x1 = x10
	f1 = f(x10)
	x2 = x20
	midf = accuracy + 1
	while (abs(midf) > accuracy):
		midf = f((x1+x2)/2)
		if f1*midf > 0:
			x1 = (x1+x2) / 2
		else:
			x2 = (x1+x2) / 2
	return (x1+x2) / 2

# Setup the two plots: wavefunction and energy
fig, (ax1, ax2) = plt.subplots(1, 2)
plt.subplots_adjust(left=0.1, bottom=0.3)
ax1.set(xlabel='x', ylabel=r"$|\psi|^2$", title='Eigenfunctions of a particle in box')
ax1.grid()
ax2.set(ylabel='Energy', title='Eigenvalues of a particle in box')
ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax2.grid(axis='y')

# Setup the widgets to control some parameters (slow and inefficient but cool)
axNp = plt.axes([0.15, 0.05, 0.7, 0.04])
axL = plt.axes([0.15, 0.1, 0.7, 0.04])
sliderNl = Slider(axNp, '# of levels', 1, 10, valinit=Nlevels, valstep=1)
sliderL = Slider(axL, 'L', 1, 5, valinit=L, valstep=0.1)
def update(_):
	global Nlevels
	global L
	Nlevels = sliderNl.val
	L = sliderL.val
	ax1.clear()
	ax2.clear()
	compute()
	fig.canvas.draw()
	ax1.set(xlabel='x', ylabel=r"$|\psi|^2$", title='Eigenfunctions of a particle in box')
	ax1.grid()
	ax2.set(ylabel='Energy', title='Eigenvalues of a particle in box')
	ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
	ax2.grid(axis='y')
sliderNl.on_changed(update)
sliderL.on_changed(update)

# Compute the method for each level and plot the results
def compute():
	print('Computing {} levels for L = {}...'.format(Nlevels, L))
	xs = np.linspace(-L, L, Npoints)
	dx = 2*L / Npoints
	E0 = Estep
	for i in range(0, Nlevels):
		# Find the bracketing energy values: eigenvalue E is in [E1, E2]
		E2 = E0
		prevBoundary = numerov(E2, dx)[-1]
		boundary = prevBoundary
		while (prevBoundary*boundary > 0):
			prevBoundary = boundary
			E2 += Estep
			boundary = numerov(E2, dx)[-1]
		E1 = E2 - Estep

		# Bisect between E1 and E2
		E = bisect(lambda x: numerov(x, dx)[-1], E1, E2)
		print('E{} = {}'.format(i, E))
		E0 = E + Estep

		# Apply algorithm for the next eigenfunction and calculate modulus squared
		psi = numerov(E, dx)
		psi2 = np.power(psi, 2)

		# Plot the results, shifting psi^2 for readability
		ax1.plot(xs, psi2 + i)
		ax2.plot([E]*2)

compute()
plt.show()
