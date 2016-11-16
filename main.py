import db_queries as db
import numpy as np
import gaussianmixtureEM as gEM
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

q = db.Queries()

data = q.avg_distance_and_interval()

distance = [int(i[0]) for i in data]
days = [int(i[1]) for i in data]

xs = np.column_stack((distance, days))

np.random.seed(123)

'''
# create data set
n = 1000
_mus = np.array([[0,4], [-2,0]])
_sigmas = np.array([[[3, 0], [0, 0.5]], [[1,0],[0,2]]])
_pis = np.array([0.6, 0.4])
xs = np.concatenate([np.random.multivariate_normal(mu, sigma, int(pi*n))
                    for pi, mu, sigma in zip(_pis, _mus, _sigmas)])
'''

# initial guesses for parameters
pis = np.random.random(2)
pis /= pis.sum()
mus = np.column_stack(([1000, 280],[50, 4]))
sigmas = np.array([np.eye(2)] * 2)
print('pis: ', pis)
print('mus: ', mus)
print('sigmas: ', sigmas)

ll_new, pis, mus, sigmas = gEM.em_gmm_orig(xs,pis,mus,sigmas)
print('pis: ', pis)
print('mus: ', mus)
print('sigmas: ', sigmas)

'''
plt.style.use('custom538')

fig, ax = plt.subplots()
fig.set_size_inches(8,4)

x = np.linspace(0,8500, 8500)

ax.plot(x, mlab.normpdf(x, mus[0],sigmas[0]))

plt.tight_layout()
plt.savefig("figure2.png")
plt.show()

'''

def gaussian_2d(x, y, x0, y0, xsig, ysig):
    return np.exp(-0.5*(((x-x0) / xsig)**2 + ((y-y0) / ysig)**2))

delta = 0.025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = gaussian_2d(X, Y, 0., 0., 1., 1.)
Z2 = gaussian_2d(X, Y, 1., 1., 1.5, 0.5)
# difference of Gaussians
Z = 10.0 * (Z2 - Z1)

# Create a contour plot with labels using default colors.  The
# inline argument to clabel will control whether the labels are draw
# over the line segments of the contour, removing the lines beneath
# the label
plt.clf()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Simplest default with labels')
plt.show()
