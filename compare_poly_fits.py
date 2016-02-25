import matplotlib.pyplot as plt
import eugene as eu
import numpy as np
import pdb

xdata = [np.linspace(0,10,100)]
ydata = 10. - 2. * xdata[0] + 0.5 * xdata[0]**2 + np.random.rand(100) * 10

#pdb.set_trace()

params, cov = eu.compare.surface_fit(xdata, ydata, 2)
pred1 = eu.compare.npoly_val(params, eu.compare.exponents(1,2),
        xdata)
poly = np.polyfit(xdata[0].flatten(), ydata.flatten(), 2)
pred2 = np.polyval(poly, xdata).flatten()

# pdb.set_trace()

plt.plot(xdata[0], ydata, 'bo', xdata[0], pred1, 'ro', xdata[0], pred2, 'g.')
plt.show()

xdata = [np.linspace(0,10,100)]
ydata = (-7.17959 + 82.6262 * xdata[0] - 46.9803 * xdata[0]**2 - 2.15732 *
    xdata[0]**3 - 51.0863 * xdata[0]**4 + 11.3268 * xdata[0]**5 - 14.0821 *
    xdata[0]**6)

params, cov = eu.compare.surface_fit(xdata, ydata, 7)
pred1 = eu.compare.npoly_val(params, eu.compare.exponents(1,7),
        xdata)
poly = np.polyfit(xdata[0].flatten(), ydata.flatten(), 7)
pred2 = np.polyval(poly, xdata).flatten()

print eu.compare.exponents(1,7)
print params
print poly[::-1]
print "differences between fit parameters: {}".format(params - poly[::-1])

plt.plot(xdata[0], ydata, 'bo', xdata[0], pred1, 'ro', xdata[0], pred2, 'g+')
plt.show()


