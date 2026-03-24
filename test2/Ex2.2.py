import numpy as np

#为加以区分，代码中tau表示地震过程产生的应力，e表示地震过程中产生的应变，sigma代表长期应变过程中的应力，sigma2表示每年的应力
#theta为度数单位，theta2为弧度单位
theta = np.arange(0, 175, 10)
theta2 = np.deg2rad(theta)

e11 = -0.26
e22 =  0.92
e12 = -0.69
E = np.array([[e11, e12],
              [e12, e22]])

tau11 = 0.3294e4
tau12 = -4.56435e4
tau22 = 8.1351e4
tau = np.array([[tau11, tau12],
                [tau12, tau22]])

sigma11 = 9.19e6
sigma12 = 3.31e5
sigma22 = 1.19e6
sigma = np.array([[sigma11, sigma12],
                [sigma12, sigma22]])

sigma2 = sigma/1000.0


#(c)问
print("\nc问\n")


#算特征值和特征向量
vals, vecs = np.linalg.eigh(E)

print("principal strains =", vals)
print("eigenvectors =\n", vecs)

#算方位角
for i in range(2):
    east = vecs[0, i]
    north = vecs[1, i]
    azimuth = np.degrees(np.arctan2(east, north)) % 360
    print(f"axis {i+1} azimuth = {azimuth:.2f} deg")


#(f)问
print("\nf问\n")

theta = np.arange(0, 175, 10)
theta2 = np.deg2rad(theta)

s = np.sin(theta2)
c = np.cos(theta2)

normal = np.zeros(len(theta))
shear = np.zeros(len(theta))

normal2 = np.zeros(len(theta))
shear2 = np.zeros(len(theta))


print(f"{'Azimuth':>8} {'Normal(Landers)':>20} {'Shear(Landers)':>20}{'Normal(long-term)':>20} {'Shear(long-term)':>20}")

for i in range(len(theta2)):

    #断层法向量
    p = np.array([c[i], -s[i]])
    #断层走向
    f = np.array([s[i],c[i]])
    #地震断层上的力
    t = tau @ p
    normal[i] = np.dot(t, p)
    shear[i] = np.dot(t, f)

    #长期应变层位上的力
    t2 = sigma @ p
    normal2[i] = np.dot(t2, p)
    shear2[i] = np.dot(t2, f)
    print(f"{theta[i]:8d} {normal[i]:20.3e} {shear[i]:20.3e}{normal2[i]:20.3e} {shear2[i]:20.3e}")

imax = np.argmax(np.abs(shear))

print("\nMaximum shear stress(Landers Earthquake):")
print("Azimuth =", theta[imax])
print("Shear =", shear[imax])

imax2 = np.argmax(np.abs(shear2))

print("\nMaximum shear stress(long-term stress):")
print("Azimuth =", theta[imax2])
print("Shear =", shear2[imax])


#g问

print("\n(g)问\n")

mu = 0.2

def g(sigma, theta_deg,mu):

    theta = np.deg2rad(theta_deg)
    p = np.array([np.cos(theta), -np.sin(theta)])
    f = np.array([np.sin(theta), np.cos(theta)])
    t = sigma @ p
    normal = np.dot(t, p)
    shear = np.dot(t, f)
    return abs(shear) + mu * normal


print(f"{'Azimuth':>8} {'ΔCFF(Pa)':>15}")

for i in theta:

    cff = g(sigma2,i,mu)
    print(f"{i:8d} {cff:15.3e}")


#(h)问


print("\n(h)问\n")


print(f"{'Azimuth':>8} {'CFFa':>15} {'CFF1000':>15}{'CFF_1000+L':>19}{'dt':>16} {'shear':>8} {'shear2':>8}")

for k, i in enumerate(theta):

    CFFa = g(sigma/1000.0, i, mu)
    CFF1000 = g(sigma, i, mu)
    CFF_1000L = g(sigma+tau, i, mu)

    dt = (CFF_1000L - CFF1000) / CFFa

    sign1 = "+" if shear[k] > 0 else "-"
    sign2 = "+" if shear2[k] > 0 else "-"

    print(f"{i:8d} {CFFa:15.3e} {CFF1000:15.3e} {CFF_1000L:18.3e} {dt:15.3f} {sign1:>8} {sign2:>8}")