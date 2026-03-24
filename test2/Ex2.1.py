import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

#画图中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

theta = np.arange(0 , 95, 5) #度数
theta2 = np.deg2rad(theta)   #弧度

ts = np.arange(19)
tn = np.arange(19)

s = np.sin(theta2)
c = np.cos(theta2)

#法1，斜截面应力法
ts = ( -40 * s - 10 * c ) *c - ( - 10 * s - 60 * c ) *s
tn = ( -40 * s - 10 * c ) *s + ( - 10 * s - 60 * c ) *c
print("斜截面应力法")
print(f"ts:{ts}\ntn:{tn}")

plt.figure()

plt.scatter(theta, ts, label="Shear stress ts")
plt.plot(theta, ts)

plt.scatter(theta, tn, label="Normal stress tn")
plt.plot(theta, tn)

plt.xlabel("Angle (degree)")
plt.ylabel("Stress")
plt.title("应力 vs 断层倾角(斜截面应力法)")

plt.legend()
plt.show()

#法2. 坐标变换法
sigma = np.array([[-40, -10],
                  [-10, -60]])
theta = np.arange(0, 91, 5)

tn2 = []
ts2 = []

for th in theta2:

    # 法向和切向
    n = np.array([np.sin(th),  np.cos(th)])
    f = np.array([np.cos(th), -np.sin(th)])

    # 旋转矩阵
    Q = np.array([n, f])

    # 坐标变换
    sigma2 = Q @ sigma @ Q.T

    tn2.append(sigma2[0,0])
    ts2.append(sigma2[0,1])

tn2 = np.array(tn2)
ts2 = np.array(ts2)

print("坐标变换法")
print("tn =", tn2)
print("ts =", ts2)

plt.figure()

plt.scatter(theta, ts2, label="Shear stress ts")
plt.plot(theta, ts2)

plt.scatter(theta, tn2, label="Normal stress tn")
plt.plot(theta, tn2)

plt.xlabel("Angle (degree)")
plt.ylabel("Stress")
plt.title("应力 vs 断层倾角(坐标旋转法)")

plt.legend()
plt.show()

#求主应力及对应方向

# 特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(sigma)

print("主应力为：")
print(eigenvalues)

print("\n对应的主方向单位向量为：")
f1 = np.array([eigenvectors[0,0],eigenvectors[1,0]])
f2 = np.array([eigenvectors[0,1],eigenvectors[1,1]])
print(f1,f2)

#求最大剪应力及方向

tau_max =np.fabs( (eigenvalues[0] - eigenvalues[1])/2 )
print("\n最大剪应力 =", tau_max)

f3 = (f1+f2)/np.sqrt(2)

print("最大剪应力的单位向量为：")
print(f3)