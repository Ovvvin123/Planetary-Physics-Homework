import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("query.csv")
#震级数据
mag = df.iloc[:, 4]
#与震级对应的时间数据
time = pd.to_datetime(df.iloc[:, 0])
year = time.dt.year

#区间
x = np.arange(5.0, 10, 0.5)

def f(mag):

    counts = [0] * (len(x) - 1)

    #每个区间内的地震数
    for value in mag:
        if 5.0 <= value <= 10:
            index = int((value - 5.0) / 0.5)

            if index == len(x) - 1:
                index -= 1

            counts[index] += 1

    counts = np.array(counts)

    #大于该震级的地震数量
    N = []
    for i in range(len(counts)):
        s = sum(counts[i:])
        N.append(s)

    N = np.array(N)

    # 避免log0
    mask = N > 0
    N = N[mask]

    logN = np.log10(N)

    #区间中心
    x_mid = [(x[i] + x[i+1]) / 2 for i in range(len(x)-1)]
    x_mid = np.array(x_mid)

    x_mid = x_mid[mask]

    #线性拟合
    model = LinearRegression().fit(x_mid.reshape(-1,1), logN)

    a = model.intercept_
    b = model.coef_[0]

    return x_mid, logN,counts, a, b

#2010-2020数据
x_mid, logN,counts, a, b = f(mag)

for i in range(len(counts)): print(f"{x[i]} - {x[i+1]} : {counts[i]}")

#画柱状图
labels = []
for i in range(len(x) - 1):
    labels.append(f"{x[i]}-{x[i+1]}")

plt.figure()
plt.bar(labels, counts)
plt.xlabel("Magnitude ")
plt.ylabel("Count")
plt.title("Earthquake Frequency (M ≥ 5.0)")
plt.show()


print(f"2010-2020年数据拟合结果：logN = {a:.3f} + ({b:.3f})M")


plt.figure()
plt.scatter(x_mid, logN)
x_line = np.linspace(min(x_mid), max(x_mid), 100)
y_line = a + b * x_line
plt.plot(x_line, y_line)
plt.xlabel("Magnitude")
plt.ylabel("log N")
plt.title("2010-2020 Magnitude vs log N (Magnitude >5.0) ")
plt.show()


#2010-2014数据


mask1 = (year >= 2010) & (year <= 2014)

mag1 = mag[mask1]

x_mid1, logN1, counts1, a1, b1 = f(mag1)

print(f"2010-2014年数据拟合结果:logN = {a1:.3f} + ({b1:.3f})M")


#2015-2020数据

mask2 = (year >= 2015) & (year <= 2020)

mag2 = mag[mask2]

x_mid2, logN2, counts2, a2, b2 = f(mag2)

print(f"2015-2020年数据拟合结果:logN = {a2:.3f} + ({b2:.3f})M")


# 对比图
plt.figure()

plt.scatter(x_mid1, logN1, label="2010-2014 Data")
x_line1 = np.linspace(min(x_mid1), max(x_mid1), 100)
y_line1 = a1 + b1 * x_line1
plt.plot(x_line1, y_line1, label="2010-2014 Fit")


plt.scatter(x_mid2, logN2, label="2015-2020 Data")
x_line2 = np.linspace(min(x_mid2), max(x_mid2), 100)
y_line2 = a2 + b2 * x_line2
plt.plot(x_line2, y_line2, label="2015-2020 Fit")


plt.xlabel("Magnitude")
plt.ylabel("log N")
plt.title("2010-2014 and 2015-2020 Magnitude vs log N (Magnitude >5.0)")

plt.legend()
plt.show()

