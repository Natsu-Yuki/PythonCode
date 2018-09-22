###   移动坐标轴（中心位置）  ###

1. 导入相关的包
import numpy as np
import matplotlib.pyplot as plt

2. 获取 figure 和 axis
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111)

plt.show()

3. 隐藏上边和右边
上下左右，四个边属于当前轴对象（axis）；

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

4. 移动另外两个轴
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))

5. 填充数据
theta = np.arange(0, 2*np.pi, 2*np.pi/100)
ax.plot(np.cos(theta), np.sin(theta))
plt.show()

7. 其他设置
plt.style.use('ggplot')
ax.set_xticks([-1.2, 1.2])
ax.set_yticks([-1.2, 1.2])

####   3D数据   ###
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as axes
import numpy as np


fig=plt.figure()
ax=axes(fig)

x=np.arange(-4,4,0.25)
y=np.arange(-4,4,0.25)
x,y=np.meshgrid(x,y)
r=np.sqrt(x**2+y**2)
z=np.sin(r)

ax.plot_surface(x,y,z,rstride=1,cstride=2,cmap=plt.get_cmap('rainbow'))


ax.contour(x,y,z,zdir='z',offset=-2,cmap=plt.get_cmap('rainbow'))
ax.set_zlim(-2,2)


plt.show()