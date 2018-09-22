import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
file=r"C:\Users\Natsu\Desktop\train-public\raw_images/airport00006.jpg"
image = Image.open(file)
r,g,b=image.split()
r=np.array(r)
g=np.array(g)
b=np.array(b)
p=plt.figure()

##############################
ax_1=p.add_subplot(2,2,1)

r_arr=r[:200,700:900]
g_arr=g[:200,700:900]
b_arr=b[:200,700:900]


r = Image.fromarray(r_arr)
g = Image.fromarray(g_arr)
b = Image.fromarray(b_arr)


image = Image.merge("RGB", (r, g, b))
plt.imshow(image)

(m,n)=r_arr.shape
print(m,n)

r_arr.astype(np.int16)
g_arr.astype(np.int16)
b_arr.astype(np.int16)

arr=[]
for i in range(m):
    for j in range(n):

        value=r_arr[i][j]**2+g_arr[i][j]**2+b_arr[i][j]**2

        arr.append([value])

from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=5)
kmeans.fit(arr)
label=kmeans.predict(arr)
for i in range(m):
    print(i/m)
    for j in range(n):
        if label[i*n+j]==0:
            r_arr[i][j] = 255
            g_arr[i][j] = 0
            b_arr[i][j] = 0
        elif label[i*n+j]==1:
            r_arr[i][j] = 0
            g_arr[i][j] = 255
            b_arr[i][j] = 0
        elif  label[i*n+j]==2:
            r_arr[i][j] = 0
            g_arr[i][j] = 0
            b_arr[i][j] = 255
        else :
            r_arr[i][j] = 255
            g_arr[i][j] = 255
            b_arr[i][j] = 0





ax_2=p.add_subplot(2,2,2)
plt.title('Unsupervised classification')
r = Image.fromarray(r_arr)
g = Image.fromarray(g_arr)
b = Image.fromarray(b_arr)
image = Image.merge("RGB", (r, g, b))
plt.imshow(image)


###############################################################################
image = Image.open(file)
r_r,g_r,b_r=image.split()
r_r=np.array(r_r)
g_r=np.array(g_r)
b_r=np.array(b_r)

ax_3=p.add_subplot(2,2,3)



r = Image.fromarray(r_r)
g = Image.fromarray(g_r)
b = Image.fromarray(b_r)
image = Image.merge("RGB", (r, g, b))
plt.imshow(image)

r_r.astype(np.int16)
g_r.astype(np.int16)
b_r.astype(np.int16)
(m,n)=r_r.shape
print(m,n)
arr=[]


for i in range(m):
    for j in range(n):

        value=r_r[i][j]**2+g_r[i][j]**2+b_r[i][j]**2

        arr.append([value])

label=kmeans.predict(arr)
print(len(arr)/847)
print(m,n)
for i in range(m):
    print(i/m)
    for j in range(n):
        if label[i*n+j]==0:
            r_r[i][j] = 255
            g_r[i][j] = 0
            b_r[i][j] = 0
        elif label[i*n+j]==1:
            r_r[i][j] = 0
            g_r[i][j] = 255
            b_r[i][j] = 0
        elif  label[i*n+j]==2:
            r_r[i][j] = 0
            g_r[i][j] = 0
            b_r[i][j] = 255
        else :
            r_r[i][j] = 255
            g_r[i][j] = 255
            b_r[i][j] = 0

ax_4=p.add_subplot(2,2,4)

r = Image.fromarray(r_r)
g = Image.fromarray(g_r)
b = Image.fromarray(b_r)

image1 = Image.merge("RGB", (r, g, b))
plt.imshow(image1)

########################################

plt.show()

''''






from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=3)
kmeans.fit(arr)
label=kmeans.labels_
for i in range(m):
    print(i/m)
    for j in range(n):

        if label[i*100+j]==0:
            r_arr[i][j]=255
            g_arr[i][j] = 0
            b_arr[i][j] = 0

        elif label[i*100+j]==1:
            r_arr[i][j]=0
            g_arr[i][j] = 255
            b_arr[i][j] = 0
        else:
            r_arr[i][j] = 0
            g_arr[i][j] = 0
            b_arr[i][j] = 255


ax_2=p.add_subplot(2,2,2)
r = Image.fromarray(r_arr)
g = Image.fromarray(g_arr)
b = Image.fromarray(b_arr)
image = Image.merge("RGB", (r, g, b))
plt.imshow(image)
plt.show()










kmeans=KMeans(n_clusters=4)
kmeans.fit(arr)
label=kmeans.labels_
for i in range(m):
    print(i/m)
    for j in range(n):

        if label[i*100+j]==0:
            r_arr[i][j]=255
            g_arr[i][j] = 0
            b_arr[i][j] = 0

        elif label[i*100+j]==1:
            r_arr[i][j]=0
            g_arr[i][j] = 255
            b_arr[i][j] = 0
        elif label[i * 100 + j] == 2:
            r_arr[i][j] = 0
            g_arr[i][j] = 255
            b_arr[i][j] = 255
        else:
            r_arr[i][j] = 0
            g_arr[i][j] = 0
            b_arr[i][j] = 255


ax_3=p.add_subplot(2,2,3)
r = Image.fromarray(r_arr)
g = Image.fromarray(g_arr)
b = Image.fromarray(b_arr)
image = Image.merge("RGB", (r, g, b))
plt.imshow(image)


kmeans=KMeans(n_clusters=5)
kmeans.fit(arr)
label=kmeans.labels_
for i in range(m):
    print(i/m)
    for j in range(n):

        if label[i*100+j]==0:
            r_arr[i][j]=255
            g_arr[i][j] = 0
            b_arr[i][j] = 0

        elif label[i*100+j]==1:
            r_arr[i][j]=0
            g_arr[i][j] = 255
            b_arr[i][j] = 0
        elif label[i * 100 + j] == 2:
            r_arr[i][j] = 0
            g_arr[i][j] = 255
            b_arr[i][j] = 255
        elif label[i * 100 + j] == 3:
            r_arr[i][j] = 0
            g_arr[i][j] = 255
            b_arr[i][j] = 255
        elif label[i * 100 + j] ==4:
            r_arr[i][j] = 255
            g_arr[i][j] = 255
            b_arr[i][j] = 255
        else:
            r_arr[i][j] = 0
            g_arr[i][j] = 0
            b_arr[i][j] = 255


ax_4=p.add_subplot(2,2,4)
r = Image.fromarray(r_arr)
g = Image.fromarray(g_arr)
b = Image.fromarray(b_arr)
image = Image.merge("RGB", (r, g, b))
plt.imshow(image)
class Bunch(dict):
    def __init__(self,*args,**kwds):
        super(Bunch,self).__init__(*args,**kwds)
        self.__dict__=self
x=Bunch(index=array_np)



'''











