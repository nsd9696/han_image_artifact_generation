import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import poisson
from scipy.io import loadmat

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, rescale
##Load image
img = plt.imread('Lenna.png')

#gray image generation
# img = np.mean(img,axis=2,keepdims=True)
sz = img.shape
cmap="gray" if sz[2] == 1 else None
# plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
# plt.title("Ground Truth")
# plt.show()

#1-1 uniform sampling
ds_y = 2
ds_x = 4
msk = np.zeros(sz)
msk[::ds_y, ::ds_x, :] = 1

dst = img*msk
plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(msk), cmap=cmap, vmin=0, vmax=1)
plt.title("Uniform Sampling Mask")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title("Sampling image")
plt.show()

#1-2 Random sampling
# rnd = np.random.rand(sz[0],sz[1],sz[2])
# prob = 0.5
# msk = (rnd>prob).astype(np.float)

rnd = np.random.rand(sz[0],sz[1],1)
prob = 0.5
msk = (rnd>prob).astype(np.float)
msk = np.tile(msk,(1,1,sz[2]))

dst = img*msk

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(msk), cmap=cmap, vmin=0, vmax=1)
plt.title("Random Sampling Mask")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title("Sampling image")
plt.show()

#Gaussian Sampling
ly = np.linspace(-1,1,sz[0])
lx = np.linspace(-1,1,sz[1])

x,y = np.meshgrid(lx,ly)

x0 = 0
y0 = 0
sgmx = 1
sgmy = 1
a = 1

gaus= a*np.exp(-((x-x0)**2/(2*sgmx**2) + (y-y0)**2/(2*sgmy**2)))
gaus = np.tile(gaus[:,:,np.newaxis], (1,1,sz[2]))
msk = (rnd<gaus).astype(np.float)

dst = img*msk

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(msk), cmap=cmap, vmin=0, vmax=1)
plt.title("Gaussian Sampling Mask")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title("Sampling image")
plt.show()

#Noise
##2-1Random Noise
sgm = 60.0

noise = sgm/255.0* np.random.rand(sz[0],sz[1],sz[2])
dst = img+noise

plt.subplot(131)
plt.imshow(np.squeeze(img),cmap=cmap,vmin=0,vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(noise),cmap = cmap, vmin=0,vmax=1)
plt.title("Random Noise")

plt.subplot(133)
plt.imshow(np.squeeze(dst),cmap = cmap, vmin=0,vmax=1)
plt.title("Sampling Image")
plt.show()

#2-2Poisson Noise
dst = poisson.rvs(255.0 * img)/255.0
noise = dst - img

plt.subplot(131)
plt.imshow(np.squeeze(img),cmap=cmap,vmin=0,vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(noise),cmap = cmap, vmin=0,vmax=1)
plt.title("Poisson Noise")

plt.subplot(133)
plt.imshow(np.squeeze(dst),cmap = cmap, vmin=0,vmax=1)
plt.title("Sampling Image")
plt.show()
#3 Super-resolution
dw = 1/5.0
order=1
dst_dw = rescale(img, scale=(dw,dw,1),order=order)
dst_up = rescale(dst_dw, scale=(1/dw, 1/dw, 1),order=order)

plt.subplot(131)
plt.imshow(np.squeeze(img),cmap=cmap, vmin=0,vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(dst_dw),cmap=cmap, vmin=0,vmax=1)
plt.title("Down Scaling")

plt.subplot(133)
plt.imshow(np.squeeze(dst_up),cmap=cmap, vmin=0,vmax=1)
plt.title("Up Scaling")
plt.show()



