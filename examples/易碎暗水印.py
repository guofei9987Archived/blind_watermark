import cv2
import numpy as np

# jpg 是有损的，会破坏整除信息，统一使用png
img1 = cv2.imread('img1.jpg')
img2 = cv2.imread('img2.png')
img2 = cv2.resize(img2, img1.shape[-2::-1])

# %%
n = 3
password = 1
img1_new = (img1 >> n << n)
img2_new = img2 >> (8 - n)

# %%
# 加密
np.random.seed(password)
random_factor = np.random.randint(low=0, high=2 ** 3, size=img2_new.shape)
img2_new2 = img2_new ^ random_factor

# %%
img_combine = img1_new + img2_new2
# img_combine = img1_new + img2_new
cv2.imwrite('combine.png', img_combine)

# %% 提取水印

n = 3
password = 1

img = cv2.imread('combine.png')
# %% 解密
img_ex = img << (8 - n) >> (8 - n)

np.random.seed(password)
random_factor = np.random.randint(low=0, high=2 ** n, size=img2_new.shape)
img_ex = img_ex ^ random_factor

# cv2.imwrite('img_extract.png', (img % 8) * 32)
cv2.imwrite('img_extract.png', img_ex << (8 - n))
