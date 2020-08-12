from blind_watermark import att
import numpy as np
import cv2

# %%
# 攻击
att.shelter_att('output/打上水印的图.png', 'output/多遮挡攻击.png', ratio=0.1, n=10)

# %%多遮挡攻击.png
from blind_watermark import WaterMark

bwm1 = WaterMark(password_wm=1, password_img=1)
bwm1.extract(filename='output/多遮挡攻击.png', wm_shape=(128, 128), out_wm_name='output/多遮挡攻击_提取水印.png')

