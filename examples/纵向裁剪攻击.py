from blind_watermark import att
import numpy as np

# 一次纵向裁剪打击
att.cut_att_height('output/打上水印的图.png', 'output/纵向裁剪打击.png')

att.anti_cut_att('output/纵向裁剪打击.png', 'output/纵向裁剪打击_填补.png', origin_shape=(1200, 1920))

# %%纵向裁剪打击.png
from blind_watermark import WaterMark

bwm1 = WaterMark(4399, 2333, 36, 20, wm_shape=(128, 128))
bwm1.extract("output/纵向裁剪打击_填补.png", "output/纵向裁剪打击_提取水印.png")
