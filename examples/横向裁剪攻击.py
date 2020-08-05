from blind_watermark import att
import numpy as np

# 一次横向裁剪打击
att.cut_att_width('output/打上水印的图.png', 'output/横向裁剪攻击.png', ratio=0.5)
att.anti_cut_att('output/横向裁剪攻击.png', 'output/横向裁剪攻击_填补.png', origin_shape=(1200, 1920))

# %%提取水印
from blind_watermark import WaterMark

bwm1 = WaterMark(4399, 2333, 36, 20, wm_shape=(128, 128))
bwm1.extract("output/横向裁剪攻击_填补.png", "output/横向裁剪攻击_提取水印.png")
