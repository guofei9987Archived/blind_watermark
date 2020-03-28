from blind_watermark import att

# 缩放攻击
att.resize_att('output/打上水印的图.png', 'output/缩放攻击.png', out_shape=(1200, 1920))
att.resize_att('output/缩放攻击.png', 'output/缩放攻击_还原.png', out_shape=(1200, 1920))

# %%提取水印
from blind_watermark import WaterMark

bwm1 = WaterMark(4399, 2333, 36, 20, wm_shape=(128, 128))
bwm1.extract("output/缩放攻击_还原.png", "output/缩放攻击_提取水印.png")
