from blind_watermark import att

# 旋转攻击
att.rot_att('output/打上水印的图.png', 'output/旋转攻击.png', angle=45)
att.rot_att('output/旋转攻击.png', 'output/旋转攻击_还原.png', angle=-45)

# %%提取水印
from blind_watermark import WaterMark

bwm1 = WaterMark(4399, 2333, 36, 20, wm_shape=(128, 128))
bwm1.extract("output/旋转攻击_还原.png", "output/旋转攻击_提取水印.png")
