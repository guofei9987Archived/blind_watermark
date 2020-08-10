from blind_watermark import att

# 缩放攻击
att.resize_att('output/打上水印的图.png', 'output/缩放攻击.png', out_shape=(800, 600))
att.resize_att('output/缩放攻击.png', 'output/缩放攻击_还原.png', out_shape=(1920, 1200))
# out_shape 是分辨率，需要颠倒一下
# %%提取水印
from blind_watermark import WaterMark

bwm1 = WaterMark(password_wm=1, password_img=1, wm_shape=(128, 128))
bwm1.extract("output/缩放攻击_还原.png", "output/缩放攻击_提取水印.png")
