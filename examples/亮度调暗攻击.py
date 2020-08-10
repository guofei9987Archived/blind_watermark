from blind_watermark import att

#%% 亮度调低攻击
att.bright_att('output/打上水印的图.png', 'output/亮度调低攻击.png',ratio=0.9)


#%% 提取水印
from blind_watermark import WaterMark

bwm1 = WaterMark(password_wm=1, password_img=1, wm_shape=(128, 128))
bwm1.extract('output/亮度调低攻击.png', 'output/亮度调低攻击_提取水印.png')
