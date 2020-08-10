# 除了嵌入图片，也可以嵌入比特类数据
from blind_watermark import WaterMark

bwm1 = WaterMark(1, 1)

# 读取原图
bwm1.read_ori_img('pic/原图.jpg')

# 读取水印
bwm1.read_wm([True, False, True, True, True, False], mode='bit')

# 打上盲水印
bwm1.embed('output/打上水印的图.png')

# %% 解水印

# 注意设定水印的长宽wm_shape
bwm1 = WaterMark(1, 1, wm_shape=6)
wm_extract = bwm1.extract('output/打上水印的图.png', mode='bit')
print(wm_extract)
