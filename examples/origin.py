from blind_watermark import WaterMark

bwm1 = WaterMark(4399, 2333, mod=36, mod2=20, block_shape=(4, 4))

# 读取原图
bwm1.read_ori_img('pic/原图.jpg')

# 读取水印
bwm1.read_wm('pic/水印.png')

# 打上盲水印
bwm1.embed('output/打上水印的图.png')


# %% 解水印


bwm1 = WaterMark(4399, 2333, 36, 20, wm_shape=(128, 128), block_shape=(4, 4))
# 注意需要设定水印的长宽wm_shape
bwm1.extract('output/打上水印的图.png', 'output/解出的水印.png')

