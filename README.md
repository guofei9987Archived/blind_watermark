# blind-watermark
基于傅里叶变换的数字盲水印  

## 如何使用

嵌入水印
```python
from blind_watermark import WaterMark

bwm1 = WaterMark(4399, 2333, 36, 20)

# 读取原图
bwm1.read_ori_img('pic/原图.jpg')

# 读取水印
bwm1.read_wm('pic/水印.png')

# 打上盲水印
bwm1.embed('output/打上水印的图.png')
```


提取水印
```python
from blind_watermark import WaterMark

bwm1 = WaterMark(4399, 2333, 36, 20, wm_shape=(128, 128))
# 注意需要设定水印的长宽wm_shape
bwm1.extract('output/打上水印的图.png', 'output/解出的水印.png')
```

## 效果展示

|原图|水印|
|--|--|
|![原图](/examples/pic/原图.jpg)|![水印](/examples/pic/水印.png)|

|嵌入后的图|提取的水印|
|--|--|
|![](/examples/output/打上水印的图.png)|![提取的水印](https://github.com/guofei9987/pictures_for_blog/blob/master/blind_watermark/%E8%A7%A3%E5%87%BA%E7%9A%84%E6%B0%B4%E5%8D%B0.png?raw=true)|


### 各种攻击后的效果

|攻击方式|攻击后的图片|提取的水印|
|--|--|--|
|多遮挡|![多遮挡](https://github.com/guofei9987/pictures_for_blog/blob/master/blind_watermark/%E5%A4%9A%E9%81%AE%E6%8C%A1%E6%94%BB%E5%87%BB.png?raw=true)|![多遮挡_提取水印](https://github.com/guofei9987/pictures_for_blog/blob/master/blind_watermark/%E5%A4%9A%E9%81%AE%E6%8C%A1%E6%94%BB%E5%87%BB_%E6%8F%90%E5%8F%96%E6%B0%B4%E5%8D%B0.png?raw=true)|
|横向裁剪|![](https://github.com/guofei9987/pictures_for_blog/blob/master/blind_watermark/%E6%A8%AA%E5%90%91%E8%A3%81%E5%89%AA%E6%94%BB%E5%87%BB_%E5%A1%AB%E8%A1%A5.png?raw=true)|![](https://github.com/guofei9987/pictures_for_blog/blob/master/blind_watermark/%E6%A8%AA%E5%90%91%E8%A3%81%E5%89%AA%E6%94%BB%E5%87%BB_%E6%8F%90%E5%8F%96%E6%B0%B4%E5%8D%B0.png?raw=true)|
|纵向裁剪|![纵向裁剪](https://github.com/guofei9987/pictures_for_blog/blob/master/blind_watermark/%E7%BA%B5%E5%90%91%E8%A3%81%E5%89%AA%E6%89%93%E5%87%BB_%E5%A1%AB%E8%A1%A5.png?raw=true)|![纵向裁剪](https://github.com/guofei9987/pictures_for_blog/blob/master/blind_watermark/%E7%BA%B5%E5%90%91%E8%A3%81%E5%89%AA%E6%89%93%E5%87%BB_%E6%8F%90%E5%8F%96%E6%B0%B4%E5%8D%B0.png?raw=true)|
|椒盐攻击|![](https://github.com/guofei9987/pictures_for_blog/blob/master/blind_watermark/%E6%A4%92%E7%9B%90%E6%94%BB%E5%87%BB.png?raw=true)|![](https://github.com/guofei9987/pictures_for_blog/blob/master/blind_watermark/%E6%A4%92%E7%9B%90%E6%94%BB%E5%87%BB_%E6%8F%90%E5%8F%96%E6%B0%B4%E5%8D%B0.png?raw=true)|
|亮度提高10%|![](https://github.com/guofei9987/pictures_for_blog/blob/master/blind_watermark/%E4%BA%AE%E5%BA%A6%E8%B0%83%E9%AB%98%E6%94%BB%E5%87%BB.png?raw=true)|![](https://github.com/guofei9987/pictures_for_blog/blob/master/blind_watermark/%E4%BA%AE%E5%BA%A6%E8%B0%83%E9%AB%98%E6%94%BB%E5%87%BB_%E6%8F%90%E5%8F%96%E6%B0%B4%E5%8D%B0.png?raw=true)|
|亮度调低10%|![](https://github.com/guofei9987/pictures_for_blog/blob/master/blind_watermark/%E4%BA%AE%E5%BA%A6%E8%B0%83%E4%BD%8E%E6%94%BB%E5%87%BB.png?raw=true)|![](https://github.com/guofei9987/pictures_for_blog/blob/master/blind_watermark/%E4%BA%AE%E5%BA%A6%E8%B0%83%E4%BD%8E%E6%94%BB%E5%87%BB_%E6%8F%90%E5%8F%96%E6%B0%B4%E5%8D%B0.png?raw=true)|


### 还没调整好的
缩放攻击，旋转攻击
