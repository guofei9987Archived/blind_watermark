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
|![原图](https://img1.github.io/blind_watermark/原图.jpg)|![水印](https://img1.github.io/blind_watermark/水印.png)|

|嵌入后的图|提取的水印|
|--|--|
|![](https://img1.github.io/blind_watermark/打上水印的图.png)|![提取的水印](https://img1.github.io/blind_watermark/%E8%A7%A3%E5%87%BA%E7%9A%84%E6%B0%B4%E5%8D%B0.png)|


### 各种攻击后的效果

|攻击方式|攻击后的图片|提取的水印|
|--|--|--|
|多遮挡<br>[多遮挡攻击.py](examples/多遮挡攻击.py)|![多遮挡](https://img1.github.io/blind_watermark/%E5%A4%9A%E9%81%AE%E6%8C%A1%E6%94%BB%E5%87%BB.png?raw=true)|![多遮挡_提取水印](https://img1.github.io/blind_watermark/%E5%A4%9A%E9%81%AE%E6%8C%A1%E6%94%BB%E5%87%BB_%E6%8F%90%E5%8F%96%E6%B0%B4%E5%8D%B0.png?raw=true)|
|横向裁剪10%<br>[横向裁剪攻击.py](examples/横向裁剪攻击.py)|![横向裁剪](https://img1.github.io/blind_watermark/%E6%A8%AA%E5%90%91%E8%A3%81%E5%89%AA%E6%94%BB%E5%87%BB_%E5%A1%AB%E8%A1%A5.png?raw=true)|![](https://img1.github.io/blind_watermark/%E6%A8%AA%E5%90%91%E8%A3%81%E5%89%AA%E6%94%BB%E5%87%BB_%E6%8F%90%E5%8F%96%E6%B0%B4%E5%8D%B0.png?raw=true)|
|纵向裁剪10%<br>[纵向裁剪攻击.py](examples/纵向裁剪攻击.py)|![纵向裁剪](https://img1.github.io/blind_watermark/%E7%BA%B5%E5%90%91%E8%A3%81%E5%89%AA%E6%89%93%E5%87%BB_%E5%A1%AB%E8%A1%A5.png?raw=true)|![纵向裁剪](https://img1.github.io/blind_watermark/%E7%BA%B5%E5%90%91%E8%A3%81%E5%89%AA%E6%89%93%E5%87%BB_%E6%8F%90%E5%8F%96%E6%B0%B4%E5%8D%B0.png?raw=true)|
|缩放攻击（1200X1920->600X800）<br>[缩放攻击.py](examples/缩放攻击.py)|![](https://img1.github.io/blind_watermark/%E7%BC%A9%E6%94%BE%E6%94%BB%E5%87%BB.png?raw=true)|![](https://img1.github.io/blind_watermark/%E7%BC%A9%E6%94%BE%E6%94%BB%E5%87%BB_%E6%8F%90%E5%8F%96%E6%B0%B4%E5%8D%B0.png?raw=true)|
|椒盐攻击<br>[椒盐击.py](examples/椒盐.py)|![](https://img1.github.io/blind_watermark/%E6%A4%92%E7%9B%90%E6%94%BB%E5%87%BB.png?raw=true)|![](https://img1.github.io/blind_watermark/%E6%A4%92%E7%9B%90%E6%94%BB%E5%87%BB_%E6%8F%90%E5%8F%96%E6%B0%B4%E5%8D%B0.png?raw=true)|
|亮度提高10%<br>[亮度调高攻击.py](examples/亮度调高攻击.py)|![](https://img1.github.io/blind_watermark/%E4%BA%AE%E5%BA%A6%E8%B0%83%E9%AB%98%E6%94%BB%E5%87%BB.png?raw=true)|![](https://img1.github.io/blind_watermark/%E4%BA%AE%E5%BA%A6%E8%B0%83%E9%AB%98%E6%94%BB%E5%87%BB_%E6%8F%90%E5%8F%96%E6%B0%B4%E5%8D%B0.png?raw=true)|
|亮度调低10%<br>[亮度调暗攻击.py](examples/亮度调暗攻击.py)|![](https://img1.github.io/blind_watermark/%E4%BA%AE%E5%BA%A6%E8%B0%83%E4%BD%8E%E6%94%BB%E5%87%BB.png?raw=true)|![](https://img1.github.io/blind_watermark/%E4%BA%AE%E5%BA%A6%E8%B0%83%E4%BD%8E%E6%94%BB%E5%87%BB_%E6%8F%90%E5%8F%96%E6%B0%B4%E5%8D%B0.png?raw=true)|


<!-- ### 未能抵抗的攻击


- 90度旋转打击：缩放还原后无法复原，但旋转回来后可以（旋转回来后就是原图）。
- 加白色边框攻击：攻击时增加5个纯白像素，还原时用resize，效果很差。
- 裁剪图像中的一部分，然后把这部分放大。无法还原。正确的做法是使用循环填充
- 旋转攻击：还没做 -->
