import numpy as np
import cv2
from pywt import dwt2, idwt2
from scipy.stats import pearsonr


class WaterMark:
    def __init__(self, password_wm=1, password_img=1, mod=36, mod2=20, wm_shape=None, block_shape=(4, 4)):
        self.block_shape = block_shape  # 2或4
        self.password_wm, self.password_img = password_wm, password_img  # 打乱水印和打乱原图分块的随机种子
        self.mod, self.mod2 = mod, mod2  # 用于嵌入算法的除数,mod/mod2 越大鲁棒性越强,但输出图片的失真越大
        self.wm_shape = wm_shape  # 水印的大小

    def init_block_index(self):
        # 四维分块后的前2个维度：
        block_shape0, block_shape1 = self.ha_block_shape[0], self.ha_block_shape[1]
        self.length = block_shape0 * block_shape1
        print('最多可嵌入{}kb信息，水印含{}kb信息'.format(self.length / 1000, self.wm_size / 1000))
        if self.wm_size > self.length:
            print("水印的大小超过图片的容量")
        # self.part_shape 是取整后的ha二维大小,用于嵌入时忽略右边和下面对不齐的细条部分。
        self.part_shape = (block_shape0 * self.block_shape[0], block_shape1 * self.block_shape[1])
        self.block_index = [(i, j) for i in range(block_shape0) for j in range(block_shape1)]

    def read_img(self, filename):
        self.img = cv2.imread(filename).astype(np.float32)

        self.img_shape = self.img.shape[:2]
        self.img_YUV = cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV)

        # 如果不是偶数，那么补上白边
        self.img_YUV = cv2.copyMakeBorder(self.img_YUV,
                                          0, self.img_YUV.shape[0] % 2, 0, self.img_YUV.shape[1] % 2,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))

        self.ha_Y, self.coeffs_Y = dwt2(self.img_YUV[:, :, 0], 'haar')
        self.ha_U, self.coeffs_U = dwt2(self.img_YUV[:, :, 1], 'haar')
        self.ha_V, self.coeffs_V = dwt2(self.img_YUV[:, :, 2], 'haar')
        self.ha_Y = self.ha_Y.astype(np.float32)
        self.ha_U = self.ha_U.astype(np.float32)
        self.ha_V = self.ha_V.astype(np.float32)

        self.ha_shape = self.ha_Y.shape
        # 转换成4维分块
        self.ha_block_shape = (self.ha_shape[0] // self.block_shape[0], self.ha_shape[1] // self.block_shape[1],
                               self.block_shape[0], self.block_shape[1])
        strides = 4 * np.array([self.ha_shape[1] * self.block_shape[0], self.block_shape[1], self.ha_shape[1], 1])
        self.ha_Y_block = np.lib.stride_tricks.as_strided(self.ha_Y.copy(), self.ha_block_shape, strides)
        self.ha_U_block = np.lib.stride_tricks.as_strided(self.ha_U.copy(), self.ha_block_shape, strides)
        self.ha_V_block = np.lib.stride_tricks.as_strided(self.ha_V.copy(), self.ha_block_shape, strides)

    def read_ori_img(self, filename):
        self.read_img(filename)

    def read_img_wm(self, filename):
        # 读入图片格式的水印，并转为一维 bit 格式
        self.wm = cv2.imread(filename)[:, :, 0]
        self.wm_shape = self.wm.shape[:2]

        # 加密信息只用bit类，抛弃灰度级别
        self.wm_bit = self.wm.flatten() > 128

    def read_wm(self, wm_content, mode='img'):
        if mode == 'img':
            self.read_img_wm(filename=wm_content)
        else:
            self.wm_bit = np.array(wm_content)
        self.wm_size = self.wm_bit.size
        # 水印加密:
        self.random_wm = np.random.RandomState(self.password_wm)
        self.random_wm.shuffle(self.wm_bit)

    def block_add_wm(self, block, index, i):

        i = i % self.wm_size

        wm_1 = self.wm_bit[i]
        block_dct = cv2.dct(block)

        # 加密（打乱顺序）
        block_dct_shuffled = block_dct.flatten()[index].reshape(self.block_shape)
        U, s, V = np.linalg.svd(block_dct_shuffled)
        s[0] = (s[0] // self.mod + 1 / 4 + 1 / 2 * wm_1) * self.mod
        if self.mod2:
            s[1] = (s[1] // self.mod2 + 1 / 4 + 1 / 2 * wm_1) * self.mod2

        block_dct_shuffled = np.dot(U, np.dot(np.diag(s), V))

        block_dct_flatten = block_dct_shuffled.flatten()

        block_dct_flatten[index] = block_dct_flatten.copy()
        block_dct = block_dct_flatten.reshape(self.block_shape)

        return cv2.idct(block_dct)

    def embed(self, filename):
        self.init_block_index()

        embed_ha_Y_block = self.ha_Y_block.copy()
        embed_ha_U_block = self.ha_U_block.copy()
        embed_ha_V_block = self.ha_V_block.copy()

        self.random_dct = np.random.RandomState(self.password_img)
        index = np.arange(self.block_shape[0] * self.block_shape[1])

        for i in range(self.length):
            self.random_dct.shuffle(index)
            embed_ha_Y_block[self.block_index[i]] = self.block_add_wm(
                embed_ha_Y_block[self.block_index[i]], index, i)
            embed_ha_U_block[self.block_index[i]] = self.block_add_wm(
                embed_ha_U_block[self.block_index[i]], index, i)
            embed_ha_V_block[self.block_index[i]] = self.block_add_wm(
                embed_ha_V_block[self.block_index[i]], index, i)

        embed_ha_Y_part = np.concatenate(np.concatenate(embed_ha_Y_block, 1), 1)
        embed_ha_U_part = np.concatenate(np.concatenate(embed_ha_U_block, 1), 1)
        embed_ha_V_part = np.concatenate(np.concatenate(embed_ha_V_block, 1), 1)

        embed_ha_Y = self.ha_Y.copy()
        embed_ha_Y[:self.part_shape[0], :self.part_shape[1]] = embed_ha_Y_part
        embed_ha_U = self.ha_U.copy()
        embed_ha_U[:self.part_shape[0], :self.part_shape[1]] = embed_ha_U_part
        embed_ha_V = self.ha_V.copy()
        embed_ha_V[:self.part_shape[0], :self.part_shape[1]] = embed_ha_V_part

        # 逆变换回去
        embed_ha_Y = idwt2((embed_ha_Y.copy(), self.coeffs_Y), "haar")
        embed_ha_U = idwt2((embed_ha_U.copy(), self.coeffs_U), "haar")
        embed_ha_V = idwt2((embed_ha_V.copy(), self.coeffs_V), "haar")

        # 合并3通道
        embed_img_YUV = np.stack([embed_ha_Y, embed_ha_U, embed_ha_V], axis=2)

        # 之前如果不是2的整数，增加了白边，这里去除掉
        embed_img_YUV = embed_img_YUV[:self.img_shape[0], :self.img_shape[1]]
        embed_img = cv2.cvtColor(embed_img_YUV, cv2.COLOR_YUV2BGR)

        embed_img[embed_img > 255] = 255
        embed_img[embed_img < 0] = 0

        cv2.imwrite(filename, embed_img)

        print('隐水印嵌入成功，保存到文件 ', filename)
        for i in range(3):
            diff, _ = pearsonr(self.img[:, :, i].flatten(), embed_img[:, :, i].flatten())
            print('通道{}的相似度是{}'.format(i, diff))
        print('(相似度越接近1越好)')

    def block_get_wm(self, block, index):
        block_dct = cv2.dct(block)
        block_dct_flatten = block_dct.flatten().copy()
        block_dct_flatten = block_dct_flatten[index]
        block_dct_shuffled = block_dct_flatten.reshape(self.block_shape)

        U, s, V = np.linalg.svd(block_dct_shuffled)
        wm_1 = (s[0] % self.mod > self.mod / 2) * 1
        if self.mod2:
            wm_2 = (s[1] % self.mod2 > self.mod2 / 2) * 1
            wm = (wm_1 * 3 + wm_2 * 1) / 4
        else:
            wm = wm_1
        return wm

    def extract(self, filename, out_wm_name=None, mode='img'):
        if not self.wm_shape:
            print("水印的形状未设定")
            return 0
        self.wm_size = np.prod(self.wm_shape)

        self.read_img(filename)
        self.init_block_index()

        extract_wm = np.array([])
        self.random_dct = np.random.RandomState(self.password_img)

        index = np.arange(self.block_shape[0] * self.block_shape[1])
        for i in range(self.length):
            self.random_dct.shuffle(index)
            wm_Y = self.block_get_wm(self.ha_Y_block[self.block_index[i]], index)
            wm_U = self.block_get_wm(self.ha_U_block[self.block_index[i]], index)
            wm_V = self.block_get_wm(self.ha_V_block[self.block_index[i]], index)
            wm = round((wm_Y + wm_U + wm_V) / 3)

            if i < self.wm_size:
                extract_wm = np.append(extract_wm, wm)
            else:
                times, ii = i // self.wm_size, i % self.wm_size
                extract_wm[ii] = (extract_wm[ii] * times + wm) / (times + 1)

        # 水印提取完成后，解密
        wm_index = np.arange(extract_wm.size)
        self.random_wm = np.random.RandomState(self.password_wm)
        self.random_wm.shuffle(wm_index)
        extract_wm[wm_index] = extract_wm.copy()

        if mode == 'img':
            cv2.imwrite(out_wm_name, 255 * extract_wm.reshape(self.wm_shape[0], self.wm_shape[1]))
        else:
            return extract_wm
