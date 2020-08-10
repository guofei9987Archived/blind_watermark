import numpy as np
import copy
import cv2
from pywt import dwt2, idwt2


class WaterMark:
    def __init__(self, password_wm=1, password_img=1, mod=36, mod2=20, wm_shape=None, block_shape=(4, 4)):
        self.block_shape = np.array(block_shape)
        self.password_wm, self.password_img = password_wm, password_img  # 打乱水印和打乱原图分块的随机种子
        self.mod, self.mod2 = mod, mod2  # 用于嵌入算法的除数,mod/mod2 越大鲁棒性越强,但输出图片的失真越大
        self.wm_shape = wm_shape  # 水印的大小

        # init data
        self.img, self.img_YUV = None, None  # self.img 是原图，self.img_YUV 对像素做了偶数化（加白）
        self.ca, self.hvd, = [np.array([])] * 3, [np.array([])] * 3  # 每个通道 dct 的结果
        self.ca_block = [np.array([])] * 3  # 每个 channel 存一个四维 array，代表四维分块后的结果
        self.ca_part = [np.array([])] * 3  # 四维分块后，有时因不整除而少一部分，self.ca_part 是少这一部分的 self.ca

        self.wm_size, self.block_num = 0, 0  # 水印的个数，原图片可插入信息的个数

    def init_block_index(self):
        self.block_num = self.ca_block_shape[0] * self.ca_block_shape[1]
        assert self.wm_size < self.block_num, IndexError(
            '最多可嵌入{}kb信息，水印含{}kb信息，溢出'.format(self.block_num / 1000, self.wm_size / 1000))
        # self.part_shape 是取整后的ca二维大小,用于嵌入时忽略右边和下面对不齐的细条部分。
        self.part_shape = self.ca_block_shape[:2] * self.block_shape
        self.block_index = [(i, j) for i in range(self.ca_block_shape[0]) for j in range(self.ca_block_shape[1])]

    def read_img(self, filename):
        # 读入图片->YUV化->加白边使像素变偶数->四维分块
        self.img = cv2.imread(filename).astype(np.float32)
        self.img_shape = self.img.shape[:2]
        self.img_YUV = cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV)

        # 如果不是偶数，那么补上白边
        self.img_YUV = cv2.copyMakeBorder(self.img_YUV,
                                          0, self.img_YUV.shape[0] % 2, 0, self.img_YUV.shape[1] % 2,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))

        self.ca_shape = [(i + 1) // 2 for i in self.img_shape]

        self.ca_block_shape = (self.ca_shape[0] // self.block_shape[0], self.ca_shape[1] // self.block_shape[1],
                               self.block_shape[0], self.block_shape[1])
        strides = 4 * np.array([self.ca_shape[1] * self.block_shape[0], self.block_shape[1], self.ca_shape[1], 1])

        for channel in range(3):
            self.ca[channel], self.hvd[channel] = dwt2(self.img_YUV[:, :, channel], 'haar')
            # 转为4维度
            self.ca_block[channel] = np.lib.stride_tricks.as_strided(self.ca[channel].astype(np.float32),
                                                                     self.ca_block_shape, strides)

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
        # dct->flatten->加密->逆flatten->svd->打水印->逆svd->逆dct
        wm_1 = self.wm_bit[i % self.wm_size]
        block_dct = cv2.dct(block)

        # 加密（打乱顺序）
        block_dct_shuffled = block_dct.flatten()[index].reshape(self.block_shape)
        U, s, V = np.linalg.svd(block_dct_shuffled)
        s[0] = (s[0] // self.mod + 1 / 4 + 1 / 2 * wm_1) * self.mod
        if self.mod2:
            s[1] = (s[1] // self.mod2 + 1 / 4 + 1 / 2 * wm_1) * self.mod2

        block_dct_flatten = np.dot(U, np.dot(np.diag(s), V)).flatten()
        block_dct_flatten[index] = block_dct_flatten.copy()
        return cv2.idct(block_dct_flatten.reshape(self.block_shape))

    def embed(self, filename):
        self.init_block_index()

        self.random_dct = np.random.RandomState(self.password_img)
        index = np.arange(self.block_shape[0] * self.block_shape[1])

        embed_ca = copy.deepcopy(self.ca)
        embed_YUV = [np.array([])] * 3
        for channel in range(3):
            for i in range(self.block_num):
                block_idx = self.block_index[i]
                self.random_dct.shuffle(index)
                self.ca_block[channel][block_idx] = self.block_add_wm(self.ca_block[channel][block_idx], index, i)

            # 4维分块变2维
            self.ca_part[channel] = np.concatenate(np.concatenate(self.ca_block[channel], 1), 1)
            # 4维分块时右边和下边不能整除的长条保留，其余是主体部分，换成 embed 之后的频域的数据
            embed_ca[channel][:self.part_shape[0], :self.part_shape[1]] = self.ca_part[channel]
            # 逆变换回去
            embed_YUV[channel] = idwt2((embed_ca[channel], self.hvd[channel]), "haar")

        # 合并3通道
        embed_img_YUV = np.stack(embed_YUV, axis=2)
        # 之前如果不是2的整数，增加了白边，这里去除掉
        embed_img_YUV = embed_img_YUV[:self.img_shape[0], :self.img_shape[1]]
        embed_img = cv2.cvtColor(embed_img_YUV, cv2.COLOR_YUV2BGR)
        embed_img = np.clip(embed_img, a_min=0, a_max=255)
        cv2.imwrite(filename, embed_img)
        return embed_img

    def block_get_wm(self, block, index):
        # dct->flatten->加密->逆flatten->svd->解水印
        block_dct_shuffled = cv2.dct(block).flatten()[index].reshape(self.block_shape)

        U, s, V = np.linalg.svd(block_dct_shuffled)
        wm = (s[0] % self.mod > self.mod / 2) * 1
        if self.mod2:
            tmp = (s[1] % self.mod2 > self.mod2 / 2) * 1
            wm = (wm * 3 + tmp * 1) / 4
        return wm

    def extract(self, filename, out_wm_name=None, mode='img'):
        assert self.wm_shape, 'wm_shape（水印形状）未指定'
        self.wm_size = np.prod(self.wm_shape)
        self.read_img(filename)
        self.init_block_index()

        self.random_dct = np.random.RandomState(self.password_img)
        index = np.arange(self.block_shape[0] * self.block_shape[1])
        wm_extract = np.zeros(shape=(3, self.block_num))  # 3个channel，length 个分块提取的水印，全都记录下来
        wm = np.zeros(shape=self.wm_size)  # 最终提取的水印，是 wm_extract 循环嵌入+3个 channel 的平均
        for channel in range(3):
            for i in range(self.block_num):
                self.random_dct.shuffle(index)
                wm_extract[channel, i] = self.block_get_wm(self.ca_block[channel][self.block_index[i]], index)

        for i in range(self.wm_size):
            wm[i] = wm_extract[:, i::self.wm_size].mean()

        # 水印提取完成后，解密
        wm_index = np.arange(self.wm_size)
        np.random.RandomState(self.password_wm).shuffle(wm_index)
        wm[wm_index] = wm.copy()

        if mode == 'img':
            cv2.imwrite(out_wm_name, 255 * wm.reshape(self.wm_shape[0], self.wm_shape[1]))
        return wm
