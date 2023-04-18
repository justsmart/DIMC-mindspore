import mindspore as ms
from mindspore import nn, Tensor
import mindspore.numpy as mnp
import mindspore.ops as ops
import mindspore.context as context
context.set_context(device_target="GPU")
from mindspore.nn import Dense
from mindspore.ops import MatMul,ReduceSum,ReduceMean,Abs,Log,Zeros,BinaryCrossEntropy,Mul
mul = Mul()
matmul=MatMul()
log=Log()
mean=ReduceMean()
class encoder(nn.Cell):
    def __init__(self, n_dim, dims, n_z):
        super(encoder, self).__init__()
        # print(n_dim,dims[0])
        self.enc_1 = nn.Dense(n_dim, dims[0])
        self.enc_2 = nn.Dense(dims[0], dims[1])
        self.enc_3 = nn.Dense(dims[1], dims[2])
        self.z_layer = nn.Dense(dims[2], n_z)
        self.z_b0 = nn.BatchNorm1d(n_z)
        self.relu = nn.ReLU()
    def construct(self, x):
        enc_h1 = self.relu(self.enc_1(x))
        enc_h2 = self.relu(self.enc_2(enc_h1))
        enc_h3 = self.relu(self.enc_3(enc_h2))
        z = self.z_b0(self.z_layer(enc_h3))
        return z
class decoder(nn.Cell):
    def __init__(self, n_dim, dims, n_z):
        super(decoder, self).__init__()
        self.dec_0 = nn.Dense(n_z, n_z)
        self.dec_1 = nn.Dense(n_z, dims[2])
        self.dec_2 = nn.Dense(dims[2], dims[1])
        self.dec_3 = nn.Dense(dims[1], dims[0])
        self.x_bar_layer = nn.Dense(dims[0], n_dim)
        self.relu = nn.ReLU()
    def construct(self, z):
        r = self.relu(self.dec_0(z))
        dec_h1 = self.relu(self.dec_1(r))
        dec_h2 = self.relu(self.dec_2(dec_h1))
        dec_h3 = self.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)
        return x_bar
class AE(nn.Cell):

    def __init__(self, n_stacks, n_input, n_z, nLabel):
        super(AE, self).__init__()
        dims = []
        for n_dim in n_input:

            linshidims = []
            for idim in range(n_stacks - 2):
                linshidim = round(n_dim * 0.8)
                linshidim = int(linshidim)
                linshidims.append(linshidim)
            linshidims.append(1500)
            dims.append(linshidims)
        self.encoder_list = nn.CellList([encoder(n_input[i], dims[i], n_z) for i in range(len(n_input))])
        self.decoder_list = nn.CellList([decoder(n_input[i], dims[i], n_z) for i in range(len(n_input))])
        self.relu = nn.ReLU()
        self.regression = nn.Dense(n_z, nLabel)
        self.act = nn.Sigmoid()
        self.ExpandDims=ms.ops.ExpandDims()

    def construct(self, mul_X, we):
        # print('ss')

        we = we.float()
        summ=0
        individual_zs = []
        for enc_i, enc in enumerate(self.encoder_list):
            z_i = enc(mul_X[enc_i])
            individual_zs.append(z_i)
            summ += matmul(mnp.diagflat(we[:, enc_i]),z_i)
        wei = 1 / we.sum(1)
        z = matmul(mnp.diagflat(wei),summ)


        x_bar_list = []
        for dec_i, dec in enumerate(self.decoder_list):
            # x_bar_list.append(dec(individual_zs[dec_i]))
            x_bar_list.append(dec(z))

        yLable = self.act(self.regression(self.relu(z)))
        # print("网络构建成功！")
        return x_bar_list, yLable,z,individual_zs


class DIMV(nn.Cell):

    def __init__(self,
                 n_stacks,
                 n_input,
                 n_z,
                 Nlabel):
        super(DIMV, self).__init__()

        self.ae = AE(
            n_stacks=n_stacks,
            n_input=n_input,
            n_z=n_z,
            nLabel=Nlabel)

    # def construct(self, x0, x1, x2, x3, x4, x5, we):

    #     return self.ae(x0, x1, x2, x3, x4, x5, we)
    def construct(self, mul_X, we):
        return self.ae(mul_X, we)
        
if __name__=='__main__':
    inp = [Tensor(np.ones([10,15]),ms.float32),Tensor(np.ones([10,20]),ms.float32)]
    MatrixDiag = nn.MatrixDiag()
    a = Tensor([0.2,0.3,1],ms.float32)
    print(np.log(a))
    we = Tensor(np.ones([10,2]),ms.float32)
    net = DIMV(n_stacks=4,
                 n_input=[15,20],
                 n_z=5,
                 Nlabel=5)
    oup=net(inp,we)
    print(oup[1].shape)