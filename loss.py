import mindspore as ms
from mindspore import nn, Tensor
import mindspore.numpy as mnp
import mindspore.ops as ops
import numpy as np
# This code is inspired by https://github.com/SubmissionsIn/MFLVC
import time
class Loss(nn.Cell):
    def __init__(self, t):
        super(Loss, self).__init__()
        self.t = t

        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward_contrast(self, v1, v2, we1, we2):
        t1=time.time()
        normalize = ops.L2Normalize(-1)
        # mask_miss_inst = we1.mul(we2).numpy() # mask the unavailable instances
        
        # mask_miss_inst = Tensor(np.where(mask_miss_inst))
        # mask_miss_inst=mask_miss_inst.squeeze(0)
        # v1= v1[mask_miss_inst]
        # v2= v2[mask_miss_inst]
        t2=time.time()

        mask_miss_inst = we1.mul(we2).numpy()
        # print(mask_miss_inst.shape)
        if (mask_miss_inst==1).sum()==0:
            return 0
        d= v1.shape[-1]
        # v1 = Tensor(v1.asnumpy()[mask_miss_inst==1])
        # v2 = Tensor(v2.asnumpy()[mask_miss_inst==1])
        v1 = ops.masked_select(v1,mask_miss_inst.unsqueeze(-1)==1).reshape(-1,d)
        v2 = ops.masked_select(v2,mask_miss_inst.unsqueeze(-1)==1).reshape(-1,d)
        # print(v2.shape)
        n = v1.shape[0]
        N = 2 * n
        if n == 0:
            return 0
        v1 = normalize(v1) #normalize two vectors
        v2 = normalize(v2)
        mask = np.ones((N, N)) # get mask
        # mask = mask.fill_diagonal_(0)
        # Meye = mnp.eye(N)
        # mask = np.where(Meye == 1, 0, mask)
        for i in range(N // 2):
            mask[i, N // 2 + i] = 0
            mask[N // 2 + i, i] = 0
        for i in range(N):
            mask[i,i]=0
        h = ops.concat((v1, v2), axis=0)
        sim_mat = ops.matmul(h, h.T) / self.t
        t3=time.time()

        # print(sim_mat.shape,Meye.shape)
        # sim_mat = mnp.where(Meye==1,0,sim_mat)
        # print(ops.diagonal(sim_mat))
        # targets1 = Tensor(np.array(range(N//2,N)),ms.int32)
        # targets2 = Tensor(np.array(range(0,N//2)),ms.int32)
        # targets = ops.concat((targets1,targets2),axis=0)
        # t4=time.time()

        # print(targets.shape)
        positive_pairs = ops.concat((ops.diagonal(sim_mat, n), ops.diagonal(sim_mat, -n)), axis=0)
        positive_pairs = positive_pairs.reshape(N, 1)
        # sim_mat = sim_mat.asnumpy()
        # print(sim_mat.shape)
        # negative_pairs = Tensor(sim_mat[mask==1]).reshape(N, -1)
        negative_pairs = ops.masked_select(sim_mat,mask==1).reshape(N, -1)
        targets = Tensor(mnp.zeros(N),ms.int32)
        # print(positive_pairs.shape,negative_pairs.shape)
        logits = ops.concat((positive_pairs, negative_pairs), axis=1)
        
        loss = self.criterion(sim_mat, targets)
        t5=time.time() 

        return loss/N

