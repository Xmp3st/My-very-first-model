import torch

a = torch.rand(2, 3)
#a = torch.tensor([2, 2])
# tensor() receives original data
#           tensor([1.2, 2.3])
# Tensor() receives data (from list)
#           Tensor([1.2, 2,3])
#       or generates data (zeros) (from multiple parameters)
#           Tensor(3, 4)

print('a =', a)
#a = tensor([[0.8234, 0.8564, 0.4126], [0.3079, 0.0613, 0.0924]])
print('dim =', len(a.size())) #a.dim()
print('size =', a.size())
print('shape =', a.shape)
print('type =',a.type())
# dim = 2
# size = torch.Size([2, 3])
# shape = torch.Size([2, 3])
# type = torch.FloatTensor

b = torch.Tensor(4, 5) # Don't use this! Use FloatTensor()
print(b)
print(b.shape)
# tensor([[0., 0.]])
# torch.Size([1, 2])

print(torch.empty(20))
print(torch.IntTensor(2, 3))

import numpy as np
c = torch.tensor(np.zeros(10), dtype = torch.float)
print(c)
print(c.type())

#torch.rand_like(a) #生成shape与a相同的tensor
#torch.randn(2, 3)
#torch.full([2, 3], 7) #生成2x3，全为7
#torch.full([], x)
#torch.arange(0, n, 2)
#torch.linspace(0, 5, steps=6)
#torch.logspace(-1, 0, steps=3)
#torch.ones(3, 4)
#     zeros(3, 4)
#       eye(3, 4)
#       eye(3)
#torch.randperm(0, 10)
#
#               dim     range
#a.index_select(2, arange(28))
