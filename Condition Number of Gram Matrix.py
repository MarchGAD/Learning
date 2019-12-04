
import torch
import torch.nn.functional as F
'''
question from https://zhuanlan.zhihu.com/p/47415565
    事实上，由于格拉姆矩阵 G 的条件数是 X 的平方，因此格拉姆矩阵会比描述矩阵更不稳定。
    因此，通过0.5矩阵幂可以使得格拉姆矩阵的条件数等于描述矩阵的条件数，稳定训练过程。
'''
# set Gram matrix as:
#   G := 1 / (HW) * X * X.T
# get condition number
def cond(A, p=2):
    assert A.size(0) == A.size(1) and len(A.size()) == 2
    return torch.norm(A, p=p) * torch.norm(torch.inverse(A), p=p)

# generate a matrix
a = torch.diag(torch.tensor([1, 10, 3, 4, -5, -100, -1000]).float())
print('condition number is', cond(a).item())
a = a @ a.transpose(0, 1)
k = cond(a)
print('condition number of the Gram matrix is ', k.item())
b = cond(torch.sqrt(F.relu(a)) - torch.sqrt(F.relu(-a)))
print('condition number of the Gram matrix after power normalise', b.item())
a = torch.diag(torch.tensor([1, 10, 3, 4, -5, -100, -1000]).float())
a[0, 2] = 10
a[1, 3] = 100
print('condition number is', cond(a).item())
a = a @ a.transpose(0, 1)
k = cond(a)
print('condition number of the Gram matrix is ', k.item())
b = cond(torch.sqrt(F.relu(a)) - torch.sqrt(F.relu(-a)))
print('condition number of the Gram matrix after power normalise', b.item())
print('My result is: power normalise can just reduce the condition number.\n'
      'when the matrix is a DIAGONAL MATRIX, it can make them same together.\n'
      'However, not all square matrices can make that happen.\n')