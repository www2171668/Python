import torch

b = torch.Tensor([[1,2,3],[4,5,6]])
print(b)
index_1 = torch.LongTensor([[0,1],[2,0]])
index_2 = torch.LongTensor([[0,1,1],[0,0,0]])
print(torch.gather(b, dim=1, index=index_1))
print(torch.gather(b, dim=0, index=index_2))

# index是索引，在行还是列上索引要看dim
# 如tensor([[1., 2., 3.],[4., 5., 6.]])，指定dim=1，也就是横向，那么索引就是列号。
# index的大小是输出的大小，如[[0,1],[2,0]]index，第一行，1列指的是2， 0列指的是1，同理，第二行为6，4

# gather在one-hot为输出的多分类问题中，可以把最大值坐标作为index传进去，然后提取到每一行的正确预测结果，这也是gather可能的一个作用。