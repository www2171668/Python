import torch
from torch.autograd import Variable
from torch.nn import functional

data_num = 100
x = torch.unsqueeze(torch.linspace(-1,1,data_num), dim=1)
y0 = torch.zeros(50) 
y1 = torch.ones(50)
y = torch.cat((y0, y1), ).type(torch.LongTensor)

def get_acc(labels, outputs):
    '''get_acc: Get the classification accuracy
        Inputs: labels: the ground truth; 
                outputs: the predicted labels of a DNN 
        Outputs: the accuracy 
    '''
    _, predicted = torch.max(outputs.data, 1)
    data_num = y.shape[0]*1.0
    #item() to get a Python number from a tensor containing a single value:
    correct_num = (predicted == labels).sum().item()
    accuracy = correct_num/data_num
    return accuracy

def save():
    mynet = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 2)
    )
    print(mynet)

    optimizer = torch.optim.SGD(mynet.parameters(), lr=0.01)
    loss_func = torch.nn.CrossEntropyLoss()

    for t in range(1000):
        out = mynet(x)
        loss = loss_func(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 100 == 0:
            acc = get_acc(y, out)
            print('acc is:', acc)
    
    torch.save(mynet, 'net.pkl' )
    #torch.save(mynet.state_dict(), 'net_params.pkl')

def restore_net():
    # load a trained DNN, and test its accuracy
    loaded_net = torch.load('net.pkl')
    print(loaded_net)

    out = loaded_net(x)
    acc = get_acc(y, out)
    print('acc of loaded_net is:', acc)

save()
restore_net()
