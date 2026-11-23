from nandnet import NANDnet
from torch import optim, tensor, float32, nn,save
from sys import argv

nnet = NANDnet()
optimizer = optim.AdamW(nnet.parameters(), lr=0.01)

x = tensor(
    [
        [0,0],
        [1,0],
        [0,1],
        [1,1]
    ],
    dtype=float32
)

y = tensor(
    [
        [1],
        [1],
        [1],
        [0]
    ],
    dtype=float32
)

def train(x,y, epoch):
    global nnet, optimizer

    prediction = nnet(x)
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(prediction,y)
    print(f'Epoch: {epoch} | Loss rounded to 5: {loss:.5f}')
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    save(nnet, 'nandnet.pt')
def loop(epochs):
    for epoch in range(1, epochs+1):
        train(x,y,epoch)

if len(argv) == 2:
    epoch_amm = argv[1]
    if epoch_amm.isalnum():
        epoch_amm = int(epoch_amm)

        loop(epoch_amm)

else:
    print('nandnet_train: parameter: epochs required.')