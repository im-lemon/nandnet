from torch import load,tensor, float32, sigmoid

def modelinference():
    x = int(input('>> '))
    y = int(input('>> '))

    model = load('nandnet.pt', weights_only=False)
    e = model(tensor([x,y], dtype=float32))

    return e

def inference():
    md = sigmoid(modelinference())
    if md >= 0.5:
        print('nandnet: 1')
    else:
        print('nandnet: 0')
inference()

while True:
    inference()