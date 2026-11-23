from torch import load

m = load('nandnet.pt', weights_only=False)
print(sum(p.numel() for p in m.parameters()))