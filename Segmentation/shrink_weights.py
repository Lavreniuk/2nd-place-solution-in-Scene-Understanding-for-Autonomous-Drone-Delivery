import torch
import argparse
import copy

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('weights', type=str, default='', help='path to weights')
    args = parser.parse_args()
    path = args.weights
    a = torch.load(path, map_location='cpu')
    del a['optimizer']
    b = copy.copy(a['state_dict'])

    k=0
    for i in a['state_dict'].keys():
        if 'ema_' in i:
            b.pop(i)
            k+=1
    a['state_dict'] = b
    print(f'{k} weights changed')
    torch.save(a, path.replace('.pth', '_sh.pth'))