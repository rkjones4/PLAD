import numpy as np
import torch
import h5py
import sys

# Logic for loading voxelized ShapeNet data

V_DIM = 32
B_DIM = V_DIM * 2
voxel_inds = ((np.indices((V_DIM, V_DIM, V_DIM)).T + .5) / (V_DIM//2)) -1.
flat_voxel_inds = torch.from_numpy(voxel_inds.reshape(-1, 3)).float()

ndim = V_DIM
a = torch.arange(ndim).view(1,1,-1) * 2 * 1 + torch.arange(ndim).view(1,-1,1) * 2 * ndim * 2 + torch.arange(ndim).view(-1,1,1) * 2 * ((ndim * 2) ** 2)
b = a.view(-1,1).repeat(1, 8)
rs_inds = b + torch.tensor([[0,1,ndim*2,ndim*2+1,(ndim*2)**2,((ndim*2)**2)+1, ((ndim*2)**2)+(ndim*2), ((ndim*2)**2)+(ndim*2)+1]])

def vis_voxels(voxels, name):
    pos_inds = voxels[:,:,:].flatten().nonzero().flatten()
    pos_pts = flat_voxel_inds[pos_inds]
    writeSPC(pos_pts, name)

def convert(voxels):
    return voxels.flatten()[rs_inds].max(dim=1).values.view(V_DIM, V_DIM, V_DIM).T    
    
def get_voxels(FILENAME, num):
    data = h5py.File(FILENAME, 'r')
    raw_voxels = torch.from_numpy(data['voxels'][:num]).flip(dims=[3])

    voxels = []

    for i in range(min(num, raw_voxels.shape[0])):
        voxels.append(convert(raw_voxels[i]))
        
    return voxels

def test_sn(FILENAME, num, name):
    voxels = get_voxels(FILENAME, num)
    for i, voxel in enumerate(voxels):
        vis_voxels(voxel, f'{name}_{i}.obj') 
    
if __name__ == '__main__':
    test_sn(sys.argv[1], 10, sys.argv[2])
