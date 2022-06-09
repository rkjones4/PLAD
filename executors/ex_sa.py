import numpy as np
import sys
import random
import torch

import executors.ShapeAssembly as SA

device = torch.device('cuda')

# 3D CSG Executor Logic


# Define Language Tokens + Expected Voxel Dimension

DIM = 32

MAX_PRIMS = 10
MAX_SYM = 4

a = (torch.arange(DIM).float() / (DIM-1.)) - .5
b = a.unsqueeze(0).unsqueeze(0).repeat(DIM, DIM, 1)
c = a.unsqueeze(0).unsqueeze(2).repeat(DIM, 1, DIM)
d = a.unsqueeze(1).unsqueeze(2).repeat(1, DIM, DIM)
pts = torch.stack((b,c,d), dim=3).view(-1, 3).to(device)

FACES = ['left','right','bot','top','back','front']
BBOX_FACE_DIST = np.array([0.,0.,0.5,0.5,0.,0.])
GEN_FACE_DIST = np.array([.125,.125,0.25,0.25,.125,.125])

AXES = ['X','Y','Z']
AXES_DIST = np.array([0.6,.2,.2])

PART_TYPES = ['Prim,Move,Reflect', 'Prim,Move,Translate', 'Prim,Move']
PART_DIST = np.array([0.25, 0.1, 0.65])

MOVE_TYPES = ['Attach','Attach,Attach','Squeeze']
MOVE_DIST = np.array([0.4,0.25,0.35])

MIN_UNIQUE_VOXELS = 8

START_TOKEN = 'START'

# TOKENS describe the semantics of the language
# {name: (input types, output type)}
# name -> name of the token
# input types -> expected types of the arguments to the token
# output types -> the output type of the token

TOKENS = {
    START_TOKEN: ('Part', '', ''),
    'END': ('', 'Part', ''),
        
    'Cuboid': ('fnum,fnum,fnum','Prim','Cuboid('),
    'Attach': ('cind,face,fpos,fpos','Att','attach('),
    'Squeeze': ('cind,cind,face,fpos','Sq','squeeze('),
    'Reflect': ('axis','Sym','reflect('),
    'Translate': ('axis,snum,fnum','Sym','translate('),
}

for i in range(1,DIM):
    TOKENS[f'{i}'] = ('', 'fnum', (i * 1.)/DIM)

TOKENS[f'bbox'] = ('', 'cind', f'bbox')
for i in range(0,MAX_PRIMS):
    TOKENS[f'cube{i}'] = ('', 'cind', f'cube{i}')


for i in range(1,10):
    for j in range(1,10):
        TOKENS[f'fpos_{i}_{j}'] = ('', 'fpos', (
            (i * 1.)/10., (j * 1.)/10.
        ))

for face in FACES:
    TOKENS[face] = ('', 'face',face)

for axis in AXES:
    TOKENS[axis] = ('', 'axis',axis)

for i in range(1, MAX_SYM+1):
    TOKENS[f'snum_{i}'] = ('', 'snum', i)
    
I2T = {i:t for i,t in enumerate(TOKENS)}
T2I = {t:i for i,t in I2T.items()}




## end defining language

# Global var that rejects semantic violations during synthetic data creation
DO_CHECKS = False

# The expected dimension of the input data
def get_input_shape():
    return [DIM, DIM, DIM]

# Visualize helper function
def vis_voxels(voxels, fn):
    with open(fn,'w') as f:
        for i, pt in enumerate(pts):
            if flat_voxels[i]:
                x,y,z = pt.tolist()
                f.write(f'v {x} {y} {z}\n')

# Start LANG Execution Logic

def makeCuboidLine(params, cc):
    x = float(params[0]) *1. / DIM
    y = float(params[1]) *1. / DIM
    z = float(params[2]) *1. / DIM
    return f'cube{cc} = Cuboid({x}, {y}, {z}, False)'

def makeAttParams(face,fpos1,fpos2,flip):
    _,x1,y1 = fpos1.split('_')
    _,x2,y2 = fpos2.split('_')
    x1 = float(x1) / 10.
    y1 = float(y1) / 10.
    x2 = float(x2) / 10.
    y2 = float(y2) / 10.
    
    att = [None,None,None,None,None,None]

    if face == 'left':
        I = 3
        J = 0
        A = 1
        B = 2
        C = 4
        D = 5
        
    elif face == 'right':
        I = 0
        J = 3
        A = 1
        B = 2
        C = 4
        D = 5

    elif face == 'bot':
        I = 4
        J = 1
        A = 0
        B = 2
        C = 3
        D = 5

    elif face == 'top':
        I = 1
        J = 4
        A = 0
        B = 2
        C = 3
        D = 5

    elif face == 'back':
        I = 5
        J = 2
        A = 0
        B = 1
        C = 3
        D = 4

    elif face == 'front':
        I = 2
        J = 5
        A = 0
        B = 1
        C = 3
        D = 4

    if flip and face == 'bot':
        att[I] = 0.
        att[J] = 0.
    elif flip and face == 'top':
        att[I] = 1.
        att[J] = 1.
    else:
        att[I] = 1.0
        att[J] = 0.0
        
    att[A] = x1
    att[B] = y1
    att[C] = x2
    att[D] = y2
        
    return att
    
def makeAttachLine(params, cc):
    cind = params[0]
    att_prms = makeAttParams(params[1],params[2],params[3],cind == 'bbox')
    
    return f'attach(cube{cc-1}, {cind}, {att_prms[0]}, {att_prms[1]}, {att_prms[2]}, {att_prms[3]}, {att_prms[4]}, {att_prms[5]})' 

def makeSqueezeLine(params, cc):

    cind1 = params[0]
    cind2 = params[1]
    face = params[2]
    _,x1,y1 = params[3].split('_')
    x = float(x1) / 10.
    y = float(y1) / 10.
    
    return f'squeeze(cube{cc-1}, {cind1}, {cind2}, {face}, {x}, {y})'

def makeReflectLine(params, cc):
    return f'reflect(cube{cc-1}, {params[0]})'

def makeTranslateLine(params, cc):
    num = params[1].split('_')[1]
    d = float(params[2]) * 1. / DIM
    return f'translate(cube{cc-1}, {params[0]}, {num}, {d})'

def tokens_to_lines(tokens):
    lines = []

    start = 0

    cc = 0
    
    while start < len(tokens):        
        t = tokens[start]

        if t == 'START':
            bbDim = float(tokens[start+1]) * 1. / DIM
            lines.append(f'bbox = Cuboid(1., {bbDim}, 1., False)')
            start += 2
            
        elif t == 'Cuboid':
            lines.append(makeCuboidLine(tokens[start+1:start+4], cc))
            start += 4
            cc += 1
            
        elif t == 'Attach':
            lines.append(makeAttachLine(tokens[start+1:start+5], cc))
            start += 5

        elif t == 'Squeeze':
            lines.append(makeSqueezeLine(tokens[start+1:start+5], cc))
            start += 5

        elif t == 'Reflect':
            lines.append(makeReflectLine(tokens[start+1:start+2], cc))
            start += 2

        elif t == 'Translate':
            lines.append(makeTranslateLine(tokens[start+1:start+4], cc))
            start += 4

        elif t == 'END':
            break
            
        else:
            assert False
                        
    return lines                                     

def make_voxels(cubes):
        
    cubes = cubes.to(device)
    ucubes = cubes.unsqueeze(0)

    rotmat = torch.cat(
        (
            (cubes[:, 6:9]).unsqueeze(2),
            (cubes[:, 9:12]).unsqueeze(2),
            (cubes[:, 12:15]).unsqueeze(2),                
        ), dim = 2
    )
        
    cent_pts = pts.unsqueeze(1) - ucubes[:,:,3:6]    
    rot_pts = torch.matmul(rotmat, cent_pts.unsqueeze(-1)).squeeze()    
    cube_sdfs = (
        rot_pts.abs() - ( ucubes[:,:,:3] / 2.)
    ).max(dim=2).values

    exp_voxels = (cube_sdfs <= 0.)
    
    flat_voxels = exp_voxels.float().sum(dim=1)

    num = ((flat_voxels == 1.).view(-1, 1) & exp_voxels).sum(dim=0)

    if (num.min().item() < MIN_UNIQUE_VOXELS) and DO_CHECKS:
        return None

    vox = flat_voxels > 0.
    
    return vox.view(DIM,DIM,DIM)


# Given an expr in the language, return a visual representation
def execute(prog, ret_cubes = False, render=None):

    with torch.no_grad():
        tokens = prog.split()
        
        lines = tokens_to_lines(tokens)

        P = SA.Program()
        
        for line in lines:            
            P.execute(line)

        if render is not None:
            P.render(render)
            return
        
        cubes = P.getCubeGeo()

        if ret_cubes:
            return cubes
                
        voxels = make_voxels(cubes)                

        if voxels is not None:        
            return voxels.cpu().numpy()
        else:
            return voxels


# END LANG Execution Logic

# START SYNTHETIC PROG CREATION LOGIC

O2T = {}

for token, (inp_types, out_type, out_values) in TOKENS.items():
    if token in ['START','END']:
        continue
    
    if out_type not in O2T:
        O2T[out_type] = []

    O2T[out_type].append(token)


# Helper function to sample a bounded normal distribution
def norm_sample(mean, std, mi, ma):
    v = None

    if mi == ma:
        return mi
    
    while True:        
        v = round(max(mean,1) + (np.random.randn() * max(std,1)))
        if v >= mi and v <= ma:
            break

    return v


def samplePartType():
    return np.random.choice(PART_TYPES,p=PART_DIST)

def sampleMoveType():    
    return np.random.choice(MOVE_TYPES,p=MOVE_DIST)

def sampleCubInd(prev_prims, last_cind):
    l = ['bbox']
    for i in [f'cube{i}' for i  in range(prev_prims)]:
        if i != last_cind:
            l.append(i)

    return random.sample(l,1)[0]

def sampleFace(last, last_face):
    if last == 'bbox':
        dist = BBOX_FACE_DIST
    else:
        dist = GEN_FACE_DIST 

    if last_face is not None:
        dist = dist.copy()
        dist[FACES.index(last_face)] = 0.
        dist /= dist.sum()
        
    return np.random.choice(FACES, p=dist)

def sampleAxis():
    return np.random.choice(AXES, p=AXES_DIST)

def sampleBBDim():

    d = norm_sample(
        20,
        8,
        8,
        31
    )
    
    return str(d)
        

def sampleFNum():
    H = DIM//2
    
    center = norm_sample(H,H/3,2,DIM-2)    
    
    d = norm_sample(
        min(DIM-center, center) * 0.75,
        min(DIM-center, center) / 2,
        2, 
        min(DIM-center,center) * 2 - 2,
    )    
    return str(d)

def sampleFPos(paired):
    r = random.random() <= 0.5

    if r:
        if paired:
            return f'fpos_5_5'
        else:
            return random.sample(O2T['fpos'],1)[0]


    i = norm_sample(5,2,1,9)
    j = norm_sample(5,2,1,9)

    return f'fpos_{i}_{j}'

def samplePart(prev_prims):    
    q = [samplePartType()]

    r = []

    c = 0

    last_cind = None
    last_face = None
    lastT = None
    
    while len(q) > 0:
        p = q.pop(0)
        
        c += 1
        
        if ',' in p:
            q = p.split(',') + q
            
        elif p in TOKENS:
            r.append(p)
            ninp = TOKENS[p][0]

            if ninp == '':
                continue
            
            _q = []            
            for i in ninp.split(','):
                _q.append(i)
            q = _q + q
                
        elif p in O2T:
            # State dependant command
                        
            if p == 'cind':
                n = sampleCubInd(prev_prims, last_cind)
                last_cind = n
                
            elif p == 'face':
                n = sampleFace(last_cind, last_face)
                last_face = n
                
            elif p == 'axis':
                n = sampleAxis()

            elif p == 'fnum':
                n = sampleFNum()

            elif p == 'fpos':
                n = sampleFPos(lastT == 'fpos')
                
            else:
                n = random.sample(O2T[p],1)[0]
                
            q = [n] + q
            lastT = p
            
        else:
            assert p == 'Move'
            n = sampleMoveType()
            q = [n] + q

    return r


def sample_prog(max_tokens, max_prims, min_prims):
    
    num_prims = random.randint(min_prims, max_prims)

    tokens = ['START', sampleBBDim()]

    for i in range(num_prims):
        tokens += samplePart(i)        
        
    tokens.append('END')
    
    if len(tokens) > max_tokens:
        return None, None

    voxels = execute(' '.join(tokens))
    
    if voxels is None:
        return None, None
    
    return tokens, voxels


# Main entrypoint to generate a dataste/batch of synthetic programs
# Returns data, either a list of tokens, or a list of tupled (voxels, tokens)

def prog_random_sample(num, max_tokens, max_prims, min_prims, ret_voxels=True):
        
    data = []
    c = 0

    global DO_CHECKS
    DO_CHECKS = True    
    
    while len(data) < num:
        try:
            tokens, voxels = sample_prog(max_tokens, max_prims, min_prims)
            c += 1
        except Exception as e:
            print(f"FAILED SAMPLE PROG WITH {e}")
            continue
        
        if tokens is None:
            continue

        if ret_voxels:
            data.append((voxels, tokens))
        else:
            data.append(tokens)

    DO_CHECKS = False
            
    return data
    
if __name__ == '__main__':    
    import time
    
    t = time.time()
    
    prog_random_sample(int(sys.argv[1]), 100, 8, 2, False)

    print(f'{time.time() - t}')

