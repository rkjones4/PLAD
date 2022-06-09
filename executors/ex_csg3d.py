import numpy as np
import sys
import random
import torch

device = torch.device('cuda')

# 3D CSG Executor Logic


# Define Language Tokens + Expected Voxel Dimension

DIM = 32
TPREC = 32
TOFF = 1

a = torch.arange(DIM).float()
b = a.unsqueeze(0).unsqueeze(0).repeat(DIM, DIM, 1)
c = a.unsqueeze(0).unsqueeze(2).repeat(DIM, 1, DIM)
d = a.unsqueeze(1).unsqueeze(2).repeat(1, DIM, DIM)
pts = torch.stack((b,c,d), dim=3).view(-1, 3).to(device)

START_TOKEN = 'START'

# TOKENS describe the semantics of the language
# {name: (input types, output type)}
# name -> name of the token
# input types -> expected types of the arguments to the token
# output types -> the output type of the token

TOKENS = {
    START_TOKEN: ('shape', 'prog'),

    'sphere': ('fnum,fnum,fnum,fnum,fnum,fnum','shape'),
    'cube': ('fnum,fnum,fnum,fnum,fnum,fnum','shape'),
        
    'union': ('shape,shape','shape'),
    'inter': ('shape,shape','shape'),
    'diff': ('shape,shape','shape'),    
}

for i in range(TOFF,DIM,DIM//TPREC):
    TOKENS[f'{i}'] = ('', 'fnum')

# Assigning each token a unique index
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
        for i in range(0, DIM):
            for j in range(0, DIM):
                for k in range(0, DIM):
                    x,y,z = (i-DIM//2.) / (DIM//2.), (j-DIM//2.) / (DIM//2.), (k-DIM//2.) / (DIM//2.)
                    if voxels[i,j,k]:
                        f.write(f'v {x} {y} {z}\n')

# Start LANG Execution Logic
                        
# Cube primitive
class Cube:
    def __init__(self):
        self.name = 'cube'
        self.center = np.array([0, 0, 0])
        self.dims = np.array([1, 1, 1])

    def voxelize(self):
        pts1 = pts - torch.from_numpy(self.center).view(1,3).to(device)
        pts2 = pts1.abs() - (torch.from_numpy(self.dims).view(1,3).to(device) / 2.)
        voxels = (pts2.max(dim=1).values <= 0).view(DIM, DIM, DIM).transpose(0,2).cpu().numpy()
        return voxels                                          

# Sphere primitive
class Sphere:
    def __init__(self):
        self.name = 'sphere'
        self.center = np.array([0, 0, 0])
        self.dims = np.array([1, 1, 1])

    def voxelize(self):
        pts1 = pts - torch.from_numpy(self.center).view(1,3).to(device)
        pts2 = pts1 / (torch.from_numpy(self.dims).view(1,3).to(device) / 2.)
        voxels = (pts2.norm(dim=1) <= 1).view(DIM, DIM, DIM).transpose(0, 2).cpu().numpy()
        return voxels

# A primitive is either a sphere or cube, and is initilized with position and scale
class Primitive:
    def __init__(self, name, pos, scale):
        assert name in ['sphere','cube']

        if name == 'sphere':
            self.p = Sphere()

        elif name == 'cube':
            self.p = Cube()

        pa,pb,pc = [int(p) for p in pos]
        sa,sb,sc = [int(s) for s in scale]

        self.move(np.array([pa,pb,pc]))

        self.scale(np.array([sa,sb,sc]))
        
    def move(self, vec):
        self.p.center += vec

        if DO_CHECKS:
            if not ((self.p.center >= 0).all() and (self.p.center < DIM)).all():
                assert False, 'ex_error bad center val'
        
    def scale(self, vec):
        self.p.dims *= vec

        if DO_CHECKS:
            if not ((self.p.dims > 0).all() and (self.p.dims < DIM)).all():
                assert False, 'ex_error bad dim val'

    def voxelize(self):
        return self.p.voxelize()

# Represent program a tree of Nodes        
class Node:
    def __init__(self, token, parent=None, cind=None):
        self.parent = parent
        self.cind = cind
        self.token = token
        inp, self.out_type = TOKENS[token]

        self.inp_types = []
        for _inp in inp.split(','):
            if len(_inp) > 0:
                self.inp_types.append(_inp.split('|'))

        self.children = []

        self.prim = None
        self.voxels = None

    # Does this need need any more arguments
    def is_finished(self):
        return len(self.children) == len(self.inp_types)

    # Add token to program at this node
    def add(self, token):

        if len(self.inp_types) == 0:
            return self.ret()

        assert TOKENS[token][1] in self.inp_types[len(self.children)]
        
        c = Node(token, self, len(self.children))
        
        self.children.append(c)

        return c.ret()

    # Find a node that is not finished consuming input arguments
    def ret(self):
        r = self

        while(r.parent is not None and r.is_finished()):
            r = r.parent
            
        return r        

    # Turn primitive creating nodes into actual Primitive objects
    def prim_replace(self):
        if self.token in ['sphere', 'cube']:
            self.prim = Primitive(
                self.token,
                [self.children[0].token, self.children[1].token, self.children[2].token],
                [self.children[3].token, self.children[4].token, self.children[5].token]
            )
            self.children = []
            
        for c in self.children:
            c.prim_replace()

    # Convert operations into voxels
    def voxelize(self):
        
        if self.prim is not None:
            self.voxels = self.prim.voxelize()
            
        elif self.token in ['union','inter','diff']:
            self.children[0].voxelize()
            self.children[1].voxelize()

            a = self.children[0].voxels
            b = self.children[1].voxels
            
            if self.token == 'union':
                self.voxels = a | b
            elif self.token == 'inter':
                self.voxels = a & b
            elif self.token == 'diff':
                self.voxels = a & (a!=b)

            if DO_CHECKS:
                if (self.voxels == a).all() or (self.voxels == b).all() or (self.voxels == False).all():
                    assert False, 'ex_error bad op result'
                
        elif self.token == 'START':
            self.children[0].voxelize()
            self.voxels = self.children[0].voxels

        else:
            assert False

# Given an expr in the language, return a visual representation
def execute(expr):
    tokens = expr.split()
    
    c = Node(tokens[0])
    
    for t in tokens[1:]:
        c = c.add(t)

    assert c.token == 'START'

    c.prim_replace()
    c.voxelize()

    return c.voxels


# END LANG Execution Logic

# START SYNTHETIC PROG CREATION LOGIC

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

# Sample a random primtive 
def sample_primitive():

    H = DIM//2
    
    center = [
        norm_sample(H,H/3,2,DIM-2),
        norm_sample(H,H/3,2,DIM-2),
        norm_sample(H,H/3,2,DIM-2)
    ]
        
    dims = [
        norm_sample(
            min(DIM-center[0], center[0]) * 0.75,
            min(DIM-center[0], center[0]) / 2,
            2, 
            min(DIM-center[0],center[0]) * 2 - 2,
        ),
        norm_sample(
            min(DIM-center[1], center[1]) * 0.75,
            min(DIM-center[1], center[1]) / 2,
            2, 
            min(DIM-center[1],center[1]) * 2 - 2,
        ),
        norm_sample(
            min(DIM-center[2],center[2]) * 0.75,
            min(DIM-center[2],center[2]) / 2,            
            2, 
            min(DIM-center[2],center[2]) * 2 - 2,
        ),            
    ]

    stype = random.sample(['sphere','cube'], 1)[0]
    return (stype, center, dims)
    
def sample_primitives(n):
    prims = []
    for _ in range(n):
        prims.append(sample_primitive())
    return prims

# identify which primitives might overlap (based on bounding box)
def calc_overlaps(prims):    
    o = {i:set() for i in range(len(prims))}

    for i, (_, ac, ad) in enumerate(prims):
        for j, (_, bc, bd) in enumerate(prims):
            if i >= j:
                continue

            amin = np.array([ac[i] - ad[i]//2 for i in range(3)])
            amax = np.array([ac[i] + ad[i]//2 for i in range(3)])
            bmin = np.array([bc[i] - bd[i]//2 for i in range(3)])
            bmax = np.array([bc[i] + bd[i]//2 for i in range(3)])

            if (amax > bmin).all() and (amin < bmax).all():
                o[i].add(j)
                o[j].add(i)
                
    return o

# Helper function to see if every primitive in sa has some overlap with some primitive in sb
def checkOverlap(overlaps, sa, sb):    
    for pa in sa:
        found = False
        for pb in sb:
            if pa in overlaps[pb]:
                found = True
                break
                        
        if not found:
            return False
        
    return True

# Sample a tree of CSG operations, where valid operations are determined by overlaps
def op_sample(overlaps):

    shapes = [[o] for o in overlaps.keys()]
    order = []

    while len(shapes) > 1:
        ia = random.randint(0, len(shapes)-1)
        sa = shapes.pop(ia)
        ib = random.randint(0, len(shapes)-1)
        sb = shapes.pop(ib)

        ops = ['union']

        a2b = checkOverlap(overlaps, sa, sb)
        b2a = checkOverlap(overlaps, sb, sa)

        # Diff requires every prim in b to overlap with some prim in a
        if b2a:
            ops.append('diff')

        # Intersection requires overlaps from both b to a , and a to b
        if b2a and a2b:
            ops.append('inter')

        # After consolidating all valid possible operations, sample one at random
        op = random.sample(ops, 1)[0]
        
        sc = sa + sb
        sc.sort()
        shapes.append(sc)

        order.append((op, tuple(sa), tuple(sb), tuple(sc)))
        
    return order

# Convert value to token string
def ne(v):
    return str(int((round((v - TOFF) / (DIM//TPREC)) * (DIM//TPREC)) + TOFF))

# Turn a primitive into tokens
def make_prim_tokens(p):
    t = [p[0], ne(p[1][0]), ne(p[1][1]), ne(p[1][2]), ne(p[2][0]), ne(p[2][1]), ne(p[2][2])]
    
    return t

# Given primitives, and sampled oprations, return a tokenized version of the program
def make_op_prog(prims, ops):    

    tm = {tuple([i]): make_prim_tokens(p) for i,p in enumerate(prims)}

    last = None

    for op, a, b, c in ops:
        te = [op] + tm[a] + tm[b]
        tm[c] = te
        last = c

    return ['START'] + tm[last]

# Helper function to sample prims, then create programs given those prims
def sample_prim_progs(max_tokens, max_prims, min_prims, num_op_samples):

    num_prims = random.randint(min_prims, max_prims)
    
    prims = sample_primitives(num_prims)
    
    prim_overlaps = calc_overlaps(prims)
    
    ops = []
    
    for _ in range(num_op_samples):            
        ops.append(op_sample(prim_overlaps))

    progs = []

    for op in ops:
        tokens = make_op_prog(prims, op)

        if len(tokens) <= max_tokens:
            prog = ' '.join(tokens)
            progs.append((prog, tokens))
            
    return progs
        
# Main entrypoint to generate a dataste/batch of synthetic programs
# Returns data, either a list of tokens, or a list of tupled (voxels, tokens)

def prog_random_sample(num, max_tokens, max_prims, min_prims, num_op_samples, ret_voxels=True, vis_progs=False):
        
    data = []
    c = 0

    global DO_CHECKS
    DO_CHECKS = True
    
    while len(data) < num:

        samples = sample_prim_progs(max_tokens, max_prims, min_prims, num_op_samples)
        
        for expr, tokens in samples:
            
            if len(expr) == 0:
                continue
                                        
            try:
                voxels = execute(expr)
                if ret_voxels:
                    data.append((voxels, tokens))
                else:
                    data.append(tokens)

                if vis_progs:
                    print(f'prog {c} : {expr}')
                    vis_voxels(voxels, f'prog_{c}.obj')
                    c += 1
                    
            except Exception as e:
                if e.args[0].split(' ')[0] == 'ex_error':
                    continue
                elif e.args[0].split(' ')[0] == 'prog_error':
                    continue            
                else:
                    raise e

    DO_CHECKS = False
    
    return data
    
if __name__ == '__main__':    
    import time
    
    t = time.time()
    
    prog_random_sample(int(sys.argv[1]), 100, 12, 2, 1, False, False)

    print(f'{time.time() - t}')

