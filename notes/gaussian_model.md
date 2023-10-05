# GaussianModel

- The only init is spherical harmonic (SH) degree, which is taken in as `max_sh_degree`. Preceding that is an `active_sh_degree` set to 0, which is increased to `max_sh_degree` later (`iter % 1000 == 0` thing):
```py
# Every 1000 its we increase the levels of SH up to a maximum degree
if iteration % 1000 == 0:
    gaussians.oneupSHdegree()
    
# ...
def oneupSHdegree(self):
    if self.active_sh_degree < self.max_sh_degree:
        self.active_sh_degree += 1

```
All other inits:
```py
def __init__(self, sh_degree : int):
    self.active_sh_degree = 0
    self.max_sh_degree = sh_degree  
    self._xyz = torch.empty(0)
    self._features_dc = torch.empty(0)
    self._features_rest = torch.empty(0)
    self._scaling = torch.empty(0)
    self._rotation = torch.empty(0)
    self._opacity = torch.empty(0)
    self.max_radii2D = torch.empty(0)
    self.xyz_gradient_accum = torch.empty(0)
    self.denom = torch.empty(0)
    self.optimizer = None
    self.percent_dense = 0
    self.spatial_lr_scale = 0
    self.setup_functions()
```
- `setup_functions()` sets several function names:
1.   `scaling_activation` -> `torch.exp`
2.   `scaling_inverse_activation` -> `torch.log`
3.   `covariance_activation` = func below vvv
```py
"""
build_scaling_rotation, strip_symmetric: utils/general_utils.py
"""
def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm

# ...

# sort of reshape thing but we don't know the shape of L
def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

# rotation building
# analysis in progress

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

```
4.   `opacity_activation` = `torch.sigmoid`
5.   `inverse_opacity_activation` = `inverse_sigmoid` (`utils/general_utils.py`):
```py
"""
derive from y = 1 / (1 + exp(-x))
y(1 + exp(-x)) = 1
1 + exp(-x) = 1 / y
exp(-x) = (1 - y) / y
-x = ln((1 - y) / y)
x = ln(y / (1 - y))
"""
def inverse_sigmoid(x):
    return torch.log(x/(1-x))
```
6.   `rotation_activation` = `torch.nn.functional.normalize`

**activations tldr:**
- scaling: exp
- inverse scaling: log
- covariance: (check code)
- opacity: sigmoid
- inverse opacity: inverse sigmoid
- rotation: normalize

## training_setup
Setups:
- percent_dense
- xyz_gradient_accum
```py
# get_expon_lr_func: utils/general_utils.py
# basically exponential learning rate
def training_setup(self, training_args):
    self.percent_dense = training_args.percent_dense
    self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
    self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    l = [
        {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
        {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
        {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
        {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
        {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
        {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
    ]

    self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                lr_delay_mult=training_args.position_lr_delay_mult,
                                                max_steps=training_args.position_lr_max_steps)

```

## update_learning_rate
```py
def update_learning_rate(self, iteration):
    ''' Learning rate scheduling per step '''
    for param_group in self.optimizer.param_groups:
        if param_group["name"] == "xyz":
            lr = self.xyz_scheduler_args(iteration)
            param_group['lr'] = lr
            return lr

```