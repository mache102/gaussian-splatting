# train.py
init arguments with `arguments/__init__.py`  
... gui stuff

```py
training()
    input args:
    dataset, 
    opt, 
    pipe, 
    testing_iterations, 
    saving_iterations, 
    checkpoint_iterations, 
    checkpoint, 
    debug_from

    - log stuff

    # initialize gaussians, then scenes
    # scene/__init__.py
    # scene/gaussian_model.py
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
```
## for each training iter:

- try to connect to network gui
- if not connected, send renders etc.
```py
if network_gui.conn == None:
    network_gui.try_connect()
while network_gui.conn != None:
    try:
        net_image_bytes = None
        custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        if custom_cam != None:
            net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
            net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        network_gui.send(net_image_bytes, dataset.source_path)
        if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
            break
    except Exception as e:
        network_gui.conn = None

# network stuff: gaussian_renderer/network_gui.py
```

- update lr `gaussians.update_learning_rate(iteration)`
- increase max degree of SH every 1k lvls 
```py
# Every 1000 its we increase the levels of SH up to a maximum degree
if iteration % 1000 == 0:
    gaussians.oneupSHdegree()
```

- pick a random camera 
This appears to be always done at scale=1.0 (default, scale not provided)
`(samples, ...)`


& render
```py
# Pick a random Camera
if not viewpoint_stack:
    viewpoint_stack = scene.getTrainCameras().copy()
viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

# Render
if (iteration - 1) == debug_from:
    pipe.debug = True
render_pkg = render(viewpoint_cam, gaussians, pipe, background)
image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

# render(): 
# gaussian_renderer/__init__.py
```

- calc loss
```py
# Loss
gt_image = viewpoint_cam.original_image.cuda()
Ll1 = l1_loss(image, gt_image)
loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
loss.backward()

# l1_loss, ssim: utils/loss_utils.py
```

## end of train iter: update stuff
```py
with torch.no_grad():
    # Progress bar

    # Log and save
    
    # Densification
    if iteration < opt.densify_until_iter:
        # Keep track of max radii in image-space for pruning
        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
        
        if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            gaussians.reset_opacity()

    # Optimizer step
    if iteration < opt.iterations:
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none = True)

    if (iteration in checkpoint_iterations):
        print("\n[ITER {}] Saving Checkpoint".format(iteration))
        torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

```