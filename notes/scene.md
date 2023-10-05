# Scene

`scene/__init__.py`

## \_\_init\_\_ (load scenes)

**Steps:**
1. If a specific iteration is provided, set the loaded iter to that. Otherwise, search for the max iter in the model path: `point_cloud_{iter}`, `max(iter)`. (`searchForMaxIteration` is in `utils/system_utils.py`)
2. Create empty dicts for train and test cams.
3. Look for scene type: Colmap thing or blender. Loading in `scene/dataset_readers.py` (see `Blender scene loading` and `Colmap scene loading` below)
```py
"""
fyi:
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
"""
if os.path.exists(os.path.join(args.source_path, "sparse")):
    scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
    print("Found transforms_train.json file, assuming Blender data set!")
    scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
else:
    assert False, "Could not recognize scene type!"
```
4. If this is a new scene, copy the ply file to the model path and create a `cameras.json` file. `camera_to_JSON` is in `utils/camera_utils.py`. This ensures that the ply file is in the model path and that the cameras are in the same order as the ply file, and can be used to load the scene later.
5. Load cams for each resolution scale
6. Load ply if it exists `gaussians.load_ply`, otherwise create a point cloud `gaussians.create_from_pcd` from scene info.

### Blender scene loading

```py
def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    # read transforms_train/val/test.json
    # returns List[CameraInfo]
    # CameraInfo:
    # CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
    #            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1])
    # R, T = rotation, transform matrices
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            # ...
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    # read train and test transforms with readCamerasFromTransforms
    # get normalizer
    # generate random point cloud if no ply -> pcd
    # ...

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info
```
### Colmap scene loading

See `def readColmapSceneInfo(path, images, eval, llffhold=8):` in `scene/dataset_readers.py`  
Returns `SceneInfo` obj as well

___
Besides the main init, the remaining functions are as follows:
- save pcd
- get train/test cams
```py
def save(self, iteration):
    point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
    self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

def getTrainCameras(self, scale=1.0):
    return self.train_cameras[scale]

def getTestCameras(self, scale=1.0):
    return self.test_cameras[scale]
```
... and that's it for `scene/__init__.py`!