# HIBERTools

>This is a toolbox of reading, loading and visualizing HIBER dataset. We also provide tools to generate lmdb format for fast training.

## File Structure

Our HIBER dataset has 5 main categories collected in 10 different environment.

- **ACTION** : Single person performing four actions (**stand, walk, sit, squat**) with normal light and no occlusions. 
- **DARK** : Single or multiple people walk randomly in darkness.
- **MULTI** : Multiple people walk randomly with normal light and no occlusions.
- **OCCLUSION** : Single or multiple people walk randomly with three kind of occlusions (**styrofoam occlusion, carton occlusion and yoga mat occlusion**).
- **WALK** : Single person walk randomly with normal light and no occlusions.

Train, val and test subset of HIBER all have the same file structure as follows.

```text
.
├── ACTION
│   ├── ANNOTATIONS
│   │   ├── 2DPOSE
│   │   │   ├── 01_01.npy
│   │   │   ├── 01_02.npy
│   │   │   └── ...
│   │   ├── 3DPOSE
│   │   │   ├── 01_01.npy
│   │   │   ├── 01_02.npy
│   │   │   └── ...
│   │   ├── BOUNDINGBOX
│   │   │   ├── HORIZONTAL
│   │   │   │   ├── 01_01.npy
│   │   │   │   ├── 01_02.npy
│   │   │   │   └── ...
│   │   │   └── VERTICAL
│   │   │       ├── 01_01.npy
│   │   │       ├── 01_02.npy
│   │   │       └── ...
│   │   └── SILHOUETTE
│   │       ├── 01_01
│   │       │   │── 0000.jpg
│   │       │   └── ...
│   │       └── 01_02
│   │           |── 0000.jpg
│   │           └── ...
│   ├── HORIZONTAL_HEATMAPS
│   │   ├── 01_01
│   │   │   |── 0000.jpg
│   │   │   └── ...
│   │   └── 01_02
│   │       |── 0000.jpg
│   │       └── ...
│   └── VERTICAL_HEATMAPS
│       ├── 01_01
│       │   |── 0000.jpg
│       │   └── ...
│       └── 01_02
│           |── 0000.jpg
│           └── ...
├── DARK (Same as Action directory, omitted here)
├── MULTI (Same as Action directory, omitted here)
├── OCCLUSION (Same as Action directory, omitted here)
└── WALK (Same as Action directory, omitted here)
```

## Data format

>Note that: Optical data is collected under the setting of **10 fps** and RF data is collected under **20 fps**.

### Variable definition
* env : the environment number, candidate value is [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].
* grp_idx : the group index, the data is collected group by group, each group of data elapses one minute.
* frame_idx: the optical frame index in each group.
* radar_frame_idx: the RF frame index in each group.
* grp_frame_num : the number of optical frames in each group, it should be 600, We slice out the last ten frames and make it **590** for the purpose of alignment.
* num_of_person : the number of people present in the current environment.
* num_kp : the number of human keypoints.

### Shape explanation

* 2DPOSE :
    * file name : "%02d_%02d.npy" % (env, grp_idx)
    * shape : (grp_frame_num, num_of_person, num_kp, 2) *# (x, y)*
* 3DPOSE :
    * file name : "%02d_%02d.npy" % (env, grp_idx)
    * shape : (grp_frame_num, num_of_person, num_kp, 3) *# (x, y, z)*
* BOUNDINGBOX :
    * file name : "%02d_%02d.npy" % (env, grp_idx)
    * shape : (grp_frame_num, num_of_person, 4) *# xyxy*
* SILHOUETTE :
    * directory name : "%02d_%02d" % (env, grp_idx)
    * file name : "%04d.jpg" % frame_idx
    * shape: (1248, 1640) *# optical camera resolution*
* HORIZONTAL_HEATMAPS / VERTICAL_HEATMAPS :
    * directory name : "%02d_%02d" % (env, grp_idx)
    * file name : "%04d.jpg" % radar_frame_idx
    * shape: (160, 200) *# radar frame resolution*

## Install
```text
pip install git+https://github.com/wuzhiwyyx/HIBER.git#subdirectory=HIBERTools
```

## Usage
TODO