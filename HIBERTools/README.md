# HIBERTools

>This is a toolbox of reading, loading and visualizing HIBER dataset. We also provide tools to generate lmdb format for fast training.

## Dataset File Structure

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
│   │       │   │── 0000.npy
│   │       │   └── ...
│   │       └── 01_02
│   │           |── 0000.npy
│   │           └── ...
│   ├── HORIZONTAL_HEATMAPS
│   │   ├── 01_01
│   │   │   |── 0000.npy
│   │   │   └── ...
│   │   └── 01_02
│   │       |── 0000.npy
│   │       └── ...
│   └── VERTICAL_HEATMAPS
│       ├── 01_01
│       │   |── 0000.npy
│       │   └── ...
│       └── 01_02
│           |── 0000.npy
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
* num_kp : the number of human keypoints, 14 in our case.

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
    * file name : "%04d.npy" % frame_idx
    * shape: (num_of_person, 1248, 1640) *# related to optical camera resolution, height is 1248, width is 1640*
* HORIZONTAL_HEATMAPS / VERTICAL_HEATMAPS :
    * directory name : "%02d_%02d" % (env, grp_idx)
    * file name : "%04d.npy" % radar_frame_idx
    * shape: (160, 200, 2) *# related to radar frame resolution, height is 160, width is 200*

## Install
```text
pip install git+https://github.com/wuzhiwyyx/HIBER.git#subdirectory=HIBERTools
```

## Usage

### Create dataset object and demonstrate statistics
```py
import HIBERTools as hiber

root_path = '/data/hiber'
dataset = hiber.HIBERDataset(root_path, subsets=['WALK'])

info = dataset.info()
print(info)
```
You will get the results
```text
HIBERDataset statistic info:

Main categories and their corresponding number of groups:
Each group contains 590 optical frames (annotations) and 1180 RF frames.
If you want to know precise group list of each categories, please access datasetobj.statistics attribute.

{
    "WALK": 152,
    "MULTI": 120,
    "ACTION": 143,
    "OCCLUSION": 244,
    "DARK": 31
}

Detailed categories and their corresponding number of groups:
These detailed categories are all included in main categories.
If you want to know precise group list of each detailed categories, please access datasetobj.details attribute.

{
    "occlusion_styrofoam": 84,
    "occlusion_carton": 84,
    "occlusion_yoga": 76,
    "occlusion_multi": 112,
    "occlusion_action": 6,
    "dark_multi": 12,
    "dark_light_change": 7
}
```

### Visualize data

```text
import random

seq = random.randint(0, len(dataset) - 1)
data_item = dataset[seq]
hiber.visualize(data_item, 'result.jpg')
```
You will get the results
![Visualize data](../images/vis_data.jpg)
TODO