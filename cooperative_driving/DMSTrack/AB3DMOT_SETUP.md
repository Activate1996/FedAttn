# AB3DMOT and Xinshuo_PyToolbox Setup

The evaluation utilities use AB3DMOT for 3D box operations. AB3DMOT is not redistributed in this reviewer package because its original licence restricts third-party redistribution. AB3DMOT also depends on Xinshuo_PyToolbox.

## Installation

From the repository root, activate the evaluation environment and obtain both dependencies:

```bash
conda activate dmstrack3
git clone https://github.com/xinshuoweng/AB3DMOT.git DMSTrack/AB3DMOT
git clone https://github.com/xinshuoweng/Xinshuo_PyToolbox.git DMSTrack/AB3DMOT/Xinshuo_PyToolbox
```

The expected directory structure is:

```text
DMSTrack/
└── AB3DMOT/
    ├── AB3DMOT_libs/
    └── Xinshuo_PyToolbox/
        └── xinshuo_io/
```

## Verification

From the repository root, verify that both dependencies can be imported:

```bash
PYTHONPATH="$PWD/DMSTrack:$PWD/DMSTrack/AB3DMOT:$PWD/DMSTrack/AB3DMOT/Xinshuo_PyToolbox" python -c "from AB3DMOT.AB3DMOT_libs.box import Box3D; import xinshuo_io; print('AB3DMOT dependencies: OK')"
```

A successful installation prints:

```text
AB3DMOT dependencies: OK
```

AB3DMOT and Xinshuo_PyToolbox remain subject to their original licence terms.
