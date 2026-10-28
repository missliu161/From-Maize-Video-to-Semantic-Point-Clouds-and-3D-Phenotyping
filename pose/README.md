# Pose estimation with VGGSfM

VGGSfM is not vendored in this bundle. Clone and install the official
repository in a separate environment (see the bundle's top-level README).
VGGSfM downloads its pretrained checkpoint automatically on first use unless
the upstream configuration is changed.

`extract_video_frames.py` samples the input video at 2 frames per second by default.
`run_vggsfm.py` executes the upstream checkout's `demo.py`, requests all input frames as
query frames by default, and moves the resulting COLMAP files under `sparse/0/`
for the 2DGS loader. The runner refuses to overwrite an existing `sparse/0`
model. The input scene must have `images/` and the project code must be in the
separate `2dgs/` folder.
