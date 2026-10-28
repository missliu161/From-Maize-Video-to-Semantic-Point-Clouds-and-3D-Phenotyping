# Third-party code and licenses

The `2dgs/` implementation is based on the 2D Gaussian Splatting project:

- Upstream project: <https://github.com/hbb1/2d-gaussian-splatting>
- Surfel rasterizer: <https://github.com/hbb1/diff-surfel-rasterization>
- Simple KNN extension: <https://gitlab.inria.fr/bkerbl/simple-knn>
- VGGSfM (installed separately; not vendored here): <https://github.com/facebookresearch/vggsfm>

The main project license is included as `LICENSE.md`; extension-specific
license files are included beside each vendored extension. The local copy of
the surfel rasterizer is from the working tree used for the experiments and
contains local modifications relative to its Git `HEAD`; the `.git` metadata
and prebuilt binaries are intentionally excluded. Review the upstream and
local license texts before redistribution.
