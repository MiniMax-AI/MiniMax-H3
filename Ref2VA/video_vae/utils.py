# SPDX-License-Identifier: Apache-2.0
# Module helpers for the MiniMax H3 visual VAE (inference-only bundle).


def apply_spatial_parallel(module, enabled, chunk_dim=-1):
    from .conv import SpatialParallelConv3d
    from .norm import FusedGroupNorm3D, SpatialParallelGroupNorm

    if hasattr(module, "set_spatial_parallel"):
        module.set_spatial_parallel(enabled)
    for m in module.modules():
        if enabled and isinstance(m, FusedGroupNorm3D):
            raise NotImplementedError("FusedGroupNorm3D is incompatible with SP")
        if isinstance(m, SpatialParallelGroupNorm):
            m.spatial_parallel = enabled
        elif isinstance(m, SpatialParallelConv3d):
            m.spatial_parallel = enabled
            m.chunk_dim = chunk_dim
