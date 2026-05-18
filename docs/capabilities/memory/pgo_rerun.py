# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render voxels-only vs PGO vs two-pass global maps side-by-side in rerun.
Also dump the two-pass global maps for the relocalization test.

All three clouds are logged under distinct entity paths so they can be
toggled / recolored independently in the viewer.
"""

import rerun as rr

from dimos.mapping.pgo import PGOMapTransformer, pgo_then_voxels
from dimos.mapping.voxels import VoxelMapTransformer
from dimos.memory2.store.sqlite import SqliteStore
from dimos.msgs.sensor_msgs.PointCloud2 import register_colormap_annotation
from dimos.utils.data import get_data, get_data_dir

store = SqliteStore(path=get_data("go2_hongkong_office.db"))
lidar = store.streams.lidar

t0 = lidar.first().ts
slice_lidar = lidar.before(t0 + 200)

print("computing voxels-only baseline...")
voxels_map = slice_lidar.transform(VoxelMapTransformer(device="CUDA:0", emit_every=0)).last().data

print("computing PGO map (re-projected body clouds)...")
pgo_map = (
    slice_lidar.transform(PGOMapTransformer(emit_every=0, global_map_voxel_size=0.05)).last().data
)

print("computing two-pass map (PGO trajectory + voxel rebuild)...")
twopass_map = pgo_then_voxels(slice_lidar, voxel_size=0.05)

(get_data_dir() / "go2_hongkong_office_twopass_map.pc2.lcm").write_bytes(twopass_map.lcm_encode())

rr.init("pgo_compare", spawn=True)
register_colormap_annotation()
rr.log("voxels", voxels_map.to_rerun(), static=True)
rr.log("pgo", pgo_map.to_rerun(), static=True)
rr.log("twopass", twopass_map.to_rerun(), static=True)
