# PGO experiments

We will load lidar frames from a recording and experiment with how
pose-graph optimization (PGO) reassembles the map. To start, no PGO at
all — just feed lidar frames through `VoxelMapTransformer` to get a
single global pointcloud, using whatever pose each frame already
carries (raw odometry). This gives us the baseline drift to improve
against.

## Load the recording

```python session=pgo
from dimos.memory2.store.sqlite import SqliteStore
from dimos.utils.data import get_data

store = SqliteStore(path=get_data("go2_hongkong_office.db"))
lidar = store.streams.lidar
print(lidar.summary())
```

<!--Result:-->
```
Stream("lidar"): 4235 items, 2026-05-06 08:12:09 — 2026-05-06 08:21:27 (558.0s)
```

## Baseline global map (first 3 minutes, no PGO)

Slice to a 120 s window. We splice a small generator-based transform
between `slice_lidar` and `VoxelMapTransformer` to time the downstream
consumer per-frame — yielding hands control to the voxel transformer's
`add_frame` call; on resume we measure elapsed and append to a parallel
numerical side-stream (same `ts`, value = ms).

```python session=pgo
import time
from dimos.mapping.voxels import VoxelMapTransformer
from dimos.memory2.store.memory import MemoryStore
from dimos.memory2.transform import measure_gpu_mem, measure_time

t0 = lidar.first().ts
slice_lidar = lidar#.before(t0 + 500)

mem = MemoryStore()
frame_ms = mem.stream("frame_ms", float)

global_map = (
    slice_lidar
    .transform(measure_time(frame_ms))
    .transform(VoxelMapTransformer(device="CUDA:0", emit_every=0))
    .last().data
)

vals = [o.data for o in frame_ms]
print(f"frames={len(vals)}  total={sum(vals) / 1000:.1f}s  "
      f"mean={sum(vals) / len(vals):.1f}ms  max={max(vals):.1f}ms")

from dimos.memory2.vis.space.space import Space
Space().add(global_map).to_svg("assets/pgo_baseline_map.svg")
```

<!--Result:-->
```
07:00:38.423 [inf][dimos/mapping/voxels.py       ] VoxelGrid using device: CUDA:0
frames=4235  total=7.9s  mean=1.9ms  max=60.4ms
```

![output](assets/pgo_baseline_map.svg)

## PGO trajectory overlaid on voxels map

For realtime use, building an extra global map per pipeline is too
expensive. The interesting signal from PGO is its *corrected
trajectory* — overlay it on the voxels-only map (which is fast and
accurate locally) and you can see exactly where PGO undid drift.

`pgo_trajectories(stream)` returns two `Path` messages: ``drifted`` is
the raw odometry pose at each keyframe (the input to PGO), ``corrected``
is iSAM2's optimized pose after all loop closures have settled.

```python session=pgo
from dimos.memory2.vis.space.elements import Polyline
from dimos.mapping.pgo import pgo_trajectories

loop_score = mem.stream("loop_score", float)
pose_jump_m = mem.stream("pose_jump_m", float)

drifted_path, corrected_path, pgo_map = pgo_trajectories(
    slice_lidar,
    loop_score=loop_score,
    pose_jump=pose_jump_m,
    global_map_voxel_size=0.05,  # also rebuild global map from corrected keyframes
)
print(f"keyframes: {len(corrected_path.poses)}")
jumps = [o.data for o in pose_jump_m]
print(f"loops fired: {loop_score.count()}  "
      f"max pose_jump: {max(jumps, default=0):.3f} m")

(
    Space()
    .add(global_map)
    .add(Polyline(msg=drifted_path, color="#e74c3c", width=0.08))   # red
    .add(Polyline(msg=corrected_path, color="#2ecc71", width=0.08)) # green
    .to_svg("assets/pgo_trajectories.svg")
)

# PGO map (keyframe body clouds through corrected poses) + corrected path
(
    Space()
    .add(pgo_map)
    .add(Polyline(msg=corrected_path, color="#2ecc71", width=0.08))
    .to_svg("assets/pgo_map.svg")
)
```

<!--Result:-->
```
07:00:50.300 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0598 source=125 target=78
07:00:50.507 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0229 source=134 target=96
07:00:53.617 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0534 source=289 target=248
07:00:53.752 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0193 source=298 target=240
07:00:53.969 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0738 source=308 target=68
07:00:54.189 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0352 source=317 target=60
07:00:54.442 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0222 source=324 target=51
07:00:54.687 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0164 source=333 target=38
07:00:54.945 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0199 source=341 target=29
07:00:55.331 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0421 source=353 target=19
07:00:57.364 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0609 source=435 target=404
07:00:57.606 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0328 source=442 target=415
07:00:57.974 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0287 source=452 target=407
07:01:00.039 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.165 source=539 target=498
07:01:00.260 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0483 source=547 target=499
07:01:00.650 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0422 source=562 target=498
07:01:01.042 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0462 source=576 target=498
07:01:01.376 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0301 source=588 target=490
07:01:01.634 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0256 source=596 target=484
07:01:01.887 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0472 source=605 target=481
07:01:04.360 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0178 source=710 target=668
07:01:04.668 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0456 source=719 target=668
07:01:04.942 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.043 source=726 target=666
07:01:06.333 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0216 source=772 target=734
07:01:06.675 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0171 source=782 target=741
07:01:07.184 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0175 source=798 target=741
07:01:07.572 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0236 source=808 target=734
07:01:07.806 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0259 source=815 target=727
07:01:08.079 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0224 source=825 target=720
07:01:09.640 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0677 source=896 target=10
07:01:09.983 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0516 source=906 target=343
07:01:10.488 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0394 source=920 target=32
07:01:10.792 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0308 source=928 target=339
07:01:11.096 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0217 source=939 target=38
07:01:11.336 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0129 source=946 target=53
07:01:11.646 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0155 source=958 target=56
07:01:11.952 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.014 source=970 target=61
07:01:12.257 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0113 source=980 target=74
07:01:12.517 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0151 source=989 target=91
07:01:12.837 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0114 source=999 target=99
07:01:13.051 [inf][dimos/mapping/pgo.py          ] Loop closure detected score=0.0111 source=1006 target=105
keyframes: 1007
loops fired: 41  max pose_jump: 1.082 m
```


![output](assets/pgo_trajectories.svg)

![output](assets/pgo_map.svg)

## Per-frame voxels ingest coste

```python session=pgo output=assets/pgo_frame_ms.svg
from dimos.memory2.transform import smooth
from dimos.memory2.vis.plot.plot import Plot

(
    Plot()
    .add(frame_ms.offset(10).transform(smooth(20)),
         label="voxels add_frame (ms)")
    .to_svg("{output}")
)
```

<!--Result:-->
![output](assets/pgo_frame_ms.svg)

## Loop closures: spatial correction & ICP fitness

Each loop closure event yields one sample on each side-stream:

- `pose_jump_m` — the worst per-keyframe translation correction PGO
  applied (i.e. how far the most-shifted past pose moved). A small
  number means the graph was already nearly-consistent; a big number
  means PGO undid significant accumulated drift.
- `loop_score` — ICP fitness of the matched submap pair (lower is
  better).

```python session=pgo output=assets/pgo_loop_events.svg
(
    Plot()
    .add(pose_jump_m, label="max pose shift (m)", gap_fill=30.0)
    .add(loop_score, label="ICP fitness", gap_fill=30.0)
    .to_svg("{output}")
)
```

<!--Result:-->
![output](assets/pgo_loop_events.svg)
