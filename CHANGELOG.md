# RTX Mega Geometry SDK Change Log

## 2.1.0

Platform

* Build for Windows on Arm64 via the `vs2022-arm64` preset, either cross-compiled from
  an x64 host or natively; binaries go to `bin_arm64` and installs to `_install_arm64`
* The release zip carries both architectures, `bin` and `bin_arm64`, next to one shared
  `assets` folder; the `run_*.bat` scripts pick the one that matches the machine
* Update Streamline to 2.14.1, NVAPI to R615 and Nsight Aftermath to 2026.3, the first
  releases that ship Arm64 binaries
* Update DXC to 1.9.2609; Arm64 build machines use the native Arm64 shader compilers
* Require CMake 4.1 or later
* Show the CPU architecture in the window title, e.g. `(D3D12 arm64)`

Bug Fixes

* The DLSS preset menus only offer presets Streamline still supports; the RR entry
  labelled Preset G selected Preset F

## 2.0.0

Adds a second geometry path: **Cluster LOD**. Pre-baked triangle clusters are selected
each frame from a continuous LOD hierarchy and streamed into VRAM on demand, so a scene's
source geometry no longer has to fit in VRAM — detail is bounded by a memory budget
rather than by mesh count. It shares the material table, TLAS and scene graph with the
existing tessellation path, so a single scene can use both.

The path is scene-driven — load a `.gltf` or `.glb` model and it runs. See
[ClusterLOD.md](docs/ClusterLOD.md).

Cluster LOD

* Offline bake, cached to a per-geometry `_nvsngeo` cluster cache so that only the first
  load of an asset pays for it, optionally with arithmetic-packed vertex data unpacked
  per group at load
* GPU-driven residency streaming against fixed geometry and CLAS pools, with a persistent
  CLAS allocator, plus a preloaded reference path
* Adaptive LOD error raises the effective pixel error while the pools run hot, so a scene
  larger than its budget settles coarser instead of thrashing
* BLAS sharing, caching and merging, all on by default, to keep acceleration-structure
  build cost from scaling with instance count
* Frustum and Hierarchical-Z occlusion culling feeding back into LOD selection

Content

* Load glTF 2.0 scenes: PBR metallic-roughness materials, KTX2 textures, alpha-masked and
  two-sided geometry, `EXT_mesh_gpu_instancing`, and `KHR_materials_transmission` /
  `KHR_materials_ior` for thin-walled glass
* Cap material texture memory, applied at scene load by dropping high-resolution mips
* Shade Cluster LOD hits with normal maps, with a tangent frame derived from the hit
  triangle's du/dv rather than baked per-vertex tangents
* Replicate a scene's instances into a grid to build a heavy scene from a small asset

UI and Profiling

* Add a Streaming profiler tab: pool occupancy, resident groups and clusters, transfer
  and load rates, and per-frame BLAS builds
* Split the BVH profiler tab into ClusterTess BVH and ClusterLOD BVH; each is hidden when
  its geometry path is not in the scene
* Add the Geometry Inspector window: every geometry in the scene with per-LOD residency
  detail, selectable by picking a surface in the viewport
* Add Cluster LOD colour modes for LOD level, cluster group, BLAS source and whether a
  BLAS came from the cache
* Add a Cluster LOD section to the Settings window covering shading, LOD selection,
  culling and BLAS reuse, and a Bake Config window for the bake settings
* Add a VRAM Budget window that owns every memory budget — texture, Cluster LOD pool and
  tessellation — and plots current against proposed consumption over the card's VRAM,
  including the driver-reported process total. Budgets that change what is read off disk
  commit through a confirmation dialog and a scene reload; the pool budgets apply as they
  are edited

## 1.0.1

Improvements
* Add smooth vertex normal support which allows for lower tessellation rates and includes memory profiler row for vertex normals
* Filter out degenerate normals for backface culling to improve backface test and reduce triangle count

Debugging Improvements
* Improve shader debug with stable cluster index output from path tracing shader
* Add surface highlighting feature where path-tracer will blink highlight a selected debug surface
* Add utilities to shader debug to force output

Bug Fixes
* Fix regression in surface 1-ring culling caused by 1D thread-ordering change, which resulted in incorrect lanes doing work for unrelated waves
* Fix crash when refreshing media list after changing json/obj filters with an asset already loaded

## 1.0.0

Vulkan Support
* Requires Vulkan SDK 1.4.313 or greater which uses Cluster SPIRV intrinsics
* Convert all bindless arrays to use ResourceDescriptorHeap via Vulkan's mutable descriptor extension
* Fix validation warnings and Vulkan shutdown crashes

Minor Changes
* Expose isolation level in the UI
* Add smooth single crease sharpness, which prevents transition artifacts between single and multicrease edges. Only visible if the isolation level is lowered.
* Improve Profiler "Frame" Tab, add average times in tool tip, expose motion vector pass time
* Update to Streamline v2.8.0

Bug Fixes
* Fix thread-ordering to be compliant with SM6.6 1D quad lane ordering.
* Fix micro-triangle view toggle when in DLAA mode.
* Fix cases where a surface resulted in over U16_MAX clusters
* Fix tessellation for when there was per-material displacement scaling
* Fix crash for some malformed OBJ files with '0' values for some indices.

## 0.9.2

Performance Improvements 

Test Scene: amy_kitchenset.scene.json (default camera: 79M microtriangles) 
Hardware: RTX 5090 @ 4K Render Resolution (r572.83)

* Compute Cluster tiling: 4.0ms -> 1.0ms (300% speedup)
    * Coalesced UAV writes for structs, unaligned members were causing UAV readbacks
    * Coalesce per wave atomics into a groupshared atomic to reduce pressure on global/UAV atomics by 4x
* Fill clusters: 4.9ms to -> 1.0ms (390% speedup)
    * Fixed cases where the compiler was unable to unroll loops due to dynamic loop counts
    * Specialized shaders by subdivision surface type, with a special path for Pure BSpline surfaces. Prefetch all control points into shared memory to be used wave wide.

Minor Fixes

* Fix SpecularHitT guide buffer to DLSS-RR to improve coherence of specular rays.

## 0.9.1

Stability
* Fix crash on scenes with multiple subdivision mesh instances when the topology quality color mode was selected
* Fix a crash if scene with audio is loaded and no audio devices are present.

Topology Quality
* Add button in the Subdivision Evaluator tab to switch to topology quality view if issues are detected
* Fix Subdivision Evaluator UI not resetting upon scene load.

UI
* Application Window Size/Maximized/Fullscreen and Window state is now saved to and restored from imgui.ini. Delete imgui.ini to reset layout

Minor
* Make initial VRAM check non-fatal but add warnings about performance degradation, memory budgets.
* Made localToWorld transform use Matrix3x4 for consistency
* Style clean-up
* Update donut version

## 0.9.0

Initial beta release.
