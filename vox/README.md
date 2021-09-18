# 1. binvox
## Introduction
**binvox** is a straight-forward program that reads a 3D model file, 
rasterizes it into a binary 3D voxel grid, and writes the resulting voxel file.
## Usage

- input formats:
    - nearly 100% VRML 2.0 support
    - will parse Wavefront OBJ, Geomview OFF, Autocad DXF, PLY and STL, if they contain polygons only
VRML 1.0 support added in version 1.08 (still in alpha).
- output formats:
    - [.binvox](https://www.patrickmin.com/binvox/binvox.html), HIPS, [MIRA](https://www.patrickmin.com/binvox/mira.html), VTK, a "raw" file format, [minecraft](https://www.patrickmin.com/minecraft) .schematic, [Gmsh](https://geuz.org/gmsh/) .msh, and [nrrd](http://teem.sourceforge.net/nrrd/)
- rasterizes to a cube grid of up to 512×512×512.

# 2. viewvox
## Introduction
**viewvox** is a program that reads a 3D voxel file as produced by binvox or thinvox and shows it in a window. You can then use the mouse to move the camera around the model.

## Usage

- input formats:
    - [.binvox](https://www.patrickmin.com/binvox/binvox.html), as produced by the binvox 3D mesh voxelizer
    - [MIRA](https://www.patrickmin.com/binvox/mira.html)
    - [nrrd](http://teem.sourceforge.net/nrrd/)
- example
```markdown
viewvox   [-ki] <model filename>

    -ki: keep internal voxels (removed by default)

    Mouse left button = rotate
    middle      = pan
    right       = zoom
    Key   r     = reset view
    arrow keys  = move 1 voxel step along x (left, right) or y (up, down)
    =,-         = move 1 voxel step along z
    q           = quit
    a           = toggle alternating colours
    p           = toggle between orthographic and perspective projection
    u           = set z axis up
    x, y, z     = set camera looking down X, Y, or Z axis
    X, Y, Z     = set camera looking up X, Y, or Z axis
    1           = toggle show x, y, and z coordinates
    s           = show single slice
    n           = show both/above/below slice neighbour(s)
    t           = toggle neighbour transparency
    j           = move slice down
    k           = move slice up
    G           = toggle show grid
    g           = toggle show grid at slice level
    Alt + g     = change grid orientation
```
# 3. meshconv
## Introduction
meshconv converts to and from several popular 3D file formats. Currently only geometry conversion is supported.

## Usage
- input and output formats:
    - nearly 100% VRML 2.0 support (input 2.0, output 1.0 and 2.0)
    - simple geometry only support for Wavefront OBJ, Geomview OFF, Autocad DXF, PLY, 3DS (reading only), three.js JSON (writing only), XML3D (writing only), and STL.
- example
    - Run meshconv without parameters for a usage summary.
    - Example: to convert a VRML 2.0 model to PLY, triangulated: 
      ```shell script
       meshconv -c ply -tri mymodel.wrl
      ```

# 4. Reference
```markdown
@article{nooruddin03,
  author = {Fakir S. Nooruddin and Greg Turk},
  title = {Simplification and Repair of Polygonal Models Using Volumetric Techniques},
  journal = {IEEE Transactions on Visualization and Computer Graphics},
  volume = {9},
  number = {2},
  pages = {191--205},
  year = {2003}
}
```
```markdown
@Misc{binvox,
  author = {Patrick Min},
  title =  {binvox},
  howpublished = {{\tt http://www.patrickmin.com/binvox} or {\tt https://www.google.com/search?q=binvox}},
  year =  {2004 - 2019},
  note = {Accessed: yyyy-mm-dd}
}
```

```markdown
@Misc{meshconv,
  author = {Patrick Min},
  title =  {meshconv},
  howpublished = {{\tt http://www.patrickmin.com/meshconv} or {\tt https://www.google.com/search?q=meshconv}},
  year =  {1997 - 2019},
  note = {Accessed: yyyy-mm-dd}
}
```

