# newton-clips

> Clips from [newton-physics](https://github.com/newton-physics/newton) simulation to Unreal Engine 5 runtime

- exchange the simulation data with [NewtonClips](https://github.com/doidio/NewtonClips), a twin UE5 plugin
- replace renderers in `newton-physics` and convert simulation data
- support `newton-physics` examples with the least code change

## Install

```
pip install newton-clips
```

## Getting started

1. run `newtonclips/example.py`
2. find generated simulation data in `newtonclips.SAVE_DIR`
3. use this directory in [NewtonClips](https://github.com/doidio/NewtonClips)

## How to run `newton.examples`

```python
import newtonclips  # replace newton renderers implicitly
newtonclips.SAVE_DIR = '.clips'  # set directory to save simulation data

# make sure you have installed the necessary external libraries
from newton.examples import example_anymal_c_walk_on_sand as example
import runpy
runpy.run_path(example.__file__, run_name='__main__')
```
