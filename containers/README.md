# Description
The blueprints needed to build a container that runs RL Benchmarks is provided here.  The containers are guranteed to work on any Linux OS system.

# Build

To build the `rl_benchmarks.sif` image you need to execute the following:

```bash
singularity build core.sif core.def
singularity build miniconda.sif miniconda.def
singularity build mujoco.sif mujoco.def
singularity build rl_benchmarks.sid rl_benchmarks.def
```

>[!NOTE]
You can execute `build.sh` which will automatically build the images in the order described above.

If you have a Ray Cluster up you can pass the Ray address and Ray temporary directory into the **%runscript** of the container:

```bash
singularity run --tmp-sandbox --writable --nv tests/speed_test.py <IP:PORT> <TMP_DIR>
singularity run --tmp-sandbox --writable --nv tests/solve_test.py <IP:PORT> <TMP_DIR>
```
