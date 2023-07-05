[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/OpenSenseAction/OPENSENSE_sandbox_environment/main?urlpath=git-pull?repo=https://github.com/OpenSenseAction/OPENSENSE_sandbox%26urlpath=lab/tree/OPENSENSE_sandbox/notebooks/index.ipynb&branch=main)  :point_left: Click here to run the examples online. (please retry if the build process does not finish after some minutes)

<img src="https://user-images.githubusercontent.com/102827/174779884-a2fb0971-4850-4ad6-93eb-2c53b922b408.svg" alt="drawing" width="300"/>

# OPENSENSE software sandbox
A collection of software packages for processing data from opportunistic rainfall sensors, developed within the COST Action [OPENSENSE](https://opensenseaction.eu/)

This is **currently WIP** and just a showcase of how existing codebases and existing open datasets can be combined in a reproducible environment and run online via mybinder.

## Run code online

Even though this repo is WIP, feel free to try out the preliminary notebooks by clicking on the button "lauch binder" above. Note that it can take some minutes to build the online environment in case mybinder has no cached version of it.

## Run environment locally

To run the code locally and/or to contribute to this repository, you need to set up the `conda` environment defined in the file `environment.yml` (located in the [seperated environment repository](https://github.com/OpenSenseAction/OPENSENSE_sandbox_environment)) which is also used by mybinder to build the environment that you run online.

First, you need to have `conda` installed. If this is you first installation of `conda` we recommend to start with the `mambaforge` installer which is available for Windows, Linux and Mac [here](https://github.com/conda-forge/miniforge#mambaforge). Note that "mamba" is just a faster implementation of "conda", and "forge" refers to the fact that you will use the community packages from the [conda-forge "channel"](https://conda-forge.org/), which has a  larger choice of scientific Python and R packages than the default conda "channel".

With `conda` or `mamba` set up, follow these steps:
1. Clone this repo and its git submodules to your machine. Or, if you plan to contribute, first create a fork of it and clone from this fork (you have to adjust the URL below). Note that `git clone` will create a new directory OPENSENSE_sandbox in the directory that you are currently in and place the repo content there.
   ```
   git clone --recursive https://github.com/OpenSenseAction/OPENSENSE_sandbox.git
   ```
2. Stay in the directory of step 1, or change to it if required. Then download the `environment.yml` file from the [seperated environment repository](https://github.com/OpenSenseAction/OPENSENSE_sandbox_environment) via `wget` like this
    ```
   wget https://raw.githubusercontent.com/OpenSenseAction/OPENSENSE_sandbox_environment/main/environment.yml
    ```
    or via `curl`
   ```
   curl -O https://raw.githubusercontent.com/OpenSenseAction/OPENSENSE_sandbox_environment/main/environment.yml
   ```
   If neither `wget` nor `curl` are available, you can also download the environment file via the browser, but make sure to save it in the directory to which you cloned the sandbox repo in step 1.
3. Create the `conda` environment. Note that you have to be in a terminal/shell where `conda` or `mamba` is available. Please refer to the `conda` docs to find out how to achieve that.
   ```
   mamba env create environment.yml
   ```
4. Activate the env.
   ```
   conda activate opensense_sandbox
   ```
5. Install `jupyter-lab` in addition. It is not in then `environment.yml` because mybinder installs it by default.
   ```
   mamba install -c conda-forge jupyterlab
   ```
6. Run `jupyter-lab`. It will open in your default browser.
   ```
   jupyter-lab
   ```
7. Go to the index notebook in `notebooks/index.ipynb` and run the cell with the init script, using the line of code for local installation which is commented out by default. This will clone the code of the submodules, add them to the conda env and install the R packages.

## Contributing

We encourages everyone to contribute to the developement of the OPENSENSE_sandbox.

The easiest way is to fork the repository and submit a pull request (PR). Each PR automatically gets its own mybinder button to test everything in its defined environment. PRs will be iterated with and merged by the repository maintainers. If required we can also have dev branches in this repository.

Note that, if you have to change the dependencies, you can update your local `conda` env based on an updated `environment.yml` file like this (see [here](https://stackoverflow.com/a/43873901/356463))
```
conda activate myenv
conda env update --file environment.yml --prune
```

## Code of Conduct

Contributors to the OPENSENSE_sandbox are expected to act respectfully toward others in accordance with the [OSGeo Code of Conduct](http://www.osgeo.org/code_of_conduct).

## Contributions and Licensing

All contributions shall comply to the project [license](LICENSE). The individual included packages might have their own license, which has to be compatible with the one of the project license.
