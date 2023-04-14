# PhD topic

This repository contains the code I used for figures and analysis presented on the defense of the topic of the doctoral thesis.

To reproduce the results, the easiest way is to create a local environment by using `conda` as
```shell
conda create --name phd-dissertation-topic python=3.9.12
```
and, inside the environment, within `code` repository, run the following command
```shell
pip install -r requirements.txt
```
to install all dependencies listed in `requirements.txt`.

## Contents

| Directory | Subdirectory/Contents | Description |
|:---:|:---:|:---:|
| `artwork` |  | All visuals used for the defense of the topic of the doctoral thesis. |
| 1 | file name tba | Desc tba. |
| `code` |  | Notebooks and files used for generating most of the figures within `artwork` directory. |
| 1 | `data` | Data placeholder. |
| 2 | `figures` | Placeholder for visuals generated by using Jupyter notebooks listed below. |
| 3 | `src` | A module containing the set of documented Python functions used for data manipulation, processing, numerical analysis and visualization. |
| 4 | 00-canonical-tissue-models.ipynb | Planar, spherical and cylindrical tissue model for EMF exposure assessment. |
| 5 | 01-evaluation-surfaces.ipynb | Planar, spherical and cylindrical evaluation surface used for the extraction of the spatially averaged absorbed/incident power density. |
| 6 | 02-unit-normal-estimation-pca.ipynb | Principal component analysis for the estimation of the unit vector field normal to any surface. |
| 7 | 03-curvature-normal-estimation-interp.ipynb | Estimation of the surface vector field normal to any observed surface. |
| 8 | 04-anatomical-tissue-model.ipynb | Estimation of the surface (curvature) normals on the surface of the realistic ear model. Model is given as a 3-D point cloud whose coordinates are stored in ear.xyz file within `data` directory. |
| 9 | 05-hotspot-region-detection.ipynb | Automatic detection of the peak spatially averaged power density on a non-planar surface over which the distribution of the absorbed EMF is inhomogeneous. |
| `defense` |  | Placeholder for the presentation used during the defense of the topic of the doctoran thesi. |
| 1 | PrezentacijaTemeDoktorskogRada.pptx | PowerPoint slides. |
| `docs` |  | Misc files related to the application for the defense. |
| `papers` |  | Papers of which I am the principal author or co-author and are related to the topic of the doctoral research. All listed papers are published before the official notification of the topic of the doctoral thesis to the FESB/UniSt Committee for postgraduate studies (March 10, 2023). |


 ## License

 [MIT](https://en.wikipedia.org/wiki/MIT_License) except for figures extracted directly from journals in `artwork` directory which are protected under [CC-BY](https://en.wikipedia.org/wiki/Creative_Commons_license) license protection.
