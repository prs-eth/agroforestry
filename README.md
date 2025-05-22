# The unrealized potential of agroforestry for an emissions-intensive agricultural commodity

This is the code accompanying our paper by Alexander Becker, Jan D. Wegner, Evans Dawoe, Konrad Schindler, William J. Thompson, Christian Bunn, Rachael D. Garrett, Fabio Castro, Simon P. Hart, Wilma J. Blaser-Hart.

[[Link to paper]](https://arxiv.org/abs/2410.20882) 
[[Link to interactive maps]](https://albecker.users.earthengine.app/view/agroforestry)
[[Link to checkpoints & maps]](https://share.phys.ethz.ch/~pf/albecker/agroforestry/)

## Code organization

```
* ./shade: code for training and inference of the shade cover model
    * reproject.py: build a reprojected dataset from raw input images
    * train_gbr.py: train a GBR regressor from the dataset
    * predict.py: run inference with a trained model
* ./agbd: code adapted from Lanfranchi et al. (2022) for biomass estimation
```

## Getting started
The code requires Python 3.9 (i.e. installed via conda), then install all requirements:
```
pip install -r shade/requirements.txt
```

## Citation
```
@article{becker2024agroforestry,
  title={The unrealized potential of agroforestry for an emissions-intensive agricultural commodity},
  author={Becker, Alexander and Wegner, Jan D and Dawoe, Evans and Schindler, Konrad and Thompson, William J and Bunn, Christian and Garrett, Rachael D and Castro, Fabio and Hart, Simon P and Blaser-Hart, Wilma J},
  journal={arXiv preprint arXiv:2410.20882},
  year={2024}
}
```
