# Deformation-Flow-Based-Two-stream-Network


## Introduction   

This is the implementation of the proposed method in "Deformation Flow Based Two-Stream Network for Lip Reading". Our paper can be found [here](https://arxiv.org/pdf/2003.05709.pdf).

## Dependencies
* Python 3.5+
* PyTorch 1.0+
* Others
## Dataset
This model is trained on LRW with grayscale lip images.   
## Training And Testing
You can train or test the model as follow:
```
python train.py options.toml
```
or
```
python test.py options.toml
```


## Reference

If this work is useful for your research, please cite our work:

```
  title={Deformation Flow Based Two-Stream Network for Lip Reading},
  author={Xiao, Jingyun and Yang, Shuang and Zhang, Yuanhang and Shan, Shiguang and Chen, Xilin},
  booktitle={2020 15th IEEE International Conference on Automatic Face & Gesture Recognition (FG 2020)},
  year={2020},
  organization={IEEE}
}
```
