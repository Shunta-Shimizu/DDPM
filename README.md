# DDPM
## Denoising-Diffusion-Probabilistic-Models

### Installation
```
git clone https://github.com/Shunta-Shimizu/DDPM.git
cd DDPM
conda create -n ddpm python=3.9
pip install requirements.txt
```

### Dataset
- CelebA
- ImageNet
- Places365 Standard

### Train
```
python train.py --train_data_dir ./ --save_model_dir ./ 
````

### Test
```
python test.py --test_data_dir ./ --model_path ./ --save_result_dir ./
```

### Tasks
- test.pyを書く
- train.py test.pyでコマンドライン引数を受け取れるようにする
- Residual block, Attention Blockを導入したU-netの構築
