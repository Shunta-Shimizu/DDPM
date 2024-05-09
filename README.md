# DDPM
## Denoising-Diffusion-Probabilistic-Models

### 環境構築
```
git clone https://github.com/Shunta-Shimizu/DDPM.git
conda create -n ddpm python=3.9
pip install requirements.txt
```

### データセット
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
- train.py test.pyをコマンドラインから引数を受け取れるようにする
- Residual block, Attention Blockを導入したU-netの構築
