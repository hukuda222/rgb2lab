# rgb2lab

2018年度 PFNサマーインターンシップの選考課題、chianer/問題1を解いたものです。

https://github.com/pfnet/intern-coding-tasks/tree/master/2018/chainer

RGBの画像をLab色の画像に変換するものです。
openCVでランダムに生成したRGB色の画像と、それをopenCVでLab色に変換したものを教師データとして学習します。


学習には以下のように実行します。学習終了後にmodel.npzを出力します。
```
python train_RGB2Lab.py
```
以下のように実行することで、ランダムに生成したRGB色の画像とそれをこれを用いてLab色に変換した画像、openCVでLab色にした画像を出力します
```
python train_RGB_Lab.py --model model.npz --image
```

train_Lab2RGB.pyは逆にLab色の画像をRGBの画像に変換します。

### ランダムに生成したRGBの画像
<img src="https://raw.githubusercontent.com/hukuda222/rgb2lab/master/result/input_image.png" width="200"/>

### これを用いてLab色に変換した画像
<img src="https://raw.githubusercontent.com/hukuda222/rgb2lab/master/result/output_image.png" width="200"/>

### openCVでLab色に変換した画像
<img src="https://raw.githubusercontent.com/hukuda222/rgb2lab/master/result/ideal_image.png" width="200"/>
