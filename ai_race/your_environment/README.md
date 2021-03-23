# Cチーム 4Q

## 概要

4Qの課題が「１つのモデルで春夏秋冬全てのコースを走らせること」なので、走行環境に依存しないモデルが求められる。
そこで、CチームではAutoencoderによる事前学習モデルを採用し、環境変化にロバストなモデルの作成を目指す。

## モデル構造

追記

## 学習の流れ

### 0. データの取得

#### a. VAEの学習データ

* 実行スクリプト: `vehicle_pos.py`

* 使用方法:

  追記

#### b. 模倣学習の学習データ

* 実行スクリプト: `keyboard_con_pygame_videosave.py`

* 使用方法:

  追記

### 1. VAE (Variational Autoencoder)の学習

まずは、事前学習としてVAE(または単なるAE)を学習させる。
VAEでは、春夏秋冬の4コースそれぞれのカメラ画像を入力とし、通常コースのカメラ画像を教師データとする。
そうすることにより、春夏秋冬の異なる画像から通常コースと共通する特徴量を抽出することができる。

#### a. データを用意

`/home/jetson/`の直下に春夏秋冬＋通常コースのデータを用意する。
具体的には、以下のようなファイル構造を作成する。
また、同じ位置・角度で撮影したカメラ画像のファイル名は全てのコースで同じものにしておく。

```bash
$ cd /home/jetson
$ tree .
.
├── catkin_ws
├── Image_from_rosbag
├── normal
│   └── images
│      └── *.jpg
├── spring
│   └── images
│       └── *.jpg
├── summer
│   └── images
│       └── *.jpg
├── autumn
│   └── images
│       └── *.jpg
└── winter
    └── images
        └── *.jpg
```

#### b. 学習スクリプトを実行

学習を実行するためのコマンドをいかに示す。
事前学習では単なるAEではなく、VAEを使用する。
そのため、オプションとして必ず`--variational`をつける。

```bash
$ cd /home/jetson/catkin_ws/src/ai_race/ai_race/your_environment/scripts
$ python3 train_ae.py --variational --num_z 256 --wd 0.0
```

**オプションの説明**

`model_name`: 保存するモデルのファイル名\
`model_path`: 保存するモデルのディレクトリ\
`variational`: VAE (store_true) or AE (otherwise)\
`result_path`: reconstructed imagesを保存するディレクトリ\
`batch_size`: バッチサイズ\
`num_z`: 潜在変数の数\
`n_epoch`: エポック数\
`lr`: 学習率\
`wd`: 正則化(L2)の大きさ\
`save_model_interval`: モデルを保存する頻度\
`inf_model_interval`: 推論して、reconstructed imagesを保存する頻度\
`num_inf`:　reconstructed imagesの数

### 2. 模倣学習

上記、VAEの学習で保存したモデルを転移学習する形で、模倣学習を行う。

#### a. データを用意

`/home/jetson/`の直下に春夏秋冬コースを走行して取得したデータを用意する。
具体的には、以下のようなファイル構造を作成する。

```bash
$ cd /home/jetson
$ tree .
.
├── catkin_ws
├── Image_from_rosbag
│   ├── _2021-*
│   │   ├── _2021-*.csv
│   │   └── images
│   │       └── *.jpg
│   ├── _2021-*
│   │   ├── _2021-*.csv
│   │   └── images
│   │       └── *.jpg
│   ├── _2021-*
│   │   ├── _2021-*.csv
│   │   └── images
│   │       └── *.jpg
│   ├── _2021-*
│   │   ├── _2021-*.csv
│   │   └── images
│   │       └── *.jpg
│   └── _2021-*
│       ├── _2021-*.csv
│       └── images
│           └── *.jpg
├── normal
├── spring
├── summer
├── autumn
└── winter
```

#### b. 学習スクリプトを実行

```bash
$ cd /home/jetson/catkin_ws/src/ai_race/ai_race/your_environment/scripts
$ python3 train_race.py  --num_z 256 --wd 0.0 --pretrained_model /home/jetson/ai_race/ai_race/your_environment/scripts/models/vae_mse_z256_ckpt_25.pth
```

**オプションの説明**

`data_path`: 学習データが保存されているパス\
`model_name`: 保存するモデルのファイル名\
`model_path`: 保存するモデルのディレクトリ\
`pretrained_model`: 事前学習モデルのパス\
`batch_size`: バッチサイズ\
`num_z`: 潜在変数の数\
`n_epoch`: エポック数\
`lr`: 学習率\
`wd`: 正則化(L2)の大きさ\
`save_model_interval`: モデルを保存する頻度\

## 学習結果

### 1. VAEの学習結果

|     | Target | Normal | Spring | Summer | Autumn | Winter |
| --- | --- | --- | --- | --- | --- | --- |
| 画像 | ![001373_00085_-2600_0000](https://user-images.githubusercontent.com/52629908/111936268-ac33b080-8b08-11eb-8ef4-6d44f73ec076.jpg) | ![001373_00085_-2600_0000_pred_normal](https://user-images.githubusercontent.com/52629908/111936350-d4231400-8b08-11eb-8718-14e9d69662a6.jpg) | ![001373_00085_-2600_0000_pred_spring](https://user-images.githubusercontent.com/52629908/111936389-e735e400-8b08-11eb-9700-1dc9d81343b3.jpg) | ![001373_00085_-2600_0000_pred_summer](https://user-images.githubusercontent.com/52629908/111936413-f157e280-8b08-11eb-9198-1c11a1b2880e.jpg) | ![001373_00085_-2600_0000_pred_autumn](https://user-images.githubusercontent.com/52629908/111936415-f2890f80-8b08-11eb-8612-5bfee0bcd944.jpg) | ![001373_00085_-2600_0000_pred_winter](https://user-images.githubusercontent.com/52629908/111936410-f0bf4c00-8b08-11eb-8cc1-b9370bda9629.jpg) |


### 2. 模倣学習の学習結果

追記

## ファイル構造

実行環境のファイル構造を以下に示す。ソースコードはyour_environmentのpkgに実装する。

```bash
$ cd /home/jetson/catkin_ws/src/ai_race/ai_race/your_environment
$ tree.
.
├── CMakeLists.txt
├── README.md
├── launch
│   └── sim_environment.launch
├── package.xml
└── scripts
    ├── autoencoder.py
    ├── inference_from_image.py
    ├── make_datasets.py
    ├── models
    │   ├── control_z256_ckpt_25.pth
    │   └── vae_mse_z256_ckpt_25.pth
    ├── recognet.py
    ├── train_ae.py
    ├── train_funcs.py
    ├── train_race.py
    ├── utils.py
    └── vehicle_pos.py
```
