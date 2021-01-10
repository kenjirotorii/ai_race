# Reinforcement Learning

## 概要

強化学習を実装するパッケージ。今回はネット上に実装例が豊富にある`Deep Q-learning(DQN)`というモデルを使用した。

## 追加の設定

車両の位置情報を取得するための gazebo プラグイン`pd3`を使用する。(`/gazebo/model_state`でも位置情報を取得できるが更新頻度がデフォルトで 1000 Hz と大きいので今回は使用しない)

`ai_race/ai_race/sim_environment/urdf/wheel_robot.urdf.xacro`に以下の設定を追加する。

```
<gazebo>
    <plugin name="ground_truth" filename="libgazebo_ros_p3d.so">
        <frameName>world</frameName>
        <bodyName>base_link</bodyName>
        <topicName>/tracker</topicName>
        <updateRate>10.0</updateRate>
    </plugin>
</gazebo>
```

## 動作環境

Gazebo を起動した状態でネットワークの学習も同時にやるので、できるだけ処理能力のあるマシンで実行することが望ましいが、jetson nano でも学習スピードは遅いものの自動走行できるくらいには学習させることができる。

### 動作確認済み

#### 1. Docker on Ubuntu 18.04

Ubuntu 18.04 上の Docker で動作することを確認済。高田さん docker を少し修正する必要あり。（環境によって異なる）

- CPU : Ryzen 7 3700X
- GPU : RTX2060 Super
- Memory: 32 GB

#### 2. Jetson Nano

jetson nano での動作確認済み

## 実行方法

強化学習の実行方法を以下に示す。

### 1. Gazebo の起動

```bash
$ cd ~/catkin_ws/src/ai_race/scripts
$ bash prepare.sh
```

### 2. 強化学習スクリプトを実行

```bash
$ cd ~/catkin_ws/src/ai_race/ai_race/reinforcement_learning/scripts
$ python car_operation.py
```

## 強化学習のパラメータ設定

強化学習スクリプトの実行部でいくつかのパラメータを設定する必要がある。

```python
if __name__ == "__main__":

    SAVE_MODEL_PATH = '../model_weight/dqn_20210109.pth'
    LOAD_MODEL_PATH = '../model_weight/dqn_20210108.pth'

    # parameters
    NUM_ACTIONS = 2
    CAPACITY = 2500
    BATCH_SIZE = 32
    LR = 0.0005
    GAMMA = 0.99
    TARGET_UPDATE = 2
```

`SAVE_MODEL_PATH` : モデルの保存先のパス

`LOAD_MODEL_PATH` : 読み込むモデルのパス

`NUM_ACTIONS` : 操作の数(car_operation.py では`2`でしか動作しない。car_operation_analog.py では自由に設定可能)

`CAPACITY` : Experience Replay でのメモリの容量

`LR` : ネットワークの学習率

`GAMMA` : 割引率

`TARGET_UPDATE` : target のネットワークを更新する頻度

また、Q-learning ではオンライン学習が望ましいが、シミュレータを動かしながらそれをするのにはある程度のマシンスペックが必要となるため、マシンスペックの低い jetson nano でも動作できるようにオンライン学習をするかを設定できるようにした。(`online=False`によりオンライン学習をしないように設定できる)

```python
car_bot = CarBot(save_model_path=SAVE_MODEL_PATH, pretrained=False, load_model_path=LOAD_MODEL_PATH, online=True)
```

オフライン学習の場合、各エピソードの終了時にまとめてネットワークの学習を行うようになっている。

## 報酬関数

強化学習の最も難しい部分が報酬関数の設計である。強化学習が上手くいくかは報酬関数の設計次第といっても過言ではない。

### ベースラインの報酬関数

車両の中心とセンターラインの距離に応じた報酬を与えている。センターラインから大きく離れていなければ報酬`+1`を与え、外枠ギリギリの場合には報酬`0`を、外枠をはみ出した場合には報酬`-1`を与える。また、センターラインから大きく離れた場合にはコースアウトとしてその episode を終了する。

```python
def get_reward(self, pose):
    # distance from center line
    dist_from_center = distance_from_centerline(pose)

    if dist_from_center < 0.4:
        return 1.0
    elif dist_from_center < 0.45:
        return 0.0
    elif dist_from_center < 0.6:
        return -1.0
    else:
        course_out = True
        return -1.0
```

### ベストスコアの報酬関数

できる限りインラインのギリギリを走行させたいので、インラインからの距離に応じた報酬を与えた。

```python
def get_reward(self, pose):
    # distance from innner line
    dist_from_inline = distance_from_inline(pose)

    if dist_from_inline < -0.10:
        self.course_out = True
        rospy.loginfo('Course Out !!')
        return -1.0
    elif dist_from_inline < 0:
        return -1.0
    elif dist_from_inline < 0.25:
        return 1.0
    elif dist_from_inline < 0.5:
        return 0.0
    elif dist_from_inline < 0.9:
        return -1.0
    else:
        self.course_out = True
        rospy.loginfo('Course Out !!')
        return -1.0
```

## 学習済モデル

推論のスクリプトで動作確認できているモデル

- `dqn_202104_2.pth` : 27-28 周/4 分
- `dqn_202109.pth` : 30-31 周/4 分
- `dqn_202109_jetson.pth` : 29-30 周/4 分
