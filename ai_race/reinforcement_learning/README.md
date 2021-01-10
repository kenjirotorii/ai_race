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

## 報酬関数

強化学習の最も難しい部分が報酬関数の設計である。強化学習が上手くいくかは報酬関数の設計次第といっても過言ではない。
以下に上手く学習できたときの報酬関数を示す。これはあくまで一例であり、ベストなものではない。

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

上の例では、車両の中心とセンターラインの距離に応じた報酬を与えている。センターラインから大きく離れていなければ報酬`+1`を与え、外枠ギリギリの場合には報酬`0`を、外枠をはみ出した場合には報酬`-1`を与える。また、センターラインから大きく離れた場合にはコースアウトとしてその episode を終了する。
