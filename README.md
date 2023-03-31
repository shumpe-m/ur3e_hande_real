# ur3e_hande_real
## Docker set up
- Dockerを使ったセットアップは次のコードをターミナル上で使用。
```
git clone https://github.com/shumpe-m/ur3e_hande_real.git
cd ur3e_hande_real/Docker
./build.sh
./run.sh
```

## Examples

それぞれのターミナルで以下のコードを起動

### ur driver
URの出荷時のキャリブレーションを書き出す。
```
roslaunch ur_calibration calibration_correction.launch robot_ip:=192.168.1.103 target_filename:="${HOME}/my_robot_calibration.yaml"
```

先程のキャリブレーションを読み出しながらur3e_bringupを起動。ipアドレスはTPで設定している値。

```
roslaunch ur_robot_driver ur3e_bringup.launch robot_ip:=192.168.1.103 kinematics_config:="${HOME}/my_robot_calibration.yaml"
```

起動後、TPで再生ボタンを押しリモートにすることでPC側から通信を行えるようになる。


### rviz

rvizを起動。

```
roslaunch ur_hande_moveit_config start_moveit.launch
```

### script

まず、rviz内の接触回避するオブジェクトを呼び出す。

```
rosrun ur_control_scripts rviz_setup.py 
```

続いて、Moveit！もしくはソケット通信を用いてURへ命令を送るスクリプトを起動することで動かすことができる。

```
rosrun ur_control_scripts main.py 
```

