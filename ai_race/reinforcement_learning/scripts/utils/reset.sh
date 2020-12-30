#!/bin/bash

rostopic pub -1 /gazebo/set_model_state gazebo_msgs/ModelState '{model_name: wheel_robot_spawn, pose: { position: { x: 1.6, y: 0, z: 0 }, orientation: {x: 0, y: 0, z: 0.7071067811865476, w: 0.7071067811865476} }, twist: { linear: { x: 0, y: 0, z: 0 }, angular: { x: 0, y: 0, z: 0}  }, reference_frame: world }'

bash ~/catkin_ws/src/ai_race/judge/request_to_judge.sh -s init