# markerLocation

  一个基于marker的6dof定位程序，marker使用了AprilGrid。


## compile
 Put the source code in the catkin workspace，then execute compile command.
 ```
  cd ~/catkin_ws/src
  git clone [source code]
  cd ~/catkin_ws
  catkin_make --pkg marker_location
```
    
## run
-  First,you should run the camera node which publish image topic like "/cam0/image_raw".
```
  rosrun your_camera_pkg your_camera_node
```
- Then run the marker_location node
```
  source ./devel/setup.bash
  roslaunch marker_location test.launch
```
    
## others
  This program tested on Ubuntu16.04.
