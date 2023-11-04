# Create a pgm image and the associated yaml as used by ros1 move_base from a point cloud map in a pcd file.
All credits to [Hinson-A](https://github.com/Hinson-A/pcd2pgm_package).


# Build
```
catkin_make pcd2pgm
```

# Example use
```
rosrun pcd2pgm pcd2topic _pcd_file:=src/pcd2pgm/data/909-1.2-ds.pcd _thre_z_min:=0.2 _thre_z_max:=0.6
rosrun map_server map_saver # save the pgm and yaml

```

The resulting pgm looks like [this](./data/map.pgm).


