# Graph Gain Exploration

## Concave-Hull Induced Graph-Gain for Fast and Robust Robotic Exploration

The official implementation of "Concave-Hull Induced Graph-Gain for Fast and Robust Robotic Exploration"

<img width="613" alt="image" src="https://github.com/IMRL/Graph_Gain_Exploration/assets/67741955/04eec4af-a2ce-4417-b34e-9b0b6b85e1b4">

## Update

06/02/2023, The paper has been submitted for presentation on RAL 2023.
07/01/2023, The code has been released.

## Installation

Follow the dependence of the following work: 

- [LeGO-LOAM](https://github.com/RobustFieldAutonomyLab/LeGO-LOAM)
- [autonomous_exploration_development_environment](https://github.com/HongbiaoZ/autonomous_exploration_development_environment)
- [Lidar-road-atlas](https://github.com/IMRL/Lidar-road-atlas)
- [DSV Planner](https://github.com/HongbiaoZ/dsv_planner)

## Usage

Launch the following nodes in sequence:

- Start Lidar and robot control drive
- roslaunch vehicle_simulator system_real_robot.launch 
- roslaunch lidar_atlas run.launch
- rosrun chunkmap_terrain chunkmap_terrain_node
- roslaunch dsv_launch dsvp.launch