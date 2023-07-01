cmake_minimum_required(VERSION 2.8.3)
project(lego_loam)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=skylake")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)

find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  rospy
  cv_bridge
  image_transport

  pcl_ros
  pcl_conversions

  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  cloud_msgs
  visualization_msgs
  nodelet
  #ndt_omp
)

find_package(GTSAM REQUIRED QUIET)
find_package(PCL REQUIRED QUIET)
find_package(OpenCV REQUIRED QUIET)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS cloud_msgs
  DEPENDS PCL
)

include_directories(
  include
  .
  ./ORB_SLAM2/include
  ./ORB_SLAM2
	${catkin_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${GTSAM_INCLUDE_DIR}
)

link_directories(
	include
	${OpenCV_LIBRARY_DIRS}
	${PCL_LIBRARY_DIRS}
	${GTSAM_LIBRARY_DIRS}
)

# message(FATAL_ERROR ${GTSAM_VERSION})
#
add_executable(imageProjection src1/imageProjection.cpp)
add_dependencies(imageProjection ${catkin_EXPORTED_TARGETS} cloud_msgs_gencpp)
target_link_libraries(imageProjection ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(featureAssociation src1/featureAssociation.cpp)
add_dependencies(featureAssociation ${catkin_EXPORTED_TARGETS} cloud_msgs_gencpp)
target_link_libraries(featureAssociation ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

# add_executable(mapOptmization src/mapOptmization.cpp)
# target_link_libraries(mapOptmization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam)

# add_executable(mapOptmization src1/mapOptmization_loop.cpp ORB_SLAM2/wy_keyframedb.cpp ORB_SLAM2/wy_keyframe.cpp include/fftm/fftm.cpp src1/LidarIris.cpp)

# add_executable(mapOptmization src1/mapOptmization_loop_iris.cpp include/fftm/fftm.cpp src1/LidarIris.cpp)
add_executable(mapOptmization src1/mapOptmization_loop_iris_submap.cpp include/fftm/fftm.cpp src1/LidarIris.cpp)
target_link_libraries(mapOptmization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam
)

add_executable(transformFusion src1/transformFusion.cpp)
target_link_libraries(transformFusion ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})
