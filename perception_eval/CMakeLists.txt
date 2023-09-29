cmake_minimum_required(VERSION 3.5)
project(perception_eval)

find_package(ament_cmake_auto REQUIRED)
find_package(ament_cmake_python REQUIRED)
ament_auto_find_build_dependencies()

set(ROS_REQUIREMENTS "${CMAKE_CURRENT_SOURCE_DIR}/ros_requirements.txt")
execute_process(COMMAND bash -c "pip install -r ${ROS_REQUIREMENTS}")

ament_python_install_package(${PROJECT_NAME})

ament_auto_package()
