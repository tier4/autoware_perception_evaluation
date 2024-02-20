cmake_minimum_required(VERSION 3.5)
project(simple_lanelet_loader)

find_package(ament_cmake_auto REQUIRED)
find_package(ament_cmake_python REQUIRED)
ament_auto_find_build_dependencies()

ament_python_install_package(${PROJECT_NAME})

ament_auto_package()
