cmake_minimum_required(VERSION 3.8)
project(flipper_training)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# The Python package the ROS nodes import from.
ament_python_install_package(marv_rl_training
  PACKAGE_DIR marv_rl_training
)

install(PROGRAMS
  ros2/flipper_policy_node.py
  ros2/mitriakov_policy_node.py
  ros2/send_goal.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
