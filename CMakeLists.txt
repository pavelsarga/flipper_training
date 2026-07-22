cmake_minimum_required(VERSION 3.8)
project(flipper_training)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python package (only the ros2 subpackage for ROS nodes)
ament_python_install_package(${PROJECT_NAME}
  PACKAGE_DIR flipper_training
)

# Install Python executables
install(PROGRAMS
  ros2/flipper_policy_node.py
  ros2/send_goal.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
