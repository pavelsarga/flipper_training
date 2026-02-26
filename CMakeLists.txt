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
  flipper_training/ros2/flipper_policy_node.py
  flipper_training/ros2/send_goal.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(DIRECTORY sota_configs/
  DESTINATION share/${PROJECT_NAME}/config
)

# Install robot configs (needed by flipper_training package)
install(DIRECTORY robots/
  DESTINATION lib/python3.10/site-packages/robots
)

# Install robot configs (needed by flipper_training package)
install(DIRECTORY meshes/
  DESTINATION lib/python3.10/site-packages/meshes
)

install(PROGRAMS
    ros2/flipper_policy_node.py
    DESTINATION lib/${PROJECT_NAME}
)

ament_package()
