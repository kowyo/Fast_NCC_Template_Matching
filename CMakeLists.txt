cmake_minimum_required(VERSION 3.10)
project(AngleEnhancedTemplateMatching)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories for OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable for the project
add_executable(${PROJECT_NAME} main.cpp)

# Link OpenCV libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
