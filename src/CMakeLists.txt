cmake_minimum_required(VERSION 2.8)
project(vehicletracker)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

set(SOURCES vehicle-tracker.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
add_definitions(-Wall -Wextra -g)
add_executable(vehicletracker ${SOURCES} ${HEADERS})
target_link_libraries(vehicletracker ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(vehicletracker ${Boost_LIBRARIES})

install(TARGETS vehicletracker RUNTIME DESTINATION bin)
