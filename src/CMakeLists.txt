find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

set(SOURCES vehicle-tracker.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
add_definitions(-Wall -Wextra -pedantic -Werror -g)
add_executable(vehicletracker ${SOURCES} ${HEADERS})
target_link_libraries(vehicletracker ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# Needed for boost log lib. Another option would be -DBOOST_LOG_DYN_LINK.
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost REQUIRED COMPONENTS log)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(vehicletracker ${Boost_LIBRARIES})

install(TARGETS vehicletracker RUNTIME DESTINATION bin)
