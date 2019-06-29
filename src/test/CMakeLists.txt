find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

set(SOURCES_MISC misc.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
add_definitions(-Wall -Wextra -pedantic -Werror -g)
add_executable(misc ${SOURCES_MISC} ${HEADERS})
target_link_libraries(misc ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# Needed for boost log lib. Another option would be -DBOOST_LOG_DYN_LINK.
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost REQUIRED COMPONENTS log)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(misc ${Boost_LIBRARIES})

install(TARGETS misc RUNTIME DESTINATION bin)
