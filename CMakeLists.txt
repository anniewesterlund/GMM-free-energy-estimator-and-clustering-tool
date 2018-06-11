cmake_minimum_required (VERSION 3.8.0)
project (GMM_free_energy)
set(CMAKE_BUILD_TYPE Release)

include_directories(include)
file(GLOB SOURCES "src/run_GMM.cpp")

add_executable(GMM_free_energy ${SOURCES})

# The version number.
set (GMM_free_energy_VERSION_MAJOR 1)
set (GMM_free_energy_VERSION_MINOR 0)

SET(CMAKE_CXX_COMPILER g++-6.1.0)

# Set the location for library installation: /usr/local/bin
install(TARGETS GMM_free_energy DESTINATION bin)