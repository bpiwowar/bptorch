CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)
project(bptorch)

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")

FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(CUDA 4.0 REQUIRED)
MESSAGE(STATUS "${BLAS_LIBRARIES}")

add_definitions(
  "-DFOLLY_NO_CONFIG"
  "-DFOLLY_HAVE_CLOCK_GETTIME"
  )

add_subdirectory(thpp/thpp)


IF(APPLE)
  # work around for mac os x bug:
  # http://stackoverflow.com/questions/16286588/cuda-5-0-cmake-and-make-failing-on-osx-10-8-3
  if (NOT DEFINED CUDA_HOST_COMPILER AND EXISTS /usr/bin/gcc)
    set(CUDA_HOST_COMPILER /usr/bin/gcc CACHE FILEPATH "Host side compiler used by NVCC")
    message(STATUS "Setting CMAKE_HOST_COMPILER to /usr/bin/gcc instead of ${CMAKE_C_COMPILER}.")
  endif()
ENDIF()


SET(src-cpp src/fblualib/LuaUtils.cpp src/cpu/HSM.cpp src/cpu/HSoftMax.cpp src/cpu/init.cpp src/cuda/HSMHost.cpp src/cuda/HSoftMaxHost.cpp)


SET(src-cuda src/cuda/init.cu src/cuda/HSM.cu src/cuda/HSoftMax.cu ${src-cpp})
SET(luasrc src/lua/init.lua src/lua/HSM.lua src/lua/HSoftMax.lua src/lua/MOperator.lua)
source_group(lua FILES ${luasrc})

SET_SOURCE_FILES_PROPERTIES(${src-cpp} PROPERTIES COMPILE_FLAGS -std=c++11 )


INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}/THC" "${CMAKE_CURRENT_SOURCE_DIR}/thpp" "${CMAKE_CURRENT_SOURCE_DIR}/thpp/thpp" "${CMAKE_CURRENT_SOURCE_DIR}/src")
LINK_DIRECTORIES("${Torch_INSTALL_LIB}")

LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_20")

CUDA_ADD_LIBRARY(bptorch MODULE ${src-cuda} ${luasrc})
TARGET_LINK_LIBRARIES(bptorch luaT THC TH luajit)


### Torch packages supposes libraries prefix is "lib"
SET_TARGET_PROPERTIES(bptorch PROPERTIES
  PREFIX "lib"
  IMPORT_PREFIX "lib")

INSTALL(TARGETS bptorch
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

INSTALL(
  FILES
  ${luasrc}
  DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/bptorch")
