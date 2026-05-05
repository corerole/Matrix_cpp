cmake_minimum_required (VERSION 3.10)

if(MSVC)
  if (POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
  endif()
endif()

project ("Matrix_test")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

#include_directories("${CMAKE_CURRENT_LIST_DIR}/src")
#include_directories("${CMAKE_CURRENT_LIST_DIR}/src/Matrix")
#add_executable (Matrix_test "src/main.cpp")
include("${CMAKE_CURRENT_LIST_DIR}/src/main.cmake")