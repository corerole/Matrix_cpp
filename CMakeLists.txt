cmake_minimum_required (VERSION 4.00)

#if(MSVC)
#  if (POLICY CMP0141)
#    cmake_policy(SET CMP0141 NEW)
#    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
#  endif()
#endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project ("Matrix_test" LANGUAGES CXX)
add_executable(Matrix_test)
include("${CMAKE_CURRENT_LIST_DIR}/dep/dep.cmake")
target_sources(Matrix_test PUBLIC "${CMAKE_CURRENT_LIST_DIR}/main.cpp")