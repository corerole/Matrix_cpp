if(USE_HANDMADE_STD_MODULE)
	include("${CMAKE_CURRENT_LIST_DIR}/std/std.cmake")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/Matrix/Matrix.cmake")

#file(GLOB_RECURSE CMAKE_MODULES_INCLUDE_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/*/*.cmake")
#foreach(cmake_module_file ${CMAKE_MODULES_INCLUDE_FILES})
#    include("${cmake_module_file}")
#endforeach()