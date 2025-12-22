include("${CMAKE_CURRENT_LIST_DIR}/Matrix/Matrix.cmake")

add_executable(
	Matrix_test
	"${CMAKE_CURRENT_LIST_DIR}/main.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/main.hpp"
)

target_link_libraries(Matrix_test PRIVATE Matrix MatrixHelpers)

