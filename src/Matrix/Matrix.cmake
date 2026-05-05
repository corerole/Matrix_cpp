add_library(Matrix STATIC "${CMAKE_CURRENT_LIST_DIR}/Matrix.cpp" "${CMAKE_CURRENT_LIST_DIR}/Matrix.hpp")
add_library(MathMatrix STATIC "${CMAKE_CURRENT_LIST_DIR}/MathMatrix.cpp" "${CMAKE_CURRENT_LIST_DIR}/MathMatrix.hpp")
target_link_libraries(MathMatrix PRIVATE Matrix)

add_library(MatrixHelpers STATIC "${CMAKE_CURRENT_LIST_DIR}/Matrix_helpers.cpp" "${CMAKE_CURRENT_LIST_DIR}/Matrix_helpers.hpp")
target_link_libraries(MatrixHelpers PRIVATE MathMatrix)