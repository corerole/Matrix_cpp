#include "main.hpp"

int main() {
	Matrix<float> A(4, 4);
	A[0][0] = 2.0f; A[0][1] = -0.5f; A[0][2] = 1.0f; A[0][3] = 0.0f;
	A[1][0] = 1.0f; A[1][1] = -1.0f; A[1][2] = 0.0f; A[1][3] = 2.0f;
	A[2][0] = 0.0f; A[2][1] = 1.0f; A[2][2] = 2.0f; A[2][3] = -1.0f;
	A[3][0] = -2.0f; A[3][1] = 0.0f; A[3][2] = 1.0f; A[3][3] = -1.0f;
	auto&& sub = A.get_submatrix(1, 1, 3, 3);

	helpers_tests();
	return 0;
}
