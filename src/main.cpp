#include "main.hpp"
#include <memory_resource>

template<typename T> using pmrMatrix = Matrix<T, std::pmr::polymorphic_allocator<T>>;

int main() {
	/*
	*  Ctors;
	*/
	using mtx = Matrix<char>;
	mtx a1;
	mtx a2(3, 4); // def
	mtx a3(a2); // copy
	mtx a4(std::move(a3)); // move
	mtx a5 = a2; // ass copy
	mtx a6 = std::move(a2); // ass move


	constexpr size_t BS = 4000;
	std::array<std::byte, BS> buf;
	std::pmr::monotonic_buffer_resource mbr(buf.data(), buf.size());
	std::pmr::polymorphic_allocator alloc(&mbr);

	pmrMatrix<float> A(4, 4, alloc);
	A[0][0] = 2.0f; A[0][1] = -0.5f; A[0][2] = 1.0f; A[0][3] = 0.0f;
	A[1][0] = 1.0f; A[1][1] = -1.0f; A[1][2] = 0.0f; A[1][3] = 2.0f;
	A[2][0] = 0.0f; A[2][1] = 1.0f; A[2][2] = 2.0f; A[2][3] = -1.0f;
	A[3][0] = -2.0f; A[3][1] = 0.0f; A[3][2] = 1.0f; A[3][3] = -1.0f;

	return 0;
}
