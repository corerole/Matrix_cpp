#include "main.hpp"
#include <chrono>
#include <cmath>
#include <numbers>
#include <random>

template<auto V> struct End {
	inline constexpr bool operator==(auto&& x) const { return *x == V; }
};

template<typename T> using distribution = std::conditional_t<
	std::floating_point<T>,
	std::uniform_real_distribution<T>,
	std::uniform_int_distribution<T>
>;

int main() {
	auto mt = std::mt19937(std::random_device{}());

	auto gen = [&mt]<typename T = unsigned>() -> T {
		auto min = static_cast<T>(1);
		auto max = static_cast<T>(100);
		auto dist = distribution<T>(min, max);
		return std::invoke(dist, mt);
	};

	Matrix<float> A(4, 4);
	A[0][0] = 2.0f; A[0][1] = -0.5f; A[0][2] = 1.0f; A[0][3] = 0.0f;
	A[1][0] = 1.0f; A[1][1] = -1.0f; A[1][2] = 0.0f; A[1][3] = 2.0f;
	A[2][0] = 0.0f; A[2][1] = 1.0f; A[2][2] = 2.0f; A[2][3] = -1.0f;
	A[3][0] = -2.0f; A[3][1] = 3.0f; A[3][2] = 1.0f; A[3][3] = -1.0f;
	std::cout << "Matrix A:" << std::endl;
	A.print();


	Matrix<float> B(4, 4);
	B[0][0] = 1.0f; B[0][1] = 2.0f; B[0][2] = 3.0f; B[0][3] = 3.0f;
	B[1][0] = -4.0f; B[1][1] = 5.0f; B[1][2] = 6.0f; B[1][3] = 3.0f;
	B[2][0] = 1.0f; B[2][1] = 0.0f; B[2][2] = -2.0f; B[2][3] = 3.0f;
	B[3][0] = 3.0f; B[3][1] = 3.0f; B[3][2] = 3.0f; B[3][3] = 3.0f;
	std::cout << "Matrix B:" << std::endl;
	B.print();

	Matrix<float> C(3, 3);
	C[0][0] = 1.0f;  C[0][1] = 2.0f; C[0][2] = 3.0f;
	C[1][0] = -4.0f; C[1][1] = 5.0f; C[1][2] = 6.0f;
	C[2][0] = 1.0f;  C[2][1] = 0.0f; C[2][2] = -2.0f;


#if 0
	auto x = A.col(0);
	std::cout << "ColProxy iterator: ";
	std::for_each(x.begin(), x.end(), [](auto&& it) { std::cout << it << " "; });
	std::cout << std::endl;

	std::cout << "ColProxy const_iterator: ";
	std::for_each(x.cbegin(), x.cend(), [](auto&& it) { std::cout << it << " "; });
	std::cout << std::endl;

	std::cout << "ColProxy reverse_iterator: ";
	std::for_each(x.rbegin(), x.rend(), [](auto&& it) { std::cout << it << " "; });
	std::cout << std::endl;

	std::cout << "ColProxy const_reverse_iterator: ";
	std::for_each(x.crbegin(), x.crend(), [](auto&& it) { std::cout << it << " "; });
	std::cout << std::endl;

	auto y = A[0];
	std::cout << "RowProxy iterator: ";
	std::for_each(y.begin(), y.end(), [](auto&& it) { std::cout << it << " "; });
	std::cout << std::endl;

	std::cout << "RowProxy const_iterator: ";
	std::for_each(y.cbegin(), y.cend(), [](auto&& it) { std::cout << it << " "; });
	std::cout << std::endl;

	std::cout << "RowProxy reverse_iterator: ";
	std::for_each(y.rbegin(), y.rend(), [](auto&& it) { std::cout << it << " "; });
	std::cout << std::endl;

	std::cout << "RowProxy const_reverse_iterator: ";
	std::for_each(y.crbegin(), y.crend(), [](auto&& it) { std::cout << it << " "; });
	std::cout << std::endl;
#endif

#if 0
	constexpr auto accumulate = [](const std::ranges::input_range auto& coll) -> auto {
		using type = typename std::ranges::range_value_t<std::decay_t<decltype(coll)>>;
		auto acc = static_cast<type>(0);
		for (auto&& x : coll) { acc += x; }
		return acc;
	};

	auto r = A.row(3);
	auto&& sr = std::ranges::subrange(r.begin() + 1, End<-1.0f>{});
	auto&& cv = std::ranges::common_view{ sr };
	auto b = cv.begin();
	auto e = cv.end();

	// for (auto&& z : sr) { std::cout << z << " "; }
	// for (; b != e; ++b) { std::cout << *b << std::endl; }
	// std::cout << std::endl;
	std::cout << accumulate(sr) << std::endl;
#if 0
	auto c = A.col(3);
	auto&& csr = std::ranges::subrange(c.begin() + 1, End<-1.0f>{});
	for (auto&& z : csr) { std::cout << z << std::endl; }
#endif
#endif

#if 1
	constexpr size_t M = 500;
	constexpr size_t N = 600;
	constexpr size_t K = 700;
	constexpr auto much = 2000U;

	auto cgen = [&mt]() {
		auto dist = std::uniform_real_distribution(-100.0L, 100.0L);
		return std::invoke(dist, mt);
	};

	auto J = Matrix<long double>(much, much);
	std::ranges::generate(J, cgen);
	auto P = Matrix<long double>(M, K);
	std::ranges::generate(P, gen);
	auto O = Matrix<long double>(K, N);
	std::ranges::generate(O, gen);

	std::cout << "Filled! " << std::endl;
#endif

#if 0
	size_t c_n = 3;
	auto c = A.get_col_as_vector(c_n);
	std::cout << " A.col " << c_n << " : ";
	for (auto&& x : c) { std::cout << x << " "; }
	std::cout << std::endl;

	size_t r_n = 2;
	auto r = A.get_row_as_vector(r_n);
	std::cout << " A.row " << r_n << " : ";
	for (auto&& x : r) { std::cout << x << " "; }
	std::cout << std::endl;
#endif

#if 0
	size_t x_n = 3;
	std::cout << "A.get_row_as_matrix_row(" << x_n << ") " << std::endl;
	A.get_row_as_matrix_row(x_n).print();

	std::cout << "A.get_row_as_matrix_col(" << x_n << ") " << std::endl;
	A.get_row_as_matrix_col(x_n).print();

	std::cout << "A.get_col_as_matrix_row(" << x_n << ") " << std::endl;
	A.get_col_as_matrix_row(x_n).print();

	std::cout << "A.get_col_as_matrix_col(" << x_n << ") " << std::endl;
	A.get_col_as_matrix_col(x_n).print();
#endif

#if 0
	std::ranges::for_each(A.row_const_reverse_range(), [](auto&& it) {
		std::ranges::for_each(it, [](auto&& _it) {
			std::cout << _it << " ";
		});
		std::cout << std::endl;
	});
#endif

#if 0
	std::cout << "Infinity norm: " << helpers::infinity_norm(A) << std::endl;
	std::cout << "Mean col: " << helpers::mean(A.col(3)) << std::endl;
	std::cout << "Mean A: " << helpers::mean(A) << std::endl;
#endif
	// auto&& [X, Y, Z] = helpers::svd_jacobi(A);

#if 0
	// test concept utils::arithmetic_input_range
	struct Z { int* a = nullptr; };
	auto false_ = std::vector<Z>(10);
	auto true_ = std::vector<int>(10);
	auto f = helpers::max(false_);
	auto t = helpers::max(true_);
#endif

#if 0
	// test concept utils::Container
	std::ranges::for_each(helpers::col_means(A), [](auto&& it) { std::cout << it << " "; });
	std::cout << std::endl;
	std::ranges::for_each(helpers::row_means(A), [](auto&& it) { std::cout << it << " "; });
	std::cout << std::endl;
#endif

#if 0
	std::cout << helpers::matrix_euclid_max_col_norm(A) << std::endl;
#endif

	using complex_t = typename std::complex<long double>;
	auto x = std::vector<complex_t>(10);
	std::ranges::for_each(x, [&gen] (auto&& it) { it = gen(); });
	auto F = Matrix<complex_t>(10, 10);
	auto z = helpers::matrix_euclid_max_col_norm(F);

	auto euclid_n = helpers::euclid_norm(x);
	std::cout << euclid_n << std::endl;

#if 0
	auto E = utils::to_complex(A);
#endif

	auto&& [U, Sigma, Vh] = helpers::svd_jacobi_real(J);
	U.print();
	Sigma.print();
	Vh.print();
	// auto&& [Q, R] = helpers::householder_qr_decomposition(J);

	return 0;
}
