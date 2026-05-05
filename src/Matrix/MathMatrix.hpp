#ifndef MATHMATRIX_HPP
#define MATHMATRIX_HPP
#pragma once

#include "Matrix.hpp"

namespace math_matrix {
	template<typename T, size_t SZ> using array_like_row = std::array<T, SZ>;
	template<typename T, size_t SZ> using array_like_col = std::array<T, SZ>;
	template<typename T, typename Alloc> using vector_like_row = std::vector<T, Alloc>;
	template<typename T, typename Alloc> using vector_like_col = std::vector<T, Alloc>;

	template<typename T> struct is_Complex : std::false_type {};
	template<typename T> struct is_Complex<std::complex<T>> : std::true_type {};
	template<typename T> constexpr inline bool is_Complex_v = is_Complex<T>::value;
	template<typename T> concept ComplexLike = is_Complex_v<T>;

	template<typename T> struct is_matrix : std::false_type {};
	template<typename T, typename Alloc> struct is_matrix<Matrix<T, Alloc>> : std::true_type {};
	template<typename T> inline constexpr bool is_matrix_v = is_matrix<T>::value;
	
	template<typename T> struct is_vector : std::false_type {};
	template<typename T, typename Alloc> struct is_vector<std::vector<T, Alloc>> : std::true_type {};
	template<typename T, size_t SZ> struct is_vector<std::array<T, SZ>> : std::true_type {};
	template<typename T> inline constexpr bool is_vector_v = is_vector<T>::value;

	template<typename T> concept arithmetic = std::is_arithmetic_v<T>;

	template<typename T> struct is_allocator : std::true_type {};
	template<typename T> inline constexpr bool is_allocator_v = is_allocator<T>::value;
	template<typename T> concept AllocatorLike = is_allocator_v<T>;

	template<typename T> struct is_Submatrix : std::false_type {};
	template<typename T> struct is_Submatrix<Submatrix<T>> : std::true_type {};
	template<typename T> struct is_Submatrix<const_Submatrix<T>> : std::true_type {};
	template<typename T> inline constexpr bool is_Submatrix_v = is_Submatrix<T>::value;
	template<typename T> concept SubmatrixLike = is_Submatrix_v<T>;

	template<typename T> struct is_Row : std::false_type {};
	template<typename T> struct is_Row<RowProxy<T>> : std::true_type {};
	template<typename T> struct is_Row<const_RowProxy<T>> : std::true_type {};
	template<typename T> inline constexpr bool is_Row_v = is_Row<T>::value;
	template<typename T> concept RowLike = is_Row_v<T>;

	template<typename T> struct is_Col : std::false_type {};
	template<typename T> struct is_Col<ColProxy<T>> : std::true_type {};
	template<typename T> struct is_Col<const_ColProxy<T>> : std::true_type {};
	template<typename T> inline constexpr bool is_Col_v = is_Col<T>::value;
	template<typename T> concept ColLike = is_Col_v<T>;

	template<typename T> concept MatrixLike = is_matrix_v<T> || is_Submatrix_v<T>;
	template<typename T> concept VectorLike = is_vector_v<T>;
	template<typename T> concept ScalarLike = arithmetic<T> || ComplexLike<T>;

	void print(const MatrixLike auto& mtx) {
		int precision = 6;
		std::ostream& os = std::cout;
		const auto m = mtx.rows();
		const auto n = mtx.cols();
		os << std::fixed << std::setprecision(precision);
		for (size_t i = 0; i < m; ++i) {
			os << "[ ";
			for (size_t j = 0; j < n; ++j) {
				os << std::setw(10) << mtx[i][j];
				if (j + 1 < n) os << ' ';
			}
			os << " ]" << std::endl;
		}
		os << std::endl;
	}

#if 0
	auto operator/(const MatrixLike auto& lhs, const ScalarLike auto& rhs) {
		auto new_mtx = lhs;
		using mtx_value_type = std::decay_t<decltype(new_mtx)>::value_type;
		const auto val = static_cast<mtx_value_type>(rhs);
		std::ranges::for_each(new_mtx.def_range(), [&val](auto& it) { it /= val; });
		return new_mtx;
	}
#else
	auto operator/(const MatrixLike auto& mtx, const ScalarLike auto& scalar) {
		auto r = mtx.rows();
		auto c = mtx.cols();
		auto tmp = mtx;
		for (size_t i = 0; i < r; ++i) {
			auto&& tmp_row_i = tmp.row(i);
			for (size_t j = 0; j < c; ++j) {
				auto& x = tmp_row_i[j];
				x = x / scalar;
			}
		}
		return tmp;
	}
#endif

	auto operator-(const MatrixLike auto& lhs, const MatrixLike auto& rhs) {
		using lhs_value_type = std::decay_t<decltype(lhs)>::value_type;
		using rhs_value_type = std::decay_t<decltype(rhs)>::value_type;
		using res_value_type = std::common_type_t<lhs_value_type, rhs_value_type>;
		Matrix<res_value_type, std::pmr::polymorphic_allocator<res_value_type>> res(lhs.rows(), lhs.cols());

#if 0
		auto l_b = res.begin();
		auto l_e = res.end();
		auto r_b = rhs.begin();

		for (; l_b != l_e; ++l_b, ++r_b) {
			*l_b -= static_cast<lhs_value_type>(*r_b);
		}
#else
		auto cols = res.cols();
		auto rows = res.rows();
		for (size_t i = 0; i < rows; ++i) {
			auto n_row = res.row(i);
			auto l_row = lhs.row(i);
			auto r_row = rhs.row(i);
			for (size_t j = 0; j < cols; ++j) {
				n_row[j] = l_row[j] - r_row[j];
			}
		}
#endif

		return res;
	}

	auto operator+(const MatrixLike auto& lhs, const MatrixLike auto& rhs) {
		auto new_mtx = lhs;
		using lhs_value_type = std::decay_t<decltype(new_mtx)>::value_type;
		using rhs_value_type = std::decay_t<decltype(rhs)>::value_type;

		auto l_b = new_mtx.begin();
		auto l_e = new_mtx.end();
		auto r_b = rhs.begin();

		for (; l_b != l_e; ++l_b, ++r_b) {
			*l_b += static_cast<lhs_value_type>(*r_b);
		}

		return new_mtx;
	}

#if 0
	auto operator+(MatrixLike auto& it, const MatrixLike auto& rhs) {
		return it - scalar;
	}
#endif

	inline auto operator*(const MatrixLike auto& lhs, const ScalarLike auto& rhs) {
		using mtx_value_type = std::decay_t<decltype(lhs)>::value_type;
		Matrix<mtx_value_type> new_mtx(lhs); //lhs.rows(), lhs.cols());
		const auto val = static_cast<mtx_value_type>(rhs);
		std::ranges::for_each(new_mtx.def_range(), [&val](auto& it) { it *= val; });
		return new_mtx;
	}

	inline auto operator*(const ScalarLike auto& lhs, const MatrixLike auto& rhs) {
		return rhs * lhs;
	}

	auto col_x_row(std::ranges::random_access_range auto&& col, std::ranges::random_access_range auto&& row, const auto& allocator) {
		using col_value_type = std::ranges::range_value_t<std::decay_t<decltype(col)>>;
		using row_value_type = std::ranges::range_value_t<std::decay_t<decltype(row)>>;
		using res_value_type = std::common_type_t<col_value_type, row_value_type>;
		using res_allocator_type = rebind_allocator<std::decay_t<decltype(allocator)>, res_value_type>;

		Matrix<res_value_type, res_allocator_type> res(col.size(), row.size(), allocator);

		for (std::size_t i = 0; i < col.size(); ++i) {
			auto&& res_col_i = res.col(i);
			auto& col_i = *(col.begin() + i);
			for (std::size_t j = 0; j < row.size(); ++j) {
				auto& row_j = *(row.begin() + j);
				res_col_i[j] = col_i * row_j;
			}
		}
		return res;
	}


	auto col_x_row(std::ranges::random_access_range auto&& col, std::ranges::random_access_range auto&& row) {
		return col_x_row(col, row, std::pmr::polymorphic_allocator<std::byte>{});
	}

	auto operator*(const ColLike auto& col, const RowLike auto& row) {
		return col_x_row(col, row);
	}

	auto col_x_mtx(const ColLike auto& col, const MatrixLike auto& mtx) {
		return col_x_row(col, mtx.row(0));
	}

	auto operator*(const ColLike auto& col, const MatrixLike auto& mtx) {
		return col_x_mtx(col, mtx);
	}

	auto row_x_col(const RowLike auto& row, const ColLike auto& col) {
		using value_type = std::decay_t<decltype(row)>::value_type;
		constexpr auto zero = static_cast<value_type>(0);
		auto result = zero;
		for (size_t i = 0; i < row.size(); ++i) {
			const auto& fst = row[i];
			const auto& snd = col[i];
			result += fst * snd;
		}
		return result;
	}

	auto operator*(const RowLike auto& row, const ColLike auto& col) {
		return row_x_col(row, col);
	}


	constexpr void vec_x_scalar_(std::ranges::forward_range auto&& r, const ScalarLike auto& scalar) {
		using value_type = std::ranges::range_value_t<decltype(r)>;
		const auto val = static_cast<value_type>(scalar);
		std::ranges::for_each(r, [val](auto& it) { it *= val; });
	}

	inline auto vec_x_scalar(
		const VectorLike auto& vec,
		const ScalarLike auto& scalar,
		const AllocatorLike auto& allocator
	) {
		using vec_value_type = std::decay_t<decltype(vec)>::value_type;
		using scalar_value_type = std::decay_t<decltype(scalar)>;
		using result_value_type = std::common_type_t<vec_value_type, scalar_value_type>;
		using result_allocator_type = rebind_allocator<std::decay_t<decltype(allocator)>, result_value_type>;
		std::vector<result_value_type, result_allocator_type> result(vec.begin(), vec.end());
		vec_x_scalar_(result, scalar);
		return result;
	}

	inline auto vec_x_scalar(
		const VectorLike auto& vec,
		const ScalarLike auto& scalar
	) {
		return vec_x_scalar(vec, scalar, std::pmr::polymorphic_allocator<std::byte>{});
	}

	inline auto operator*(const VectorLike auto& vec, const ScalarLike auto& scalar) {
		return vec_x_scalar(vec, scalar);
	}

	inline auto row_x_mtx(std::ranges::random_access_range auto&& row, const MatrixLike auto& mtx, const AllocatorLike auto& allocator) {
		using row_value_type = std::ranges::range_value_t<decltype(row)>;
		using mtx_value_type = std::decay_t<decltype(mtx)>::value_type;
		using res_value_type = std::common_type_t<row_value_type, mtx_value_type>;
		using res_allocator_type = rebind_allocator<std::decay_t<decltype(allocator)>, res_value_type>;
		Matrix<res_value_type, res_allocator_type> result_(1, mtx.cols());

		const auto rows = mtx.rows();
		const auto cols = mtx.cols();
		auto result = result_.row(0);
#if 1
		for (std::size_t j = 0; j < cols; ++j) {
			auto mtx_col = mtx.col(j);
			auto& res_j = result[j];
			for (std::size_t i = 0; i < rows; ++i) {
				auto& row_i = *(row.begin() + i);
				res_j += row_i * mtx_col[i];
			}
		}
#else
		for (size_t i = 0; i < rows; ++i) {
			auto mtx_row = mtx.row(i);
			for (size_t j = 0; j < cols; ++j) {
				result[j] += row[i] * mtx_row[j];
			}
		}
#endif

		return result_;
	}

	inline auto row_x_mtx(std::ranges::random_access_range auto&& row, const MatrixLike auto& mtx) {
		return row_x_mtx(row, mtx, std::pmr::polymorphic_allocator<std::byte>{});
	}

	inline auto operator*(const RowLike auto& row, const MatrixLike auto& mtx) {
		return row_x_mtx(row, mtx);
	}

	constexpr void vec_x_vec_(
		std::ranges::forward_range auto&& v1,
		std::ranges::forward_range auto&& v2,
		std::ranges::forward_range auto&& v3
	) {
		for (auto&& [e_v1, e_v2, e_v3] : std::views::zip(v1, v2, v3)) {
			e_v3 = e_v1 * e_v3;
		}
	}
	
	inline auto vec_x_vec(
		std::ranges::forward_range auto&& v1,
		std::ranges::forward_range auto&& v2,
		const AllocatorLike auto& allocator
	) {
		using v1_value_type = std::ranges::range_value_t<decltype(v1)>;
		using v2_value_type = std::ranges::range_value_t<decltype(v1)>;
		using res_value_type = std::common_type_t<v1_value_type, v2_value_type>;
		using res_allocator_type = rebind_allocator<std::decay_t<decltype(allocator)>, res_value_type>;
		std::vector<res_value_type, res_allocator_type> result(v1.size());
		vec_x_vec_(v1, v2, result);
		return result;
	}

	inline auto vec_x_vec(
		std::ranges::forward_range auto&& v1,
		std::ranges::forward_range auto&& v2
	) {
		return vec_x_vec(v1, v2, std::pmr::polymorphic_allocator<std::byte>{});
	}

#if 1
	inline auto operator*(
		const VectorLike auto& v1,
		const VectorLike auto& v2
	) {
		return vec_x_vec(v1, v2);
	}
#endif

	inline auto mtx_x_col(
		const MatrixLike auto& mtx,
		std::ranges::random_access_range auto&& col,
		const AllocatorLike auto& allocator
	) {
		using col_value_type = std::ranges::range_value_t<std::decay_t<decltype(col)>>;
		using mtx_value_type = std::decay_t<decltype(mtx)>::value_type;
		using res_value_type = std::common_type_t<col_value_type, mtx_value_type>;
		using res_allocator_type = rebind_allocator<std::decay_t<decltype(allocator)>, res_value_type>;
		Matrix<res_value_type, res_allocator_type> res(mtx.rows(), 1, allocator);
		auto cols = mtx.cols();
		auto rows = mtx.rows();
		auto rcol = res.col(0);

		for (size_t i = 0; i < rows; ++i) {
			auto mtx_row = mtx.row(i);
			auto& col_i = rcol[i];
			for (size_t j = 0; j < cols; ++j) {
				auto& col_j = *(col.begin() + j);
				col_i += mtx_row[j] * col_j;
			}
		}

		return res;
	}

	inline auto mtx_x_col(
		const MatrixLike auto& mtx,
		std::ranges::random_access_range auto&& col
	) {
		return mtx_x_col(mtx, col, std::pmr::polymorphic_allocator<std::byte>{});
	}

	auto operator*(const MatrixLike auto& mtx, const ColLike auto& col) {
		return mtx_x_col(mtx, col);
	}

#if 0
	inline auto vec_x_mat(
		std::ranges::random_access_range auto&& lhs,
		const MatrixLike auto& rhs,
		const AllocatorLike auto& allocator
	) {
		using lhs_value_type = std::ranges::range_value_t<decltype(lhs)>;
		using rhs_value_type = std::decay_t<decltype(rhs)>::value_type;
		using res_value_type = std::common_type_t<lhs_value_type, rhs_value_type>;
		using res_allocator_type = rebind_allocator<std::decay_t<decltype(allocator)>, res_value_type>;

		auto m = lhs.size();
		auto n = rhs.cols();

		Matrix<res_value_type, res_allocator_type> res(m, n, allocator);

		auto l_b = lhs.begin();
		for (size_t i = 0; i < m; ++i) {
			auto& x = *(l_b + i);
			auto rhs_row = rhs[i];
			auto res_row = res[i];
			for (size_t j = 0; j < n; ++j) {
				res_row[j] = x * rhs_row[j];
			}
		}

		return res;
	}


	inline auto vec_x_mat(std::ranges::random_access_range auto&& lhs, const MatrixLike auto& rhs) {
		return vec_x_mat(lhs, rhs, std::pmr::polymorphic_allocator<std::byte>{});
	}
#endif


	auto operator*(const MatrixLike auto& A, const MatrixLike auto& B) {
		using A_val_type = std::decay_t<decltype(A)>::value_type;
		using B_val_type = std::decay_t<decltype(B)>::value_type;
		using Result_type = std::common_type_t<A_val_type, B_val_type>;
		return mult<Result_type>(A, B, std::pmr::polymorphic_allocator<std::byte>{});
	}

	template<typename T, typename U>
	constexpr auto dummy_mult(const Matrix<T>& A, const Matrix<U>& B)
		-> Matrix<std::common_type_t<T, U>>
	{
		using ResultType = std::common_type_t<T, U>;
		if (A.cols() != B.rows()) {
			throw std::runtime_error("Matrix dimensions do not match for multiplication");
		}
		using size_type = size_t;
		auto result = Matrix<ResultType>(A.rows(), B.cols());

		for (size_type i = 0; i < A.rows(); ++i) {
			for (size_type j = 0; j < B.cols(); ++j) {
				ResultType sum = ResultType{};
				for (size_type k = 0; k < A.cols(); ++k) {
					sum += A[i][k] * B[k][j];
				}
				result[i][j] = sum;
			}
		}

		return result;
	}

	template<typename T, typename U, typename W> concept MultiplyAddable = requires(T t, U u, W w) { w += t * u; };
	constexpr void mult_(const MatrixLike auto& A, const MatrixLike auto& B, MatrixLike auto& C)
		requires MultiplyAddable<
			typename std::decay_t<decltype(A)>::value_type,
				typename std::decay_t<decltype(B)>::value_type,
				typename std::decay_t<decltype(C)>::value_type
		>
	{
		using size_type = std::size_t;
		//A : 1 x n | B: n x 1
		const auto M = A.rows(); // C.rows()
		const auto N = B.cols(); // C.cols()
		const auto K = A.cols(); // B.rows()

		for (size_type i = 0; i < N; ++i) {
			auto&& c = C.col(i);
			auto&& Bci = B.col(i);
			for (size_type j = 0; j < K; ++j) {
				auto&& a_col = A.col(j);
				auto&& b_val = Bci[j];
				for (size_type k = 0; k < M; ++k) {
					c[k] += a_col[k] * b_val;
				}
			}
		}
	}

	template<typename result_type>
	inline auto mult(const MatrixLike auto& A, const MatrixLike auto& B, const auto& allocator) {
		using A_value_type = std::decay_t<decltype(A)>::value_type;
		using B_value_type = std::decay_t<decltype(B)>::value_type;
		using C_alloc_type = rebind_allocator<std::decay_t<decltype(allocator)>, result_type>;

		const auto M = A.rows();
		const auto N = B.cols();

		Matrix<result_type, C_alloc_type> C(M, N, allocator);
		mult_(A, B, C);
		return C;
	}

#if 1
	template<typename result_type>
	inline auto mult(const MatrixLike auto& A, const MatrixLike auto& B) {
		using A_value_type = std::decay_t<decltype(A)>::value_type;
		using A_alloc_type = std::decay_t<decltype(A)>::allocator_type;
		using B_value_type = std::decay_t<decltype(B)>::value_type;
		using C_alloc_type = rebind_allocator<A_alloc_type, result_type>;
		return mult<result_type>(A, B, C_alloc_type{});
	}
#endif

} // ns

#endif