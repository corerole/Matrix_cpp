#ifndef MATRIX_HELPERS_HPP
#define MATRIX_HELPERS_HPP
#pragma once

#include "MathMatrix.hpp"

#include <numeric>
#include <random>
#include <functional>
#include <exception>

using namespace math_matrix;

void helpers_tests();

namespace test_utils {
	using value_type = long double;
	using size_type = size_t;

	template<typename T> using distribution = std::conditional_t<
		std::floating_point<T>,
		std::uniform_real_distribution<T>,
		std::uniform_int_distribution<T>
	>;

	// template<std::ranges::input_range T> using range_val = std::ranges::value_type_t<>;

	static auto mt = std::mt19937(std::random_device{}());

	template<typename T>
	inline auto gen(T min, T max) {
		auto dist = distribution<T>(min, max);
		return std::invoke(dist, mt);
	};

	template<std::ranges::input_range R, std::same_as<std::ranges::range_value_t<R>> T>
	void fill_matrix(R&& r, T min = (-1000.0L), T max = (1000.0L)) {
		std::ranges::generate(r, [min, max]() { return gen(min, max); });
	}

	template<auto V> struct End {
		inline constexpr bool operator==(auto&& x) const { return *x == V; }
	};

	template<typename Func, typename Value> concept _projector = std::is_same_v<std::invoke_result_t<Func, Value>, std::add_rvalue_reference_t<Value>>;
	template<typename Func, typename It> concept projector = _projector<Func, typename std::iterator_traits<It>::value_type>;
	template<
		std::input_iterator ItFst,
		std::sentinel_for<ItFst> S,
		std::input_iterator ItSnd,
		std::predicate<ItFst, ItSnd> Pred,
		projector<ItFst> ProjFst,
		projector<ItSnd> ProjSnd
	>
	bool equal(ItFst fb, S fe, ItSnd sb, Pred&& pred, ProjFst&& proj_fst = std::identity{}, ProjSnd&& proj_snd = std::identity{}) {
		for (; fb != fe; ++fb, ++sb) {
			if (!(
				std::invoke(
					pred,
					std::invoke(proj_fst, *fb),
					std::invoke(proj_snd, *sb)
				)
				)) return false;
		}
		return true;
	};

}

namespace utils {
	template<typename T> concept custom_complex = requires (T a) { { a.real }; { a.imag }; };
	template<typename T> concept std_complex = std::is_same_v<T, std::complex<typename T::value_type>>;
	template<typename T> concept complex = std_complex<T> || custom_complex<T>;
	template<typename T> concept arithmetic = std::is_arithmetic_v<T>;

	template<typename T> concept Container = requires(const T a, T b) {
		typename T::value_type;
		typename T::iterator;
		typename T::const_iterator;
		typename T::size_type;
		{ a.begin() } -> std::same_as<typename T::const_iterator>;
		{ b.begin() } -> std::same_as<typename T::iterator>;
		{ a.end() } -> std::same_as<typename T::const_iterator>;
		{ b.end() } -> std::same_as<typename T::iterator>;
		{ a.size() } -> std::convertible_to<typename T::size_type>;
		{ a.empty() } -> std::convertible_to<bool>;
		requires std::convertible_to<typename T::iterator, typename T::const_iterator>;
	};


	template<typename Range> concept arithmetic_input_range = std::ranges::input_range<Range> && arithmetic<typename std::ranges::iterator_t<Range>::value_type>;
	template<typename Range> concept complex_input_range = std::ranges::input_range<Range> && complex<typename std::ranges::iterator_t<Range>::value_type>;

	inline auto to_complex(const MatrixLike auto& A, const AllocatorLike auto& allocator) {
		using matrix = std::decay_t<decltype(A)>;
		using value_type = matrix::value_type;
		using allocator_type = std::decay_t<decltype(allocator)>;
		using result_value_type = std::complex<value_type>;
		using result_allocator_type = rebind_allocator<allocator_type, result_value_type>;
		using result_matrix = rebind_container_t<matrix, result_value_type, result_allocator_type>;
		result_matrix cm(A.rows(), A.cols(), allocator);
		std::ranges::copy(A, cm.begin());
		return cm;
	}

	inline auto to_complex(const MatrixLike auto& A) {
		return to_complex(A, std::pmr::polymorphic_allocator<std::byte>{});
	}

	constexpr auto pow(auto base, auto exp) {
		return std::pow(base, exp);
	}

	constexpr auto sqrt(auto&& val) {
		return std::sqrt(std::forward<std::decay_t<decltype(val)>>(val));
	}

#if 0
	// trubles with ADL
	constexpr void swap_r(std::ranges::input_range auto&& fst, std::ranges::input_range auto&& snd) {
		auto sb = snd.begin();
		std::ranges::for_each(fst, [&sb](auto&& it) {
			auto& sbr = *sb;
			std::swap(it, sbr);
			++sb;
		});
	}
#else
	void swap_r(std::ranges::input_range auto&& fst, std::ranges::input_range auto&& snd) {
		auto fb = fst.begin();
		auto fe = fst.end();
		auto sb = snd.begin();
		for (; fb != fe; ++fb, ++sb) {
			std::iter_swap(fb, sb);
		}
	}
#endif

}

namespace helpers {

	constexpr void transpose_(const MatrixLike auto& A, MatrixLike auto& result) {
		const auto rows = A.rows();
		const auto cols = A.cols();
		using size_type = std::size_t;

		for (size_type i = 0; i < rows; ++i) {
			auto&& rci = result.col(i);
			auto&& ari = A.row(i);
			for (size_type j = 0; j < cols; ++j) {
				rci[j] = ari[j];
			}
		}
	}

	inline auto transpose(const MatrixLike auto& A, const AllocatorLike auto& allocator) {
		using value_type = std::decay_t<decltype(A)>::value_type;
		using allocator_type = rebind_allocator<std::decay_t<decltype(allocator)>, value_type>;
		const auto r = A.rows();
		const auto c = A.cols();
		Matrix<value_type, allocator_type> result(c, r, allocator);
		transpose_(A, result);
		return result;
	}

	inline auto transpose(const MatrixLike auto& A) {
		return transpose(A, std::pmr::polymorphic_allocator<std::byte>{});
	}

	constexpr void identity_(auto& mtx) {
		using value_type = std::decay_t<decltype(mtx)>::value_type;
		constexpr auto one = static_cast<value_type>(1);
		std::ranges::for_each(mtx.diagonal_range(), [one](auto&& it) { it = one; });
	}

	template<typename value_type>
	inline auto identity(size_t n, const AllocatorLike auto& allocator) {
		using Alloc = rebind_allocator<std::decay_t<decltype(allocator)>, value_type>;
		Matrix<value_type, Alloc> result(n, n, allocator);
		identity_(result);
		return result;
	}

	template<typename value_type>
	inline auto identity(size_t n) {
		return identity<value_type>(n, std::pmr::polymorphic_allocator<std::byte>{});
	}

	constexpr auto sum(std::ranges::forward_range auto&& r) {
		using value_type = std::ranges::range_value_t<decltype(r)>;
		return std::accumulate(r.begin(), r.end(), static_cast<value_type>(0));
	}

	constexpr auto abs_sum(std::ranges::forward_range auto&& r) {
		using value_type = std::ranges::range_value_t<decltype(r)>;
		auto sum = std::accumulate(r.begin(), r.end(),
			static_cast<value_type>(0),
			[](auto&& init, auto&& it) -> auto&& {
				init += std::abs(it);
				return std::move(init);
			}
		);
		return sum;
	}

	constexpr auto& max(utils::arithmetic_input_range auto&& r) {
		using value_type = std::ranges::range_value_t<decltype(r)>;
		auto max_val = &(*r.begin());
		std::ranges::for_each(r, [&max_val](auto&& it) {
			max_val = (((*max_val) > (it)) ? max_val : &it);
		});
		return *max_val;
	}

	constexpr void scale_matrix_self(MatrixLike auto& A) {
		using mtx = std::decay_t<decltype(A)>;
		using value_type = mtx::value_type;
		constexpr auto zero = static_cast<value_type>(0);
		constexpr auto one = static_cast<value_type>(1);
		auto max_val = max(A.def_range());
		auto scale = one / max_val;
		A = A * scale;
	}

	inline auto scale_matrix(const MatrixLike auto& A, const auto& allocator) {
		using value_type = std::decay_t<decltype(A)>::value_type;
		using alloc = rebind_allocator<std::decay_t<decltype(allocator)>, value_type>;
		Matrix<value_type, alloc> B(A, allocator);
		scale_matrix_self(B);
		return B;
	}

	inline auto scale_matrix(const MatrixLike auto& A) {
		using alloc = std::decay_t<decltype(A)>::allocator_type;
		return scale_matrix(A, alloc{});
	}

	// return non-abs val&
	constexpr auto& max_abs(utils::arithmetic_input_range auto&& r) {
		using value_type = std::ranges::range_value_t<decltype(r)>;
		auto max_val = &(*r.begin());
		std::ranges::for_each(r, [&max_val](auto&& it) {
			// max_val = &(std::max(it, *max_val)); // std::max return const &Ty_
			max_val = (((*max_val) > (std::abs(it))) ? max_val : &it);
		});
		return *max_val;
	}

	constexpr auto euclid_norm(std::ranges::forward_range auto&& r) {
		using value_type = std::ranges::range_value_t<decltype(r)>;
		if constexpr (ComplexLike<value_type>) {
			using T = value_type::value_type;
			constexpr auto zero = T(0);
			auto acc = zero;
			std::ranges::for_each(r, [&acc](auto&& it) {
				const auto t = std::abs(it);
				acc += t * t;
			});
			return utils::sqrt(acc);
		} else {
			auto acc = static_cast<value_type>(0);
			std::ranges::for_each(r, [&acc](auto&& it) { acc += it * it; });
			return utils::sqrt(acc);
		}
	}
#if 0
	constexpr auto euclid_norm(std::ranges::forward_range auto&& r) {
		using value_type = std::ranges::range_value_t<decltype(r)>::value_type;
		auto acc = static_cast<value_type>(0);
		std::ranges::for_each(r, [&acc](auto&& it) {
			const auto t = std::abs(it);
			acc += t * t;
		});
		return utils::sqrt<value_type>(acc);
	}
#endif

	auto inverse(const MatrixLike auto& A) {
		using value_type = std::decay_t<decltype(A)>::value_type;

		auto X = identity<value_type>(A.rows());
		auto tmp = A;

		auto add_scaled_row = [](auto&& fst, auto&& snd, auto scale) {
			auto&& f_b = fst.begin();
			auto&& f_e = fst.end();
			auto&& s_b = snd.begin();

			for (; f_b != f_e; ++f_b, ++s_b) {
				auto&& elem1 = *f_b;
				auto&& elem2 = *s_b;
				elem1 -= elem2 * scale;
			}
		};

		for (auto&& diag_elem : tmp.diagonal_range()) {
			auto&& [elem_row_idx, elem_col_idx] = tmp.get_elem_indices(diag_elem);

			auto&& elem_col = tmp.col(elem_col_idx);
			auto&& elem_row = tmp.row(elem_row_idx);

			auto&& under_sub = std::ranges::subrange(elem_col.begin() + elem_col_idx, elem_col.end());
			auto&& max_val = max_abs(under_sub);

			auto&& [max_val_row_idx, max_val_col_idx ] = tmp.get_elem_indices(max_val);
			auto&& max_val_row = tmp.row(max_val_row_idx);

			if (elem_row_idx != max_val_row_idx) {
				utils::swap_r(tmp.row(elem_row_idx), tmp.row(max_val_row_idx));
				utils::swap_r(X.row(elem_row_idx), X.row(max_val_row_idx));
			}

			auto norm = 1 / diag_elem;
			auto&& x_row = X.row(elem_row_idx);
			auto&& tmp_row = tmp.row(elem_row_idx);
			auto func = [norm](auto&& it) { it *= norm; };
			std::ranges::for_each(x_row, func);
			std::ranges::for_each(tmp_row, func);

			for (size_t i = 0; i < X.rows(); ++i) {
				if (i == elem_col_idx) { continue; }
				auto factor = tmp[i][elem_col_idx];
				add_scaled_row(X.row(i), x_row, factor);
				add_scaled_row(tmp.row(i), tmp_row, factor);
			}
		}

		return tmp; //std::make_pair(tmp, X);
	}
	
	inline auto matrix_euclid_col_norm(const MatrixLike auto& A, const auto& alloc) {
		using mtx = std::decay_t<decltype(A)>;
		using value_type = mtx::value_type;
		using allocator = std::decay_t<decltype(alloc)>;
		using reb_alloc = rebind_allocator<allocator, value_type>;
		std::vector<value_type, reb_alloc> x(alloc);
		x.reserve(A.cols());
		std::ranges::for_each(A.col_range(), [&x](auto&& it) { x.push_back(euclid_norm(it)); });
		return x;
	}

	inline auto matrix_euclid_col_norm(const MatrixLike auto& A) {
		return matrix_euclid_col_norm(A, std::pmr::polymorphic_allocator<std::byte>{});
	}

	template<utils::complex T, typename Alloc = std::allocator<typename T::value_type>>
	constexpr auto complex_matrix_euclid_col_norm(const Matrix<T>& A) {
		using value_type = typename std::allocator_traits<Alloc>::value_type;
		auto x = std::vector<value_type, Alloc>();
		x.reserve(A.rows());
		std::ranges::for_each(A.col_range(), [&x](auto&& it) { x.push_back(euclid_norm(it)); });
		return x;
	}

	template<typename T>
	constexpr auto matrix_euclid_max_col_norm(const Matrix<T>& A) {
		if constexpr (utils::complex<T>) {
			using value_type = Matrix<T>::value_type::value_type;
			auto max_val = std::numeric_limits<value_type>::lowest();
			std::ranges::for_each(A.col_range(), [&max_val](auto&& it) {
				max_val = std::max(max_val, euclid_norm(it));
			});
			return max_val;
		} else {
			auto max_val = std::numeric_limits<T>::lowest();
			std::ranges::for_each(A.col_range(), [&max_val](auto&& it) {
				max_val = std::max(max_val, euclid_norm(it));
			});
			return max_val;
		}
	}

	constexpr auto manhattan_norm(std::ranges::input_range auto&& r) {
		using value_type = std::ranges::range_value_t<decltype(r)>;
		return abs_sum(r);
	}

	constexpr auto lp_norm(std::ranges::input_range auto&& cnt, utils::arithmetic auto exp) {
		using value_type = typename std::decay_t<decltype(cnt)>::value_type;
		constexpr const auto zero = static_cast<value_type>(0);
		constexpr const auto one = static_cast<value_type>(1);
		const auto snd = one / exp;
		auto fst = zero;
		std::ranges::for_each(cnt, [&exp, &fst](auto&& it) {
			const auto x = std::abs(it);
			fst += utils::pow(x, exp);
		});
		return utils::pow(fst, snd);
	}

	constexpr auto infinity_norm(std::ranges::input_range auto&& r) { return max(r); }

	constexpr auto mean(const utils::Container auto& cnt) {
		using value_type = typename std::decay_t<decltype(cnt)>::value_type;
		const auto s = cnt.size();
		const auto sum_ = sum(cnt);
		return (sum_ / static_cast<value_type>(s));
	}

	constexpr auto abs_mean(const utils::Container auto& cnt) {
		const auto s = cnt.size();
		using value_type = typename std::decay_t<decltype(cnt)>::value_type;
		const auto acc = abs_sum(cnt);
		return acc / static_cast<value_type>(s);
	}

	constexpr auto variance(std::ranges::input_range auto&& r) {
		const auto m = mean(r);
		const auto rmo = static_cast<decltype(m)>(r.size() - 1);
		auto acc = std::accumulate(r.begin(), r.end(),
			static_cast<decltype(m)>(0),
			[m](auto&& init, auto&& it) {
				init += (it - m) * (it - m);
			}
		);
		return acc / rmo;
	}

	constexpr auto standard_deviation(std::ranges::input_range auto&& r) {
		return utils::sqrt(variance(r));
	}

	template<typename T>
	constexpr auto trace(const Matrix<T>& A) {
		return sum(A.diagonal_range());
	}

	template<typename T, typename Alloc = std::allocator<T>>
	constexpr std::vector<T, Alloc> col_means(const Matrix<T>& A) {
		auto x = A.col_range()
			| std::views::transform([](auto&& it) {
					return mean(it);
				})
			| std::ranges::to<std::vector<T, Alloc>>();
		return x;
	}

	template<typename T, typename Alloc = std::allocator<T>>
	constexpr std::vector<T, Alloc> row_means(const Matrix<T>& A) {
		auto x = A.row_range()
			| std::views::transform([](auto&& it) {
					return mean(it);
				})
			| std::ranges::to<std::vector<T, Alloc>>();
		return x;
	}

	static constexpr void calc_m_(
		std::ranges::input_range auto&& c,
		std::ranges::input_range auto&& result
	) {
		auto sub = std::ranges::subrange(c.begin() + 1, c.end());
		const auto x = *c.begin();
		for (size_t i = 0; i < sub.size(); ++i) {
			result[i] = sub[i] / x;
		}
	}

	template<typename Alloc_ = std::pmr::polymorphic_allocator<std::byte>>
	static inline auto calc_m(
		std::ranges::input_range auto&& c,
		const Alloc_& allocator = {}
	) {
		using value_type = std::ranges::range_value_t<decltype(c)>;
		using allocator_type = rebind_allocator<std::decay_t<decltype(allocator)>, value_type>;
		std::vector<value_type, allocator_type> res(c.size() - 1, allocator);
		calc_m_(c, res);
		return res;
	}

	static constexpr void row_substruction_(
		std::ranges::input_range auto&& fst,
		std::ranges::input_range auto&& snd,
		utils::arithmetic auto m,
		std::ranges::input_range auto&& res
	) {
		auto f_b = fst.begin();
		auto s_b = snd.begin();
		for (auto&& elem : res) {
			auto& fb = *f_b;
			auto& sb = *s_b;
			elem = sb - (fb * m);
			++f_b;
			++s_b;
		}
	}

	static inline auto row_substruction(
		std::ranges::input_range auto&& fst,
		std::ranges::input_range auto&& snd,
		utils::arithmetic auto m,
		const auto& alloc
	) {
		using value_type = std::ranges::range_value_t<decltype(fst)>;
		using allocator_type = rebind_allocator<std::decay_t<decltype(alloc)>, value_type>;
		std::vector<value_type, allocator_type> res(fst.size(), alloc);
		row_substruction_(fst, snd, m, res);
		return res;
	}

	static inline auto row_substruction(
		std::ranges::input_range auto&& fst,
		std::ranges::input_range auto&& snd,
		utils::arithmetic auto m) {
		return row_substruction(fst, snd, m, std::pmr::polymorphic_allocator<std::byte>{});
	}

	static constexpr auto determinant_impl(const MatrixLike auto& A) {
		using matrix = std::decay_t<decltype(A)>;
		using size_type = matrix::size_type;
		using value_type = matrix::value_type;

		auto B = A;

		auto my_copy = [](
			std::ranges::input_range auto&& src,
			std::ranges::input_range auto&& dst
		) {
				auto s_b = src.begin();
				auto s_e = src.end();
				auto d_b = dst.begin();
				for (; s_b != s_e; ++s_b, ++d_b) {
					*d_b = *s_b;
				}
		};

		auto frwz = []<typename value_type_>(
			const MatrixLike auto& mtx,
			const value_type_& elem
		) {
				auto&& [elem_row_idx, elem_col_idx] = mtx.get_elem_indices(elem);
				auto&& elem_row = mtx.row(elem_row_idx);

				auto current_to_last = std::ranges::subrange(mtx.row_begin() + elem_row_idx, mtx.row_end());
				auto pred = [elem_col_idx](auto&& row) { return row[elem_col_idx] != 0 ? true : false; };
				auto rows_iter = std::ranges::find_if(current_to_last, pred);
				if (rows_iter == current_to_last.end()) {
					return elem_row_idx;
				}
				auto& x = (*rows_iter).front();
				return mtx.get_elem_row_index(x);
		};

		int sign = 1;

		auto diag_range = B.diagonal_range();
		for (auto&& diag_elem : diag_range) {
			auto&& submatrix = B.get_submatrix(diag_elem, B[B.rows() - 1][B.cols() - 1]);
			auto&& sub_col = submatrix.col(0);

			if (diag_elem == 0) {
				sign = -sign;
				auto idx = frwz(B, diag_elem);
				if (B.get_elem_row_index(diag_elem) == idx) {
					throw std::runtime_error(" matrix is singular! ");
				}
				auto row_src = submatrix.row(0);
				auto row_dst = submatrix.row(idx);
				utils::swap_r(row_src, row_dst);
			}

			auto m_arr = calc_m(sub_col);
			auto&& first_row = submatrix.row(0);
			size_t i = 0;
			for (auto&& m : m_arr) {
				auto&& second_row = submatrix.row(i + 1);
				auto&& new_second_row = row_substruction(first_row, second_row, m);
				my_copy(new_second_row, second_row);
				++i;
			}
		}

		auto diag_first = *B.diagonal_begin() * sign;
		auto diag_second = B.diagonal_begin() + 1;
		auto diag_last = B.diagonal_end();
		auto s_to_l = std::ranges::subrange(diag_second, diag_last);
		for (auto&& d_elem : s_to_l) {
			diag_first *= d_elem;
		}

		return diag_first;
	}

	constexpr auto determinant(const MatrixLike auto& A) {
		const auto r = A.rows();
		const auto c = A.cols();

		if (r != c) { throw std::runtime_error("Determinant is only defined for square matrices"); }
		if (r == 1) { return A[0][0]; }
		if (r == 2) {
			const auto res = A[0][0] * A[1][1] - A[0][1] * A[1][0];
			return res;
		}

		if (r == 3) {
			const auto res = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
				A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
				A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
			return res;
		}

		return determinant_impl(A);
	}

	constexpr void col_centered_(const MatrixLike auto& A, MatrixLike auto& result) {
		using size_type = std::size_t;
		const auto r = A.rows();
		const auto c = A.cols();

		for (size_type i = 0; i < c; ++i) {
			auto ac = A.col(i);
			auto rc = result.col(i);
			auto cmean = mean(ac);
			for (size_type j = 0; j < r; ++j) {
				rc[j] = ac[j] - cmean;
			}
		}
	}

	inline auto col_centered(const MatrixLike auto& A, const auto& allocator) {
		using value_type = std::decay_t<decltype(A)>::value_type;
		using allocator_type = rebind_allocator<std::decay_t<decltype(allocator)>, value_type>;
		Matrix<value_type, allocator_type> result(A, allocator);
		col_centered_(A, result);
		return result;
	}

	inline auto col_centered(const MatrixLike auto& A) {
		return col_centered(A, std::pmr::polymorphic_allocator<std::byte>{});
	}

	inline auto covariance(const MatrixLike auto& A) {
		using matrix = std::decay_t<decltype(A)>;
		using size_type = matrix::size_type;
		using value_type = matrix::value_type;

		const auto r = A.rows();
		const auto c = A.cols();

		if (r <= 1) { throw std::runtime_error("Covariance requires at least 2 observations"); }

		auto centered = col_centered(A);

		return (transpose(centered) * centered) / static_cast<value_type>(r - 1);
	}

	template<std::floating_point Real, utils::complex Complex = std::complex<Real>>
	constexpr std::pair<Real, Complex> jacobi_rotation(
		Real a_pp,       // норма столбца p
		Real a_qq,       // норма столбца q
		Complex a_pq,    // скалярное произведение столбцов p и q
		Real abs_a_pq    // |a_pq| (уже вычисленный модуль)
	) {
		auto result = std::pair<Real, Complex>();
		auto& [c, s] = result;

		// Особые случаи
		if (abs_a_pq == Real(0)) {
			c = Real(1);
			s = Complex(0);
			return result;
		}

		// Вычисляем параметр tau
		Real delta = (a_qq - a_pp) / Real(2);
		Real tau = delta / abs_a_pq;

		// Стабильная формула для t = tan(theta/2)
		Real sign_tau = (tau >= Real(0)) ? Real(1) : Real(-1);
		Real abs_tau = std::abs(tau);
		Real t = sign_tau / (abs_tau + std::sqrt(Real(1) + tau * tau));

		// Косинус и синус угла вращения
		c = Real(1) / std::sqrt(Real(1) + t * t);
		Real sin_magnitude = t * c;  // вещественная амплитуда sin

		// Фаза для комплексного синуса
		Complex phase = a_pq / Complex(abs_a_pq);  // e^{i*arg(a_pq)}
		s = Complex(sin_magnitude) * std::conj(phase);  // s = |s| * e^{-i*arg(a_pq)}

		return result;
	}

	template<std::floating_point value_type>
	constexpr std::pair<value_type, value_type> jacobi_rotation_dummy(
		value_type a_pp,
		value_type a_qq,
		value_type a_pq,
		value_type abs_a_pq
	) {
		constexpr auto zero = static_cast<value_type>(0);
		constexpr auto one = static_cast<value_type>(1);
		constexpr auto two = static_cast<value_type>(2);

		auto result = std::pair<value_type, value_type>(zero, zero);
		auto& [c, s] = result;

		if (abs_a_pq == zero) {
			c = one;
			s = zero;
			return result;
		}

		const auto delta = (a_qq - a_pp) / two;
		const auto tau = delta / abs_a_pq;
		const auto sign_tau = (tau >= zero) ? one : -one;
		const auto abs_tau = std::abs(tau);
		const auto t = sign_tau / (abs_tau + std::sqrt(one + tau * tau));
		c = one / std::sqrt(one + t * t);
		const auto sin_magnitude = t * c;
		const auto phase = a_pq / abs_a_pq;
		s = sin_magnitude * phase;

		return result;
	}

	template<typename T>
	constexpr auto pq_c(auto&& pcol, auto&& qcol) {
		constexpr auto zero = static_cast<T>(0);
		std::tuple<T, T, T> res(zero, zero, zero);
		auto& [pq, pp, qq] = res;
		auto qcb = qcol.begin();
		std::ranges::for_each(pcol, [&pq, &pp, &qq, &qcb](auto&& it) {
			auto&& ap = it;
			auto&& aq = *qcb;
			++qcb;
			pq += ap * aq;
			pp += ap * ap;
			qq += aq * aq;
		});
		return res;
	};

#if 0
	constexpr auto svd_jacobi_real(const MatrixLike auto& A) {
		using matrix = std::decay_t<decltype(A)>;
		using size_type = matrix::size_type;
		using value_type = matrix::value_type;

		const auto m = A.rows();
		const auto n = A.cols();
		const auto p = std::min(m, n);

		auto A_c = A;

		Matrix<value_type, std::pmr::polymorphic_allocator<value_type>> U(m, p);
		Matrix<value_type, std::pmr::polymorphic_allocator<value_type>> Sigma(p, p);
		Matrix<value_type, std::pmr::polymorphic_allocator<value_type>> Vh(p, n);

		auto V = identity<value_type>(n);

		constexpr auto eps = std::numeric_limits<value_type>::epsilon();
		constexpr auto zero = static_cast<value_type>(0);
		constexpr auto one = static_cast<value_type>(1);

		auto sweeps = 0;
		while (true) {
			bool any_rot = false;

			for (size_type pcol = 0; pcol + 1 < n; ++pcol) {
				auto Apc = A_c.col(pcol);
				auto Vpc = V.col(pcol);
				for (size_type qcol = pcol + 1; qcol < n; ++qcol) {
					auto Aqc = A_c.col(qcol);
					auto Vqc = V.col(qcol);

					auto [a_pq, a_pp, a_qq] = pq_c<value_type>(Apc, Aqc);
					auto abs_a_pq = std::abs(a_pq);


					if (a_pp == zero && a_qq == zero) continue;

					if (abs_a_pq <= (eps * std::sqrt(a_pp * a_qq))) continue;

					auto [c, s] = jacobi_rotation_dummy(a_pp, a_qq, a_pq, abs_a_pq);

					if ( (std::abs(s) < zero) && ((std::abs(c - one) < eps)))	continue;

					any_rot = true;

					for (size_type i = 0; i < m; ++i) {
						auto& ap = Apc[i];
						auto& aq = Aqc[i];

						ap = c * ap - s * aq;
						aq = s * ap + c * aq;
					}

					for (size_type i = 0; i < n; ++i) {
						auto& vp = Vpc[i];
						auto& vq = Vqc[i];

						vp = c * vp - s * vq;
						vq = s * vp + c * vq;
					}
				}
			}

			if (!any_rot) {	break;}
		} 


		// --- 5) after convergence: singular values = norms of columns of A_c ---
		auto sigma = matrix_euclid_col_norm(A_c);

		// --- 6) Build U (m x p) as normalized columns of A_c ---
#if 1
#if 0
		for (size_type j = 0; j < p; ++j) {
			auto s = sigma[j];
			auto Ucj = U.col(j);
			auto Acj = A_c.col(j);
			if (s > eps) {
				for (size_type i = 0; i < m; ++i) {	U[i][j] = A_c[i][j] / s; }
			}	else {
				for (size_type i = 0; i < m; ++i) {	U[i][j] = zero;	}
				// put 1 on diagonal position if exists
				if (j < m) {	U[j][j] = one; }
			}
		}
#else
		for (size_type j = 0; j < p; ++j) {
			auto&& s = sigma[j];
			auto&& Ucj = U.col(j);
			auto&& Acj = A_c.col(j);
			for (size_type i = 0; i < m; ++i) {
				Ucj[i] = Acj[i] / s;
			}
		}
#endif
#endif

		// --- 7) Build Sigma matrix (p x p diagonal, real) ---
#if 0
		for (size_type i = 0; i < p; ++i) {
			for (size_type j = 0; j < p; ++j) {
				Sigma[i][j] = zero;
			}
			Sigma[i][i] = sigma[i];
		}
#else
#if 0
#if 0
		std::ranges::copy(Sigma.diagonal_range(), sigma);
#else
		std::ranges::for_each(Sigma.diagonal_range(), 
		[
			sb = sigma.begin()
		](auto&& it) mutable {
			it = *sb;
			++sb;
		});
#endif
#else
		std::copy(sigma.begin(), sigma.end(), Sigma.diagonal_begin());
#endif
#endif

		// --- 8) Build Vh (p x n) = first p rows of V^H ---
		for (size_type i = 0; i < p; ++i) {
			auto Vhri = Vh.row(i);
			auto Vci = V.col(i);
			for (size_type j = 0; j < n; ++j) {
				Vhri[j] = Vci[j];
			}
		}

		// --- 9) Optional: sort singular values descending with permutation of U and Vh ---
		for (size_type i = 0; i < p; ++i) {
			size_type imax = i;
			auto smax = sigma[i];
			for (size_type j = i + 1; j < p; ++j) {
				if (sigma[j] > smax) {
					smax = sigma[j];
					imax = j;
				}
			}

			if (imax != i) {
				auto tmp_s = sigma[i];
				sigma[i] = sigma[imax];
				sigma[imax] = tmp_s;

				for (size_type r = 0; r < m; ++r) {
					std::swap(U[r][i], U[r][imax]);
				}

				for (size_type c = 0; c < n; ++c) {
					std::swap(Vh[i][c], Vh[imax][c]);
				}
			}
		}

		for (size_type i = 0; i < p; ++i) {
			for (size_type j = 0; j < p; ++j) {
				Sigma[i][j] = zero;
			}
			Sigma[i][i] = sigma[i];
		}


		return std::make_tuple(U, Sigma, Vh );
	}


	template<std::floating_point Real, typename Alloc = std::allocator<Real>>
	auto svd_jacobi(const Matrix<Real>& A)
		-> std::tuple< Matrix<std::complex<Real>>, Matrix<Real>, Matrix<std::complex<Real>> >
	{

		using matrix = std::decay_t<decltype(A)>;
		using size_type = matrix::size_type;
		using value_type = matrix::value_type;

		constexpr auto max_sweeps = static_cast<size_type>(50);
		constexpr auto tol_factor = static_cast<value_type>(1e-06);

		using complex_t = std::complex<Real>;

		const auto m = A.rows();
		const auto n = A.cols();
		const auto p = std::min(m, n);

		// --- 1) Build complex copy of A (so same code works for real/complex T) ---
#if 0
		auto A_c = Matrix<complex_t>(m, n);
		for (size_type i = 0; i < m; ++i) {
			for (size_type j = 0; j < n; ++j) {
				A_c[i][j] = complex_t(A[i][j]);
			}
		}
#else
		auto A_c = utils::to_complex(A);
#endif

		// --- 2) Accumulate right singular vectors in V (n x n identity) ---
#if 0
		auto V = Matrix<complex_t>(n, n);
		for (size_type i = 0; i < n; ++i) {
			for (size_type j = 0; j < n; ++j) {
				V[i][j] = complex_t((i == j) ? Real(1) : Real(0));
			}
		}
#else
		auto V = identity<complex_t>(n); // alloc?
#endif
		// --- 3) Precompute tolerance and eps ---
		constexpr auto eps = std::numeric_limits<Real>::epsilon();

		// matrix norm (use max column 2-norm)
#if 0
		auto max_col_norm = static_cast<Real>(0);
		for (size_type j = 0; j < n; ++j) {
			auto col_norm_sq = static_cast<Real>(0);
			for (size_type i = 0; i < m; ++i) {
				auto t = std::abs(A_c[i][j]);
				col_norm_sq += t * t;
			}

			auto col_norm = std::sqrt<Real>(col_norm_sq);
			if (col_norm > max_col_norm) {
				max_col_norm = col_norm;
			}
		}
		const Real tol_abs = std::max(Real(1e-300), eps * std::max<Real>(Real(1), max_col_norm) * tol_factor);
#endif
#if 0
		const Real tol_abs = std::max(Real(1e-300), eps * std::max<Real>(Real(1), matrix_euclid_max_col_norm(A_c)) * tol_factor);
#else
#if 0
		const Real tol_abs = eps * matrix_euclid_max_col_norm(A_c) * tol_factor;
#else
		constexpr auto tol_abs = eps;
#endif
#endif
		// --- 4) One-sided Jacobi sweeps ---
		// We'll perform sweeps over pairs (p,q). Convergence when no rotation larger than threshold.
		// for (size_type sweep = 0; sweep < max_sweeps; ++sweep) {
		size_t sweep = 0u;
		while(true) {
			std::cout << "Sweep " << sweep++ << std::endl;
			bool any_rot = false;

			for (size_type pcol = 0; pcol + 1 < n; ++pcol) {
				for (size_type qcol = pcol + 1; qcol < n; ++qcol) {
					// compute Gram matrix entries for columns pcol and qcol:
					// a_pp = <col_p, col_p>, a_qq = <col_q, col_q>, a_pq = <col_p, col_q>
					complex_t a_pq = complex_t(0);
					Real a_pp = Real(0);
					Real a_qq = Real(0);

					for (size_type i = 0; i < m; ++i) {
						auto& ap = A_c[i][pcol];
						auto& aq = A_c[i][qcol];
						a_pq = std::conj(ap) * aq;
						a_pp = std::norm(ap);   // |ap|^2
						a_qq = std::norm(aq);   // |aq|^2
					}

					// If columns are already (nearly) orthogonal, skip
					Real abs_a_pq = std::abs(a_pq);
					if (abs_a_pq <= tol_abs * std::sqrt(a_pp * a_qq)) {
						continue;
					}

					// --- compute Jacobi rotation that diagonalizes the 2x2 Hermitian Gram block ---
					// G = [a_pp, a_pq; conj(a_pq), a_qq]
					// We want unitary J = [[c, s], [-conj(s), c]] (c real, s complex)
					// such that J^H * G * J is (approximately) diagonal.

					// Special-case: if a_pp or a_qq is zero (column zero), handle simply by swapping/scaling
					if (a_pp == Real(0) && a_qq == Real(0)) {
						continue;
					}
#if 0
					// compute rotation parameters (see standard one-sided Jacobi derivation)
					// Let phi = arg(a_pq), g = |a_pq|
					const Real g = abs_a_pq;
					const Real delta = (a_qq - a_pp) / (Real)2;
					// compute t (real) using stable formula
					Real t;
					if (g == Real(0)) {
						t = Real(0);
					}	else {
						Real sign_delta = (delta >= Real(0)) ? Real(1) : Real(-1);
						Real abs_delta = std::abs(delta);
						t = sign_delta / (abs_delta / g + std::sqrt((abs_delta / g) * (abs_delta / g) + Real(1)));
						// alternative stable formula:
						// tau = (a_qq - a_pp) / (2*g)
						// t = sign(tau) / (|tau| + sqrt(1 + tau^2))
					}

					// The formula above can be numerically finicky for complex; instead use the standard tau formula:
					{
						Real tau = (a_qq - a_pp) / (Real)(2.0 * g);
						Real sign_tau = (tau >= Real(0)) ? Real(1) : Real(-1);
						t = sign_tau / (std::abs(tau) + std::sqrt(Real(1) + tau * tau));
					}

					Real c = Real(1) / std::sqrt(Real(1) + t * t);
					// s has phase -arg(a_pq)
					Real tt = t * c; // tt = s magnitude (real)
					complex_t phase = (g == Real(0)) ? complex_t(1) : (a_pq / complex_t(g)); // unit complex a_pq/|a_pq|
					complex_t s = complex_t(tt) * std::conj(phase); // s = tt * exp(-i*arg(a_pq))
#else
					auto [c, s] = jacobi_rotation(a_pp, a_qq, a_pq, abs_a_pq);
#endif
					// threshold small rotations
					if (std::abs(s) < tol_abs && (std::abs(c - Real(1)) < tol_abs)) {
						continue;
					}

					any_rot = true;

					// --- Apply rotation to columns pcol and qcol of A_c: A_c = A_c * J
					// J = [[c, s],[ -conj(s), c ]] -> column transform:
					// new_col_p = c * col_p - conj(s) * col_q
					// new_col_q = s * col_p + c * col_q
					for (size_type i = 0; i < m; ++i) {
						auto& ap = A_c[i][pcol];
						auto& aq = A_c[i][qcol];

						complex_t new_p = complex_t(c) * ap - std::conj(s) * aq;
						complex_t new_q = s * ap + complex_t(c) * aq;

						A_c[i][pcol] = new_p;
						A_c[i][qcol] = new_q;
					}

					// --- Accumulate rotation into V: V = V * J (same combination on its columns)
					for (size_type i = 0; i < n; ++i) {
						auto& vp = V[i][pcol];
						auto& vq = V[i][qcol];

						complex_t new_vp = complex_t(c) * vp - std::conj(s) * vq;
						complex_t new_vq = s * vp + complex_t(c) * vq;

						V[i][pcol] = new_vp;
						V[i][qcol] = new_vq;
					}
				} // for qcol
			} // for pcol

			if (!any_rot) {
				// converged early
				break;
			}
		} // for sweep

		// --- 5) after convergence: singular values = norms of columns of A_c ---
#if 0
		std::vector<Real> sigma(p, Real(0));
		for (size_type j = 0; j < p; ++j) {
			Real ssum = Real(0);
			for (size_type i = 0; i < m; ++i) {
				ssum += std::norm(A_c[i][j]);
			}
			sigma[j] = std::sqrt(ssum);
		}
#else
		auto sigma = complex_matrix_euclid_col_norm(A_c);
		
#endif

		// --- 6) Build U (m x p) as normalized columns of A_c ---
#if 1
		Matrix<complex_t> U(m, p);
		for (size_type j = 0; j < p; ++j) {
			Real s = sigma[j];
			if (s > tol_abs) {
				for (size_type i = 0; i < m; ++i) {
					U[i][j] = A_c[i][j] / complex_t(s);
				}
			} else {
				// if sigma is zero (or tiny), produce an orthonormal vector:
				// choose standard basis vector not yet used
				for (size_type i = 0; i < m; ++i) {
					U[i][j] = complex_t(0);
				}
				// put 1 on diagonal position if exists
				if (j < m) {
					U[j][j] = complex_t(1);
				}
			}
		}
#else
		auto U = Matrix<complex_t>(m, p);
		std::ranges::for_each();
#endif
		// --- 7) Build Sigma matrix (p x p diagonal, real) ---
		Matrix<Real> Sigma(p, p);
		for (size_type i = 0; i < p; ++i) {
			for (size_type j = 0; j < p; ++j) {
				Sigma[i][j] = Real(0);
			}
			Sigma[i][i] = sigma[i];
		}

		// --- 8) Build Vh (p x n) = first p rows of V^H ---
		// Note: V is n x n accumulated; Vh_full = V.conjugate().transposed() = V^H
		Matrix<complex_t> Vh(p, n);
		for (size_type i = 0; i < p; ++i) {
			for (size_type j = 0; j < n; ++j) {
				// V^H row i, col j is conj(V[j][i])
				Vh[i][j] = std::conj(V[j][i]);
			}
		}

		// --- 9) Optional: sort singular values descending with permutation of U and Vh ---
		// We'll sort in-place to return standard ordering sigma[0] >= sigma[1] >= ...
		for (size_type i = 0; i < p; ++i) {
			size_type imax = i;
			Real smax = sigma[i];
			for (size_type j = i + 1; j < p; ++j) {
				if (sigma[j] > smax) {
					smax = sigma[j];
					imax = j;
				}
			}
			if (imax != i) {
				// swap sigma
				Real tmp_s = sigma[i];
				sigma[i] = sigma[imax];
				sigma[imax] = tmp_s;

				// swap columns in U (m x p)
				for (size_type r = 0; r < m; ++r) {
					std::swap(U[r][i], U[r][imax]);
				}

				// swap rows in Vh (p x n)
				for (size_type c = 0; c < n; ++c) {
					std::swap(Vh[i][c], Vh[imax][c]);
				}
			}
		}

		// Put sorted sigma back into Sigma matrix
		for (size_type i = 0; i < p; ++i) {
			for (size_type j = 0; j < p; ++j) {
				Sigma[i][j] = Real(0);
			}
			Sigma[i][i] = sigma[i];
		}

		return { U, Sigma, Vh };
	}

	template<typename T>
	constexpr auto& find_max_val(const Matrix<T>& A) {
		using matrix = std::decay_t<decltype(A)>;
		using size_type = typename matrix::size_type;
		using value_type = typename matrix::value_type;

		const auto n = A.rows();
		const auto m = A.cols();

		// std::find/_if
		std::vector<typename matrix::pointer> pointers(n, nullptr);
		auto ctr = static_cast<typename matrix::difference_type>(0);
		std::generate(pointers.begin(), pointers.end(),
			[
				i = std::ref(ctr),
				d = std::addressof(A[0][0]),
				c = static_cast<matrix::difference_type>(m)
			]() {
				auto&& offset = i.get();
				return (d + ((offset++) * c));
			}
		);
		/* mt todo */
		for (auto&& x : pointers) {
			auto func = [
				off = m,
				res = std::ref(x)
			]() {
				auto&& result = res.get();
				auto b = Matrix<std::decay_t<decltype(*result)>>::iterator(result);
				auto e = Matrix<std::decay_t<decltype(*result)>>::iterator(result + off);
				std::for_each(b, e, [result_ = std::ref(result)](auto&& it) {
					auto&& result__ = result_.get();
					if (*result__ < it) { result__ = std::addressof(it); }
					});
				};
			func();
		}

		auto result = pointers[0];
		std::for_each(pointers.begin() + 1, pointers.end(), [max = std::ref(result)](auto&& it) {
			auto&& max_ = max.get();
			if (*it > *max_) { max_ = it; }
			});

		return *result;
	}

	// SVD-based inverse — replaces previous Gaussian-elim implementation
	template<typename T, typename Alloc = std::allocator<T>>
	inline auto inverse(const Matrix<T>& A) -> Matrix<T, Alloc> {
		using matrix = std::decay_t<decltype(A)>;
		using size_type = matrix::size_type;
		using value_type = matrix::value_type;

		const auto n = A.rows();
		const auto m = A.cols();

		if (n != m) { throw std::runtime_error("Matrix inverse is only defined for square matrices"); }

		using Real = value_type;
		// using complex_t = std::complex<Real>;

		constexpr auto eps = std::numeric_limits<Real>::epsilon();

		auto [U_c, Sigma, Vh_c] = svd_jacobi_real<value_type>(A);

		// Sigma is real diagonal (p x p), here p == n for square matrix
		// find largest singular value to form a relative tolerance
		auto&& max_sigma = max(Sigma); // find_max_val(Sigma);

		// tolerance: relative to largest singular value and machine eps
		auto tol = eps * std::max<Real>(Real(1), max_sigma) * static_cast<Real>(n);

		// build Sigma^{-1} as complex matrix (diagonal)
		// auto Sigma_inv = Matrix<complex_t>(Sigma.rows(), Sigma.cols());
		auto Sigma_inv = Matrix<Real, Alloc>(Sigma.rows(), Sigma.cols());

		for (size_type i = 0; i < Sigma.rows(); ++i) {
			// Sigma_inv[i][i] = complex_t(Real(1) / Sigma[i][i]); // (Sigma[i][i] <= tol) ? complex_t(Real(0)) : complex_t(Real(1) / Sigma[i][i]);
			Sigma_inv[i][i] = Real(1) / Sigma[i][i];
#if 1
			auto&& s = Sigma[i][i];
			if (s <= tol) {
				// Matrix is singular or nearly singular.
				// Option A: 
				// throw std::runtime_error("Matrix is singular or nearly singular (small singular value)");

				// Option B (pseudo-inverse): comment out throw and use
				// Sigma_inv[i][i] = complex_t(Real(0)); // leave zero -> Moore-Penrose pseudo-inverse
			}	else {
				Sigma_inv[i][i] = 1 / s; //complex_t(Real(1) / s);
			}
#else
			Sigma_inv[i][i] = complex_t(Real(1) / s);
#endif
		}

		// Recover V from Vh: V = conj(Vh)^T
		auto V = Vh_c.transpose();
		std::for_each(V.begin(), V.end(), [](auto& it) { it = it * it; }); // std::conj(it); });

		// Compute U^H (conjugate transpose of U)
		auto U_H = U_c.transpose();
		std::for_each(U_H.begin(), U_H.end(), [](auto& it) { it = it * it; });// it = std::conj(it); });

		// A^{-1} = V * Sigma_inv * U^H (complex matrix)
		auto Ainv_c = (V * Sigma_inv) * U_H;

		auto out = Matrix<T>(n, n);
		for (size_type i = 0; i < n; ++i) {
			for (size_type j = 0; j < n; ++j) {
				// out[i][j] = static_cast<T>(Ainv_c[i][j].real());
				out[i][j] = static_cast<Real>(Ainv_c[i][j]);
			}
		}

		return out;
	}
#endif

	constexpr auto householder_vector__(std::ranges::forward_range auto&& x) {
		using value_type = std::ranges::range_value_t<decltype(x)>;
		if constexpr (ComplexLike<value_type>) {
			using T = value_type::value_type;
			constexpr auto eps = std::numeric_limits<T>::epsilon();
			auto& x0 = *x.begin();
			const auto sigma = euclid_norm(x);
			const auto abs_x0 = std::abs(x0);
			if (abs_x0 < eps) {
				return value_type(sigma);
			}
			return x0 + (x0 / abs_x0) * sigma;
		} else {
			auto& x0 = *x.begin();
			const auto sigma = euclid_norm(x);
			if (x0 > 0) {
				return x0 + sigma;
			}
			return x0 - sigma;
		}
	}

	constexpr auto householder_beta(
		std::ranges::forward_range auto&& v
	) {
		using value_type = std::ranges::range_value_t<decltype(v)>;
		if constexpr (ComplexLike<value_type>) {
			constexpr auto zero = value_type(0);
			constexpr auto two = value_type(2);
			auto res = zero;
			std::ranges::for_each(v, [&res](auto&& it) { res += std::norm(it); });
			return two / res;
		} else {
			constexpr auto zero = value_type(0);
			constexpr auto two = value_type(2);
			auto res = zero;
			std::ranges::for_each(v, [&res](auto&& it) {
				res += it * it;
				});
			return two / res;
		}
	}

	constexpr void householder_vector_(
		std::ranges::forward_range auto&& x,
		std::ranges::forward_range auto&& dst
	) {
		*dst.begin() = householder_vector__(x);
		std::copy(x.begin() + 1, x.end(), dst.begin() + 1);
	}

	inline auto householder_vector(
		std::ranges::forward_range auto&& x,
		const auto& alloc
	) {
		using value_type = std::ranges::range_value_t<std::decay_t<decltype(x)>>;
		using allocator_type = rebind_allocator<std::decay_t<decltype(alloc)>, value_type>;
		std::vector<value_type, allocator_type> res(x.size(), alloc);
		householder_vector_(x, res);
		return res;
	}

	constexpr auto householder_vector(std::ranges::forward_range auto&& x) {
		return householder_vector(x, std::pmr::polymorphic_allocator<std::byte>{});
	}

	inline auto householder_reflection_matrix(std::ranges::forward_range auto&& x, const auto& alloc) {
		using size_type = std::size_t;
		using value_type = std::ranges::range_value_t<std::decay_t<decltype(x)>>;
		using allocator_type = rebind_allocator<std::decay_t<decltype(alloc)>, value_type>;
		constexpr auto two = static_cast<value_type>(2);
		allocator_type vec_alloc{};
		const auto n = x.size();
		auto I = identity<value_type>(n);
		auto v = householder_vector(x, vec_alloc);
		Matrix<value_type, allocator_type> vvT(n, n, alloc);
		for (size_type i = 0; i < n; ++i) {
			auto&& vvTrow = vvT.row(i);
			auto&& vi = v[i];
			for (size_type j = 0; j < n; ++j) {
				vvTrow[j] = vi * v[j];
			}
		}

		return I - (householder_beta(v) * vvT);
	}

	constexpr auto householder_reflection_matrix(std::ranges::forward_range auto&& x) {
		return householder_reflection_matrix(x, std::pmr::polymorphic_allocator<std::byte>{});
	}

	auto householder_qr_decomposition(const MatrixLike auto& A, const auto& alloc_Q, const auto& alloc_R) {
		using matrix = std::decay_t<decltype(A)>;
		using size_type = matrix::size_type;
		using value_type = matrix::value_type;
		using allocator_type = matrix::allocator_type;
		using allocator_type_Q = rebind_allocator<std::decay_t<decltype(alloc_Q)>, value_type>;
		using allocator_type_R = rebind_allocator<std::decay_t<decltype(alloc_R)>, value_type>;

		const auto m = A.rows();
		const auto n = A.cols();

		// if (m < n) { throw std::runtime_error("QR decomposition requires m >= n"); }

		Matrix<value_type, allocator_type_R> R(A, alloc_R);
		auto Q = identity<value_type, allocator_type_Q>(m, alloc_Q);

		for (size_type k = 0; k < std::min(m, n); ++k) {
			Matrix<value_type, allocator_type> x(1, m - k, allocator_type{});
			auto x0 = x[0];
			for (size_type i = k; i < m; ++i) {
				x0[i - k] = R[i][k];
			}

			auto Hk = householder_reflection_matrix(x0);

			// R[k:m, k:n] = Hk * R[k:m, k:n]
			const auto& R_fst_elem = R[k][k];
			const auto& R_snd_elem = R[m - 1][n - 1];
			auto&& R_block = R.get_submatrix(R_fst_elem, R_snd_elem);
			auto&& r_result = Hk * R_block;
			R.set_submatrix(R_fst_elem, r_result);

			// Q[:, k:m] = Q[:, k:m] * Hk^T
			const auto& Q_fst_elem = Q[0][k];
			const auto& Q_snd_elem = Q[m - 1][m - 1];
			auto&& Q_block = Q.get_submatrix(Q_fst_elem, Q_snd_elem);
			auto&& q_result = Q_block * Hk;
			Q.set_submatrix(Q_fst_elem, q_result);
		}

		return std::make_pair(std::move(Q), std::move(R));
	}

	auto householder_qr_decomposition(const MatrixLike auto& A) {
		using matrix = std::decay_t<decltype(A)>;
		using value_type = matrix::value_type;
		using allocator_type = matrix::allocator_type;
		return householder_qr_decomposition(A, allocator_type{}, allocator_type{});
	}

	auto hessenberg_form(const MatrixLike auto& A, const AllocatorLike auto& alloc) {
		using value_type = typename std::decay_t<decltype(A)>::value_type;
		using allocator_type = std::decay_t<decltype(alloc)>;
		using res_allocator_type = rebind_allocator<allocator_type, value_type>;
		using size_type = typename std::decay_t<decltype(A)>::size_type;

		const auto n = A.rows();
		Matrix<value_type, res_allocator_type> H(A, alloc);
		auto Q = identity<value_type>(n);

		for (size_type k = 0; k < n - 2; ++k) {
			const size_type sub_size = n - k - 1;
			Matrix<value_type, allocator_type> x_vec(1, sub_size, allocator_type{});
			for (size_type i = 0; i < sub_size; ++i) {
				x_vec[0][i] = H[k + 1 + i][k];
			}

			auto H_sub = householder_reflection_matrix(x_vec.row(0));

			auto& left_top = H[k + 1][k];
			auto& left_bottom = H[n - 1][n - 1];
			auto left_block = H.get_submatrix(left_top, left_bottom);
			auto left_result = H_sub * left_block;
			H.set_submatrix(left_top, left_result);

			auto& right_top = H[0][k + 1];
			auto& right_bottom = H[n - 1][n - 1];
			auto right_block = H.get_submatrix(right_top, right_bottom);
			auto right_result = right_block * H_sub;
			H.set_submatrix(right_top, right_result);

			const auto& Q_fst_elem = Q[0][k + 1];
			const auto& Q_snd_elem = Q[n - 1][n - 1];
			auto&& Q_block = Q.get_submatrix(Q_fst_elem, Q_snd_elem);
			auto&& q_result = Q_block * H_sub;
			Q.set_submatrix(Q_fst_elem, q_result);

		}

		return std::make_pair(std::move(Q), std::move(H));
	}

	auto hessenberg_form(const MatrixLike auto& mtx) {
		using matrix = std::decay_t<decltype(mtx)>;
		using value_type = matrix::value_type;
		using allocator_type = matrix::allocator_type;
		return hessenberg_form(mtx, allocator_type{});
	}

	auto upper_companion_matrix(std::ranges::input_range auto&& r, const auto& allocator) {
		using value_type = std::ranges::range_value_t<std::decay_t<decltype(r)>>;
		using allocator_type = rebind_allocator<std::decay_t<decltype(allocator)>, value_type>;
		const auto n = r.size();
		Matrix<value_type, allocator_type> A(n, n, allocator);
		constexpr auto one = static_cast<value_type>(1);
		constexpr auto m_one = static_cast<value_type>(-1);
		for (size_t i = 1; i < n; ++i) {
			A[i][i-1] = one;
		}

		auto&& last_col = A.col(n - 1);
		auto c_b = last_col.begin();
		auto c_e = last_col.end();
		auto r_b = r.begin();
		for (; c_b != c_e; ++c_b, ++r_b) {
			auto& cb = *c_b;
			auto& rb = *r_b;
			cb = m_one * rb;
		}
		
		return A;
	}

	auto upper_companion_matrix(std::ranges::input_range auto&& r) {
		return upper_companion_matrix(r, std::pmr::polymorphic_allocator<std::byte>{});
	}

	auto lower_companion_matrix(std::ranges::input_range auto&& r, const auto& allocator_ucm, const auto& allocator_t) {
		return transpose(upper_companion_matrix(r, allocator_ucm), allocator_t);
	}

	auto lower_companion_matrix(std::ranges::input_range auto&& r, const auto& allocator_ucm) {
		return transpose(upper_companion_matrix(r, allocator_ucm), std::pmr::polymorphic_allocator<std::byte>{});
	}

	auto lower_companion_matrix(std::ranges::input_range auto&& r) {
		return transpose(upper_companion_matrix(r), std::pmr::polymorphic_allocator<std::byte>{});
	}

	constexpr inline auto tr_det_2x2_(
		const ComplexLike auto& a11,
		const ComplexLike auto& a12,
		const ComplexLike auto& a21,
		const ComplexLike auto& a22)
	{
		auto det = a11 * a22 - a12 * a21;
		auto tr = a11 + a22;
		return std::make_pair(tr, det);
	}

	constexpr inline auto self_values_2x2_(
		const ComplexLike auto& a11,
		const ComplexLike auto& a12,
		const ComplexLike auto& a21,
		const ComplexLike auto& a22
	) {
		using complex_type = std::decay_t<decltype(a11)>;
		using value_type = complex_type::value_type;
		auto&& [tr, det] = tr_det_2x2_(a11, a12, a21, a22);
		constexpr auto zero = value_type(0);
		constexpr auto two = value_type(2);
		constexpr auto four = value_type(4);
		auto d = tr * tr - four * det;
		auto x = utils::sqrt(d);
		auto fst = ((tr - x) / two);
		auto snd = ((tr + x) / two);
		return std::make_pair(fst, snd);
	}

#if 0																				
	auto wilkinson_shift_real(const MatrixLike auto& mtx) {
		using value_type = std::decay_t<decltype(mtx)>::value_type;

		constexpr auto zero = static_cast<value_type>(0);
		constexpr auto two = static_cast<value_type>(2);
		constexpr auto four = static_cast<value_type>(4);

		const auto rows = mtx.rows();
		const auto cols = mtx.cols();

		const auto& fst = mtx[rows - 2][cols - 2];
		const auto& lst = mtx[rows - 1][cols - 1];

		auto&& sub = mtx.get_submatrix(fst, lst);

		auto& a = sub[0][0];
		auto& b = sub[0][1];
		auto& c = sub[1][0];
		auto& d = sub[1][1];

		const auto trace = a + d;
		const auto det = a * d - b * c;
		const auto disc = trace * trace - four * det;
		if (disc <= zero) { return d; }
		const auto sqrt_disc = std::sqrt(disc);
		const auto root1 = (trace + sqrt_disc) / two;
		const auto root2 = (trace - sqrt_disc) / two;
		auto mu = (std::abs(root1 - d) < std::abs(root2 - d)) ? root1 : root2;
		return mu;
	}
#endif

#if 1
	auto wilkinson_shift_complex(
		const ComplexLike auto& a,
		const ComplexLike auto& b,
		const ComplexLike auto& c,
		const ComplexLike auto& d)
	{
		using value_type = std::remove_cvref_t<decltype(a)>::value_type;
		constexpr auto two = value_type(2);
		auto delta = (a - d) / two;
		auto sigma = std::sqrt(((delta * delta) + (b * c)));
		auto mu_plus = (a + d) / two + sigma;
		auto mu_minus = (a + d) / two - sigma;
		auto fst = std::abs(mu_plus - d);
		auto snd = std::abs(mu_minus - d);
		return std::move((fst < snd) ? mu_plus : mu_minus);
	}

	constexpr auto wilkinson_shift_complex(const MatrixLike auto& A) {
		const auto n = A.rows();
		const auto& a = A.get_elem((n - 2), (n - 2));
		const auto& b = A.get_elem((n - 2), (n - 1));
		const auto& c = A.get_elem((n - 1), (n - 2));
		const auto& d = A.get_elem((n - 1), (n - 1));
		return wilkinson_shift_complex(a, b, c, d);
	}
#if 0
	constexpr auto wilkinson_shift_real(const MatrixLike auto& A) {
		auto res = wilkinson_shift_complex(A);
		if (res.imag()) { throw std::runtime_error(" wilkinson_shift_real | can't retun complex \n "); }
		return res.real();
	}
#endif

#if 0
	auto wilkinson_shift(const MatrixLike auto& A) {
		auto n = A.rows();
		auto fst = A[n - 2][n - 2];
		auto snd = A[n - 1][n - 1];
		auto&& sub = A.get_submatrix(fst, snd);
		auto&& a = sub[0][0];
		auto&& b = sub[0][1];
		auto&& c = sub[1][0];
		auto&& d = sub[1][1];
		auto x = wilkinson_shift_complex(a, b, c, d);
		auto res = x.real();
		return res;
	}
#endif
#endif

	template<typename value_type>
	inline auto first_standard_basis_vector(size_t n, const AllocatorLike auto& allocator) {
		using allocator_type = rebind_allocator<std::decay_t<decltype(allocator)>, value_type>;
		constexpr auto one = static_cast<value_type>(1);
		Matrix<value_type, allocator_type> e1(n, 1, allocator);
		e1[0][0] = one;
		return e1;
	}

	template<typename value_type>
	inline auto basis_vector(size_t n) {
		return first_standard_basis_vector<value_type>(n, std::pmr::polymorphic_allocator<std::byte>{});
	}

	inline auto francis_vector(const MatrixLike auto& A) {
		using matrix = std::decay_t<decltype(A)>;
		using value_type = matrix::value_type;
		using allocator_type = matrix::allocator_type;
		const auto n = A.rows();
		auto a = A[n - 2][n - 2];
		auto b = A[n - 2][n - 1];
		auto c = A[n - 1][n - 2];
		auto d = A[n - 1][n - 1];
		auto [tr, det] = tr_det_2x2_(a, b, c, d);
		auto I = identity<value_type>(n);
		auto e1 = basis_vector<value_type>(n);
		auto x = (A * A - tr * A + det * I) * e1; // (NxN) * col -> col
		return x; // x[n][1];
	}

	auto similarity_transformations(std::ranges::input_range auto&& r, const AllocatorLike auto& allocator) {
		using value_type = std::ranges::range_value_t<decltype(r)>;
		using allocator_type = rebind_allocator<std::decay_t<decltype(allocator)>, value_type>;
		constexpr auto zero = static_cast<value_type>(0);
		Matrix<value_type, allocator_type> v(r.size(), 1, allocator);
		auto v_col = v.col(0);
		std::copy(r.begin(), r.end(), v_col.begin());
		int sign = 1;
		if (v[0][0] < zero) { sign = -1; }
		auto x_norm = euclid_norm(r);
		v[0][0] = v[0][0] - x_norm * sign;
		auto vT = transpose(v); // -> vT[1][n];
		auto vvT = v * vT; // -> vvT[n][n]
		auto vTv = vT * v; // -> scalar
		auto&& vTv_val = vTv.get_elem(0, 0);
		auto I = identity<value_type>(v.rows()); 
		auto P0 = I - 2 * vvT / vTv_val;
		return P0;
	}

	auto similarity_transformations(std::ranges::input_range auto&& r) {
		return similarity_transformations(r, std::pmr::polymorphic_allocator<std::byte>{});
	}
#if 0
	auto francis_qr_givens(const MatrixLike auto& A) {
		using matrix_type = std::decay_t<decltype(A)>;
		using value_type = matrix_type::value_type;

		size_t n = A.cols();

		auto shift = wilkinson_shift_real(A);
		auto I = identity<value_type>(n);
		auto Tmat = A - shift * I;
		auto Q = I;
		std::vector<std::pair<value_type, value_type>> rots(n - 1);
		size_t rot_count = 0;

		auto x = Tmat.get_elem(0, 0);
		auto y = Tmat.get_elem(1, 0);
		auto denom = std::sqrt(x * x + y * y);
		auto c = x / denom;
		auto s = y / denom;
		rots[rot_count++] = std::make_pair(c, s);

		for (size_t j = 0; j < n; ++j) {
			auto& t1 = Tmat.get_elem(0, j);
			auto& t2 = Tmat.get_elem(1, j);
			auto t1_old = t1;
			t1 = c * t1 + s * t2;
			t2 = -s * t1_old + c * t2;
		}

		for (size_t i = 0; i < n; ++i) {
			auto& q1 = Q.get_elem(i, 0);
			auto& q2 = Q.get_elem(i, 1);
			auto q1_old = q1;
			q1 = c * q1 + s * q2;
			q2 = -s * q1_old + c * q2;
		}

		for (size_t k = 1; k < n - 1; ++k) {
			auto& x = Tmat.get_elem(k, (k - 1));
			auto& y = Tmat.get_elem((k + 1), (k - 1));
			auto denom = std::sqrt(x * x + y * y);
			auto c = x / denom;
			auto s = y / denom;
			rots[rot_count++] = std::make_pair(c, s);

			for (size_t j = k - 1; j < n; ++j) {
				auto& t1 = Tmat.get_elem(k, j);
				auto& t2 = Tmat.get_elem((k + 1), j);
				auto t1_old = t1;
				t1 = c * t1 + s * t2;
				t2 = -s * t1_old + c * t2;
			}

			for (size_t i = 0; i < n; ++i) {
				auto& q1 = Q.get_elem(i, k);
				auto& q2 = Q.get_elem(i, (k + 1));
				auto q1_old = q1;
				q1 = c * q1 + s * q2;
				q2 = -s * q1_old + c * q2;
			}
		}

		Tmat = A;

		for (size_t k = 0; k < rot_count; ++k) {
			auto& c = rots[k].first;
			auto& s = rots[k].second;
			for (size_t j = 0; j < n; ++j) {
				auto& t1 = Tmat.get_elem(k, j);
				auto& t2 = Tmat.get_elem((k + 1), j);
				auto t1_old = t1;
				t1 = c * t1 + s * t2;
				t2 = -s * t1_old + c * t2;
			}

			for (size_t i = 0; i < n; ++i) {
				auto& t1 = Tmat.get_elem(i, k);
				auto& t2 = Tmat.get_elem(i, (k + 1));
				auto t1_old = t1;
				t1 = c * t1 - s * t2;
				t2 = s * t1_old + c * t2;
			}
		}

		return std::make_pair(Tmat, Q);
	}
#endif

	inline constexpr auto householder_vector_francis_qr_2x2_complex(const ComplexLike auto& x0, const ComplexLike auto& x1) {
		using value_type = std::decay_t<decltype(x0)>;
		auto norm_sq = std::norm(x0) + std::norm(x1);
		auto norm = std::sqrt(norm_sq);
		constexpr auto zero = value_type(0);
		constexpr auto one = value_type(1);

		if (norm == 0) {
			return std::make_tuple(zero, zero, zero);
		}

		auto s = (x0 != zero) ? (x0 / std::abs(x0)) : one;
		auto v0 = x0 + s * norm;
		auto v1 = x1;
		auto beta = one / (norm * (norm + std::abs(x0)));
		return std::make_tuple(v0, v1, beta);
	}

	auto francis_qr_householder(const MatrixLike auto& A) {
		using matrix_type = std::decay_t<decltype(A)>;
		using value_type = matrix_type::value_type;

		constexpr auto householder2 = [](const auto& x0, const auto& x1) {
			return householder_vector_francis_qr_2x2_complex(x0, x1);
		};

		constexpr auto zero = static_cast<value_type>(0);
		const size_t n = A.cols();

		auto R = A;
		auto Q = identity<value_type>(n);

		auto shift = wilkinson_shift_complex(R);

		auto u = std::vector<value_type>(n);   // row vector v^T * A
		auto p = std::vector<value_type>(n);   // column vector A * v
		{
			auto x0 = R.get_elem(0, 0) - shift;
			auto x1 = R.get_elem(1, 0);
			auto [v0, v1, beta] = householder2(x0, x1);
			if (std::abs(beta) != 0) {
				for (size_t j = 0; j < n; ++j) {
					auto& R_k_j = R.get_elem(0, j);
					auto& R_k1_j = R.get_elem((0 + 1), j);
					u[j] = v0 * R_k_j + v1 * R_k1_j;
				}

				for (size_t j = 0; j < n; ++j) {
					auto& R_k_j = R.get_elem(0, j);
					auto& R_k1_j = R.get_elem((0 + 1), j);
					R_k_j -= beta * v0 * u[j];
					R_k1_j -= beta * v1 * u[j];
				}

				for (size_t i = 0; i < n; ++i) {
					auto& R_i_k = R.get_elem(i, 0);
					auto& R_i_k1 = R.get_elem(i, (0 + 1));
					p[i] = R_i_k * v0 + R_i_k1 * v1;
				}

				for (size_t i = 0; i < n; ++i) {
					auto& R_i_k = R.get_elem(i, 0);
					auto& R_i_k1 = R.get_elem(i, (0 + 1));
					R_i_k -= beta * p[i] * v0;
					R_i_k1 -= beta * p[i] * v1;
				}

				// --- accumulate Q = Q * H ---
				// Q * H = Q - beta * (Q * v) * v^T
				for (size_t i = 0; i < n; ++i) {
					auto& Q_i_k = Q.get_elem(i, 0);
					auto& Q_i_k1 = Q.get_elem(i, (0 + 1));
					u[i] = Q_i_k * v0 + Q_i_k1 * v1;   // (Q*v)_i
				}

				for (size_t i = 0; i < n; ++i) {
					auto& Q_i_k = Q.get_elem(i, 0);
					auto& Q_i_k1 = Q.get_elem(i, (0 + 1));
					Q_i_k -= beta * u[i] * v0;
					Q_i_k1 -= beta * u[i] * v1;
				}
			}
		}

		for (size_t k = 1; k < n - 1; ++k) {
			auto x0 = R.get_elem(k, (k - 1));
			auto x1 = R.get_elem((k + 1), (k - 1));
			auto [v0, v1, beta] = householder2(x0, x1);

			for (size_t j = 0; j < n; ++j) {
				auto& R_k_j = R.get_elem(k, j);
				auto& R_k1_j = R.get_elem((k + 1), j);
				u[j] = v0 * R_k_j + v1 * R_k1_j;
			}

			for (size_t j = 0; j < n; ++j) {
				auto& R_k_j = R.get_elem(k, j);
				auto& R_k1_j = R.get_elem((k + 1), j);
				R_k_j -= beta * v0 * u[j];
				R_k1_j -= beta * v1 * u[j];
			}

			for (size_t i = 0; i < n; ++i) {
				auto& R_i_k = R.get_elem(i, k);
				auto& R_i_k1 = R.get_elem(i, (k + 1));
				p[i] = R_i_k * v0 + R_i_k1 * v1;
			}

			for (size_t i = 0; i < n; ++i) {
				auto& R_i_k = R.get_elem(i, k);
				auto& R_i_k1 = R.get_elem(i, (k + 1));
				R_i_k -= beta * p[i] * v0;
				R_i_k1 -= beta * p[i] * v1;
			}

			// --- accumulate Q = Q * H ---
			// Q * H = Q - beta * (Q * v) * v^T
			for (size_t i = 0; i < n; ++i) {
				auto& Q_i_k = Q.get_elem(i, k);
				auto& Q_i_k1 = Q.get_elem(i, (k + 1));
				u[i] = Q_i_k * v0 + Q_i_k1 * v1;   // (Q*v)_i
			}

			for (size_t i = 0; i < n; ++i) {
				auto& Q_i_k = Q.get_elem(i, k);
				auto& Q_i_k1 = Q.get_elem(i, (k + 1));
				Q_i_k -= beta * u[i] * v0;
				Q_i_k1 -= beta * u[i] * v1;
			}
		}

		return std::make_pair(Q, R);
	}


	auto francis_qr_householder_spec_shift(const MatrixLike auto& A, const ScalarLike auto& scalar) {
		using matrix_type = std::decay_t<decltype(A)>;
		using value_type = matrix_type::value_type;

		constexpr auto householder2 = [](const auto& x0, const auto& x1) {
			return householder_vector_francis_qr_2x2_complex(x0, x1);
			};

		constexpr auto zero = static_cast<value_type>(0);
		const size_t n = A.cols();

		auto R = A;
		auto Q = identity<value_type>(n);

		auto shift = scalar;

		auto u = std::vector<value_type>(n);   // row vector v^T * A
		auto p = std::vector<value_type>(n);   // column vector A * v
		{
			auto x0 = R.get_elem(0, 0) - shift;
			auto x1 = R.get_elem(1, 0);
			auto [v0, v1, beta] = householder2(x0, x1);
			if (std::abs(beta) != 0) {
				for (size_t j = 0; j < n; ++j) {
					auto& R_k_j = R.get_elem(0, j);
					auto& R_k1_j = R.get_elem((0 + 1), j);
					u[j] = v0 * R_k_j + v1 * R_k1_j;
				}

				for (size_t j = 0; j < n; ++j) {
					auto& R_k_j = R.get_elem(0, j);
					auto& R_k1_j = R.get_elem((0 + 1), j);
					R_k_j -= beta * v0 * u[j];
					R_k1_j -= beta * v1 * u[j];
				}

				for (size_t i = 0; i < n; ++i) {
					auto& R_i_k = R.get_elem(i, 0);
					auto& R_i_k1 = R.get_elem(i, (0 + 1));
					p[i] = R_i_k * v0 + R_i_k1 * v1;
				}

				for (size_t i = 0; i < n; ++i) {
					auto& R_i_k = R.get_elem(i, 0);
					auto& R_i_k1 = R.get_elem(i, (0 + 1));
					R_i_k -= beta * p[i] * v0;
					R_i_k1 -= beta * p[i] * v1;
				}

				// --- accumulate Q = Q * H ---
				// Q * H = Q - beta * (Q * v) * v^T
				for (size_t i = 0; i < n; ++i) {
					auto& Q_i_k = Q.get_elem(i, 0);
					auto& Q_i_k1 = Q.get_elem(i, (0 + 1));
					u[i] = Q_i_k * v0 + Q_i_k1 * v1;   // (Q*v)_i
				}

				for (size_t i = 0; i < n; ++i) {
					auto& Q_i_k = Q.get_elem(i, 0);
					auto& Q_i_k1 = Q.get_elem(i, (0 + 1));
					Q_i_k -= beta * u[i] * v0;
					Q_i_k1 -= beta * u[i] * v1;
				}
			}
		}

		for (size_t k = 1; k < n - 1; ++k) {
			auto x0 = R.get_elem(k, (k - 1));
			auto x1 = R.get_elem((k + 1), (k - 1));
			auto [v0, v1, beta] = householder2(x0, x1);

			for (size_t j = 0; j < n; ++j) {
				auto& R_k_j = R.get_elem(k, j);
				auto& R_k1_j = R.get_elem((k + 1), j);
				u[j] = v0 * R_k_j + v1 * R_k1_j;
			}

			for (size_t j = 0; j < n; ++j) {
				auto& R_k_j = R.get_elem(k, j);
				auto& R_k1_j = R.get_elem((k + 1), j);
				R_k_j -= beta * v0 * u[j];
				R_k1_j -= beta * v1 * u[j];
			}

			for (size_t i = 0; i < n; ++i) {
				auto& R_i_k = R.get_elem(i, k);
				auto& R_i_k1 = R.get_elem(i, (k + 1));
				p[i] = R_i_k * v0 + R_i_k1 * v1;
			}

			for (size_t i = 0; i < n; ++i) {
				auto& R_i_k = R.get_elem(i, k);
				auto& R_i_k1 = R.get_elem(i, (k + 1));
				R_i_k -= beta * p[i] * v0;
				R_i_k1 -= beta * p[i] * v1;
			}

			// --- accumulate Q = Q * H ---
			// Q * H = Q - beta * (Q * v) * v^T
			for (size_t i = 0; i < n; ++i) {
				auto& Q_i_k = Q.get_elem(i, k);
				auto& Q_i_k1 = Q.get_elem(i, (k + 1));
				u[i] = Q_i_k * v0 + Q_i_k1 * v1;   // (Q*v)_i
			}

			for (size_t i = 0; i < n; ++i) {
				auto& Q_i_k = Q.get_elem(i, k);
				auto& Q_i_k1 = Q.get_elem(i, (k + 1));
				Q_i_k -= beta * u[i] * v0;
				Q_i_k1 -= beta * u[i] * v1;
			}
		}

		return std::make_pair(Q, R);
	}

	template<typename T> constexpr auto get_epsilon() {
		if constexpr (ComplexLike<T>) {
			using val_T = T::value_type;
			constexpr auto eps = std::numeric_limits<val_T>::epsilon() * val_T(3);
			return eps;
		}	else {
			constexpr auto eps = std::numeric_limits<T>::epsilon() * T(3);
			return eps;
		}
	}

	auto self_values(const MatrixLike auto& A, const auto& allocator) {
		using matrix = std::decay_t<decltype(A)>;
		using value_type = matrix::value_type;
		using allocator_type = matrix::allocator_type;
		using result_value_type = value_type;
		using result_allocator_type = rebind_allocator<std::decay_t<decltype(allocator)>, result_value_type>;

		auto [_, H] = hessenberg_form(A);
		constexpr auto eps = get_epsilon<value_type>();
		const auto r = H.rows();

		auto I = identity<value_type>(H.rows());
		auto Q_total = I;

		std::vector<result_value_type, result_allocator_type> result;
		result.reserve(r);
		auto m = r;
		auto H_curr = H.get_submatrix(H[0][0], H[r - 1][r - 1]);
		while (m > 2) {
			H.print();
			if (std::abs(H_curr[m - 1][m - 2]) <= eps) {
				result.push_back(H_curr[m - 1][m - 1]);
				--m;
				if (m == 0) break;
				H_curr = H_curr.get_submatrix(H_curr[0][0], H_curr[m - 1][m - 1]);
			} else if (m > 2 && std::abs(H_curr[m - 2][m - 3]) <= eps) {
				auto a = H_curr[m - 2][m - 2];
				auto b = H_curr[m - 2][m - 1];
				auto c = H_curr[m - 1][m - 2];
				auto d = H_curr[m - 1][m - 1];

				auto [lambda1, lambda2] = self_values_2x2_(a, b, c, d);

				result.push_back(lambda1);
				result.push_back(lambda2);
				m -= 2;
				if (m == 0)	break;
				H_curr = H_curr.get_submatrix(H_curr[0][0], H_curr[m - 1][m - 1]);
			} else {
				auto [Q, R] = francis_qr_householder(H_curr);
				auto Q_tmp = I;
				Q_tmp.set_submatrix(Q_tmp[0][0], Q);
				Q_total = Q_total * Q_tmp;
				H.set_submatrix(H[0][0], R);
			}
		}
		H.print();
		auto [Q, R] = francis_qr_householder(H);
		H = R;
		H.print();
		if (m == 2) {
			auto [r1, r2] = self_values_2x2_(
				H[0][0],
				H[0][1],
				H[1][0],
				H[1][1]
			);
			result.push_back(r1);
			result.push_back(r2);
			return result;
		}

		if (m == 1) {
			result.push_back(H[0][0]);
		} 

		
	}

#if 0
	auto fransic_qr(const MatrixLike auto& A) {
		using matrix = std::decay_t<decltype(A)>;
		using value_type = matrix::value_type;
		using allocator_type = matrix::allocator_type;
		auto&& [ T, Q_ ] = hessenberg_form(A);
		using result_value_type = value_type;
		using result_allocator_type = rebind_allocator<allocator_type, result_value_type>;

		constexpr auto thr = static_cast<value_type>(3);
		constexpr auto eps = std::numeric_limits<value_type>::epsilon() * thr;
		auto converged = [eps](const MatrixLike auto& mtx) {

		};

		auto I = identity<value_type, allocator_type>(A.rows(), allocator_type{});
		std::cout << "T:" << std::endl;
		T.print();
		std::cout << " ----------------------- " << std::endl;
		for (int i = 0; !converged(T); ++i) {
			auto mu = wilkinson_shift(A);
			auto I_mu = I * mu;
			T = T - I_mu;
			auto&& [Q, R] = householder_qr_decomposition(T);
			T = R * Q;
			T = T + I_mu;
			std::cout << "T" << i << ":" << std::endl;
			T.print();
			std::cout << " ----------------------- " << std::endl;
		}

		return std::make_pair(std::move(T), std::move(Q_));
	}
#endif

	auto self_values(const MatrixLike auto& A) {
		return self_values(A, std::pmr::polymorphic_allocator<std::byte>{});
	}

#if 1
	inline auto schur_decomposition(const MatrixLike auto& A, const AllocatorLike auto& allocator) {
		using matrix = std::decay_t<decltype(A)>;
		using value_type = matrix::value_type;
		using allocator_type = rebind_allocator<std::decay_t<decltype(allocator)>, value_type>;

		using value_type = typename std::decay_t<decltype(A)>::value_type;
		const size_t n = A.cols();



	}

	inline auto schur_decomposition(const MatrixLike auto& A) {
		return schur_decomposition(A, std::pmr::polymorphic_allocator<std::byte>{});
	}
#endif

	constexpr void expm_i(const MatrixLike auto& A_, MatrixLike auto& R) {
		using matrix = std::decay_t<decltype(A_)>;
		using size_type = matrix::size_type;
		using value_type = matrix::value_type;

		// expm via scaling & squaring + Padé [13/13]
		if (A_.rows() != A_.cols()) { throw std::runtime_error("expm: matrix must be square"); }

		// compute infinity norm
		auto A_norm = infinity_norm(A_);

		// theta13 from Higham (approx)
		// conservative
		constexpr auto zero_st = static_cast<size_type>(0);
		constexpr auto one = static_cast<value_type>(1);
		constexpr auto theta13 = static_cast<value_type>(4.25); // for double ~4.25
		constexpr auto theta13_for_float = static_cast<value_type>(3.7); // for float ~3.7
		constexpr auto theta = std::is_same_v<value_type, float> ? theta13_for_float : theta13;

		auto A = A_;

		int s = static_cast<int>(zero_st);
		if (A_norm > theta) {
			s = std::max(s, static_cast<int>(std::ceil(std::log2(A_norm / theta))));
			// scale A
			auto scale = std::ldexp(one, -s);
			A = A * scale;
		}

		// build U and V for Pade
		// Padé [13/13] coefficients for exp (Higham)
		// Using coefficients c0..c13
		constexpr auto c0 = static_cast<value_type>(64764752532480000.0);
		constexpr auto c1 = static_cast<value_type>(32382376266240000.0);
		constexpr auto c2 = static_cast<value_type>(7771770303897600.0);
		constexpr auto c3 = static_cast<value_type>(1187353796428800.0);
		constexpr auto c4 = static_cast<value_type>(129060195264000.0);
		constexpr auto c5 = static_cast<value_type>(10559470521600.0);
		constexpr auto c6 = static_cast<value_type>(670442572800.0);
		constexpr auto c7 = static_cast<value_type>(33522128640.0);
		constexpr auto c8 = static_cast<value_type>(1323241920.0);
		constexpr auto c9 = static_cast<value_type>(40840800.0);
		constexpr auto c10 = static_cast<value_type>(960960.0);
		constexpr auto c11 = static_cast<value_type>(16380.0);
		constexpr auto c12 = static_cast<value_type>(182.0);
		constexpr auto c13 = static_cast<value_type>(1.0);

		auto I = identity<value_type>(A.rows());

		// U = A * (c1*I + c3*A2 + c5*A4 + c7*A6 + c9*A8 + c11*A10 + c13*A12)
		// V = c0*I + c2*A2 + c4*A4 + c6*A6 + c8*A8 + c10*A10 + c12*A12

		// compute powers: A2, A4, A6, A8, A10, A12
		auto A2 = A * A;
		auto A4 = A2 * A2;
		auto A6 = A4 * A2;
		auto A8 = A4 * A4;
		auto A10 = A8 * A2;
		auto A12 = A8 * A4;

		// Prepare V
		auto V = I * c0;
		V = V * (A2 * c2);
		V = V * (A4 * c4);
		V = V * (A6 * c6);
		V = V * (A8 * c8);
		V = V * (A10 * c10);
		V = V * (A12 * c12);

		auto Umat = A * (I * c1
			+ A2 * c3
			+ A4 * c5
			+ A6 * c7
			+ A8 * c9
			+ A10 * c11
			+ A12 * c13
			);

		// compute (V - U)^{-1} * (V + U)
		auto numer = V + Umat;
		auto denom = V - Umat;

		auto denom_inv = inverse(denom);
		R = denom_inv * numer;

		// undo scaling by repeated squaring
		for (size_type i = 0; i < s; ++i) {
			R = R * R;
		}

		// return R;
	}

	inline auto expm(const MatrixLike auto& A, const AllocatorLike auto& allocator) {
		using matrix = std::decay_t<decltype(A)>;
		using value_type = matrix::value_type;
		using allocator_type = rebind_allocator<std::decay_t<decltype(allocator)>, value_type>;
		Matrix<value_type, allocator_type> R(A.rows(), A.cols());
		expm_i(A, R);
		return R;
	}

	inline auto expm(const MatrixLike auto& A) {
		return expm(A, std::pmr::polymorphic_allocator<std::byte>{});
	}

	template<typename T, typename Alloc = std::allocator<T>>
	constexpr auto lu_i(const Matrix<T>& A, Matrix<T>& L, Matrix<T>& U, Matrix<T>& P) { // -> std::tuple<Matrix<T, Alloc>, Matrix<T, Alloc>, Matrix<T, Alloc>> {
		using matrix = std::decay_t<decltype(A)>;
		using size_type = matrix::size_type;
		using value_type = matrix::value_type;

		if (!(A.is_square())) { throw std::runtime_error("LU decomposition is only defined for square matrices"); }
		const auto n = A.rows();

		L = identity<T, Alloc>(n);
		U = A;
		P = identity<T, Alloc>(n);

		auto swap_rows = [&L, &U, &P](size_type current_row, size_type max_value_row) {
			utils::swap_r(U.row(current_row), U.row(max_value_row));
			utils::swap_r(P.row(current_row), P.row(max_value_row));
			utils::swap_r(L.row(current_row), L.row(max_value_row));
		};

		auto process = [&U, &L, n](size_type k) {
			auto Uck = U.col(k);
			auto Urk = U.row(k);
			auto Lck = L.col(k);
			auto& Ukk = Urk[k];
			// auto& Ukk = Uck[k];
			for (size_type i = k + 1; i < n; ++i) {
				Lck[i] = Uck[i] / Ukk;
				auto Uri = U.row(i);
				auto& Lik = Lck[i];
				for (size_type j = k; j < n; ++j) {
					Uri[j] -= Lik * Urk[j];
				}
			}
		};

		std::ranges::for_each(U.col_range(), [&U, &swap_rows, &process](auto&& it) {
			auto [r, c] = U.get_indices(it[0]);
			auto mv_row = U.get_elem_row_index(max_abs(it));
			if (mv_row != c) { swap_rows(c, mv_row); }
			process(c);
		});

		// return std::make_tuple(P, L, U);
	}

	template <typename T, typename Alloc = std::allocator<T>>
	constexpr auto lu(const Matrix<T>&A) {
		using X = Matrix<T, Alloc>;
		auto t = std::tuple<X, X, X>();
		auto& [L, U, P] = t;
		lu_i<T, Alloc>(A, L, U, P);
		return t;
	}

	template<typename T, typename Alloc = std::allocator<T>>
	constexpr auto lu_solve(const Matrix<T>& A, const Matrix<T>& B) -> Matrix<T, Alloc> {
		using matrix = std::decay_t<decltype(A)>;
		using size_type = matrix::size_type;
		using value_type = matrix::value_type;

		if (A.rows() != A.cols()) {
			throw std::runtime_error("Matrix must be square for LU solve");
		}

		if (A.rows() != B.rows()) {
			throw std::runtime_error("Matrix and right-hand side dimensions don't match");
		}

		// Perform LU decomposition
		auto [P, L, U] = lu<T, Alloc>(A);

		const auto n = A.rows();
		const auto m = B.cols();

		// Apply permutation to B: PB
		auto PB = P * B;

		// Forward substitution for each column: LY = PB
		auto Y = Matrix<T, Alloc>(n, m);
		for (size_type col = 0; col < m; ++col) {
			auto PBcc = P.col(col);
			auto Ycc = Y.col(col);
			for (size_type i = 0; i < n; ++i) {
				auto sum = PBcc[i];
				auto Lri = L.row(i);
				auto& Lii = Lri[i];
				auto& Yci = Ycc[i];

				for (size_type j = 0; j < i; ++j) {
					sum -= Lri[j] * Ycc[j];
				}
				Yci = sum / Lii;
			}
		}

		// Backward substitution for each column: UX = Y
		auto X = Matrix<T, Alloc>(n, m);
		for (size_type col = 0; col < m; ++col) {
			auto Ycc = Y.col(col);
			auto Xcc = X.col(col);
			for (size_type i = n; i-- > 0;) {
				auto sum = Ycc[i];
				auto Uri = U.row(i);
				auto& Uii = Uri[i];
				for (size_type j = i + 1; j < n; ++j) {
					sum -= Uri[j] * Xcc[j];
				}
				Xcc[i] = sum / Uii;
			}
		}

		return X;
	}

	template<std::floating_point U, typename Alloc = std::allocator<U>>
	inline auto sqrtm_Denman_Beavers(const Matrix<U>& A) -> Matrix<U, Alloc> {
		using matrix = std::decay_t<decltype(A)>;
		using size_type = matrix::size_type;
		using value_type = matrix::value_type;

		const auto n = A.rows();
		const auto m = A.cols();
		if (n != m) { throw std::runtime_error("sqrtm: matrix must be square"); }


		constexpr auto max_iters = static_cast<size_type>(100);
		constexpr auto tol = static_cast<value_type>(1e-7);

		const auto zero = static_cast<value_type>(0);
		const auto zeroFive = static_cast<value_type>(0.5);

		auto Y = A;
		auto Z = identity<value_type, Alloc>(n);

		for (size_type k = 0; k < max_iters; ++k) {
			auto Y_next = (Y + inverse(Y)) * zeroFive;
			auto Z_next = (Z + inverse(Z)) * zeroFive;

			auto diff = Y_next - Y;
			auto err = zero;
			for (size_type i = 0; i < n; ++i) {
				for (size_type j = 0; j < n; ++j) {
					err += std::abs(diff[i][j]);
				}
			}

			if (err < tol) return Y_next;

			Y = std::move(Y_next);
			Z = std::move(Z_next);
		}

		return Y;
	}

	template<typename T, typename Alloc = std::allocator<T>>
	inline auto inverse_old(const Matrix<T>& A) -> Matrix<T, Alloc> {
		using matrix = std::decay_t<decltype(A)>;
		using size_type = matrix::size_type;
		using value_type = matrix::value_type;

		const auto n = A.rows();
		const auto m = A.cols();

		if (n != m) { throw std::runtime_error("Matrix inverse is only defined for square matrices"); }

		auto result = identity<value_type, Alloc>(n);
		auto temp = A; 

		// Gaussian elimination with partial pivoting
		for (size_type k = 0; k < n; ++k) {
			// Find pivot row
			size_type max_row = k;
			auto max_val = std::abs(temp[k][k]);

			for (size_type i = k + 1; i < n; ++i) {
				auto abs_val = std::abs(temp[i][k]);
				if (abs_val > max_val) {
					max_val = abs_val;
					max_row = i;
				}
			}


			if (max_val < std::numeric_limits<value_type>::epsilon()) {
				throw std::runtime_error("Matrix is singular or nearly singular");
			}

			// Swap rows if necessary
			if (max_row != k) {
				for (size_type j = 0; j < n; ++j) {
					std::swap(temp[k][j], temp[max_row][j]);
					std::swap(result[k][j], result[max_row][j]);
				}
			}

			// Normalize pivot row
			auto pivot = temp[k][k];
			for (size_type j = k; j < n; ++j) {
				temp[k][j] /= pivot;
			}
			for (size_type j = 0; j < n; ++j) {
				result[k][j] /= pivot;
			}

			// Eliminate below and above
			for (size_type i = 0; i < n; ++i) {
				if (i == k) continue;
				auto factor = temp[i][k];
				for (size_type j = k; j < n; ++j) {
					temp[i][j] -= factor * temp[k][j];
				}
				for (size_type j = 0; j < n; ++j) {
					result[i][j] -= factor * result[k][j];
				}
			}
		}

		return result;
	}

	template<typename T, typename Alloc = std::allocator<T>>
	inline auto matrixDiv(const Matrix<T>& A, const Matrix<T>& B) -> Matrix<T, Alloc> {
		auto&& [Q, R] = householder_qr_decomposition(B);
		// X = A * R⁻¹ * Qᵀ
		auto X = A * inverse<T, Alloc>(R) * Q.transpose();
		return X;
	}


#if 0
#if 1
	bool inBound(int i, int j, const std::vector<std::vector<int>>& matrix) {
		return 0 <= i && i < matrix.size() && 0 <= j && j < matrix[i].size();
	}
#else
	template<typename T>
	bool inBound(const std::signed_integral auto& i, const std::signed_integral auto& j, const Matrix<T>& mtx) {
		using value_type = std::decay_t<decltype(mtx)>::value_type;
		constexpr auto zero = static_cast<value_type>(0);

		constexpr bool fst = zero <= i;
		constexpr bool snd = i < mtx.size();
		constexpr bool trd = zero <= j;
		constexpr bool fth = j < mtx.row().size();

		return fst && snd && trd && fth;
	}
#endif
#endif


#if 0
	bool dfs(
		int i, int j,
		const std::pair<int, int>& end,
		std::vector<std::vector<bool>>& used,
		const std::vector<std::vector<int>>& matrix
	) {
		if (
			!inBound(i, j, matrix) ||
			matrix[i][j] == 1 ||
			used[i][j]
			) {
			return false;
		}

		used[i][j] = true;
		if (i == end.first && j == end.second) {
			return true;
		}

		return dfs(i - 1, j, end, used, matrix) || // upper
			dfs(i, j + 1, end, used, matrix) || // right
			dfs(i + 1, j, end, used, matrix) || // lower
			dfs(i, j + 1, end, used, matrix) ; // left
	}

	void colored_dfs(
		int i, int j,
		int color,
		std::vector<std::vector<bool>>& used,
		const std::vector<std::vector<int>>& matrix
	) {
		if (
			!inBound(i, j, matrix) ||
			matrix[i][j] != color ||
			used[i][j]
			) {
			return;
		}

		used[i][j] = true;

		colored_dfs(i - 1, j, color, used, matrix); // upper
		colored_dfs(i, j + 1, color, used, matrix); // right
		colored_dfs(i + 1, j, color, used, matrix); // lower
		colored_dfs(i, j + 1, color, used, matrix); // left
	}
#endif

#if 0
	bool hasPath(
		const std::vector<std::vector<int>>& matrix,
		std::pair<int, int> start,
		std::pair<int, int> end
	) {
		std::vector<std::vector<bool>> used(matrix.size(), std::vector<bool>(matrix[0].size(), false));
		return dfs(start.first, start.second, end, used, matrix);
	}
#endif


} // ns

#endif
