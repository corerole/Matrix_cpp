#pragma once
#ifndef MATRIX_HELPERS_HPP
#define MATRIX_HELPERS_HPP

#include "Matrix.hpp"

#include <numeric>

void helpers_tests();

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

	template<typename T, typename Alloc = std::allocator<std::complex<T>>>
	constexpr Matrix<std::complex<T>, Alloc> to_complex(const Matrix<T>& A) {
		auto cm = Matrix<std::complex<T>, Alloc>(A.rows(), A.cols());
		std::ranges::copy(A, cm.begin());
		return cm;
	}

	constexpr auto pow(auto base, auto exp) {
		return std::pow(base, exp);
	}

	constexpr auto sqrt(auto val) {
		return std::sqrt(val);
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
	// O(N)
#if 0
	template<utils::arithmetic T, typename Alloc = std::allocator<T>>
#else
	template<typename T, typename Alloc = std::allocator<T>>
#endif
	constexpr auto identity(size_t n) -> Matrix<T, Alloc> {
		auto result = Matrix<T, Alloc>(n, n);
		constexpr auto one = static_cast<T>(1);
		std::ranges::for_each(result.diagonal_range(), [one](auto&& it) {it = one; });
		return result;
	}

	// O(N)
	constexpr auto sum(std::ranges::input_range auto&& r) {
		using value_type = std::ranges::range_value_t<decltype(r)>;
		return std::accumulate(r.begin(), r.end(), static_cast<value_type>(0));
	}

	// O(N)
	constexpr auto abs_sum(std::ranges::input_range auto&& r) {
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

	// O(N)
	constexpr auto& max(utils::arithmetic_input_range auto&& r) {
		using value_type = std::ranges::range_value_t<decltype(r)>;
		auto max_val = &(*r.begin());
		std::ranges::for_each(r, [&max_val](auto&& it) {
			max_val = (((*max_val) > (it)) ? max_val : &it);
		});
		return *max_val;
	}

	template<utils::arithmetic value_type>
	constexpr void scale_matrix_self(Matrix<value_type>& A) {
		constexpr auto zero = static_cast<value_type>(0);
		constexpr auto one = static_cast<value_type>(1);
		auto max_val = max(A);
		auto scale = one / max_val;
		A *= scale;
	}

	template<utils::arithmetic value_type, typename Alloc = std::allocator<value_type>>
	constexpr Matrix<value_type, Alloc> scale_matrix(const Matrix<value_type>& A) {
		auto B = A;
		scale_matrix_self(B);
		return B;
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

	// O(N)
	constexpr auto euclid_norm(utils::arithmetic_input_range auto&& r) {
		using value_type = std::ranges::range_value_t<decltype(r)>;
		auto acc = static_cast<value_type>(0);
		std::ranges::for_each(r, [&acc](auto&& it) { acc += it * it; });
		return utils::sqrt<value_type>(acc);
	}

	constexpr auto euclid_norm(utils::complex_input_range auto&& r) {
		using value_type = std::ranges::range_value_t<decltype(r)>::value_type;
		auto acc = static_cast<value_type>(0);
		std::ranges::for_each(r, [&acc](auto&& it) {
			const auto t = std::abs(it);
			acc += t * t;
		});
		return utils::sqrt<value_type>(acc);
	}
	
	template<utils::arithmetic T, typename Alloc = std::allocator<T>>
	constexpr auto matrix_euclid_col_norm(const Matrix<T>& A) {
		auto x = std::vector<T, Alloc>();
		x.reserve(A.cols());
		std::ranges::for_each(A.col_range(), [&x](auto&& it) { x.push_back(euclid_norm(it)); });
		return x;
	}

	template<utils::complex T, typename Alloc = std::allocator<typename T::value_type>>
	constexpr auto complex_matrix_euclid_col_norm(const Matrix<T>& A) {
		using value_type = typename std::allocator_traits<Alloc>::value_type;
		auto x = std::vector<value_type, Alloc>();
		x.reserve(A.rows());
		std::ranges::for_each(A.col_range(), [&x](auto&& it) { x.push_back(euclid_norm(it)); });
		return x;
	}

	// O(N)
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

	// O(N)
	constexpr auto manhattan_norm(std::ranges::input_range auto&& r) {
		using value_type = std::ranges::range_value_t<decltype(r)>;
		return abs_sum(r);
	}

	// O(N)
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

	// O(N)
	constexpr auto infinity_norm(std::ranges::input_range auto&& r) { return max(r); }

	// O(N)
	constexpr auto mean(const utils::Container auto& cnt) {
		using value_type = typename std::decay_t<decltype(cnt)>::value_type;
		const auto s = cnt.size();
		const auto sum_ = sum(cnt);
		return (sum_ / static_cast<value_type>(s));
	}

	// O(N)
	constexpr auto abs_mean(const utils::Container auto& cnt) {
		const auto s = cnt.size();
		using value_type = typename std::decay_t<decltype(cnt)>::value_type;
		const auto acc = abs_sum(cnt);
		return acc / static_cast<value_type>(s);
	}

	// O(2N)
	// is sized range?
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

	// O(2N)
	constexpr auto standard_deviation(std::ranges::input_range auto&& r) {
		return utils::sqrt(variance(r));
	}

	// O(N)
	template<typename T>
	constexpr auto trace(const Matrix<T>& A) {
		return sum(A.diagonal_range());
	}

	// O(N)
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


	template<typename T>
	static constexpr auto determinant_impl(const Matrix<T>& A) {
		using matrix = std::decay_t<decltype(A)>;
		using size_type = matrix::size_type;
		using value_type = matrix::value_type;
#if 0
		constexpr auto eps = std::numeric_limits<value_type>::epsilon();

		auto temp = A;
		auto det = static_cast<value_type>(1);
		const auto r = temp.rows();

#if 0
		for (size_type k = 0; k < r; ++k) {
			auto tck = temp.col(k);
			auto trk = temp.row(k);
			auto tck_r = std::ranges::subrange(tck.begin() + k, tck.end());
			auto&& max_row = temp.get_elem_row(max_abs(tck_r));

			if (max_row != k) {
#if 1
				auto temp_mr = temp.row(max_row);
				auto temp_kr = temp.row(k);
				auto temp_mr_r = std::ranges::subrange(temp_mr.begin() + k, temp_mr.end());
				auto temp_kr_r = std::ranges::subrange(temp_kr.begin() + k, temp_kr.end());
				utils::swap_r(temp_kr_r, temp_mr_r);
#else
				for (size_type j = k; j < r; ++j) {
					std::swap(temp[k][j], temp[max_row][j]);
				}
#endif
				det = -det;
			}


			
			auto&& tkk = temp[k][k];
			for (size_type i = k + 1; i < r; ++i) {
				auto factor = tck[i] / tkk;
				for (size_type j = k + 1; j < r; ++j) {
					temp[i][j] -= factor * trk[j];
				}
			}

			det *= temp[k][k];
		}
#endif
		auto&& last_elem = temp[temp.rows() - 1][temp.rows() - 1];
		std::ranges::for_each(temp.diagonal_range(), [&temp, &det, &last_elem](auto&& it) {
			auto&& sub = temp.submatrix(it, last_elem);
			auto&& itr = sub.get_elem_row(it);
			auto&& max_row_elem = max_abs(itr);
			if (max_row_elem != it) {
				utils::swap_r(sub.get_elem_row(max_row_element), itr);
				det = -det;
			}

			std::ranges::for_each(sub.row_range(), [&sub, it](auto&& row) {
				auto factor = ;
				std::ranges::for_each(row, [](auto&& elem) {
						elem -= factor * 
				});
			});
		});
#endif
		return static_cast<value_type>(0);
	}

	template<typename T, typename Alloc = std::allocator<T>>
	constexpr auto determinant(const Matrix<T>& A) -> std::decay_t<decltype(A)>::value_type {
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

	template<typename T, typename Alloc = std::allocator<T>>
	constexpr auto col_centered(const Matrix<T>& A) -> Matrix<T, Alloc> {
		using size_type = std::size_t;
		const auto r = A.rows();
		const auto c = A.cols();
		auto result = Matrix<T, Alloc>(r, c);

		for (size_type i = 0; i < c; ++i) {
			auto ac = A.col(i);
			auto rc = result.col(i);
			auto cmean = mean(ac);
			for (size_type j = 0; j < r; ++j) {
				rc[j] = ac[j] - cmean;
			}
		}

		return result;
	}

	template<typename T, typename Alloc = std::allocator<T>>
	inline auto covariance(const Matrix<T>& A) -> Matrix<T, Alloc> {
		using matrix = std::decay_t<decltype(A)>;
		using size_type = matrix::size_type;
		using value_type = matrix::value_type;

		const auto r = A.rows();
		const auto c = A.cols();

		if (r <= 1) { throw std::runtime_error("Covariance requires at least 2 observations"); }

		auto centered = col_centered(A);

		return (centered.transpose() * centered) / static_cast<value_type>(r - 1);
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

		auto delta = (a_qq - a_pp) / two;
		auto tau = delta / abs_a_pq;
		auto sign_tau = (tau >= zero) ? one : -one;
		auto abs_tau = std::abs(tau);
		auto t = sign_tau / (abs_tau + std::sqrt(one + tau * tau));
		c = one / std::sqrt(one + t * t);
		auto sin_magnitude = t * c;
		auto phase = a_pq / abs_a_pq;
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

	template<std::floating_point T, typename Alloc = std::allocator<T>>
	constexpr auto svd_jacobi_real(const Matrix<T>& A)
		-> std::tuple<Matrix<T, Alloc>, Matrix<T, Alloc>, Matrix<T, Alloc>>
	{
		using matrix = std::decay_t<decltype(A)>;
		using size_type = matrix::size_type;
		using value_type = matrix::value_type;

		const auto m = A.rows();
		const auto n = A.cols();
		const auto p = std::min(m, n);

		auto A_c = A;

		auto U = Matrix<T, Alloc>(m, p);
		auto Sigma = Matrix<T, Alloc>(p, p);
		auto Vh = Matrix<T, Alloc>(p, n);

		auto V = identity<value_type>(n);

		constexpr auto eps = std::numeric_limits<value_type>::epsilon();
		constexpr auto zero = static_cast<value_type>(0);
		constexpr auto one = static_cast<value_type>(1);

		auto sweeps = 0;
		while (true) {
			std::cout << ++sweeps << std::endl;
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
			auto s = sigma[j];
			auto Ucj = U.col(j);
			auto Acj = A_c.col(j);
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
		std::ranges::for_each(Sigma.diagonal_range(), [sb = sigma.begin()](auto&& it) mutable {
			it = *sb;
			++sb;
		});
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


		return { U, Sigma, Vh };
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

	template<typename T, typename Alloc = std::allocator<T>>
	constexpr auto householder_vector(const std::vector<T>& x) -> std::vector<T, Alloc> {
		auto sigma = euclid_norm(x);
		auto u = x;
		u[0] = x[0] - sigma;
		auto u_norm = euclid_norm(u);
		for (auto&& elem : u) {	elem = elem / u_norm; }
		return u;
	}

	template<typename T, typename Alloc = std::allocator<T>>
	constexpr auto householder_reflection_matrix(const std::vector<T>& x) -> Matrix<T, Alloc> {
		using size_type = typename std::decay_t<decltype(x)>::size_type;
		const auto n = x.size();
		auto I = identity(n);
		auto v = householder_vector(x);
		auto vvT = Matrix<T, Alloc>(n, n);
		for (size_type i = 0; i < n; ++i) {
			for (size_type j = 0; j < n; ++j) {
				vvT[i][j] = v[i] * v[j];
			}
		}
		return I - static_cast<T>(2) * vvT;
	}

#if 1
	template<typename T, typename Alloc = std::allocator<T>>
	auto householder_qr_decomposition(const Matrix<T>& A) -> std::pair<Matrix<T, Alloc>, Matrix<T, Alloc>> {
		using matrix = std::decay_t<decltype(A)>;
		using size_type = matrix::size_type;
		using value_type = matrix::value_type;

		const auto m = A.rows();
		const auto n = A.cols();

		if (m < n) { throw std::runtime_error("QR decomposition requires m >= n"); }

		auto R = Matrix<value_type, Alloc>(A);
		auto Q = identity<value_type, Alloc>(m);

		for (size_type k = 0; k < std::min(m, n); ++k) {
			// 1.
			auto x = std::vector<value_type, Alloc>(m - k);
			for (size_type i = k; i < m; ++i) {
				x[i - k] = R[i][k];
			}

			// 2. 
			auto Hk = householder_reflection_matrix<value_type, Alloc>(x);

			// 3.
			// R[k:m, k:n] = Hk * R[k:m, k:n]
			auto R_block = R.get_submatrix(k, k, m - k, n - k);
			R.set_submatrix(k, k, Hk * R_block);

			// 4. Update Q: Q[:, k:m] = Q[:, k:m] * Hk^T
			// Hk^T = Hk
			auto Q_block = Q.get_submatrix(0, k, m, m - k);
			Q.set_submatrix(0, k, Q_block * Hk);
		}

		return { Q, R };
	}


#else
	template<typename U, typename Alloc = std::allocator<U>>
	auto householder_qr_decomposition(const Matrix<U>& A) -> std::pair<Matrix<U, Alloc>, Matrix<U, Alloc>> {
		using matrix = std::decay_t<decltype(A)>;
		using size_type = matrix::size_type;
		using value_type = matrix::value_type;

		const auto m = A.rows();
		const auto n = A.cols();

		if (m < n) { throw std::runtime_error("QR decomposition requires m >= n"); }

		auto R = Matrix<value_type, Alloc>(A);
		auto Q = identity<value_type, Alloc>(m);

		for (size_type k = 0; k < n && k < m - 1; ++k) {
			std::vector<value_type, Alloc> x(m - k);
			for (size_type i = 0; i < m - k; ++i) {
				x[i] = R[k + i][k];
			}

			auto norm_x = static_cast<value_type>(0);
			for (const auto& val : x) {
				norm_x += val * val;
			}
			norm_x = std::sqrt(norm_x);

			if (norm_x < std::numeric_limits<value_type>::epsilon()) continue;

			auto v = x;
			v[0] += (x[0] >= U(0) ? norm_x : -norm_x);

			U norm_v = U(0);
			for (const auto& val : v) {
				norm_v += val * val;
			}
			norm_v = std::sqrt(norm_v);

			if (norm_v < std::numeric_limits<value_type>::epsilon()) continue;

			for (auto& val : v) {
				val /= norm_v;
			}

			for (size_type j = k; j < n; ++j) {
				auto dot = U(0);
				for (size_type i = 0; i < m - k; ++i) {
					dot += v[i] * R[k + i][j];
				}
				for (size_type i = 0; i < m - k; ++i) {
					R[k + i][j] -= U(2) * v[i] * dot;
				}
			}

			for (size_type i = 0; i < m; ++i) {
				auto dot = U(0);
				for (size_type j = 0; j < m - k; ++j) {
					dot += Q[i][k + j] * v[j];
				}
				for (size_type j = 0; j < m - k; ++j) {
					Q[i][k + j] -= U(2) * v[j] * dot;
				}
			}
		}

		return { Q, R };
	}
#endif

	template<typename T, typename Alloc = std::allocator<T>>
	constexpr void expm_i(const Matrix<T>& A_, Matrix<T>& R) {
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
			A *= scale;
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

		auto I = identity<T, Alloc>(A.rows());

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
		V += A2 * c2;
		V += A4 * c4;
		V += A6 * c6;
		V += A8 * c8;
		V += A10 * c10;
		V += A12 * c12;

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

	template<typename T, typename Alloc = std::allocator<T>>
	inline auto expm(const Matrix<T>& A) -> Matrix<T, Alloc> {
		auto R = Matrix<T, Alloc>();
		expm_i<T, Alloc>(A, R);
		return R;
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


} // ns

#endif
