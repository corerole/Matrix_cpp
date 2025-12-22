#pragma once
#ifndef MATRIX_HELPERS_HPP
#define MATRIX_HELPERS_HPP

#include "Matrix.hpp"

#include <numeric>

void helpers_tests();

namespace utils {
	template<typename T> concept arithmetic = std::is_arithmetic_v<T>;

	template<typename T>
	concept Container = requires(const T a, T b) {
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
	

	constexpr auto pow(auto base, auto exp) {
		return std::pow(base, exp);
	}

	constexpr auto sqrt(auto val) {
		return std::sqrt(val);
	}
}

namespace helpers {
	template<typename T, typename Alloc = std::allocator<T>>
	inline auto identity(size_t n) -> Matrix<T, Alloc> {
		auto result = Matrix<T, Alloc>(n, n);
		constexpr auto one = static_cast<T>(1);
		for (size_t i = 0; i < n; ++i) { result[i][i] = one; }
		return result;
	}

	constexpr auto sum(const utils::Container auto& cnt) {
		using value_type = typename std::decay_t<decltype(cnt)>::value_type;
		return std::accumulate(cnt.begin(), cnt.end(), static_cast<value_type>(0));
	}

	constexpr auto abs_sum(const utils::Container auto& cnt) {
		using value_type = typename std::decay_t<decltype(cnt)>::value_type;
		auto sum = std::accumulate(cnt.begin(), cnt.end(),
			static_cast<value_type>(0),
			[](auto&& init, auto&& it) {
				init += std::abs(it);
			}
		);
		return sum;
	}

	constexpr auto euclid_norm(const utils::Container auto& cnt) {
		using value_type = typename std::decay_t<decltype(cnt)>::value_type;
		auto acc = static_cast<value_type>(0);
		std::ranges::for_each(cnt, [&acc](auto&& it) { acc += it * it; });
		return utils::sqrt<value_type>(acc);
	}


	constexpr auto manhattan_norm(const utils::Container auto& cnt) {
		return abs_sum(cnt);
	}

	constexpr auto lp_norm(const utils::Container auto& cnt, utils::arithmetic auto exp) {
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

	constexpr auto infinity_norm(const utils::Container auto& cnt) {
		using value_type = typename std::decay_t<decltype(cnt)>::value_type;
		auto max = static_cast<value_type>(0);
		std::ranges::for_each(cnt, [&max](auto&& it) {
			max = (it > max) ? it : max;
		});
		return max;
	}

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

	constexpr auto variance(const utils::Container auto& cnt) {
		const auto m = mean(cnt);
		const auto rmo = static_cast<decltype(m)>(cnt.size() - 1);
		auto acc = std::accumulate(cnt.begin(), cnt.end(),
			static_cast<decltype(m)>(0),
			[m](auto&& init, auto&& it) {
				init += (it - m) * (it - m);
			}
		);
		return acc / rmo;
	}

	constexpr auto standard_deviation(const utils::Container auto& cnt) {
		return utils::sqrt(variance(cnt));
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

	template<typename T, typename Alloc = std::allocator<T>>
	inline auto determinant(const Matrix<T>& A) -> std::decay_t<decltype(A)>::value_type {
		using matrix = std::decay_t<decltype(A)>;
		using size_type = matrix::size_type;
		using value_type = matrix::value_type;

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

		auto temp = A;
		auto det = static_cast<value_type>(1);

		for (size_type k = 0; k < r; ++k) {
			size_type max_row = k;
			auto max_val = std::abs(temp[k][k]);

			for (size_type i = k + 1; i < r; ++i) {
				auto abs_val = std::abs(temp[i][k]);
				if (abs_val > max_val) {
					max_val = abs_val;
					max_row = i;
				}
			}

			if (max_val < std::numeric_limits<value_type>::epsilon()) {
				return static_cast<value_type>(0);
			}

			if (max_row != k) {
				for (size_type j = k; j < r; ++j) {
					std::swap(temp[k][j], temp[max_row][j]);
				}
				det = -det;
			}

			for (size_type i = k + 1; i < r; ++i) {
				auto factor = temp[i][k] / temp[k][k];
				for (size_type j = k + 1; j < r; ++j) {
					temp[i][j] -= factor * temp[k][j];
				}
			}

			det *= temp[k][k];
		}
		return det;
	}

	template<typename T, typename Alloc = std::allocator<T>>
	inline auto col_centered(const Matrix<T>& A) -> Matrix<T, Alloc> {
		using matrix = std::decay_t<decltype(A)>;
		using size_type = matrix::size_type;
		using value_type = matrix::value_type;

		const auto r = A.rows();
		const auto c = A.cols();
		const auto col_means = mean_col(A);

		auto result = Matrix<T, Alloc>(r, c);

		for (size_type i = 0; i < r; ++i) {
			for (size_type j = 0; j < c; ++j) {
				result[i][j] = A[i][j] - col_means[0][j];
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


	template<std::floating_point Real, typename Alloc = std::allocator<Real>>
	auto svd_jacobi(const Matrix<Real>& A)
		-> std::tuple< Matrix<std::complex<Real>>, Matrix<Real>, Matrix<std::complex<Real>> >
	{

		using matrix = std::decay_t<decltype(A)>;
		using size_type = matrix::size_type;
		using value_type = matrix::value_type;

		constexpr auto max_sweeps = static_cast<size_type>(50);
		constexpr auto tol_factor = static_cast<value_type>(1e-12);

		using complex_t = std::complex<Real>;

		const auto m = A.rows();
		const auto n = A.cols();
		const auto p = std::min(m, n);

		// --- 1) Build complex copy of A (so same code works for real/complex T) ---
		auto A_c = Matrix<complex_t>(m, n);
		for (size_type i = 0; i < m; ++i) {
			for (size_type j = 0; j < n; ++j) {
				A_c[i][j] = complex_t(A[i][j]);
			}
		}

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

		// --- 4) One-sided Jacobi sweeps ---
		// We'll perform sweeps over pairs (p,q). Convergence when no rotation larger than threshold.
		for (size_type sweep = 0; sweep < max_sweeps; ++sweep) {
			bool any_rot = false;

			for (size_type pcol = 0; pcol + 1 < n; ++pcol) {
				for (size_type qcol = pcol + 1; qcol < n; ++qcol) {
					// compute Gram matrix entries for columns pcol and qcol:
					// a_pp = <col_p, col_p>, a_qq = <col_q, col_q>, a_pq = <col_p, col_q>
					complex_t a_pq = complex_t(0);
					Real a_pp = Real(0);
					Real a_qq = Real(0);

					for (size_type i = 0; i < m; ++i) {
						const complex_t ap = A_c[i][pcol];
						const complex_t aq = A_c[i][qcol];
						a_pq += std::conj(ap) * aq;
						a_pp += std::norm(ap);   // |ap|^2
						a_qq += std::norm(aq);   // |aq|^2
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

					// compute rotation parameters (see standard one-sided Jacobi derivation)
					// Let phi = arg(a_pq), g = |a_pq|
					const Real g = abs_a_pq;
					const Real delta = (a_qq - a_pp) / (Real)2;
					// compute t (real) using stable formula
					Real t;
					if (g == Real(0)) {
						t = Real(0);
					}
					else {
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
						const complex_t ap = A_c[i][pcol];
						const complex_t aq = A_c[i][qcol];

						complex_t new_p = complex_t(c) * ap - std::conj(s) * aq;
						complex_t new_q = s * ap + complex_t(c) * aq;

						A_c[i][pcol] = new_p;
						A_c[i][qcol] = new_q;
					}

					// --- Accumulate rotation into V: V = V * J (same combination on its columns)
					for (size_type i = 0; i < n; ++i) {
						const complex_t vp = V[i][pcol];
						const complex_t vq = V[i][qcol];

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
		std::vector<Real> sigma(p, Real(0));
		for (size_type j = 0; j < p; ++j) {
			Real ssum = Real(0);
			for (size_type i = 0; i < m; ++i) {
				ssum += std::norm(A_c[i][j]);
			}
			sigma[j] = std::sqrt(ssum);
		}

		// --- 6) Build U (m x p) as normalized columns of A_c ---
		Matrix<complex_t> U(m, p);
		for (size_type j = 0; j < p; ++j) {
			Real s = sigma[j];
			if (s > tol_abs) {
				for (size_type i = 0; i < m; ++i) {
					U[i][j] = A_c[i][j] / complex_t(s);
				}
			}
			else {
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
	inline auto find_max_val(Matrix<T>&& A) {
		using matrix = std::decay_t<decltype(A)>;
		using size_type = matrix::size_type;
		using value_type = matrix::value_type;

		const auto n = A.rows();
		const auto m = A.cols();

		// std::find/_if
		std::vector<matrix::pointer> pointers(n, nullptr);
		auto ctr = static_cast<matrix::difference_type>(0);
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

		return result;
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
		using complex_t = std::complex<Real>;

		constexpr auto eps = std::numeric_limits<Real>::epsilon();

		auto [U_c, Sigma, Vh_c] = svd_jacobi<value_type>(A);

		// Sigma is real diagonal (p x p), here p == n for square matrix
		// find largest singular value to form a relative tolerance
		auto&& max_sigma = find_max_val(Sigma);

		// tolerance: relative to largest singular value and machine eps
		auto tol = eps * std::max<Real>(Real(1), max_sigma) * static_cast<Real>(n);

		// build Sigma^{-1} as complex matrix (diagonal)
		auto Sigma_inv = Matrix<complex_t>(Sigma.rows(), Sigma.cols());

		for (size_type i = 0; i < Sigma.rows(); ++i) {
			//Sigma_inv[i][i] = complex_t(Real(1) / Sigma[i][i]); // (Sigma[i][i] <= tol) ? complex_t(Real(0)) : complex_t(Real(1) / Sigma[i][i]);

#if 1
			auto&& s = Sigma[i][i];
			if (s <= tol) {
				// Matrix is singular or nearly singular.
				// Option A: 
				throw std::runtime_error("Matrix is singular or nearly singular (small singular value)");

				// Option B (pseudo-inverse): comment out throw and use
				// Sigma_inv[i][i] = complex_t(Real(0)); // leave zero -> Moore-Penrose pseudo-inverse
			}
			else {
				Sigma_inv[i][i] = complex_t(Real(1) / s);
			}
#else
			Sigma_inv[i][i] = complex_t(Real(1) / s);
#endif
		}

		// Recover V from Vh: V = conj(Vh)^T
		auto V = Vh_c.transpose();
		std::for_each(V.begin(), V.end(), [](auto& it) { it = std::conj(it); });

		// Compute U^H (conjugate transpose of U)
		auto U_H = U_c.transpose();
		std::for_each(U_H.begin(), U_H.end(), [](auto& it) { it = std::conj(it); });

		// A^{-1} = V * Sigma_inv * U^H (complex matrix)
		auto Ainv_c = (V * Sigma_inv) * U_H;

		auto out = Matrix<T>(n, n);
		for (size_type i = 0; i < n; ++i) {
			for (size_type j = 0; j < n; ++j) {
				out[i][j] = static_cast<T>(Ainv_c[i][j].real());
			}
		}

		return out;
	}

	template<typename T>
	inline auto euclid_norm(const std::vector<T>& x) -> T {
		auto sigma = static_cast<T>(0);
		for (auto&& elem : x) {	sigma += elem * elem; }
		return (x[0] >= 0) ? -std::sqrt(sigma) : std::sqrt(sigma);
	}

	template<typename T, typename Alloc = std::allocator<T>>
	inline auto householder_vector(const std::vector<T>& x) -> std::vector<T, Alloc> {
		auto sigma = euclid_norm(x);
		auto u = x;
		u[0] = x[0] - sigma;
		auto u_norm = euclid_norm(u);
		for (auto&& elem : u) {	elem = elem / u_norm; }
		return u;
	}

	template<typename T, typename Alloc = std::allocator<T>>
	inline auto householder_reflection_matrix(const std::vector<T>& x) -> Matrix<T, Alloc> {
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
	inline auto expm(const Matrix<T>& A) -> Matrix<T, Alloc> {
		using matrix = std::decay_t<decltype(A)>;
		using size_type = matrix::size_type;
		using value_type = matrix::value_type;

		// expm via scaling & squaring + Padé [13/13]
		if (A.rows() != A.cols()) { throw std::runtime_error("expm: matrix must be square"); }

		// compute infinity norm
		auto A_norm = infinity_norm(A);

		// theta13 from Higham (approx)
		// conservative
		constexpr auto zero_st = static_cast<size_type>(0);
		constexpr auto one = static_cast<value_type>(1);
		constexpr auto theta13 = static_cast<value_type>(4.25); // for double ~4.25
		constexpr auto theta13_for_float = static_cast<value_type>(3.7); // for float ~3.7
		constexpr auto theta = std::is_same_v<value_type, float> ? theta13_for_float : theta13;


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

		auto denom_inv = denom.inverse();
		auto R = denom_inv * numer;

		// undo scaling by repeated squaring
		for (size_type i = 0; i < s; ++i) {
			R = R * R;
		}

		return R;
	}

	template<typename T, typename Alloc = std::allocator<T>>
	inline auto lu(const Matrix<T>& A) -> std::tuple<Matrix<T, Alloc>, Matrix<T, Alloc>, Matrix<T, Alloc>> {
		using matrix = std::decay_t<decltype(A)>;
		using size_type = matrix::size_type;
		using value_type = matrix::value_type;

		if (A.rows() != A.cols()) {
			throw std::runtime_error("LU decomposition is only defined for square matrices");
		}
		const size_type n = A.rows();

		auto L = identity<T, Alloc>(n);
		auto U = A;
		auto P = identity<T, Alloc>(n);
		for (size_type k = 0; k < n; ++k) {
			auto max_val = std::abs(U[k][k]);
			size_type p = k;
			for (size_type i = k + 1; i < n; ++i) {
				auto val = std::abs(U[i][k]);
				if (val > max_val) {
					max_val = val;
					p = i;
				}
			}

			if (max_val < std::numeric_limits<value_type>::epsilon()) {
				throw std::runtime_error("Matrix is singular or nearly singular");
			}

			if (p != k) {
				for (size_type j = 0; j < n; ++j) {
					std::swap(U[k][j], U[p][j]);
				}
				for (size_type j = 0; j < n; ++j) {
					std::swap(P[k][j], P[p][j]);
				}
				for (size_type j = 0; j < k; ++j) {
					std::swap(L[k][j], L[p][j]);
				}
			}

			for (size_type i = k + 1; i < n; ++i) {
				L[i][k] = U[i][k] / U[k][k];
				for (size_type j = k; j < n; ++j) {
					U[i][j] -= L[i][k] * U[k][j];
				}
			}
		}

		return std::make_tuple(P, L, U);
	}

	template<typename T, typename Alloc = std::allocator<T>>
	inline auto lu_solve(const Matrix<T>& A, const Matrix<T>& B) -> Matrix<T, Alloc> {
		using matrix = std::decay_t<decltype(A)>;
		using size_type = matrix::size_type;
		using value_type = matrix::value_type;

		if (A.rows() != A.cols()) {
			throw std::runtime_error("Matrix must be square for LU solve");
		}

		if (A.rows() != B.rows()) {
			throw std::runtime_error("Matrix and right-hand side dimensions don't match");
		}

		// Perform LU decomposition with pivoting
		auto [P, L, U] = lu<T, Alloc>(A);

		const auto n = A.rows();
		const auto m = B.cols();

		// Apply permutation to B: PB
		auto PB = P * B;

		// Forward substitution for each column: LY = PB
		auto Y = Matrix<T, Alloc>(n, m);
		for (size_type col = 0; col < m; ++col) {
			for (size_type i = 0; i < n; ++i) {
				auto sum = PB[i][col];
				for (size_type j = 0; j < i; ++j) {
					sum -= L[i][j] * Y[j][col];
				}
				Y[i][col] = sum / L[i][i];
			}
		}

		// Backward substitution for each column: UX = Y
		auto X = Matrix<T, Alloc>(n, m);
		for (size_type col = 0; col < m; ++col) {
			for (size_type i = n; i-- > 0; ) {
				auto sum = Y[i][col];
				for (size_type j = i + 1; j < n; ++j) {
					sum -= U[i][j] * X[j][col];
				}
				X[i][col] = sum / U[i][i];
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