import std;
import matrix;
import math_matrix;
import matrix_helpers; 

using namespace matrix;
using namespace math_matrix;

 // using mtx = Matrix<float>;

void test_ctors() {	
	Matrix<float> a1(3, 4); // def
	Matrix<float> a2(a1); // copy
	Matrix<float> a4 = a1; // ass copy
	Matrix<float> a3(std::move(a1)); // move
	Matrix<float> a5 = std::move(a2); // ass move
	auto x = a1.begin();
}

void operator_test(const MatrixLike auto& mtx) {
	auto A = mtx;
	auto B = A;

	// A -= B; // -> A&
	A - B; // -> C
	// A += B;
	A + B;
	// A *= B;
	A * B;
  // A /= B;
  // A / B;

//	B -= 2;
//	B - 2;
//	B += 2;
//	B + 2;
//	B *= 2;
	B * 2;
//	B /= 2;
	B / 2;
}

int main() {
	
		using value_type = float;
		constexpr size_t cols = 6, rows = 6;
		constexpr size_t SZ = cols * rows * sizeof(value_type) + alignof(value_type);
		std::array<std::byte, SZ> buf;

		std::pmr::monotonic_buffer_resource mbr(buf.data(), buf.size(), std::pmr::null_memory_resource());
		Matrix<value_type> A(cols, rows, &mbr);

		A[0][0] = 2.0f;		A[0][1] = -0.5f;	A[0][2] = 1.0f; A[0][3] = 0.0f;		A[0][4] = 0.5f;		A[0][5] = -2.0f;
		A[1][0] = -1.0f;	A[1][1] = -1.0f;	A[1][2] = 0.0f; A[1][3] = 2.0f;		A[1][4] = 2.0f;		A[1][5] = -0.5f;
		A[2][0] = 1.0f;		A[2][1] = 1.0f;		A[2][2] = 2.0f; A[2][3] = -1.0f;	A[2][4] = -1.0f;	A[2][5] = 1.0f;
		A[3][0] = -2.0f;	A[3][1] = 0.0f;		A[3][2] = 1.0f; A[3][3] = -1.0f;	A[3][4] = -2.0f;	A[3][5] = 2.0f;
		A[4][0] = 0.5f;		A[4][1] = -1.0f;	A[4][2] = 0.0f; A[4][3] = 2.0f;		A[4][4] = -0.5f;	A[4][5] = -1.0f;
		A[5][0] = -0.5f;	A[5][1] = 1.0f;		A[5][2] = 2.0f; A[5][3] = -1.0f;	A[5][4] = 1.0f;		A[5][5] = 0.5f;

		A.print();
#if 0
		MathMatrix<value_type> G(cols, rows);
		std::copy(A.begin(), A.end(), G.begin());
		
		auto H = G.transpose();
#endif
		{
			operator_test(A);
		}

		{
			test_ctors();
		}

		{
			std::cout << std::endl << "SUBMATRIX" << std::endl;
			auto submtx = A.get_submatrix(A[1][1], A[2][2]);
			std::cout << "cols: " << submtx.cols() << std::endl;
			std::cout << "rows: " << submtx.rows() << std::endl;
			auto print_submtx = [](auto&& submtx) {
				auto print = [](auto&& it) { std::cout << it << " "; };
				for (auto&& row : submtx.row_range()) {
					std::ranges::for_each(row, print);
					std::cout << std::endl;
				}
			};
			print_submtx(submtx);
			std::cout << "---------------------" << std::endl;
		}

		{
			std::cout << "Identity: " << std::endl;
			auto z = matrix_helpers::identity<value_type>(cols);
			z.print();
		}

		{
			std::cout << "Identity with alloc: " << std::endl;
			auto x = matrix_helpers::identity<value_type>(cols, std::allocator<value_type>{});
			x.print();
		}

		{
			std::cout << "Transpose: " << std::endl;
			auto b = matrix_helpers::transpose(A);
			b.print();
		}

		{
			std::cout << "Sum : " << matrix_helpers::sum(A.def_range()) << std::endl;
			std::cout << "Abs_sum : " << matrix_helpers::abs_sum(A.def_range()) << std::endl;
			std::cout << "Max : " << matrix_helpers::max(A.def_range()) << std::endl;
		}

		{
			std::cout << std::endl << "Scaled_self MTX: \n";
			auto tmp = A;
			matrix_helpers::scale_matrix_self(tmp);
			tmp.print();
		}

		{
			std::cout << std::endl << "Scaled MTX: \n";
			auto B = matrix_helpers::scale_matrix(A);
			B.print();
		}

		{
			std::cout << std::endl << "MTX max_abs: ";
			auto&& max_abs = matrix_helpers::max_abs(A.def_range());
			std::cout << max_abs << std::endl;
		}

		{
			std::cout << "MTX euclid_norm: ";
			auto&& euclid_norm = matrix_helpers::euclid_norm(A.def_range());
			std::cout << euclid_norm << std::endl;
		}

		{
			std::cout << "MTX determinant: ";
			auto&& determinant = matrix_helpers::determinant(A);
			std::cout << determinant << std::endl;
		}

		{
			auto D = A;
			std::cout << "MTX col_centered_ " << std::endl;
			matrix_helpers::col_centered_(A, D);
			D.print();
			std::cout << "----------------------" << std::endl;
		}

		{
			std::cout << "MTX inverse: " << std::endl;
			auto&& inv = matrix_helpers::inverse(A);
			inv.print();
			auto test_mtx = inv * A;
			std::cout << "test: " << std::endl;
			test_mtx.print();
			std::cout << "----------------------" << std::endl;
		}

		{
			std::cout << "MTX covariance: " << std::endl;
			auto cov = matrix_helpers::covariance(A);
			cov.print();
			std::cout << "----------------------" << std::endl;
		}
#if 0
		{
			std::cout << "MTX SVD: " << std::endl;
			auto&& [U, Sigma, Vh] = matrix_helpers::svd_jacobi_real(A);
			std::cout << "U:" << std::endl;
			U.print();
			std::cout << "Sigma:" << std::endl;
			Sigma.print();
			std::cout << "Vh:" << std::endl;
			Vh.print();
			std::cout << "test:" << std::endl;
			auto res = U * Sigma * Vh;
			res.print();
			std::cout << "----------------------" << std::endl;
		}
#endif
		{
			std::cout << "MTX set_submtx: " << std::endl;
			Matrix<value_type, std::pmr::polymorphic_allocator<value_type>> test(2, 2);
			test[0][0] = 1; test[0][1] = 2;
			test[1][0] = 3; test[1][1] = 4;
			auto m4x4 = A;
			auto& elem = m4x4[1][1];
			m4x4.set_submatrix(elem, test);
			m4x4.print();
			std::cout << "----------------------" << std::endl;
		}

		{
			std::cout << "MTX householder_qr_decomposition:" << std::endl;
			auto&& [Q, R] = matrix_helpers::householder_qr_decomposition(A);
			std::cout << "Q:" << std::endl;
			Q.print();
			std::cout << "R:" << std::endl;
			R.print();
			std::cout << "test QtQ:" << std::endl;
			auto Qt = matrix_helpers::transpose(Q);
			auto QtQ = Qt * Q;
			QtQ.print();
			std::cout << "test QxR:" << std::endl;
			auto QxR = Q * R;
			QxR.print();
			std::cout << "----------------------" << std::endl;
		}

		{
			std::cout << "MTX hessenberg_form:" << std::endl;
			auto [_ , Hf] = matrix_helpers::hessenberg_form(A);
			Hf.print();
			std::cout << "----------------------" << std::endl;
		}

		{
			std::cout << "MTX upper_companion_matrix: " << std::endl;
			std::array<value_type, 9> a = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
			auto ucm = matrix_helpers::upper_companion_matrix(a);
			ucm.print();
		}

		{
			std::cout << "MTX lower_companion_matrix: " << std::endl;
			std::array<value_type, 9> a = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
			auto lcm = matrix_helpers::lower_companion_matrix(a);
			lcm.print();
		}

		auto print_vec = [](auto&& x) {
			std::ranges::for_each(x, [](auto&& it) { std::cout << it << " "; });
			std::cout << std::endl;
		};

#if 0
		{
			std::cout << "MTX vec_x_mat: " << std::endl;
			std::array<int, 4> x = { 1, 2, 3, 4 };
			auto res = vec_x_mat(x, A);
			res.print();
			std::cout << "----------------------" << std::endl;
		}
#endif

		{
			std::cout << "MTX row_x_col: " << std::endl;
			std::array<int, 5> row = { 1, 2, 3, 4, 5 };	
			std::cout << "row: ";
			print_vec(row);
			std::array<int, 5> col = { 1, 2, 3, 4, 5 };
			std::cout << "col: ";
			print_vec(col);
			auto res = col_x_row(col, row);
			res.print();
			std::cout << "----------------------" << std::endl;
		}

		{
			std::cout << "MTX row_x_mtx: " << std::endl;
			std::array<int, 6> row = { 1, 2, 3, 4, 5, 6 };
			std::cout << "row: ";
			print_vec(row);
			auto res = row_x_mtx(row, A);
			print_vec(res);
			std::cout << std::endl;
			std::cout << "----------------------" << std::endl;
		}

		{
			std::cout << "MTX mtx_x_col: " << std::endl;
			(A * A.col(0)).print();
			std::cout << "----------------------" << std::endl;
		}
#if 0
		{
			std::cout << "----------------------" << std::endl;
			std::cout << "A: " << std::endl;
			constexpr size_t SZ2 = sizeof(value_type) * 8 * 8;
			std::array<std::byte, SZ2> mem2;
			std::pmr::monotonic_buffer_resource mbr2(mem2.data(), mem2.size());
			Matrix<value_type> tmp(8, 8, &mbr2);
			tmp[0][0] = 1;
			tmp[1][0] = 2; tmp[1][1] = -1;
			tmp[2][2] = 2;
			tmp[3][2] = 4; tmp[3][3] = -2;
			tmp[4][4] = 3;
			tmp[5][4] = 6; tmp[5][5] = -3;
			tmp[6][6] = 4;
			tmp[7][6] = 8; tmp[7][7] = -4;
			tmp.print();
			std::cout << "MTX self_values: ";
			auto&& self_values = matrix_helpers::self_values(tmp);
			std::ranges::for_each(self_values, [](auto&& it) { std::cout << it << " "; });
			std::cout << std::endl << "----------------------" << std::endl;
		}
#endif

#if 0
		{
			std::cout << "MTX willkinson_shift_complex: ";
			auto&& a = A[A.rows() - 2][A.cols() - 2];
			auto&& b = A[A.rows() - 2][A.cols() - 1];
			auto&& c = A[A.rows() - 1][A.cols() - 2];
			auto&& d = A[A.rows() - 1][A.cols() - 1];
			std::cout << matrix_helpers::wilkinson_shift_complex(a, b, c, d) << std::endl;
		}
#endif

		{
			std::cout << "subMTX rows: ";
			auto x = A.get_submatrix(A[0][0], A[A.rows()-1][A.cols() - 1]);
			std::cout << x.rows() << " | " << x.cols() << std::endl << std::endl;
		}

		{
			std::cout << "MTX minor: " << std::endl;
			auto minor = matrix_helpers::minor(A, 0, 0);
			minor.print();
			std::cout << "----------------------" << std::endl;
		}

#if 0
		{
			std::cout << "MTX adjoint: " << std::endl;
			auto res = matrix_helpers::adjoint(A);
			res.print();
		}
#endif

#if 1
		{
			std::cout << "MTX FQR_householder: " << std::endl;
			auto Ac = utils::to_complex(A);
			std::cout << "MTX A:" << std::endl;
			A.print();
			auto [QH, H] = matrix_helpers::hessenberg_form(Ac);
			std::cout << "MTX H:" << std::endl;
			H.print();
			auto&& [Q, R] = matrix_helpers::francis_qr_householder(H);
			std::cout << "MTX R: " << std::endl;
			R.print();
			std::cout << "MTX Q: " << std::endl;
			Q.print();
			auto Q_total = QH * Q;
			std::cout << "MTX Q_total: " << std::endl;
			Q_total.print();
			auto Q_tadj = matrix_helpers::adjoint(Q_total);
			std::cout << "MTX A = Q_total * R * Q_tadj: " << std::endl;
			auto res = Q_total * R * Q_tadj;
			res.print();
		}
#endif
		

#if 0
		{
			std::cout << "MTX find_pos_test : " << std::endl;
			Matrix<double> O(3, 3);
			double fill = 1.0;
			for (auto& x : O.def_range()) { x = fill; fill += 1; }
			std::cout << "O :" << std::endl;
			O.print();
			auto cO = utils::to_complex(O);
			auto split_pos = matrix_helpers::find_split_position(cO);
		}
#endif

		{
			Matrix<value_type> X(1, 1);
			X[0][0] = 42;
			auto sub = X.get_submatrix(X[0][0], X[0][0]);
			std::cout << "One elem mtx.size() : " << sub.size() << std::endl;
		}

#if 0
		{
			std::cout << "----------------------" << std::endl;
			std::cout << "A: " << std::endl;
			auto complex_mtx = utils::to_complex(A);
			complex_mtx.print();
			auto [Q, H] = matrix_helpers::hessenberg_form(complex_mtx);
			auto [U, T] = matrix_helpers::schur(H);
			auto UQ = Q * U;
			std::cout << "Q :" << std::endl;
			UQ.print();
			std::cout << " T : " << std::endl;
			T.print();
			auto UQ_tc = matrix_helpers::conj_transpose(UQ);
			auto res = UQ * T * UQ_tc;
			std::cout << "Q * T * conj_transpose(Q) :" << std::endl;
			res.print();
			std::cout << "----------------------" << std::endl;
		}
#endif

#if 0
		{
			double a = 42.0;
			auto res = std::real(a) + std::imag(a);
			std::cout << res << std::endl;
		}
#endif

#if 0
		{
			std::cout << "Kronecker_product :" << std::endl;
			std::cout << "A: " << std::endl;
			A.print();
			Matrix<value_type> B(2, 2); 
			B[0][0] = 1; B[0][1] = 1;
			B[1][0] = 1; B[1][1] = 1;
			std::cout << "B: " << std::endl;
			B.print();
			auto res = matrix_helpers::Kronecker_product(A, B);
			res.print();
			std::cout << "----------------------" << std::endl;
		}
#endif

#if 0
		{
			Matrix<value_type> X(2, 2);
			X[0][0] = 1;
			X[0][1] = 3;
			X[1][0] = 4;
			X[1][1] = 2;

			auto [Q, H] = matrix_helpers::hessenberg_form(X);
			auto [U, T] = matrix_helpers::schur(H);
			T.print();
			auto [l1, l2] = matrix_helpers::self_values_2x2(X[0][0], X[0][1], X[1][0], X[1][1]);
			std::cout << " -2, 5 : " << std::real(l1) << " | " << std::real(l2) << std::endl;
		}
#endif

#if 1
		{
			// Parlett
			using cvt = std::complex<value_type>;
			auto cA = utils::to_complex(A);
			std::cout << "A : " << std::endl;
			A.print();

			auto [Q, H] = matrix_helpers::hessenberg_form(A);
			std::cout << "H : " << std::endl;
			H.print();

			using current_type = typename std::remove_cvref_t<decltype(H)>::value_type;
			static_assert(!ComplexLike<current_type>);
			static_assert(ComplexLike<float> != ComplexLike<std::complex<float>>);

			auto [U, T] = matrix_helpers::schur(H);
			auto Qt = matrix_helpers::transpose(Q);
			Qt = Q * Qt;
			std::cout << "Q * Qt: " << std::endl;
			Qt.print();


			std::cout << "T: " << std::endl;
			T.print();

			auto Ut = matrix_helpers::transpose(U);
			Ut = U * Ut;
			std::cout << "U * Ut: " << std::endl;
			Ut.print();

			auto QU = Q * U;
			auto QU_tc = matrix_helpers::transpose(QU);

			std::cout << "QU * T * QU_tc : " << std::endl;
			auto a_like = QU * T * QU_tc;
			a_like.print();

			auto my_sin = []<typename vT>(const vT& z) -> vT { return std::sin(z); };
			auto F = matrix_helpers::parlett(T, my_sin);
			
			std::cout << "F: " << std::endl;
			F.print();
			std::cout << "QU * F * Qtc: " << std::endl;
			auto sinA = QU * F * QU_tc;	
			sinA.print();
			
			std::cout << "A * sin(A) : " << std::endl;
			auto AsinA = A * sinA;
			AsinA.print();
			
			std::cout << "sin(A) * A : " << std::endl;
			auto sinAA = sinA * A;
			sinAA.print();

			std::cout << "sin(A) * A - A * sin(A) : " << std::endl;
			auto Zero = sinAA - AsinA;
			Zero.print();
		}
#endif

	//helpers_tests();
	return 0;
}
