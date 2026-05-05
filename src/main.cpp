#include "main.hpp"

using mtx = Matrix<float>;

void test_ctors() {	
	mtx a1(3, 4); // def
	mtx a2(a1); // copy
	mtx a4 = a1; // ass copy
	mtx a3(std::move(a1)); // move
	mtx a5 = std::move(a2); // ass move
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
		constexpr size_t cols = 4, rows = 4;
		constexpr size_t SZ = cols * rows * sizeof(value_type) + alignof(value_type);
		std::array<std::byte, SZ> buf;

		std::pmr::monotonic_buffer_resource mbr(buf.data(), buf.size(), std::pmr::null_memory_resource());
		Matrix<value_type> A(cols, rows, &mbr);

		A[0][0] = 2.0f; A[0][1] = -0.5f; A[0][2] = 1.0f; A[0][3] = 0.0f;
		A[1][0] = 1.0f; A[1][1] = -1.0f; A[1][2] = 0.0f; A[1][3] = 2.0f;
		A[2][0] = 0.0f; A[2][1] = 1.0f; A[2][2] = 2.0f; A[2][3] = -1.0f;
		A[3][0] = -2.0f; A[3][1] = 0.0f; A[3][2] = 1.0f; A[3][3] = -1.0f;
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
			auto z = helpers::identity<value_type>(cols);
			z.print();
		}

		{
			std::cout << "Identity with alloc: " << std::endl;
			auto x = helpers::identity<value_type>(cols, std::allocator<value_type>{});
			x.print();
		}

		{
			std::cout << "Transpose: " << std::endl;
			auto b = helpers::transpose(A);
			b.print();
		}

		{
			std::cout << "Sum : " << helpers::sum(A.def_range()) << std::endl;
			std::cout << "Abs_sum : " << helpers::abs_sum(A.def_range()) << std::endl;
			std::cout << "Max : " << helpers::max(A.def_range()) << std::endl;
		}

		{
			std::cout << std::endl << "Scaled_self MTX: \n";
			auto tmp = A;
			helpers::scale_matrix_self(tmp);
			tmp.print();
		}

		{
			std::cout << std::endl << "Scaled MTX: \n";
			auto B = helpers::scale_matrix(A);
			B.print();
		}

		{
			std::cout << std::endl << "MTX max_abs: ";
			auto&& max_abs = helpers::max_abs(A.def_range());
			std::cout << max_abs << std::endl;
		}

		{
			std::cout << "MTX euclid_norm: ";
			auto&& euclid_norm = helpers::euclid_norm(A.def_range());
			std::cout << euclid_norm << std::endl;
		}

		{
			std::cout << "MTX determinant: ";
			auto&& determinant = helpers::determinant(A);
			std::cout << determinant << std::endl;
		}

		{
			auto D = A;
			std::cout << "MTX col_centered_ " << std::endl;
			helpers::col_centered_(A, D);
			D.print();
			std::cout << "----------------------" << std::endl;
		}

		{
			std::cout << "MTX inverse: " << std::endl;
			auto&& inv = helpers::inverse(A);
			inv.print();
			auto test_mtx = inv * A;
			std::cout << "test: " << std::endl;
			test_mtx.print();
			std::cout << "----------------------" << std::endl;
		}

		{
			std::cout << "MTX covariance: " << std::endl;
			auto cov = helpers::covariance(A);
			cov.print();
			std::cout << "----------------------" << std::endl;
		}
#if 0
		{
			std::cout << "MTX SVD: " << std::endl;
			auto&& [U, Sigma, Vh] = helpers::svd_jacobi_real(A);
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
			auto&& [Q, R] = helpers::householder_qr_decomposition(A);
			std::cout << "Q:" << std::endl;
			Q.print();
			std::cout << "R:" << std::endl;
			R.print();
			std::cout << "test QtQ:" << std::endl;
			auto Qt = helpers::transpose(Q);
			auto QtQ = Qt * Q;
			QtQ.print();
			std::cout << "test QxR:" << std::endl;
			auto QxR = Q * R;
			QxR.print();
			std::cout << "----------------------" << std::endl;
		}

		{
			std::cout << "MTX hessenberg_form:" << std::endl;
			auto [Hf, _ ] = helpers::hessenberg_form(A);
			Hf.print();
			std::cout << "----------------------" << std::endl;
		}

		{
			std::cout << "MTX upper_companion_matrix: " << std::endl;
			std::array<value_type, 9> a = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
			auto ucm = helpers::upper_companion_matrix(a);
			ucm.print();
		}

		{
			std::cout << "MTX lower_companion_matrix: " << std::endl;
			std::array<value_type, 9> a = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
			auto lcm = helpers::lower_companion_matrix(a);
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
			std::array<int, 4> row = { 1, 2, 3, 4 };
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
			auto&& self_values = helpers::self_values(tmp);
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
			std::cout << helpers::wilkinson_shift_complex(a, b, c, d) << std::endl;
		}
#endif

		{
			std::cout << "subMTX rows: ";
			auto x = A.get_submatrix(A[0][0], A[A.rows()-1][A.cols() - 1]);
			std::cout << x.rows() << " | " << x.cols() << std::endl << std::endl;
		}

#if 0
		{
			std::cout << "MTX FQR_householder: " << std::endl;
			std::cout << "MTX T: " << std::endl;
			auto [H, _] = helpers::hessenberg_form(A);
			auto&& [Q, T] = helpers::francis_qr_householder(H);
			T.print();
			std::cout << "MTX Q: " << std::endl;
			Q.print();
		}
#endif
		
#if 0
		{
			std::cout << "MTX Q_T_Qt: " << std::endl;
			auto&& [Q, T] = helpers::francis_qr_householder(A);
			(Q * T * helpers::transpose(Q)).print();
			std::cout << "----------------------" << std::endl;
		}
#endif
		{
			std::cout << "MTX QR_householder schur: " << std::endl;
			auto tmp = utils::to_complex(A);
			auto [Q, T] = helpers::schur(tmp);
			std::cout << "Q_total :" << std::endl;
			Q.print();
			std::cout << " T : " << std::endl;
			T.print();
			auto Qh = helpers::transpose(Q);
			std::ranges::for_each(Qh.def_range(), [](auto&& it) { it = std::conj(it); });

			auto res = Q * T * Qh;
			std::cout << "Q * T * transpose(Q) :" << std::endl;
			res.print();
			std::cout << "----------------------" << std::endl;
		}

	//helpers_tests();
	return 0;
}
