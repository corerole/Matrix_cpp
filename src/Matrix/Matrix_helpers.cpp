#include "Matrix_helpers.hpp"

#include <random>
#include <exception>

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
	void fill_matrix(R&& r,  T min = (-1000.0L), T max = (1000.0L)) {
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
	bool equal(ItFst fb, S fe, ItSnd sb, Pred&& pred, ProjFst&& proj_fst = std::identity{}, ProjSnd&& proj_snd = std::identity{} ) {
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

static bool identity_test() {
	for (size_t i = 0; i < 50; ++i) {
		auto n = test_utils::gen(2L, 500L);
		auto A = helpers::identity<test_utils::value_type>(n);

		auto init0 = static_cast<test_utils::value_type>(0);
		init0 = std::accumulate(A.cbegin(), A.cend(), init0);

		auto init1 = static_cast<test_utils::value_type>(0);
		init1 = std::accumulate(A.diagonal_cbegin(), A.diagonal_cend(), init1);
		if (init0 != init1) { return false; }
	}

	return true;
}

static bool swap_r_test() {
	for (size_t i = 0; i < 50; ++i) {
		auto n = test_utils::gen(2L, 500L);
		auto m = test_utils::gen(2L, 500L);

		auto A = Matrix<test_utils::value_type>(n, m);
		test_utils::fill_matrix(A, -10.0L, 10.0L);
		auto A_c = A;

		auto B = Matrix<test_utils::value_type>(n, m);
		test_utils::fill_matrix(B, -10.0L, 10.0L);
		auto B_c = B;

		utils::swap_r(A_c, B_c);

		constexpr auto proj = []<typename T>(T&& val) -> T&& {
			return std::forward<T>(val);
		};

		auto a = test_utils::equal(A_c.begin(), A_c.end(), B.begin(),
			[](auto&& fst, auto&& snd) -> bool { return fst == snd; }, proj, proj);
		auto b = test_utils::equal(B_c.begin(), B_c.end(), A.begin(),
			[](auto&& fst, auto&& snd) -> bool { return fst == snd; }, proj, proj);
		if (!(a && b)) { return false; }
	}
	return true;
}

static bool euclid_norm_test() {
	constexpr auto eps = std::numeric_limits<test_utils::value_type>::epsilon();
	constexpr auto ten = static_cast<test_utils::value_type>(10.0L);
	auto precision = static_cast<test_utils::value_type>(1);
	constexpr auto three_eps = eps * 3;
	constexpr long double relevant_precision = 1e-7L;
	for (; precision > three_eps; precision /= ten) {
		auto n = test_utils::gen(2L, 500L);
		auto m = test_utils::gen(2L, 500L);
		auto A = Matrix<test_utils::value_type>(n, m);
		test_utils::fill_matrix(A, -1000.0L, 1000.0L);

		auto A_norm = helpers::euclid_norm(A);
		if (A_norm < 0) {
			return false;
		}
		auto B = -A;
		auto B_norm = helpers::euclid_norm(B);

		bool check = std::abs(B_norm - A_norm) < precision;
		if (!check) {
			if (precision < relevant_precision) { return true; }
			return false;
		}

		auto scale = static_cast<test_utils::value_type>(3);
		A *= scale;
		auto A_scaled_norm = helpers::euclid_norm(A);
		bool check2 = ((A_scaled_norm - (A_norm * scale)) < precision);
		if (!check2) {
			if (precision < relevant_precision) { return true; }
			return false;
		}
	}

	return true;
}

static bool manhattan_norm_test() {
	constexpr auto eps = std::numeric_limits<test_utils::value_type>::epsilon();
	constexpr auto ten = static_cast<test_utils::value_type>(10.0L);
	auto precision = static_cast<test_utils::value_type>(1);
	constexpr auto three_eps = eps * 3;
	constexpr long double relevant_precision = 1e-5L;
	for (; precision > three_eps; precision /= ten) {
		const auto n = test_utils::gen(2L, 500L);
		const auto m = test_utils::gen(2L, 500L);
		auto A = Matrix<test_utils::value_type>(n, m);
		test_utils::fill_matrix(A, -1000.0L, 1000.0L);

		auto A_norm = helpers::manhattan_norm(A);
		if (A_norm < 0) { return false; }
		auto B = -A;
		auto B_norm = helpers::manhattan_norm(B);

		bool check = std::abs(B_norm - A_norm) < precision;
		if (!check) {
			// std::cout << " | precision: " << precision << " | " << std::endl;
			if (precision < relevant_precision) { return true; }
			return false;
		}

		auto scale = static_cast<test_utils::value_type>(3);
		A *= scale;
		auto A_scaled_norm = helpers::manhattan_norm(A);
		bool check2 = ((A_scaled_norm - (A_norm * scale)) < precision);
		if (!check2) {
			// std::cout << " | precision: " << precision << " | " << std::endl;
			if (precision < relevant_precision) { return true; }
			return false;
		}
	}

	return true;
}

static bool scale_matrix_test() {
	constexpr long double relevant_precision = 1e-9L;
	for (size_t i = 0; i < 50; ++i) {
		auto n = test_utils::gen(2L, 500L);
		auto m = test_utils::gen(2L, 500L);
		auto A = Matrix<test_utils::value_type>(n, m);
		test_utils::fill_matrix(A, -1000.0L, 1000.0L);
		auto max_val = helpers::max(A);
		auto B = helpers::scale_matrix(A);
		B *= max_val;
		auto A_norm = helpers::euclid_norm(A);
		auto B_norm = helpers::euclid_norm(B);
		auto x = A_norm;
		if (A_norm > B_norm) {
			x = A_norm - B_norm;
		}	else {
			x = B_norm - A_norm;
		}
		if (x > relevant_precision) { return false; }
	}
	return true;
}

static bool lp_norm_test() {
	constexpr auto eps = std::numeric_limits<test_utils::value_type>::epsilon();
	constexpr auto ten = static_cast<test_utils::value_type>(10.0L);
	auto precision = static_cast<test_utils::value_type>(1);
	constexpr auto three_eps = eps * 3;
	constexpr long double relevant_precision = 1e-8L;

	constexpr auto n = 1000L;
	constexpr auto m = 1000L;
	auto A = Matrix<test_utils::value_type>(n, m);

	for (; precision > three_eps; precision /= ten) {
		test_utils::fill_matrix(A, -1000.0L, 1000.0L);

		auto A_norm = helpers::lp_norm(A, 6);
		if (A_norm < 0) { return false; }
		auto B = -A;
		auto B_norm = helpers::lp_norm(B, 6);

		bool check = std::abs(B_norm - A_norm) < precision;
		if (!check) {
			// std::cout << " | precision: " << precision << " | " << std::endl;
			if (precision < relevant_precision) { return true; }
			return false;
		}

		auto scale = static_cast<test_utils::value_type>(3);
		A *= scale;
		auto A_scaled_norm = helpers::lp_norm(A, 6);
		bool check2 = ((A_scaled_norm - (A_norm * scale)) < precision);
		if (!check2) {
			// std::cout << " | precision: " << precision << " | " << std::endl;
			if (precision < relevant_precision) { return true; }
			return false;
		}
	}

	return true;
}

static std::vector<std::function<bool()>> get_tests() {
	std::vector<std::function<bool()>> tests = {
		identity_test,
		swap_r_test,
		euclid_norm_test,
		manhattan_norm_test,
		scale_matrix_test,
		lp_norm_test
	};
	return tests;
}

void helpers_tests() {
	auto tests = get_tests();
	auto n_tests = tests.size();
	size_t ctr = 0;
	for (auto&& test : tests) {
		auto result = false;
		try {
			result = test();
		}	catch (const std::exception& e) {
			std::cerr << e.what() << std::endl;
		}	catch (...) {}
		std::cout << ++ctr << "/" << n_tests << " : ";
		if (result) {
			std::cout << "Test pass! \n";
		} else {
			std::cout << "Test fail! \n";
		}
	}
}