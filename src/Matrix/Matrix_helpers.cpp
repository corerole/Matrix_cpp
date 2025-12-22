#include "Matrix_helpers.hpp"

static bool identity_test() { 
	
	return true;
}


static std::vector<std::function<bool()>> get_tests() {
	std::vector<std::function<bool()>> tests = {
		identity_test
	};
	return tests;
}

void helpers_tests() {
	auto tests = get_tests();
	auto n_tests = tests.size();
	size_t ctr = 0;
	for (auto&& test : tests) {
		auto result = test();
		std::cout << ++ctr << "/" << n_tests << " : ";
		if (result) {
			std::cout << "Test pass! \n";
		} else {
			std::cout << "Test fail! \n";
		}
	}
}