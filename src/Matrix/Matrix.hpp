#pragma once
#ifndef MY_MATRIX
#define MY_MATRIX

#include <iostream>
#include <memory>
#include <type_traits>
#include <iterator>
#include <cstddef>
#include <utility>
#include <algorithm>
#include <stdexcept>
#include <ranges>
#include <numeric>
#include <iomanip>
#include <vector>
#include <complex>
#include <thread>
#include <future>
#include <chrono>

template<typename T>
struct row_const_iterator {
	public:
		using value_type = T;
		using pointer = value_type*;
		using const_pointer = const value_type*;
		using const_reference = const value_type&;
		using difference_type = std::ptrdiff_t;
		using iterator_category = std::random_access_iterator_tag;
		using iterator_concept = std::contiguous_iterator_tag;

	protected:
		pointer ptr = nullptr;

	public:
		constexpr row_const_iterator() noexcept = default;
		constexpr explicit row_const_iterator(pointer p) noexcept : ptr(p) {}

		constexpr const_reference operator*() const noexcept { return *ptr; }
		constexpr const_pointer operator->() const noexcept { return ptr; }

		constexpr row_const_iterator& operator++() noexcept { ++ptr; return *this; }
		constexpr row_const_iterator operator++(int) noexcept { auto tmp = *this; ++ptr; return tmp; }

		constexpr row_const_iterator& operator--() noexcept { --ptr; return *this; }
		constexpr row_const_iterator operator--(int) noexcept { auto tmp = *this; --ptr; return tmp; }

		constexpr row_const_iterator& operator+=(difference_type n) noexcept { ptr += n; return *this; }
		constexpr row_const_iterator& operator-=(difference_type n) noexcept { ptr -= n; return *this; }

		constexpr row_const_iterator operator+(difference_type n) const noexcept { return row_const_iterator(ptr + n); }
		constexpr row_const_iterator operator-(difference_type n) const noexcept { return row_const_iterator(ptr - n); }

		constexpr difference_type operator-(const row_const_iterator& o) const noexcept { return ptr - o.ptr; }
		constexpr const_reference operator[](difference_type n) const noexcept { return *(ptr + n); }

		constexpr bool operator==(const row_const_iterator& o) const noexcept { return ptr == o.ptr; }
		constexpr auto operator<=>(const row_const_iterator& o) const noexcept { return ptr <=> o.ptr; }
};

template<typename T>
constexpr row_const_iterator<T> operator+(
	typename row_const_iterator<T>::difference_type n,
	const row_const_iterator<T>& it) noexcept {
	return it + n;
}

template<typename T>
struct row_iterator final : public row_const_iterator<T> {
	private:
		using base_t = row_const_iterator<T>;
		using base_t::ptr;
	public:
		using typename base_t::value_type;
		using typename base_t::pointer;
		using typename base_t::const_pointer;
		using typename base_t::const_reference;
		using reference = value_type&;
		using typename base_t::difference_type;

	public:
		constexpr row_iterator() noexcept = default;
		constexpr explicit row_iterator(pointer p) noexcept
			: base_t(p) {
		}

		constexpr row_iterator(const base_t& it) noexcept
			: base_t(ptr) {
		}

		constexpr const_reference operator*() const noexcept {
			return const_cast<reference>(base_t::operator*());
		}

		constexpr const_pointer operator->() const noexcept {
			return const_cast<pointer>(base_t::operator->());
		}

		constexpr reference operator[](difference_type n) noexcept { return *(ptr + n); }
		constexpr const_reference operator[](difference_type n) const noexcept { return base_t::operator[](n); }

		constexpr row_iterator& operator++() noexcept { base_t::operator++(); return *this; }
		constexpr row_iterator operator++(int) noexcept { auto tmp = *this; base_t::operator++(); return tmp; }
		constexpr row_iterator& operator--() noexcept { base_t::operator--(); return *this; }
		constexpr row_iterator operator--(int) noexcept { auto tmp = *this; base_t::operator--(); return tmp; }

		constexpr row_iterator& operator+=(difference_type n) noexcept { base_t::operator+=(n); return *this; }
		constexpr row_iterator& operator-=(difference_type n) noexcept { base_t::operator-=(n); return *this; }

		constexpr row_iterator operator+(difference_type n) const noexcept { return row_iterator(const_cast<pointer>(ptr + n)); }
		constexpr row_iterator operator-(difference_type n) const noexcept { return row_iterator(const_cast<pointer>(ptr - n)); }
		constexpr difference_type operator-(const row_iterator& o) const noexcept { return ptr - o.ptr; }
};

template<typename T>
constexpr row_iterator<T> operator+(
	typename row_iterator<T>::difference_type n,
	const row_iterator<T>& it) noexcept {
	return it + n;
}

static_assert(std::contiguous_iterator<row_iterator<int>>);
static_assert(std::contiguous_iterator<row_const_iterator<int>>);

template<typename T>
struct RowProxy final : public std::ranges::view_interface<RowProxy<T>> {
	public:
		using value_type = T;
		using pointer = value_type*;
		using const_pointer = const value_type*;
		using reference = value_type&;
		using const_reference = const value_type&;
		using size_type = std::size_t;

		using iterator = row_iterator<T>;
		using const_iterator = row_const_iterator<T>;
		using reverse_iterator = std::reverse_iterator<iterator>;
		using const_reverse_iterator = std::reverse_iterator<const_iterator>;

	private:
		pointer row_data = nullptr;
		const size_type cols = 0;
	public:
		constexpr RowProxy() noexcept = default;
		constexpr RowProxy(pointer data, size_type c) noexcept : row_data(data), cols(c) {}

	public:
		constexpr reference operator[](size_type c) {
			return row_data[c];
		}

		constexpr const_reference operator[](size_type c) const {
			return row_data[c];
		}
		
	public:
		constexpr iterator begin() noexcept { return iterator(row_data); }
		constexpr iterator end() noexcept { return iterator(row_data + cols); }
		constexpr const_iterator begin() const noexcept { return const_iterator(row_data); }
		constexpr const_iterator end() const noexcept { return const_iterator(row_data + cols); }
		constexpr const_iterator cbegin() const noexcept { return const_iterator(begin()); }
		constexpr const_iterator cend() const noexcept { return const_iterator(end()); }
		constexpr reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
		constexpr reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
		constexpr const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }
		constexpr const_reverse_iterator crend() const noexcept { return const_reverse_iterator(cbegin()); }

	public:
		constexpr bool empty() const noexcept { return cols == 0; }
		constexpr reference front() { return *row_data; }
		constexpr const_reference front() const { return *row_data; }
		constexpr reference back() { return row_data[cols - 1]; }
		constexpr const_reference back() const { return row_data[cols - 1]; }
		constexpr size_type size() const noexcept { return cols; }
		constexpr pointer data() const noexcept { return row_data; }
};

template<typename T>
struct stride_const_iterator {
	public:
		using value_type = T;
		using difference_type = std::ptrdiff_t;
		using size_type = std::size_t;
		using pointer = value_type*;
		using const_pointer = const value_type*;
		using const_reference = const value_type&;
		using iterator_category = std::random_access_iterator_tag;
		using iterator_concept = std::random_access_iterator_tag;

	protected:
		pointer ptr = nullptr;
		size_type step = 0;

	public:
		constexpr stride_const_iterator() noexcept = default;
		constexpr stride_const_iterator(pointer p, size_type s) noexcept
			: ptr(p), step(s) {
		}

		constexpr const_reference operator*() const noexcept { return *ptr; }
		constexpr const_pointer operator->() const noexcept { return ptr; }

		constexpr stride_const_iterator& operator++() noexcept { ptr += step; return *this; }
		constexpr stride_const_iterator operator++(int) noexcept { auto tmp = *this; ptr += step; return tmp; }
		constexpr stride_const_iterator& operator--() noexcept { ptr -= step; return *this; }
		constexpr stride_const_iterator operator--(int) noexcept { auto tmp = *this; ptr -= step; return tmp; }

		constexpr stride_const_iterator& operator+=(difference_type n) noexcept { ptr += n * step; return *this; }
		constexpr stride_const_iterator& operator-=(difference_type n) noexcept { ptr -= n * step; return *this; }

		constexpr stride_const_iterator operator+(difference_type n) const noexcept { return stride_const_iterator(ptr + n * step, step); }
		constexpr stride_const_iterator operator-(difference_type n) const noexcept { return stride_const_iterator(ptr - n * step, step); }

		constexpr difference_type operator-(const stride_const_iterator& o) const noexcept {
			return (ptr - o.ptr) / static_cast<difference_type>(step);
		}

		constexpr const_reference operator[](difference_type n) const noexcept { return *(ptr + n * step); }

		constexpr bool operator==(const stride_const_iterator& o) const noexcept { return ptr == o.ptr; }
		constexpr auto operator<=>(const stride_const_iterator& o) const noexcept { return ptr <=> o.ptr; }
};

template<typename T>
constexpr stride_const_iterator<T> operator+(
	typename stride_const_iterator<T>::difference_type n,
	const stride_const_iterator<T>& it) noexcept {
	return it + n;
}

template<typename T>
struct stride_iterator final : public stride_const_iterator<T> {
	private:
		using base_t = stride_const_iterator<T>;

	public:
		using value_type = T;
		using typename base_t::pointer;
		using typename base_t::const_pointer;
		using typename base_t::const_reference;
		using typename base_t::size_type;
		using reference = value_type&;
		using typename base_t::difference_type;

	public:
		constexpr stride_iterator() noexcept = default;
		constexpr stride_iterator(pointer p, std::size_t s) noexcept
			: base_t(p, s) {
		}

		constexpr stride_iterator(const base_t& it) noexcept
			: base_t(it.ptr, it.step) {
		}

		constexpr reference operator*() const noexcept {
			return const_cast<reference>(base_t::operator*());
		}
		constexpr pointer operator->() const noexcept {
			return const_cast<pointer>(base_t::operator->());
		}

		constexpr auto operator<=>(const stride_iterator& o) const noexcept {
			return base_t::operator<=>(o);
		}

		constexpr bool operator==(const stride_iterator& o) const noexcept {
			return base_t::operator==(o);
		}

		constexpr reference operator[](difference_type n) noexcept {
			return *(base_t::ptr + n * base_t::step);
		}

		constexpr const reference operator[](difference_type n) const noexcept {
			return base_t::operator[](n);
		}

		constexpr stride_iterator& operator++() noexcept { base_t::operator++(); return *this; }
		constexpr stride_iterator operator++(int) noexcept { auto tmp = *this; base_t::operator++(); return tmp; }
		constexpr stride_iterator& operator--() noexcept { base_t::operator--(); return *this; }
		constexpr stride_iterator operator--(int) noexcept { auto tmp = *this; base_t::operator--(); return tmp; }

		constexpr stride_iterator& operator+=(difference_type n) noexcept { base_t::operator+=(n); return *this; }
		constexpr stride_iterator& operator-=(difference_type n) noexcept { base_t::operator-=(n); return *this; }

		constexpr stride_iterator operator+(difference_type n) const noexcept { return stride_iterator( (base_t::ptr + n * base_t::step), base_t::step ); }
		constexpr stride_iterator operator-(difference_type n) const noexcept { return stride_iterator( (base_t::ptr - n * base_t::step), base_t::step ); }

		constexpr difference_type operator-(const stride_iterator& other) const noexcept {
			return (base_t::ptr - other.ptr) / base_t::step;
		}
};

template<typename T>
constexpr stride_iterator<T> operator+(
	typename stride_iterator<T>::difference_type n,
	const stride_iterator<T>& it) noexcept {
	return it + n;
}

static_assert(std::random_access_iterator<stride_iterator<int>>);

template<typename T>
struct ColProxy final : public std::ranges::view_interface<ColProxy<T>> {
	public:
		using value_type = T;
		using pointer = value_type*;
		using const_pointer = const value_type*;
		using reference = value_type&;
		using const_reference = const value_type&;
		using size_type = std::size_t;

	public:
		using iterator = stride_iterator<T>;
		using const_iterator = stride_const_iterator<T>;
		using reverse_iterator = std::reverse_iterator<iterator>;
		using const_reverse_iterator = std::reverse_iterator<const_iterator>;

	private:
		pointer col_data = nullptr;
		size_type rows = 0;
		size_type stride = 0;

	public:
		constexpr ColProxy() noexcept = default;
		constexpr ColProxy(pointer base, size_type r, size_type stride_) noexcept
			: col_data(base), rows(r), stride(stride_) {
		}

		constexpr reference operator[](size_type r) {
			return *(col_data + r * stride);
		}

		constexpr const_reference operator[](size_type r) const {
			return *(col_data + r * stride);
		}

	public:
		constexpr iterator begin() noexcept { return iterator(col_data, stride); }
		constexpr iterator end() noexcept { return iterator(col_data + rows * stride, stride); }
		constexpr const_iterator begin() const noexcept { return const_iterator(col_data, stride); }
		constexpr const_iterator end() const noexcept { return const_iterator(col_data + rows * stride, stride); }
		constexpr const_iterator cbegin() const noexcept { return const_iterator(begin()); }
		constexpr const_iterator cend() const noexcept { return const_iterator(end()); }
		constexpr reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
		constexpr reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
		constexpr const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }
		constexpr const_reverse_iterator crend() const noexcept { return const_reverse_iterator(cbegin()); }

	public:
		constexpr size_type size() const noexcept { return rows; }
		constexpr bool empty() const noexcept { return rows == 0; }
		constexpr reference front() { return *col_data; }
		constexpr const_reference front() const { return *col_data; }
		constexpr reference back() { return ::operator[](rows - 1); }
		constexpr const_reference back() const { return ::operator[](rows - 1); }
};

template<typename T>
struct matrix_const_default_iterator {
	public:
		using value_type = T;
		using pointer = value_type*;
		using const_pointer = const value_type*;
		using const_reference = const value_type&;
		using difference_type = std::ptrdiff_t;
		using iterator_category = std::random_access_iterator_tag;
		using iterator_concept = std::contiguous_iterator_tag;

	protected:
		pointer ptr = nullptr;

	public:
		constexpr matrix_const_default_iterator() noexcept = default;
		constexpr explicit matrix_const_default_iterator(pointer p) noexcept : ptr(p) {}

		constexpr const_reference operator*() const { return *ptr; }
		constexpr const_pointer operator->() const { return ptr; }

		constexpr matrix_const_default_iterator& operator++() noexcept { ++ptr; return *this; }
		constexpr matrix_const_default_iterator operator++(int) noexcept { auto tmp = *this; ++ptr; return tmp; }

		constexpr matrix_const_default_iterator& operator--() noexcept { --ptr; return *this; }
		constexpr matrix_const_default_iterator operator--(int) noexcept { auto tmp = *this; --ptr; return tmp; }

		constexpr matrix_const_default_iterator& operator+=(difference_type n) noexcept { ptr += n; return *this; }
		constexpr matrix_const_default_iterator& operator-=(difference_type n) noexcept { ptr -= n; return *this; }

		constexpr matrix_const_default_iterator operator+(difference_type n) const noexcept { return matrix_const_default_iterator(ptr + n); }
		constexpr matrix_const_default_iterator operator-(difference_type n) const noexcept { return matrix_const_default_iterator(ptr - n); }

		constexpr difference_type operator-(const matrix_const_default_iterator& other) const noexcept { return ptr - other.ptr; }

		constexpr const_reference operator[](difference_type n) const noexcept { return *(ptr + n); }

		constexpr bool operator==(const matrix_const_default_iterator& o) const noexcept { return ptr == o.ptr; }
		constexpr auto operator<=>(const matrix_const_default_iterator& o) const noexcept { return ptr <=> o.ptr; }
};

template<typename T>
inline constexpr matrix_const_default_iterator<T> operator+(
	std::ptrdiff_t n, const matrix_const_default_iterator<T>& it) noexcept {
	return matrix_const_default_iterator<T>(it.ptr + n);
}

static_assert(std::contiguous_iterator<matrix_const_default_iterator<int>>);

template<typename T>
struct matrix_default_iterator final : public matrix_const_default_iterator<T> {
	private:
		using base_t = matrix_const_default_iterator<T>;

	public:
		using typename base_t::value_type;
		using typename base_t::pointer;
		using typename base_t::const_pointer;
		using typename base_t::const_reference;
		using typename base_t::difference_type;
		using reference = value_type&;

	public:
		constexpr matrix_default_iterator() noexcept = default;
		constexpr explicit matrix_default_iterator(pointer p) noexcept : base_t(p) {}

		// constexpr reference operator*() { return *(base_t::ptr); }
		// constexpr const_reference operator*() const { return *(base_t::ptr); }
		constexpr reference operator*() const { return *(base_t::ptr); }

		// constexpr pointer operator->() { return base_t::ptr; }
		// constexpr const_pointer operator->() const { return base_t::ptr; }
		constexpr pointer operator->() const { return base_t::ptr; }

		// constexpr reference operator[](difference_type n) noexcept { return *(base_t::ptr + n); }
		// constexpr const_reference operator[](difference_type n) const noexcept { return base_t::operator[](n); }
		constexpr reference operator[](difference_type n) const noexcept { return *(base_t::ptr + n); }

		constexpr matrix_default_iterator& operator++() noexcept {
			base_t::operator++();
			return *this;
		}
		constexpr matrix_default_iterator operator++(int) noexcept { auto tmp = *this; base_t::operator++(); return tmp; }

		constexpr matrix_default_iterator& operator--() noexcept {
			base_t::operator--();
			return *this;
		}
		constexpr matrix_default_iterator operator--(int) noexcept { auto tmp = *this; base_t::operator--(); return tmp; }

		constexpr matrix_default_iterator& operator+=(difference_type n) noexcept {
			base_t::operator+=(n);
			return *this;
		}
		constexpr matrix_default_iterator& operator-=(difference_type n) noexcept {
			base_t::operator-=(n);
			return *this;
		}

		constexpr matrix_default_iterator operator+(difference_type n) const noexcept { return matrix_default_iterator(base_t::ptr + n); }
		constexpr matrix_default_iterator operator-(difference_type n) const noexcept { return matrix_default_iterator(base_t::ptr - n); }
		constexpr difference_type operator-(const matrix_default_iterator& o) const noexcept { return base_t::ptr - o.ptr; }

	public:
		constexpr bool operator==(const matrix_default_iterator& o) const noexcept { return base_t::operator==(o); }
		constexpr auto operator<=>(const matrix_default_iterator& o) const noexcept { return base_t::operator<=>(o); }
};

template<typename T>
inline constexpr matrix_default_iterator<T> operator+(
	std::ptrdiff_t n, const matrix_default_iterator<T>& it) noexcept {
	return matrix_default_iterator<T>(it.ptr + n);
}

static_assert(std::contiguous_iterator<matrix_default_iterator<int>>);


template<typename T>
struct matrix_const_row_iterator {
	private:
		using internal_pointer = T*;
	public:
		using value_type = RowProxy<const T>;
		using pointer = void;
		using reference = value_type;
		using difference_type = std::ptrdiff_t;
		using size_type = std::size_t;
		using iterator_category = std::random_access_iterator_tag;
		using iterator_concept = std::random_access_iterator_tag;

	protected:
		internal_pointer ptr = nullptr;
		size_type cols = 0;

	public:
		constexpr matrix_const_row_iterator() noexcept = default;
		constexpr matrix_const_row_iterator(internal_pointer ptr, size_type cols) noexcept : ptr(ptr), cols(cols) {}

		constexpr value_type operator*() const { return value_type(ptr, cols); }
		// constexpr pointer operator->() const { return ; }
		constexpr value_type operator[](size_type n) const {
			return value_type((ptr + static_cast<difference_type>(cols) * n), cols);
		}
	public:
		constexpr matrix_const_row_iterator& operator++() noexcept { ptr += cols; return *this; }
		constexpr matrix_const_row_iterator operator++(int) noexcept { auto tmp = *this; ptr += cols; return tmp; }
		constexpr matrix_const_row_iterator& operator--() noexcept { ptr -= cols; return *this; }
		constexpr matrix_const_row_iterator operator--(int) noexcept { auto tmp = *this; ptr -= cols; return tmp; }
		constexpr matrix_const_row_iterator& operator+=(difference_type n) noexcept { ptr += static_cast<std::ptrdiff_t>(cols) * n; return *this; }
		constexpr matrix_const_row_iterator& operator-=(difference_type n) noexcept { ptr -= static_cast<std::ptrdiff_t>(cols) * n; return *this; }
		constexpr matrix_const_row_iterator operator+(difference_type n) const noexcept { return matrix_const_row_iterator(ptr + n * cols, cols); }
		constexpr matrix_const_row_iterator operator-(difference_type n) const noexcept { return matrix_const_row_iterator(ptr - n * cols, cols); }
		constexpr difference_type operator-(const matrix_const_row_iterator& o) const noexcept { return (ptr - o.ptr) / static_cast<difference_type>(cols); }

	public:
		constexpr bool operator==(const matrix_const_row_iterator& o) const noexcept { return ptr == o.ptr; }
		constexpr auto operator<=>(const matrix_const_row_iterator& o) const noexcept { return ptr <=> o.ptr; }
};

template <typename T>
inline constexpr matrix_const_row_iterator<T> operator+(
	std::ptrdiff_t n, const matrix_const_row_iterator<T>& o) {
	return matrix_const_row_iterator<T>((o.ptr + static_cast<std::ptrdiff_t>(o.cols) * n), o.cols);
}

static_assert(std::random_access_iterator<matrix_const_row_iterator<int>>);

template<typename T>
struct matrix_row_iterator final : public matrix_const_row_iterator<T> {
	private:
		using base_t = matrix_const_row_iterator<T>;
		using base_t::ptr;
		using base_t::cols;
		using pointer = decltype(ptr);
	public:
		using value_type = RowProxy<T>;
		using typename base_t::reference;
		using typename base_t::difference_type;
		using typename base_t::size_type;

		constexpr matrix_row_iterator() noexcept = default;
		constexpr matrix_row_iterator(pointer p, size_type cols) noexcept : base_t(p, cols) {}
		constexpr value_type operator*() const noexcept { return value_type(ptr, cols); }

	public:
		constexpr matrix_row_iterator& operator++() { ptr += cols; return *this; }
		constexpr matrix_row_iterator operator++(int) const { auto tmp = *this; ptr += cols; return tmp; }
		constexpr matrix_row_iterator& operator--() { ptr -= cols; return *this; }
		constexpr matrix_row_iterator operator--(int) const { auto tmp = *this; ptr -= cols; return tmp; }
		constexpr matrix_row_iterator& operator+=(difference_type n) { ptr += static_cast<difference_type>(cols) * n; return *this; }
		constexpr matrix_row_iterator& operator-=(difference_type n) { ptr -= static_cast<difference_type>(cols) * n; return *this; }
		constexpr matrix_row_iterator operator+(difference_type n) const {
			return matrix_row_iterator((ptr + (n * static_cast<difference_type>(cols))), cols);
		}
		constexpr matrix_row_iterator operator-(difference_type n) const {
			return matrix_row_iterator((ptr - (n * static_cast<difference_type>(cols))), cols);
		}
		constexpr difference_type operator-(const matrix_row_iterator& o) const { return base_t::operator-(o); }

	public:
		constexpr bool operator==(const matrix_row_iterator& o) const noexcept { return base_t::operator==(o); }
		constexpr auto operator<=>(const matrix_row_iterator& o) const noexcept { return base_t::operator<=>(o); }

	public:
		constexpr value_type operator[](size_type n) const {
			return value_type((ptr + static_cast<difference_type>(cols) * n), cols);
		}
};

template <typename T>
inline constexpr matrix_row_iterator<T> operator+(
	std::ptrdiff_t n, const matrix_row_iterator<T>& o) {
	return matrix_row_iterator<T>((o.ptr + static_cast<std::ptrdiff_t>(o.cols) * n), o.cols);
}

static_assert(std::random_access_iterator<matrix_row_iterator<int>>);

template<typename T>
struct matrix_const_col_iterator {
	public:
		using internal_pointer = T*;
	public:
		using value_type = ColProxy<const T>;
		using reference = value_type;
		using difference_type = std::ptrdiff_t;
		using size_type = std::size_t;
		using iterator_category = std::random_access_iterator_tag;
		using iterator_concept = std::random_access_iterator_tag;

	protected:
		internal_pointer ptr = nullptr;
		size_type rows = 0;
		size_type stride = 0;

	public:
		constexpr matrix_const_col_iterator() noexcept = default;
		constexpr matrix_const_col_iterator(internal_pointer ptr, size_type rows, size_type stride) noexcept
			: ptr(ptr), rows(rows), stride(stride) {
		}

	public:
		constexpr value_type operator*() const { return value_type(ptr, rows, stride); }
		constexpr value_type operator[](size_type n) const { return value_type(ptr + n, rows, stride); }

	public:
		constexpr matrix_const_col_iterator& operator++() noexcept { ++ptr; return *this; }
		constexpr matrix_const_col_iterator operator++(int) noexcept { auto tmp = *this; ++ptr; return tmp; }
		constexpr matrix_const_col_iterator& operator--() noexcept { --ptr; return *this; }
		constexpr matrix_const_col_iterator operator--(int) noexcept { auto tmp = *this; --ptr; return tmp; }
		constexpr matrix_const_col_iterator& operator+=(difference_type n) noexcept { ptr += n; return *this; }
		constexpr matrix_const_col_iterator& operator-=(difference_type n) noexcept { ptr -= n; return *this; }

		constexpr matrix_const_col_iterator operator+(difference_type n) const noexcept {
			return matrix_const_col_iterator(ptr + n, rows, stride);
		}
		constexpr matrix_const_col_iterator operator-(difference_type n) const noexcept {
			return matrix_const_col_iterator(ptr - n, rows, stride);
		}
		constexpr difference_type operator-(const matrix_const_col_iterator& o) const noexcept {
			return ptr - o.ptr;
		}

	public:
		constexpr bool operator==(const matrix_const_col_iterator& o) const noexcept { return ptr == o.ptr; }
		constexpr auto operator<=>(const matrix_const_col_iterator& o) const noexcept { return ptr <=> o.ptr; }
};

template<typename T>
inline constexpr matrix_const_col_iterator<T> operator+(
		std::ptrdiff_t n, const matrix_const_col_iterator<T>& o) {
	return matrix_const_col_iterator(o.ptr + n, o.rows(), o.stride);
}

static_assert(std::random_access_iterator<matrix_const_col_iterator<int>>);

template<typename T>
struct matrix_col_iterator final : public matrix_const_col_iterator<T> {
	private:
		using base_t = matrix_const_col_iterator<T>;
		
	public:
		using typename base_t::difference_type;
		using typename base_t::size_type;
		using value_type = typename ColProxy<T>;
		using typename base_t::internal_pointer;

	private:
		using base_t::ptr;
		using base_t::rows;
		using base_t::stride;

	public:
		constexpr matrix_col_iterator() noexcept = default;
		constexpr matrix_col_iterator(internal_pointer ptr, size_type rows, size_type stride) noexcept : base_t(ptr, rows, stride) {}

	public:
		constexpr value_type operator*() const noexcept {
			return value_type(ptr, rows, stride);
		}

		constexpr value_type operator[](difference_type n) const {
			return value_type(ptr + n, rows, stride);
		}

	public:
		constexpr matrix_col_iterator& operator++() noexcept {
			base_t::operator++();
			return *this;
		}
		constexpr matrix_col_iterator operator++(int) noexcept {
			auto tmp = *this;
			base_t::operator++();
			return tmp;
		}

		constexpr matrix_col_iterator& operator--() noexcept {
			base_t::operator--();
			return *this;
		}
		constexpr matrix_col_iterator operator--(int) noexcept {
			auto tmp = *this;
			base_t::operator--();
			return tmp;
		}

		constexpr matrix_col_iterator& operator+=(difference_type n) noexcept {
			base_t::operator+=(n);
			return *this;
		}
		constexpr matrix_col_iterator& operator-=(difference_type n) noexcept {
			base_t::operator-=(n);
			return *this;
		}

		constexpr matrix_col_iterator operator+(difference_type n) const noexcept {
			return matrix_col_iterator(ptr + n, rows, stride);
		}
		constexpr matrix_col_iterator operator-(difference_type n) const noexcept {
			return matrix_col_iterator(ptr - n, rows, stride);
		}

		constexpr difference_type operator-(const matrix_col_iterator& o) const noexcept {
			return base_t::operator-(o);
		}

	public:
		constexpr bool operator==(const matrix_col_iterator& o) const noexcept { return ptr == o.ptr; }
		constexpr auto operator<=>(const matrix_col_iterator& o) const noexcept { return ptr <=> o.ptr; }
	
};

template <typename T>
inline constexpr matrix_col_iterator<T> operator+(std::ptrdiff_t n, const matrix_col_iterator<T>& o) {
	return matrix_col_iterator(o.ptr + n, o.rows, o.stride);
}

static_assert(std::random_access_iterator<matrix_col_iterator<int>>);

template<typename T>
struct matrix_const_diagonal_iterator {
	public:
		using value_type = T;
		using pointer = value_type*;
		using const_pointer = const value_type*;
		using reference = value_type&;
		using const_reference = const value_type&;
		using difference_type = std::ptrdiff_t;
		using size_type = std::size_t;
		using iterator_category = std::random_access_iterator_tag;
		using iterator_concept = std::random_access_iterator_tag;

	protected:
		pointer ptr = nullptr;
		difference_type offset = 0u;

	public:
		constexpr matrix_const_diagonal_iterator() noexcept = default;
		constexpr explicit matrix_const_diagonal_iterator(pointer p, size_type o) noexcept : ptr(p), offset(o) {}

	public:
		constexpr const_reference operator*() const { return *ptr; }
		constexpr const_pointer operator->() const { return ptr; }

		constexpr matrix_const_diagonal_iterator& operator++() noexcept { ptr += offset; return *this; }
		constexpr matrix_const_diagonal_iterator operator++(int) noexcept { auto tmp = *this; ptr += offset; return tmp; }

		constexpr matrix_const_diagonal_iterator& operator--() noexcept { ptr -= offset; return *this; }
		constexpr matrix_const_diagonal_iterator operator--(int) noexcept { auto tmp = *this; ptr -= offset; return tmp; }

		constexpr matrix_const_diagonal_iterator& operator+=(difference_type n) noexcept { ptr += n * offset; return *this; }
		constexpr matrix_const_diagonal_iterator& operator-=(difference_type n) noexcept { ptr -= n * offset; return *this; }

		constexpr matrix_const_diagonal_iterator operator+(difference_type n) const noexcept { return matrix_const_diagonal_iterator(ptr + n * offset, offset); }
		constexpr matrix_const_diagonal_iterator operator-(difference_type n) const noexcept { return matrix_const_diagonal_iterator(ptr - n * offset, offset); }

		constexpr difference_type operator-(const matrix_const_diagonal_iterator& other) const noexcept { return (ptr - other.ptr) / offset; }

		constexpr const_reference operator[](difference_type n) const noexcept { return *(ptr + n * offset); }

		constexpr bool operator==(const matrix_const_diagonal_iterator& o) const noexcept { return ptr == o.ptr; }
		constexpr auto operator<=>(const matrix_const_diagonal_iterator& o) const noexcept { return ptr <=> o.ptr; }
};

template<typename T>
inline constexpr matrix_const_diagonal_iterator<T> operator+(std::ptrdiff_t n, const matrix_const_diagonal_iterator<T>& o) {
	return matrix_const_diagonal_iterator<T>(o.offset * n + o.ptr, o.offset);
}

static_assert(std::random_access_iterator<matrix_const_diagonal_iterator<int>>);

template<typename T>
struct matrix_diagonal_iterator final : public matrix_const_diagonal_iterator<T> {
	private:
		using base_t = typename matrix_const_diagonal_iterator<T>;

	public:
		using typename base_t::value_type;
		using typename base_t::size_type;
		using typename base_t::pointer;
		using typename base_t::const_pointer;
		using typename base_t::reference;
		using typename base_t::const_reference;
		using typename base_t::difference_type;

	private:
		using base_t::ptr;
		using base_t::offset;

	public:
		constexpr matrix_diagonal_iterator() noexcept = default;
		constexpr explicit matrix_diagonal_iterator(pointer p, size_type o) : base_t(p, o) {}

	public:
		constexpr reference operator*() const { return *ptr; }
		constexpr pointer operator->() const { return ptr; }

		constexpr matrix_diagonal_iterator& operator++() noexcept { ptr += offset; return *this; }
		constexpr matrix_diagonal_iterator operator++(int) noexcept { auto tmp = *this; ptr += offset; return tmp; }

		constexpr matrix_diagonal_iterator& operator--() noexcept { ptr -= offset; return *this; }
		constexpr matrix_diagonal_iterator operator--(int) noexcept { auto tmp = *this; ptr -= offset; return tmp; }

		constexpr matrix_diagonal_iterator& operator+=(difference_type n) noexcept { ptr += n * offset; return *this; }
		constexpr matrix_diagonal_iterator& operator-=(difference_type n) noexcept { ptr -= n * offset; return *this; }

		constexpr matrix_diagonal_iterator operator+(difference_type n) const noexcept { return matrix_diagonal_iterator(ptr + n * offset, offset); }
		constexpr matrix_diagonal_iterator operator-(difference_type n) const noexcept { return matrix_diagonal_iterator(ptr - n * offset, offset); }

		constexpr difference_type operator-(const matrix_diagonal_iterator& other) const noexcept { return (ptr - other.ptr) / offset; }

		constexpr reference operator[](difference_type n) const noexcept { return *(ptr + n * offset); }

		constexpr bool operator==(const matrix_diagonal_iterator& o) const noexcept { return ptr == o.ptr; }
		constexpr auto operator<=>(const matrix_diagonal_iterator& o) const noexcept { return ptr <=> o.ptr; }
};

template<typename T>
inline constexpr matrix_diagonal_iterator<T> operator+(std::ptrdiff_t n, const matrix_diagonal_iterator<T>& o) {
	return matrix_diagonal_iterator<T>(o.offset * n + o.ptr, o.offset);
}

static_assert(std::random_access_iterator<matrix_diagonal_iterator<int>>);

template<typename T> constexpr auto maybe_null(T ptr) { return ptr ? std::addressof(*ptr) : nullptr; }

template<typename T, typename Alloc = std::allocator<T>>
struct Matrix {
	public:
		using value_type = T;
		using allocator_type = typename std::allocator_traits<Alloc>::allocator_type;
		using pointer = typename std::allocator_traits<Alloc>::pointer;
		using const_pointer = typename std::allocator_traits<Alloc>::const_pointer;
		using reference = T&;
		using const_reference = const T&;
		using size_type = typename std::allocator_traits<Alloc>::size_type;
		using difference_type = typename std::allocator_traits<Alloc>::difference_type;

	public:
		using col_proxy = ColProxy<value_type>;
		using const_col_proxy = ColProxy<const value_type>;
		using row_proxy = RowProxy<value_type>;
		using const_row_proxy = RowProxy<const value_type>;

	public:
		using iterator = matrix_default_iterator<value_type>;
		using const_iterator = matrix_const_default_iterator<value_type>;
		using reverse_iterator = std::reverse_iterator<iterator>;
		using const_reverse_iterator = std::reverse_iterator<const_iterator>;

		using row_const_iterator = matrix_const_row_iterator<value_type>;
		using row_const_reverse_iterator = std::reverse_iterator<row_const_iterator>;
		using row_iterator = matrix_row_iterator<value_type>;
		using row_reverse_iterator = std::reverse_iterator<row_iterator>;

		using col_const_iterator = matrix_const_col_iterator<value_type>;
		using col_const_reverse_iterator = std::reverse_iterator<col_const_iterator>;
		using col_iterator = matrix_col_iterator<value_type>;
		using col_reverse_iterator = std::reverse_iterator<col_iterator>;

		using diagonal_iterator = matrix_diagonal_iterator<value_type>;
		using diagonal_const_iterator = matrix_const_diagonal_iterator<value_type>;
		using diagonal_reverse_iterator = std::reverse_iterator<diagonal_iterator>;
		using diagonal_const_reverse_iterator = std::reverse_iterator<diagonal_const_iterator>;

	public:
		constexpr Matrix() noexcept = default;

		explicit Matrix(size_type rows, size_type cols)
			: _rows(rows), _cols(cols) {
			if (_rows == 0 || _cols == 0) {
				_rows = _cols = 0;
				_data = nullptr;
				return;
			}
			const size_type n = _rows * _cols;
			_data = std::allocator_traits<Alloc>::allocate(_allocator, n);
			for (size_type i = 0; i < n; ++i) {
				std::allocator_traits<Alloc>::construct(_allocator, std::addressof(_data[i]));
			}
		}

		// dtor
		~Matrix() {
			if (_data) {
				const size_type n = _rows * _cols;
				for (size_type i = 0; i < n; ++i) {
					std::allocator_traits<Alloc>::destroy(_allocator, std::addressof(_data[i]));
				}
				std::allocator_traits<Alloc>::deallocate(_allocator, _data, n);
				_data = nullptr;
			}
		}

		// copy
		Matrix(const Matrix& other)
			: _rows(other._rows), _cols(other._cols) {
			if (other._data) {
				const size_type n = _rows * _cols;
				_data = std::allocator_traits<Alloc>::allocate(_allocator, n);
				for (size_type i = 0; i < n; ++i) {
					std::allocator_traits<Alloc>::construct(_allocator, std::addressof(_data[i]), other._data[i]);
				}
			} else {
				_data = nullptr;
			}
		}

		// move
		Matrix(Matrix&& other) noexcept
			: _rows(other._rows), _cols(other._cols), _allocator(std::move(other._allocator)), _data(other._data) {
			other._rows = other._cols = 0;
			other._data = nullptr;
		}

		// copy assign
		Matrix& operator=(const Matrix& other) {
			if (this == &other) return *this;
			Matrix tmp(other);
			swap(tmp);
			return *this;
		}

		// move assign
		Matrix& operator=(Matrix&& other) noexcept {
			if (this == &other) return *this;
			this->~Matrix();
			_rows = other._rows;
			_cols = other._cols;
			_allocator = std::move(other._allocator);
			_data = other._data;
			other._rows = other._cols = 0;
			other._data = nullptr;
			return *this;
		}

		void swap(Matrix& o) noexcept {
			std::swap(_rows, o._rows);
			std::swap(_cols, o._cols);
			std::swap(_allocator, o._allocator);
			std::swap(_data, o._data);
		}

	public:
		constexpr size_type size() const noexcept { return _rows * _cols; }
		constexpr size_type rows() const noexcept { return _rows; }
		constexpr size_type cols() const noexcept { return _cols; }
		constexpr bool empty() const noexcept { return ((_rows == 0) || (_cols == 0)); }
		constexpr bool is_square() const noexcept { return _rows == _cols; }
		std::pair<size_type, size_type> get_index(const value_type& elem) const {
			auto&& x = std::addressof(elem);
			static_assert( ((x < _data) || (x >= _data + _rows * _cols)) );
#if 0
			if (x < _data || x >= _data + _rows * _cols) {
				throw std::runtime_error("get_index: element not in array bounds");
			}
#endif
			const auto offset = static_cast<size_type>(x - _data);
			return { offset / _cols, offset % _cols };
		}

		/* default */
		[[nodiscard]] constexpr pointer data() noexcept {
			return maybe_null(_data);
		}

		[[nodiscard]] constexpr const_pointer data() const noexcept {
			return maybe_null(_data);
		}

		[[nodiscard]] constexpr iterator begin() noexcept {
			return iterator(_data);
		}

		[[nodiscard]] constexpr const_iterator begin() const noexcept {
			return const_iterator(_data);
		}

		[[nodiscard]] constexpr iterator end() noexcept {
			return iterator(_data + size());
		}

		[[nodiscard]] constexpr const_iterator end() const noexcept {
			return const_iterator(_data + size());
		}

		[[nodiscard]] constexpr reverse_iterator rbegin() noexcept {
			return reverse_iterator(end());
		}

		[[nodiscard]] constexpr const_reverse_iterator rbegin() const noexcept {
			return const_reverse_iterator(end());
		}

		[[nodiscard]] constexpr reverse_iterator rend() noexcept {
			return reverse_iterator(begin());
		}

		[[nodiscard]] constexpr const_reverse_iterator rend() const noexcept {
			return const_reverse_iterator(begin());
		}

		[[nodiscard]] constexpr const_iterator cbegin() const noexcept {
			return begin();
		}

		[[nodiscard]] constexpr const_iterator cend() const noexcept {
			return end();
		}

		[[nodiscard]] constexpr const_reverse_iterator crbegin() const noexcept {
			return rbegin();
		}

		[[nodiscard]] constexpr const_reverse_iterator crend() const noexcept {
			return rend();
		}

		/* row */
		[[nodiscard]] constexpr row_iterator row_begin() noexcept {
			return row_iterator(_data, _cols);
		}

		[[nodiscard]] constexpr row_const_iterator row_begin() const noexcept {
			return row_const_iterator(_data, _cols);
		}

		[[nodiscard]] constexpr row_iterator row_end() noexcept {
			return row_iterator(_data + size(), _cols);
		}

		[[nodiscard]] constexpr row_const_iterator row_end() const noexcept {
			return row_const_iterator(_data + size(), _cols);
		}

		[[nodiscard]] constexpr row_reverse_iterator row_rbegin() noexcept {
			return row_reverse_iterator(row_end());
		}

		[[nodiscard]] constexpr row_const_reverse_iterator row_rbegin() const noexcept {
			return row_const_reverse_iterator(row_end());
		}

		[[nodiscard]] constexpr row_reverse_iterator row_rend() noexcept {
			return row_reverse_iterator(row_begin());
		}

		[[nodiscard]] constexpr row_const_reverse_iterator row_rend() const noexcept {
			return row_const_reverse_iterator(row_begin());
		}

		[[nodiscard]] constexpr row_const_iterator row_cbegin() const noexcept {
			return row_begin();
		}

		[[nodiscard]] constexpr row_const_iterator row_cend() const noexcept {
			return row_end();
		}

		[[nodiscard]] constexpr row_const_reverse_iterator row_crbegin() const noexcept {
			return row_const_reverse_iterator(row_cend());
		}

		[[nodiscard]] constexpr row_const_reverse_iterator row_crend() const noexcept {
			return row_const_reverse_iterator(row_cbegin());
		}


		/* col */
		[[nodiscard]] constexpr col_iterator col_begin() noexcept {
			return col_iterator(_data, _rows, _cols);
		}

		[[nodiscard]] constexpr col_const_iterator col_begin() const noexcept {
			return col_const_iterator(_data, _rows, _cols);
		}

		[[nodiscard]] constexpr col_iterator col_end() noexcept {
			return col_iterator(_data + _cols, _rows, _cols);
		}

		[[nodiscard]] constexpr col_const_iterator col_end() const noexcept {
			return col_const_iterator(_data + _cols, _rows, _cols);
		}

		[[nodiscard]] constexpr col_reverse_iterator col_rbegin() noexcept {
			return col_reverse_iterator(col_end());
		}

		[[nodiscard]] constexpr col_const_reverse_iterator col_rbegin() const noexcept {
			return col_const_reverse_iterator(col_end());
		}

		[[nodiscard]] constexpr col_reverse_iterator col_rend() noexcept {
			return col_reverse_iterator(col_begin());
		}

		[[nodiscard]] constexpr col_const_reverse_iterator col_rend() const noexcept {
			return col_const_reverse_iterator(col_begin());
		}

		[[nodiscard]] constexpr col_const_iterator col_cbegin() const noexcept {
			return col_begin();
		}

		[[nodiscard]] constexpr col_const_iterator col_cend() const noexcept {
			return col_end();
		}

		[[nodiscard]] constexpr col_const_reverse_iterator col_crbegin() const noexcept {
			return col_rbegin();
		}

		[[nodiscard]] constexpr col_const_reverse_iterator col_crend() const noexcept {
			return col_rend();
		}

		/* diagonal */
		[[nodiscard]] constexpr diagonal_iterator diagonal_begin() {
			if (!is_square()) { throw std::runtime_error("diagonal iterator usable only in square matrices"); }
			return diagonal_iterator(_data, _rows + 1u);
		}

		[[nodiscard]] constexpr diagonal_const_iterator diagonal_begin() const {
			if (!is_square()) { throw std::runtime_error("diagonal iterator usable only in square matrices"); }
			return diagonal_const_iterator(_data, _rows + 1u);
		}

		[[nodiscard]] constexpr diagonal_iterator diagonal_end() {
			if (!is_square()) { throw std::runtime_error("diagonal iterator usable only in square matrices"); }
			return diagonal_iterator(_data + (_rows * _cols + _rows), _rows + 1u);
		}

		[[nodiscard]] constexpr diagonal_const_iterator diagonal_end() const {
			if (!is_square()) { throw std::runtime_error("diagonal iterator usable only in square matrices"); }
			return diagonal_const_iterator(_data + _rows * _cols, _rows + 1u);
		}

		[[nodiscard]] constexpr diagonal_reverse_iterator diagonal_rbegin() {
			if (!is_square()) { throw std::runtime_error("diagonal iterator usable only in square matrices"); }
			return diagonal_reverse_iterator(diagonal_end());
		}

		[[nodiscard]] constexpr diagonal_const_reverse_iterator diagonal_rbegin() const {
			if (!is_square()) { throw std::runtime_error("diagonal iterator usable only in square matrices"); }
			return diagonal_const_reverse_iterator(diagonal_end());
		}

		[[nodiscard]] constexpr diagonal_reverse_iterator diagonal_rend() {
			if (!is_square()) { throw std::runtime_error("diagonal iterator usable only in square matrices"); }
			return diagonal_reverse_iterator(diagonal_begin());
		}

		[[nodiscard]] constexpr diagonal_const_reverse_iterator diagonal_rend() const noexcept {
			if (!is_square()) { throw std::runtime_error("diagonal iterator usable only in square matrices"); }
			return diagonal_const_reverse_iterator(diagonal_begin());
		}

		[[nodiscard]] constexpr diagonal_const_iterator diagonal_cbegin() const noexcept {
			if (!is_square()) { throw std::runtime_error("diagonal iterator usable only in square matrices"); }
			return diagonal_begin();
		}

		[[nodiscard]] constexpr diagonal_const_iterator diagonal_cend() const noexcept {
			if (!is_square()) { throw std::runtime_error("diagonal iterator usable only in square matrices"); }
			return diagonal_end();
		}

		[[nodiscard]] constexpr diagonal_const_reverse_iterator diagonal_crbegin() const noexcept {
			if (!is_square()) { throw std::runtime_error("diagonal iterator usable only in square matrices"); }
			return col_rbegin();
		}

		[[nodiscard]] constexpr diagonal_const_reverse_iterator diagonal_crend() const noexcept {
			if (!is_square()) { throw std::runtime_error("diagonal iterator usable only in square matrices"); }
			return diagonal_rend();
		}


	public:
		[[nodiscard]] constexpr ColProxy<value_type> col(size_type j) {
			// if (j >= _cols) throw std::out_of_range("Column index out of range");
			return ColProxy<value_type>{_data + j, _rows, _cols};
		}

		[[nodiscard]] constexpr ColProxy<const value_type> col(size_type j) const {
			// if (j >= _cols) throw std::out_of_range("Column index out of range");
			return ColProxy<const value_type>{_data + j, _rows, _cols};
		}

		[[nodiscard]] constexpr RowProxy<value_type> row(size_type j) {
			return RowProxy<value_type>(_data + j * _cols, _cols);
		}

		[[nodiscard]] constexpr RowProxy<const value_type> row(size_type j) const {
			return RowProxy<const value_type>(_data + j * _cols, _cols);
		}

	public:
		/* row */
		[[nodiscard]] constexpr auto row_range() noexcept {
			return std::ranges::subrange(row_begin(), row_end());
		}

		[[nodiscard]] constexpr auto row_range() const noexcept {
			return std::ranges::subrange(row_cbegin(), row_cend());
		}

		[[nodiscard]] constexpr auto row_const_range() const noexcept {
			return std::ranges::subrange(row_cbegin(), row_cend());
		}

		[[nodiscard]] constexpr auto row_reverse_range() noexcept {
			return std::ranges::subrange(row_rbegin(), row_rend());
		}

		[[nodiscard]] constexpr auto row_const_reverse_range() const noexcept {
			return std::ranges::subrange(row_crbegin(), row_crend());
		}
		
		/* col */
		[[nodiscard]] constexpr auto col_range() noexcept {
			return std::ranges::subrange(col_begin(), col_end());
		}

		[[nodiscard]] constexpr auto col_range() const noexcept {
			return std::ranges::subrange(col_cbegin(), col_cend());
		}

		[[nodiscard]] constexpr auto col_const_range() const noexcept {
			return std::ranges::subrange(col_cbegin(), col_cend());
		}

		[[nodiscard]] constexpr auto col_reverse_range() noexcept {
			return std::ranges::subrange(col_rbegin(), col_rend());
		}

		[[nodiscard]] constexpr auto col_const_reverse_range() const noexcept {
			return std::ranges::subrange(col_crbegin(), col_crend());
		}

		/* diagonal */
		[[nodiscard]] constexpr auto diagonal_range() noexcept {
			return std::ranges::subrange(diagonal_begin(), diagonal_end());
		}

		[[nodiscard]] constexpr auto diagonal_range() const noexcept {
			return std::ranges::subrange(diagonal_cbegin(), diagonal_cend());
		}

		[[nodiscard]] constexpr auto diagonal_const_range() const noexcept {
			return std::ranges::subrange(diagonal_cbegin(), diagonal_cend());
		}

		[[nodiscard]] constexpr auto diagonal_reverse_range() noexcept {
			return std::ranges::subrange(diagonal_rbegin(), diagonal_rend());
		}

		[[nodiscard]] constexpr auto diagonal_const_reverse_range() const noexcept {
			return std::ranges::subrange(diagonal_crbegin(), diagonal_crend());
		}
		
	public:
		constexpr RowProxy<value_type> operator[](size_type r) {
			// if (r >= _rows) throw std::out_of_range("Row index out of range");
			return RowProxy<value_type>{ _data + r * _cols, _cols };
		}

		constexpr const RowProxy<const value_type> operator[](size_type r) const {
			// if (r >= _rows) throw std::out_of_range("Row index out of range");
			return RowProxy<const value_type>{ _data + r * _cols, _cols };
		}

		constexpr reference at(size_type r, size_type c) {
			if (r >= _rows || c >= _cols) throw std::out_of_range("Index out of range");
			return _data[r * _cols + c];
		}
		constexpr const_reference at(size_type r, size_type c) const {
			if (r >= _rows || c >= _cols) throw std::out_of_range("Index out of range");
			return _data[r * _cols + c];
		}

	public:
		void fill(const T& value) {
			std::fill(begin(), end(), value);
		}

	public:
		void print(std::ostream& os = std::cout, int precision = 6) const {
			const auto m = rows();
			const auto n = cols();

			os << std::fixed << std::setprecision(precision);
			auto&& it = *this;
			for (size_type i = 0; i < m; ++i) {
				os << "[ ";
				for (size_type j = 0; j < n; ++j) {
					os << std::setw(10) << it[i][j];
					if (j + 1 < n) os << ' ';
				}
				os << " ]" << std::endl;
			}
			os << std::endl;
		}
		
		template<typename AllocR = Alloc>
		auto get_row_as_vector(size_type x) const {
			auto&& it = *this;
			auto r = it[x];
			return std::vector<value_type, AllocR>(r.begin(), r.end());
		}

		template<typename AllocR = Alloc>
		auto get_row_as_matrix_row(size_type x) const {
			auto&& it = *this;
			auto r = it[x];
			auto result = Matrix<value_type, AllocR>(1, it.cols());
			for (size_type i = 0; i < it.cols(); ++i) {
				result[0][i] = r[i];
			}
			return result;
		}

		template<typename AllocR = Alloc>
		auto get_row_as_matrix_col(size_type x) const {
			auto&& it = *this;
			auto r = it[x];
			auto result = Matrix<value_type, AllocR>(it.cols(), 1);
			for (size_type i = 0; i < it.cols(); ++i) {
				result[i][0] = r[i];
			}
			return result;
		}

		template<typename AllocR = Alloc>
		auto get_col_as_vector(size_type x) const {
			auto&& it = *this;
			auto c = it.col(x);
			return std::vector<value_type, AllocR>(c.begin(), c.end());
		}

		template<typename AllocR = Alloc>
		auto get_col_as_matrix_row(size_type x) const {
			auto&& it = *this;
			auto c = it.col(x);
			auto result = Matrix<value_type, AllocR>(1, it.rows());
			for (size_type i = 0; i < it.rows(); ++i) {
				result[0][i] = c[i];
			}
			return result;
		}

		template<typename AllocR = Alloc>
		auto get_col_as_matrix_col(size_type x) const {
			auto&& it = *this;
			auto c = it.col(x);
			auto result = Matrix<value_type, AllocR>(it.rows(), 1);
			for (size_type i = 0; i < it.rows(); ++i) {
				result[i][0] = c[i];
			}
			return result;
		}

		template<typename AllocR = Alloc>
		auto get_submatrix(
			size_type from_x,
			size_type from_y,
			size_type to_x,
			size_type to_y
		) const {
#if 0
			if (from_x >= _rows || from_y >= _cols || to_x >= _rows || to_y >= _cols) {
				throw std::out_of_range("Submatrix indices out of range");
			}

			if (from_x > to_x || from_y > to_y) {
				throw std::invalid_argument("Invalid submatrix range: from indices must be <= to indices");
			}
#endif
			auto&& it = *this;
			const auto N = to_x - from_x + 1;
			const auto M = to_y - from_y + 1;

			auto result = Matrix<value_type, AllocR>(N, M);

			for (size_type i = 0; i < N; ++i) {
				for (size_type j = 0; j < M; ++j) {
					result[i][j] = it[from_x + i][from_y + j];
				}
			}

			return result;
		}

		template<typename AllocR = Alloc>
		auto get_submatrix(const value_type& from_elem, const value_type& to_elem) const {
			auto&& [from_x, from_y] = get_index(from_elem);
			auto&& [to_x, to_y] = get_index(to_elem);
			return get_submatrix<AllocR>(from_x, from_y, to_x, to_y);
		}

		void set_submatrix(size_type to_x, size_type to_y, const Matrix<value_type>& A) {
			auto&& it = *this;
#if 0
			if (to_x >= it.rows() || to_y >= it.cols()) {
				throw std::out_of_range("set_submatrix: starting indices out of range");
			}

			if (to_x + A.rows() > it.rows() || to_y + A.cols() > it.cols()) {
				throw std::runtime_error("set_submatrix: submatrix does not fit in target matrix");
			}
#else
			static_assert(to_x >= it.rows() || to_y >= it.cols());
			static_assert(to_x + A.rows() > it.rows() || to_y + A.cols() > it.cols());
#endif
			
			for (size_type i = 0; i < A.rows(); ++i) {
				for (size_type j = 0; j < A.cols(); ++j) {
					it[to_x + i][to_y + j] = A[i][j];
				}
			}
		}

	public:
		Matrix transpose() const {
			auto&& it = *this;
			Matrix result(_cols, _rows);
			for (size_type i = 0; i < _rows; ++i) {
				for (size_type j = 0; j < _cols; ++j) {
					result[j][i] = it[i][j];
				}
			}
			return result;
		}

	private:

		template<typename First_, typename Second_>
		auto mult(
			const Matrix<First_>& A,
			const Matrix<Second_>& B
		) const -> Matrix<std::common_type_t<First_, Second_>> {
			using ResultType = std::common_type_t<First_, Second_>;
			auto C = Matrix<ResultType>(A.rows(), B.cols());

			using size_type = typename Matrix<ResultType>::size_type;

			const auto M = A.rows(); // C.rows()
			const auto N = B.cols(); // C.cols()
			const auto K = A.cols();

			for (size_type i = 0; i < N; ++i) {
				auto&& c = C.col(i);
				for (size_type j = 0; j < K; ++j) {
					auto&& a_col = A.col(j);
					auto&& b_val = B[j][i];
					for (size_type k = 0; k < M; ++k) {
						c[k] += a_col[k] * b_val;
					}
				}
			}

			return C;
		}

		template<typename First_, typename Second_>
		auto dummy_mult(
			const Matrix<First_>& A,
			const Matrix<Second_>& B
		) const -> Matrix<std::common_type_t<First_, Second_>> {
			if (A.cols() != B.rows()) {
				throw std::runtime_error("Matrix dimensions do not match for multiplication");
			}
			using ResultType = std::common_type_t<First_, Second_>;
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

	public:

		template<typename U>
		auto operator*(const Matrix<U>& other) const -> Matrix<std::common_type_t<T, U>> {
			auto&& it = *this;
			if (it.cols() != other.rows()) {
				throw std::runtime_error("Matrix dimensions do not match for multiplication");
			}

			return mult(it, other);
		}

		template<typename U>
		Matrix& operator*=(const Matrix<U>& other) {
			auto&& it = *this;
			if (it.cols() != other.rows()) {
				throw std::runtime_error("Matrix dimensions do not match for multiplication");
			}

			using ResultType = std::common_type_t<value_type, U>;
			it = std::move(mult(it, other));

			return *this;
		}

		// Scalar multiplication (matrix * scalar)
		template<typename U>
		auto operator*(U scalar) const -> Matrix<std::common_type_t<T, U>> {
			using ResultType = std::common_type_t<T, U>;
			Matrix<ResultType> result(_rows, _cols);

			for (size_type i = 0; i < _rows; ++i) {
				for (size_type j = 0; j < _cols; ++j) {
					result[i][j] = (*this)[i][j] * scalar;
				}
			}

			return result;
		}

		template<typename U>
		requires std::is_arithmetic_v<U>
		Matrix& operator*=(const U& scalar) {
			for (size_type i = 0; i < _rows; ++i) {
				for (size_type j = 0; j < _cols; ++j) {
					(*this)[i][j] *= scalar;
				}
			}
			return *this;
		}

		template<typename U>
		auto operator/(U scalar) const -> Matrix<std::common_type_t<T, U>> {
			using ResultType = std::common_type_t<T, U>;
			if (scalar == U{0}) {
				throw std::runtime_error("Division by zero");
			}
		
			Matrix<ResultType> result(_rows, _cols);
		
			for (size_type i = 0; i < _rows; ++i) {
				for (size_type j = 0; j < _cols; ++j) {
					result[i][j] = (*this)[i][j] / scalar;
				}
			}
		
			return result;
		}
	
		template<typename U>
		Matrix& operator/=(U scalar) {
			if (scalar == U{0}) {
				throw std::runtime_error("Division by zero");
			}
		
			for (size_type i = 0; i < _rows; ++i) {
				for (size_type j = 0; j < _cols; ++j) {
					(*this)[i][j] /= scalar;
				}
			}
			return *this;
		}

		template<typename U>
		auto operator+(const Matrix<U>& other) const -> Matrix<std::common_type_t<T, U>> {
			if (_rows != other.rows() || _cols != other.cols()) {
				throw std::runtime_error("Matrix dimensions do not match for addition");
			}

			using ResultType = std::common_type_t<T, U>;
			Matrix<ResultType> result(_rows, _cols);

			for (size_type i = 0; i < _rows; ++i) {
				for (size_type j = 0; j < _cols; ++j) {
					result[i][j] = static_cast<ResultType>((*this)[i][j]) + 
								  static_cast<ResultType>(other[i][j]);
				}
			}

			return result;
		}

		template<typename U>
		auto operator-(const Matrix<U>& other) const -> Matrix<std::common_type_t<T, U>> {
			if (_rows != other.rows() || _cols != other.cols()) {
				throw std::runtime_error("Matrix dimensions do not match for subtraction");
			}

			using ResultType = std::common_type_t<T, U>;
			Matrix<ResultType> result(_rows, _cols);

			for (size_type i = 0; i < _rows; ++i) {
				for (size_type j = 0; j < _cols; ++j) {
					result[i][j] = static_cast<ResultType>((*this)[i][j]) - 
								  static_cast<ResultType>(other[i][j]);
				}
			}

			return result;
		}

		Matrix operator+() const { return *this; }

		Matrix operator-() const {
			Matrix result(_rows, _cols);
		
			for (size_type i = 0; i < _rows; ++i) {
				for (size_type j = 0; j < _cols; ++j) {
					result[i][j] = -(*this)[i][j];
				}
			}
		
			return result;
		}

		Matrix& operator+=(const Matrix& other) {
			if (_rows != other._rows || _cols != other._cols) {
				throw std::runtime_error("Matrix dimensions do not match for addition");
			}

			for (size_type i = 0; i < _rows; ++i) {
				for (size_type j = 0; j < _cols; ++j) {
					(*this)[i][j] += other[i][j];
				}
			}

			return *this;
		}

		Matrix& operator-=(const Matrix& other) {
			if (_rows != other._rows || _cols != other._cols) {
				throw std::runtime_error("Matrix dimensions do not match for subtraction");
			}

			for (size_type i = 0; i < _rows; ++i) {
				for (size_type j = 0; j < _cols; ++j) {
					(*this)[i][j] -= other[i][j];
				}
			}

			return *this;
		}

	private:
		size_type _rows = 0;
		size_type _cols = 0;
		Alloc _allocator = Alloc();
		pointer _data = nullptr;
};

template<typename T, typename U, typename Alloc>
auto operator*(U scalar, const Matrix<T, Alloc>& matrix) -> Matrix<std::common_type_t<T, U>> {
	return matrix * scalar;
}

#if 0
template<typename T, typename Alloc>
Matrix<T> operator/(const Matrix<T, Alloc>& B, const Matrix<T, Alloc>& A) {
	if (A.rows() != A.cols()) {
		throw std::runtime_error("Matrix A must be square for division");
	}
	if (A.rows() != B.rows()) {
		throw std::runtime_error("Matrix dimensions do not match for division");
	}

	auto A_inv = A.inverse();
	return A_inv * B;
}
#endif

constexpr auto to_Matrix_col(std::ranges::range auto&& range) {
	using value_type = std::ranges::range_value_t<std::remove_cvref_t<decltype(range)>>;
	auto result = Matrix<value_type>(std::ranges::size(range), 1);
	std::ranges::copy(range, result.col(0).begin());
	return result;
}

constexpr auto to_Matrix_row(std::ranges::range auto&& range) {
	using value_type = std::ranges::range_value_t<std::remove_cvref_t<decltype(range)>>;
	auto result = Matrix<value_type>(std::ranges::size(range), 1);
	std::ranges::copy(range, result.row(0).begin());
	return result;
}

template<typename T, typename Alloc = std::allocator<T>>
struct OneColMatrix : public Matrix<T, Alloc> {
	using base_t = typename Matrix<T, Alloc>;
	OneColMatrix(std::ranges::range auto&& range) : base_t(to_Matrix_col(range)) {};
};

template<typename T, typename Alloc = std::allocator<T>>
struct OneRowMatrix : public Matrix<T, Alloc> {
	using base_t = typename Matrix<T, Alloc>;
	OneRowMatrix(std::ranges::range auto&& range) : base_t(to_Matrix_row(range)) {};
};

#endif
