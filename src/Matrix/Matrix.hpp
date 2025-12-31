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

namespace detail {
	template<typename T> concept object = requires {
		requires !std::is_const_v<T>;
		requires !std::is_volatile_v<T>;
		requires !std::is_lvalue_reference_v<T>;
	};
	
	template<object T> struct TestCont {
		public:
			using allocator_type = std::allocator<T>;
			using allocator_traits = std::allocator_traits<allocator_type>;

		public:
			using value_type = typename allocator_traits::value_type;
			using pointer = typename allocator_traits::pointer;
			using const_pointer = typename allocator_traits::const_pointer;
			using size_type = typename allocator_traits::size_type;
			using difference_type = typename allocator_traits::difference_type;
			using reference = value_type&;
			using const_reference = const value_type&;

		private:
			pointer* dummy = nullptr;
		public:
			constexpr TestCont() noexcept = default;
	};

	using tCont = typename TestCont<float>;
}

template<typename Cont>
struct row_const_iterator {
	public:
		using value_type = Cont::value_type;
		using pointer = Cont::pointer;
		using const_pointer = Cont::const_pointer;
		using size_type = Cont::size_type;
		using difference_type = Cont::difference_type;
		using reference = Cont::reference;
		using const_reference = Cont::const_reference;

	public:
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

template<typename Cont>
struct row_iterator : public row_const_iterator<Cont> {
	private:
		using base_t = row_const_iterator<Cont>;
		
	public:
		using typename base_t::value_type;
		using typename base_t::pointer;
		using typename base_t::const_pointer;
		using typename base_t::size_type;
		using typename base_t::difference_type;
		using typename base_t::reference;
		using typename base_t::const_reference;

	public:
		using base_t::ptr;

	public:
		constexpr row_iterator() noexcept = default;
		constexpr explicit row_iterator(pointer p) noexcept : base_t(p) { }

	public:
		constexpr reference operator*() const noexcept { return *ptr; }
		constexpr pointer operator->() const noexcept { return ptr; }

		constexpr reference operator[](difference_type n) const noexcept { return *(ptr + n); }

		constexpr row_iterator& operator++() noexcept {++ptr; return *this; }
		constexpr row_iterator operator++(int) noexcept { auto tmp = *this; ++ptr; return tmp; }
		constexpr row_iterator& operator--() noexcept { --ptr; return *this; }
		constexpr row_iterator operator--(int) noexcept { auto tmp = *this; --ptr; return tmp; }

		constexpr row_iterator& operator+=(difference_type n) noexcept { ptr += n; return *this; }
		constexpr row_iterator& operator-=(difference_type n) noexcept { ptr -= n; return *this; }

		constexpr row_iterator operator+(difference_type n) const noexcept { return row_iterator(ptr + n); }
		constexpr row_iterator operator-(difference_type n) const noexcept { return row_iterator(ptr - n); }
		constexpr difference_type operator-(const row_iterator& o) const noexcept { return ptr - o.ptr; }

		constexpr bool operator==(const row_iterator& o) const noexcept { return ptr == o.ptr; }
		constexpr auto operator<=>(const row_iterator& o) const noexcept { return ptr <=> o.ptr; }
};

template<typename T>
constexpr row_iterator<T> operator+(
	typename row_iterator<T>::difference_type n,
	const row_iterator<T>& it) noexcept {
	return it + n;
}

static_assert(std::contiguous_iterator<row_iterator<detail::tCont>>);
static_assert(std::contiguous_iterator<row_const_iterator<detail::tCont>>);

template<typename Cont>
struct const_RowProxy : public std::ranges::view_interface<const_RowProxy<Cont>> {
	public:
		using value_type = Cont::value_type;
		using pointer = Cont::pointer;
		using const_pointer = Cont::const_pointer;
		using size_type = Cont::size_type;
		using difference_type = Cont::difference_type;
		using reference = Cont::reference;
		using const_reference = Cont::const_reference;

	public:
		using iterator = row_const_iterator<Cont>;
		using const_iterator = iterator;
		using reverse_iterator = std::reverse_iterator<const_iterator>;
		using const_reverse_iterator = reverse_iterator;

	protected:
		pointer row_data = nullptr;
		size_type cols = 0;

	public:
		constexpr const_RowProxy() noexcept = default;
		constexpr const_RowProxy(pointer p, size_type c) noexcept : row_data(p), cols(c) {}

	public:
		constexpr const_reference operator[](size_type c) const { return row_data[c]; }

	public:
		constexpr const_iterator begin() const noexcept { return const_iterator(row_data); }
		constexpr const_iterator end() const noexcept { return const_iterator(row_data + cols); }
		constexpr const_iterator cbegin() const noexcept { return const_iterator(begin()); }
		constexpr const_iterator cend() const noexcept { return const_iterator(end()); }
		constexpr const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
		constexpr const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
		constexpr const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }
		constexpr const_reverse_iterator crend() const noexcept { return const_reverse_iterator(cbegin()); }

	public:
		constexpr bool empty() const noexcept { return cols == 0; }
		constexpr const_reference front() const { return *row_data; }
		constexpr const_reference back() const { return row_data[cols - 1]; }
		constexpr size_type size() const noexcept { return cols; }
		constexpr pointer data() const noexcept { return row_data; }
};

static_assert(std::ranges::view<const_RowProxy<detail::tCont>>);

template<typename Cont>
struct RowProxy final : public const_RowProxy<Cont> {
	private:
		using base_t = const_RowProxy<Cont>;

	public:
		using typename base_t::value_type;
		using typename base_t::pointer;
		using typename base_t::const_pointer;
		using typename base_t::size_type;
		using typename base_t::difference_type;
		using typename base_t::reference;
		using typename base_t::const_reference;

	public:
		using iterator = row_iterator<Cont>;
		using const_iterator = row_const_iterator<Cont>;
		using reverse_iterator = std::reverse_iterator<iterator>;
		using const_reverse_iterator = std::reverse_iterator<const_iterator>;

	private:
		using base_t::row_data;
		using base_t::cols;

	public:
		constexpr RowProxy() noexcept = default;
		constexpr RowProxy(pointer data, size_type c) noexcept : base_t(data, c) {};

	public:
		constexpr reference operator[](size_type c) { return row_data[c];	}
		
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

static_assert(std::ranges::view<RowProxy<detail::tCont>>);

template<typename Cont>
struct stride_const_iterator {
	public:
		using value_type = Cont::value_type;
		using pointer = Cont::pointer;
		using const_pointer = Cont::const_pointer;
		using size_type = Cont::size_type;
		using difference_type = Cont::difference_type;
		using reference = Cont::reference;
		using const_reference = Cont::const_reference;

	public:
		using iterator_category = std::random_access_iterator_tag;
		using iterator_concept = std::random_access_iterator_tag;

	protected:
		pointer ptr = nullptr;
		size_type step = 0;

	public:
		constexpr stride_const_iterator() noexcept = default;
		constexpr stride_const_iterator(pointer p, size_type s) noexcept : ptr(p), step(s) {}

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
			return (ptr - o.ptr) / step;
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

static_assert(std::random_access_iterator<stride_const_iterator<detail::tCont>>);

template<typename Cont>
struct stride_iterator final : public stride_const_iterator<Cont> {
	private:
		using base_t = stride_const_iterator<Cont>;

	public:
		using typename base_t::value_type;
		using typename base_t::pointer;
		using typename base_t::const_pointer;
		using typename base_t::size_type;
		using typename base_t::difference_type;
		using typename base_t::reference;
		using typename base_t::const_reference;

	public:
		using iterator_category = std::random_access_iterator_tag;
		using iterator_concept = std::random_access_iterator_tag;

	private:
		using base_t::ptr;
		using base_t::step;

	public:
		constexpr stride_iterator() noexcept = default;
		constexpr stride_iterator(pointer p, size_type s) noexcept : base_t(p, s) { }

		constexpr reference operator*() const noexcept { return *ptr;	}
		constexpr pointer operator->() const noexcept { return ptr; }

		constexpr reference operator[](difference_type n) const noexcept {
			return *(ptr + n * step);
		}

		constexpr stride_iterator& operator++() noexcept { ptr += step; return *this; }
		constexpr stride_iterator operator++(int) noexcept { auto tmp = *this; ptr += step; return tmp; }
		constexpr stride_iterator& operator--() noexcept { ptr -= step; return *this; }
		constexpr stride_iterator operator--(int) noexcept { auto tmp = *this; ptr -= step; return tmp; }

		constexpr stride_iterator& operator+=(difference_type n) noexcept { ptr += n * step; return *this; }
		constexpr stride_iterator& operator-=(difference_type n) noexcept { ptr -= n * step; return *this; }

		constexpr stride_iterator operator+(difference_type n) const noexcept { return stride_iterator( (ptr + n * step), step ); }
		constexpr stride_iterator operator-(difference_type n) const noexcept { return stride_iterator( (ptr - n * step), step ); }

		constexpr difference_type operator-(const stride_iterator& other) const noexcept {
			return (ptr - other.ptr) / step;
		}

	public:
		constexpr auto operator<=>(const stride_iterator& o) const noexcept { return base_t::operator<=>(o); }
		constexpr bool operator==(const stride_iterator& o) const noexcept { return base_t::operator==(o); }
};

template<typename T>
constexpr stride_iterator<T> operator+(
	typename stride_iterator<T>::difference_type n,
	const stride_iterator<T>& it) noexcept {
	return it + n;
}

static_assert(std::random_access_iterator<stride_iterator<detail::tCont>>);

template<typename Cont>
struct const_ColProxy : public std::ranges::view_interface<const_ColProxy<Cont>> {
	public:
		using value_type = Cont::value_type;
		using pointer = Cont::pointer;
		using const_pointer = Cont::const_pointer;
		using size_type = Cont::size_type;
		using difference_type = Cont::difference_type;
		using reference = Cont::reference;
		using const_reference = Cont::const_reference;

	public:
		using iterator = stride_const_iterator<Cont>;
		using const_iterator = iterator;
		using reverse_iterator = std::reverse_iterator<const_iterator>;
		using const_reverse_iterator = reverse_iterator;

	protected:
		pointer col_data = nullptr;
		size_type rows = 0;
		size_type stride = 0;

	public:
		constexpr const_ColProxy() noexcept = default;
		constexpr const_ColProxy(pointer base, size_type r, size_type stride_) noexcept
			: col_data(base), rows(r), stride(stride_) {
		}

		constexpr const_reference operator[](size_type r) const {
			return *(col_data + r * stride);
		}


	public:
		constexpr const_iterator begin() const noexcept { return const_iterator(col_data, stride); }
		constexpr const_iterator end() const noexcept { return const_iterator(col_data + rows * stride, stride); }
		constexpr const_iterator cbegin() const noexcept { return const_iterator(begin()); }
		constexpr const_iterator cend() const noexcept { return const_iterator(end()); }
		constexpr const_reverse_iterator rbegin() noexcept { return const_reverse_iterator(end()); }
		constexpr const_reverse_iterator rend() noexcept { return const_reverse_iterator(begin()); }
		constexpr const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }
		constexpr const_reverse_iterator crend() const noexcept { return const_reverse_iterator(cbegin()); }

	public:
		constexpr size_type size() const noexcept { return rows; }
		constexpr bool empty() const noexcept { return rows == 0; }
		constexpr const_reference back() const { return *(col_data + (rows - 1) * stride); }
		constexpr const_reference front() const { return *col_data; }
};

template<typename Cont>
struct ColProxy final : public const_ColProxy<Cont> {
	private:
		using base_t = const_ColProxy<Cont>;

	public:
		using typename base_t::value_type;
		using typename base_t::pointer;
		using typename base_t::const_pointer;
		using typename base_t::size_type;
		using typename base_t::difference_type;
		using typename base_t::reference;
		using typename base_t::const_reference;

	public:
		using iterator = stride_iterator<Cont>;
		using const_iterator = stride_const_iterator<Cont>;
		using reverse_iterator = std::reverse_iterator<iterator>;
		using const_reverse_iterator = std::reverse_iterator<const_iterator>;

	private:
		using base_t::col_data;
		using base_t::rows;
		using base_t::stride;

	public:
		constexpr ColProxy() noexcept = default;
		constexpr ColProxy(pointer base, size_type r, size_type stride) noexcept
			: base_t(base, r, stride) { // col_data(base), rows(r), stride(stride_) {
		}

		constexpr reference operator[](size_type r) {
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
		constexpr reference back() { return *(col_data + (rows - 1) * stride); }
		constexpr const_reference back() const { return *(col_data + (rows - 1) * stride); }
};

template<typename T> concept derived_view_base_check = std::derived_from<T, std::ranges::view_base>;
static_assert(std::ranges::view<ColProxy<detail::tCont>>);
static_assert(std::ranges::range<ColProxy<detail::tCont>>);
static_assert(std::movable<ColProxy<detail::tCont>>);
static_assert(std::ranges::enable_view<ColProxy<detail::tCont>>);
// static_assert(derived_view_base_check<ColProxy<int>>);

template<typename Cont>
struct matrix_const_default_iterator {
	public:
		using value_type = Cont::value_type;
		using pointer = Cont::pointer;
		using const_pointer =  Cont::const_pointer;
		using size_type = Cont::size_type;
		using difference_type = Cont::difference_type;
		using reference = Cont::reference;
		using const_reference = Cont::const_reference;

	public:
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
constexpr matrix_const_default_iterator<T> operator+(
	std::ptrdiff_t n, const matrix_const_default_iterator<T>& it) noexcept {
	return matrix_const_default_iterator<T>(it.ptr + n);
}

static_assert(std::contiguous_iterator<matrix_const_default_iterator<detail::tCont>>);

template<typename Cont>
struct matrix_default_iterator final : public matrix_const_default_iterator<Cont> {
	private:
		using base_t = matrix_const_default_iterator<Cont>;

	public:
		using typename base_t::value_type;
		using typename base_t::pointer;
		using typename base_t::const_pointer;
		using typename base_t::size_type;
		using typename base_t::difference_type;
		using typename base_t::reference;
		using typename base_t::const_reference;

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
constexpr matrix_default_iterator<T> operator+(
	std::ptrdiff_t n, const matrix_default_iterator<T>& it) noexcept {
	return matrix_default_iterator<T>(it.ptr + n);
}

static_assert(std::contiguous_iterator<matrix_default_iterator<detail::tCont>>);

template<typename Cont>
struct matrix_const_row_iterator {
	public:
		using internal_value_type = Cont::value_type;
		using internal_pointer = Cont::pointer;
		using internal_const_pointer = Cont::const_pointer;
		using internal_size_type = Cont::size_type;
		using internal_difference_type = Cont::difference_type;
		using internal_reference = Cont::reference;
		using internal_const_reference = Cont::const_reference;

	public:
		using value_type = const_RowProxy<Cont>;
		using pointer = void*;
		using const_pointer = const pointer;
		using size_type = internal_size_type;
		using difference_type = internal_difference_type;
		using reference = value_type;
		using const_reference = const value_type;

	public:
		using iterator_category = std::random_access_iterator_tag;
		using iterator_concept = std::random_access_iterator_tag;

	protected:
		internal_pointer ptr = nullptr;
		size_type cols = 0;

	public:
		constexpr matrix_const_row_iterator() noexcept = default;
		constexpr matrix_const_row_iterator(internal_pointer ptr, size_type cols) noexcept : ptr(ptr), cols(cols) {}

		constexpr value_type operator*() const { return value_type(ptr, cols); }
		// constexpr pointer operator->() const { return nullptr; }
		constexpr value_type operator[](size_type n) const {
			return value_type((ptr + cols * n), cols);
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
constexpr matrix_const_row_iterator<T> operator+(
	std::ptrdiff_t n, const matrix_const_row_iterator<T>& o) {
	return matrix_const_row_iterator<T>((o.ptr + static_cast<std::ptrdiff_t>(o.cols) * n), o.cols);
}

static_assert(std::random_access_iterator<matrix_const_row_iterator<detail::tCont>>);

template<typename Cont>
struct matrix_row_iterator final : public matrix_const_row_iterator<Cont> {
	private:
		using base_t = matrix_const_row_iterator<Cont>;

	public:
		using typename base_t::internal_value_type;
		using typename base_t::internal_pointer;

	public:
		using value_type = RowProxy<Cont>;
		using typename base_t::pointer;
		using typename base_t::const_pointer;
		using typename base_t::difference_type;
		using typename base_t::size_type;
		using typename base_t::reference;
		using typename base_t::const_reference;

	private:
		using base_t::ptr;
		using base_t::cols;

	public:
		constexpr matrix_row_iterator() noexcept = default;
		constexpr matrix_row_iterator(internal_pointer p, size_type cols) noexcept : base_t(p, cols) {}
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
constexpr matrix_row_iterator<T> operator+(
	std::ptrdiff_t n, const matrix_row_iterator<T>& o) {
	return matrix_row_iterator<T>((o.ptr + static_cast<std::ptrdiff_t>(o.cols) * n), o.cols);
}

static_assert(std::random_access_iterator<matrix_row_iterator<detail::tCont>>);

template<typename Cont>
struct matrix_const_col_iterator {
	public:
		using internal_pointer = Cont::pointer;
		using internal_const_pointer = Cont::const_pointer;
		using internal_reference = Cont::reference;
		using internal_value_type = Cont::value_type;
		using internal_const_reference = Cont::const_reference;
		using internal_difference_type = Cont::difference_type;
		using internal_size_type = Cont::size_type;

	public:
		using value_type = const_ColProxy<Cont>;
		using pointer = void;
		using const_pointer = const pointer;
		using size_type = internal_size_type;
		using difference_type = internal_difference_type;
		using reference = value_type;
		using const_reference = const value_type;

	public:
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
		constexpr pointer operator->() const { return nullptr;  }
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

static_assert(std::random_access_iterator<matrix_const_col_iterator<detail::tCont>>);

template<typename Cont>
struct matrix_col_iterator final : public matrix_const_col_iterator<Cont> {
	private:
		using base_t = matrix_const_col_iterator<Cont>;

	public:
		using typename base_t::internal_value_type;
		using typename base_t::internal_pointer;
		
	public:
		using value_type = ColProxy<Cont>;
		using typename base_t::pointer;
		using typename base_t::const_pointer;
		using typename base_t::difference_type;
		using typename base_t::size_type;
		using typename base_t::reference;
		using typename base_t::const_reference;

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
		constexpr matrix_col_iterator& operator++() noexcept { ++ptr; return *this;	}
		constexpr matrix_col_iterator operator++(int) noexcept { auto tmp = *this; ++ptr; return tmp;	}

		constexpr matrix_col_iterator& operator--() noexcept { --ptr;	return *this;	}
		constexpr matrix_col_iterator operator--(int) noexcept { auto tmp = *this; --ptr; return tmp;	}

		constexpr matrix_col_iterator& operator+=(difference_type n) noexcept { ptr += n;	return *this; }
		constexpr matrix_col_iterator& operator-=(difference_type n) noexcept { ptr -= n; return *this; }

		constexpr matrix_col_iterator operator+(difference_type n) const noexcept {
			return matrix_col_iterator(ptr + n, rows, stride);
		}
		constexpr matrix_col_iterator operator-(difference_type n) const noexcept {
			return matrix_col_iterator(ptr - n, rows, stride);
		}
		constexpr difference_type operator-(const matrix_col_iterator& o) const noexcept {
			return ptr - o.ptr;
		}

	public:
		constexpr bool operator==(const matrix_col_iterator& o) const noexcept { return ptr == o.ptr; }
		constexpr auto operator<=>(const matrix_col_iterator& o) const noexcept { return ptr <=> o.ptr; }
};

template <typename T>
inline constexpr matrix_col_iterator<T> operator+(std::ptrdiff_t n, const matrix_col_iterator<T>& o) {
	return matrix_col_iterator(o.ptr + n, o.rows, o.stride);
}

static_assert(std::random_access_iterator<matrix_col_iterator<detail::tCont>>);

template<typename Cont>
struct matrix_const_diagonal_iterator {
	public:
		using value_type = typename Cont::value_type;
		using pointer = typename Cont::pointer;
		using const_pointer = typename Cont::const_pointer;
		using size_type = typename Cont::size_type;
		using difference_type = typename Cont::difference_type;
		using reference = typename Cont::reference;
		using const_reference = typename Cont::const_reference;

	public:
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
constexpr matrix_const_diagonal_iterator<T> operator+(std::ptrdiff_t n, const matrix_const_diagonal_iterator<T>& o) {
	return matrix_const_diagonal_iterator<T>(o.offset * n + o.ptr, o.offset);
}

static_assert(std::random_access_iterator<matrix_const_diagonal_iterator<detail::tCont>>);

template<typename Cont>
struct matrix_diagonal_iterator final : public matrix_const_diagonal_iterator<Cont> {
	private:
		using base_t = typename matrix_const_diagonal_iterator<Cont>;

	public:
		using typename base_t::value_type;
		using typename base_t::pointer;
		using typename base_t::const_pointer;
		using typename base_t::size_type;
		using typename base_t::difference_type;
		using typename base_t::reference;
		using typename base_t::const_reference;

	private:
		using base_t::ptr;
		using base_t::offset;

	public:
		constexpr matrix_diagonal_iterator() noexcept = default;
		constexpr explicit matrix_diagonal_iterator(pointer p, size_type o) noexcept : base_t(p, o) {}

	public:
		constexpr reference operator*() const noexcept { return *ptr; }
		constexpr pointer operator->() const noexcept { return ptr; }

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
constexpr matrix_diagonal_iterator<T> operator+(std::ptrdiff_t n, const matrix_diagonal_iterator<T>& o) {
	return matrix_diagonal_iterator<T>(o.offset * n + o.ptr, o.offset);
}
static_assert(std::random_access_iterator<matrix_diagonal_iterator<detail::tCont>>);

/* SUBMATRIX ITERATORS */
template<typename Cont>
struct submatrix_row_const_iterator {
	protected:
		using internal_value_type = Cont::value_type;
		using internal_pointer = Cont::pointer;
		using internal_const_pointer = Cont::const_pointer;
		using internal_size_type = Cont::size_type;
		using internal_difference_type = Cont::difference_type;
		using internal_reference = Cont::reference;
		using internal_const_reference = Cont::const_reference;

	public:
		using value_type = const_RowProxy<Cont>;
		using pointer = void*;
		using const_pointer = const void*;
		using size_type = internal_size_type;
		using difference_type = internal_difference_type;
		using reference = value_type;
		using const_reference = const value_type;

	public:
		using iterator_category = std::random_access_iterator_tag;
		using iterator_concept = std::random_access_iterator_tag;

	protected:
		internal_pointer ptr = nullptr;
		internal_size_type matrix_cols_offset = 0;
		internal_size_type submatrix_cols_offset = 0;

	public:
		constexpr submatrix_row_const_iterator() noexcept = default;
		constexpr submatrix_row_const_iterator(
			internal_pointer ptr,
			internal_size_type matrix_cols_offset,
			internal_size_type submatrix_cols_offset
		) noexcept :
			ptr(ptr)
			, matrix_cols_offset(matrix_cols_offset)
			, submatrix_cols_offset(submatrix_cols_offset) {
		}

	public:
		constexpr value_type operator*() const noexcept { return value_type(ptr, submatrix_cols_offset); }
		// constexpr pointer operator->() const { return nullptr; }
		constexpr value_type operator[](size_type n) const noexcept {
			return value_type((ptr + matrix_cols_offset * n), submatrix_cols_offset);
		}

	public:
		constexpr submatrix_row_const_iterator& operator++() noexcept { ptr += matrix_cols_offset; return *this; }
		constexpr submatrix_row_const_iterator operator++(int) noexcept { auto tmp = *this; ptr += matrix_cols_offset; return tmp; }
		constexpr submatrix_row_const_iterator& operator--() noexcept { ptr -= matrix_cols_offset; return *this; }
		constexpr submatrix_row_const_iterator operator--(int) noexcept { auto tmp = *this; ptr -= matrix_cols_offset; return tmp; }
		constexpr submatrix_row_const_iterator& operator+=(difference_type n) noexcept {
			ptr += static_cast<difference_type>(matrix_cols_offset) * n; return *this;
		}
		constexpr submatrix_row_const_iterator& operator-=(difference_type n) noexcept {
			ptr -= static_cast<difference_type>(matrix_cols_offset) * n; return *this;
		}
		constexpr submatrix_row_const_iterator operator+(difference_type n) const noexcept {
			return submatrix_row_const_iterator(ptr + n * matrix_cols_offset, matrix_cols_offset, submatrix_cols_offset);
		}
		constexpr submatrix_row_const_iterator operator-(difference_type n) const noexcept {
			return submatrix_row_const_iterator(ptr - n * matrix_cols_offset, matrix_cols_offset, submatrix_cols_offset);
		}
		constexpr difference_type operator-(const submatrix_row_const_iterator& o) const noexcept {
			return (ptr - o.ptr) / static_cast<difference_type>(matrix_cols_offset);
		}

	public:
		constexpr bool operator==(const submatrix_row_const_iterator& o) const noexcept { return ptr == o.ptr; }
		constexpr auto operator<=>(const submatrix_row_const_iterator& o) const noexcept { return ptr <=> o.ptr; }
};
template<typename T>
constexpr submatrix_row_const_iterator<T> operator+(
	typename submatrix_row_const_iterator<T>::difference_type n,
	const submatrix_row_const_iterator<T>& o) {
	return submatrix_row_const_iterator<T>(o.ptr + o.matrix_cols_offset * n, o.matrix_cols_offset, o.submatrix_cols_offset);
}
static_assert(std::random_access_iterator<submatrix_row_const_iterator<detail::tCont>>);

template<typename Cont>
struct submatrix_row_iterator final : public submatrix_row_const_iterator<Cont> {
	private:
		using base_t = submatrix_row_const_iterator<Cont>;

	private:
		using typename base_t::internal_value_type;
		using typename base_t::internal_pointer;
		using typename base_t::internal_const_pointer;
		using typename base_t::internal_size_type;
		using typename base_t::internal_difference_type;
		using typename base_t::internal_reference;
		using typename base_t::internal_const_reference;

	public:
		using value_type = RowProxy<Cont>;
		using typename base_t::pointer;
		using typename base_t::const_pointer;
		using typename base_t::size_type;
		using typename base_t::difference_type;
		using reference = value_type;
		using const_reference = const value_type;

	private:
		using base_t::ptr;
		using base_t::matrix_cols_offset;
		using base_t::submatrix_cols_offset;

	public:
		constexpr submatrix_row_iterator() noexcept = default;
		constexpr submatrix_row_iterator(
			internal_pointer ptr,
			internal_size_type matrix_col_offset,
			internal_size_type submatrix_col_offset
		) noexcept : base_t(ptr, matrix_col_offset, submatrix_col_offset) { }

	public:
		constexpr value_type operator*() const noexcept { return value_type(ptr, submatrix_cols_offset); }
		// constexpr pointer operator->() const { return nullptr; }
		constexpr value_type operator[](size_type n) const noexcept {
			return value_type((ptr + matrix_cols_offset * n), submatrix_cols_offset);
		}

	public:
		constexpr submatrix_row_iterator& operator++() noexcept { ptr += matrix_cols_offset; return *this; }
		constexpr submatrix_row_iterator operator++(int) noexcept { auto tmp = *this; ptr += matrix_cols_offset; return tmp; }
		constexpr submatrix_row_iterator& operator--() noexcept { ptr -= matrix_cols_offset; return *this; }
		constexpr submatrix_row_iterator operator--(int) noexcept { auto tmp = *this; ptr -= matrix_cols_offset; return tmp; }
		constexpr submatrix_row_iterator& operator+=(difference_type n) noexcept {
			ptr += static_cast<difference_type>(matrix_cols_offset) * n; return *this;
		}
		constexpr submatrix_row_iterator& operator-=(difference_type n) noexcept {
			ptr -= static_cast<difference_type>(matrix_cols_offset) * n; return *this;
		}
		constexpr submatrix_row_iterator operator+(difference_type n) const noexcept {
			return submatrix_row_iterator(ptr + n * matrix_cols_offset, matrix_cols_offset, submatrix_cols_offset);
		}
		constexpr submatrix_row_iterator operator-(difference_type n) const noexcept {
			return submatrix_row_iterator(ptr - n * matrix_cols_offset, matrix_cols_offset, submatrix_cols_offset);
		}
		constexpr difference_type operator-(const submatrix_row_iterator& o) const noexcept {
			return (ptr - o.ptr) / static_cast<difference_type>(matrix_cols_offset);
		}

	public:
		constexpr bool operator==(const submatrix_row_iterator& o) const noexcept { return ptr == o.ptr; }
		constexpr auto operator<=>(const submatrix_row_iterator& o) const noexcept { return ptr <=> o.ptr; }
};
template<typename T>
constexpr submatrix_row_iterator<T> operator+(
	typename submatrix_row_iterator<T>::difference_type n,
	const submatrix_row_iterator<T>& o) noexcept {
	return submatrix_row_iterator<T>(o.ptr + o.matrix_cols_offset * n, o.matrix_cols_offset, o.submatrix_cols_offset);
}
static_assert(std::random_access_iterator<submatrix_row_iterator<detail::tCont>>);

template<typename Cont>
struct const_Submatrix { // : std::ranges::view_interface<const_Submatrix<Cont>>{
	public:
		using value_type = Cont::value_type;
		using pointer = Cont::pointer;
		using const_pointer = Cont::const_pointer;
		using size_type = Cont::size_type;
		using difference_type = Cont::difference_type;
		using reference = Cont::reference;
		using const_reference = Cont::const_reference;

	public:
#if 0
		using iterator = submatrix_const_default_iterator<Cont>;
		using const_iterator = iterator;
		using reverse_iterator = std::reverse_iterator<iterator>;
		using const_reverse_iterator = reverse_iterator;
#endif

		using col_iterator = matrix_const_col_iterator<Cont>;
		using const_col_iterator = col_iterator;
		using col_reverse_iterator = std::reverse_iterator<col_iterator>;
		using const_col_reverse_iterator = col_reverse_iterator;

		using row_iterator = submatrix_row_const_iterator<Cont>;
		using const_row_iterator = row_iterator;
		using row_reverse_iterator = std::reverse_iterator<row_iterator>;
		using const_row_reverse_iterator = row_reverse_iterator;

		using diagonal_iterator = matrix_const_diagonal_iterator<Cont>;
		using const_diagonal_iterator = diagonal_iterator;
		using diagonal_reverse_iterator = std::reverse_iterator<diagonal_iterator>;
		using const_diagonal_reverse_iterator = diagonal_reverse_iterator;

	public:
		using const_col_proxy = const_ColProxy<Cont>;
		using const_row_proxy = const_RowProxy<Cont>;

	protected:
		Cont* cont = nullptr;
		pointer first = nullptr;
		pointer last = nullptr;

	public:
		constexpr const_Submatrix() noexcept = default;
		constexpr const_Submatrix(Cont* p, pointer f, pointer l) noexcept : cont(p), first(f), last(l) {}

	public:
		constexpr size_type rows() const noexcept {
			auto&& mtx = *cont;
			auto&& f = *first;
			auto&& l = *last;
			auto&& from_y = mtx.get_elem_row_index(f);
			auto&& to_y = mtx.get_elem_row_index(l);
			return to_y - from_y + 1;
		}
		constexpr size_type cols() const noexcept {
			auto&& mtx = *cont;
			auto&& f = *first;
			auto&& l = *last;
			auto&& from_x = mtx.get_elem_col_index(f);
			auto&& to_x = mtx.get_elem_col_index(l);
			return to_x - from_x + 1; 
		}
		constexpr size_type size() const noexcept { return rows() * cols(); }
		constexpr pointer data()  const noexcept { return first; }
		constexpr bool is_square() const noexcept { return rows() == cols(); }
		constexpr bool empty() const noexcept { return ((cols() == 0) || (rows() == 0)); }

	public:
		constexpr std::pair<size_type, size_type> get_indices(const_reference elem) const noexcept {
			auto&& mtx = *cont;
			auto&& elem_ptr = std::addressof(elem);
#if 0
			if (elem_ptr < first || elem_ptr > last) {
				throw std::out_of_range("Element not in submatrix");
			}
#endif
			auto&& [elem_row, elem_col] = mtx.get_elem_indices(elem);
			// auto = cont->get_elem_col_index(elem);
			auto&& fst = *first;
			auto&& [first_row, first_col] = mtx.get_elem_indices(fst);
			// auto  = cont->get_elem_col_index(*first);
			return { elem_row - first_row, elem_col - first_col };
		}

		constexpr const_col_proxy col(size_type c) const noexcept {
			return const_col_proxy(first + c, rows(), (*cont).cols());
		}

		constexpr const_row_proxy row(size_type r) const noexcept {
			return const_row_proxy(first + r * (*cont).cols(), cols());
		}

		constexpr const_row_proxy operator[](size_type r) const noexcept {
			return row(r);
		}

		[[nodiscard]] constexpr const_row_proxy get_elem_row(const_reference elem) const noexcept {
			auto [сrow, _] = get_indices(elem);
			return row(сrow);
		}

		[[nodiscard]] constexpr const_col_proxy get_elem_col(const_reference elem) const noexcept {
			auto [_, сcol] = get_indices(elem);
			return col(сcol);
		}

	public:
#if 0
		constexpr iterator begin() noexcept { return iterator(); }
		constexpr iterator end() noexcept { return iterator(); }
		constexpr const_iterator begin() const noexcept { return const_iterator(); }
		constexpr const_iterator end() const noexcept { return const_iterator(); }
		constexpr const_iterator cbegin() const noexcept { return const_iterator(begin()); }
		constexpr const_iterator cend() const noexcept { return const_iterator(end()); }
		constexpr reverse_iterator rbegin() const noexcept { return reverse_iterator(end()); }
		constexpr reverse_iterator rend() const noexcept { return reverse_iterator(begin()); }
		constexpr const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }
		constexpr const_reverse_iterator crend() const noexcept { return const_reverse_iterator(cbegin()); }
#endif

		/* submatrix col iterator */
		constexpr col_iterator col_begin() noexcept { return col_iterator(first, rows(), (*cont).cols()); }
		constexpr col_iterator col_end() noexcept { return col_iterator(first + cols(), rows(), (*cont).cols()); }
		constexpr const_col_iterator col_begin() const noexcept { return const_col_iterator(first, rows(), (*cont).cols()); }
		constexpr const_col_iterator col_end() const noexcept { return const_col_iterator(first + cols(), rows(), (*cont).cols()); }
		constexpr const_col_iterator col_cbegin() const noexcept { return const_col_iterator(col_begin()); }
		constexpr const_col_iterator col_cend() const noexcept { return const_col_iterator(col_end()); }
		constexpr col_reverse_iterator col_rbegin() const noexcept { return col_reverse_iterator(col_end()); }
		constexpr col_reverse_iterator col_rend() const noexcept { return col_reverse_iterator(col_begin()); }
		constexpr const_col_reverse_iterator col_crbegin() const noexcept { return const_col_reverse_iterator(col_cend()); }
		constexpr const_col_reverse_iterator col_crend() const noexcept { return const_col_reverse_iterator(col_cbegin()); }

		/* submatrix row iterator */
		constexpr row_iterator row_begin() noexcept { return row_iterator(first, ((*cont).cols()), cols()); }
		constexpr row_iterator row_end() noexcept { return row_iterator(first + rows() * ((*cont).cols()), ((*cont).cols()), cols()); }
		constexpr const_row_iterator row_begin() const noexcept { return const_row_iterator(first, ((*cont).cols()), cols()); }
		constexpr const_row_iterator row_end() const noexcept { return const_row_iterator(first + rows() * ((*cont).cols()), ((*cont).cols()), cols()); }
		constexpr const_row_iterator row_cbegin() const noexcept { return const_row_iterator(row_begin()); }
		constexpr const_row_iterator row_cend() const noexcept { return const_row_iterator(row_end()); }
		constexpr row_reverse_iterator row_rbegin() noexcept { return row_reverse_iterator(row_end()); }
		constexpr row_reverse_iterator row_rend() noexcept { return row_reverse_iterator(row_begin()); }
		constexpr const_row_reverse_iterator row_crbegin() const noexcept { return const_row_reverse_iterator(row_cend()); }
		constexpr const_row_reverse_iterator row_crend() const noexcept { return const_row_reverse_iterator(row_cbegin()); }

		/* submatrix diagonal iterator */
		constexpr diagonal_iterator diagonal_begin() noexcept { return diagonal_iterator(first, ((*cont).cols() + 1)); }
		constexpr diagonal_iterator diagonal_end() noexcept { return diagonal_iterator(first + rows() * ((*cont).cols() + 1), ((*cont).cols() + 1)); }
		constexpr const_diagonal_iterator diagonal_begin() const noexcept { return const_diagonal_iterator(first, ((*cont).cols() + 1)); }
		constexpr const_diagonal_iterator diagonal_end() const noexcept { return const_diagonal_iterator(first + rows() * ((*cont).cols() + 1), ((*cont).cols() + 1)); }
		constexpr const_diagonal_iterator diagonal_cbegin() const noexcept { return const_diagonal_iterator(diagonal_begin()); }
		constexpr const_diagonal_iterator diagonal_cend() const noexcept { return const_diagonal_iterator(diagonal_end()); }
		constexpr diagonal_reverse_iterator diagonal_rbegin() const noexcept { return diagonal_reverse_iterator(diagonal_end()); }
		constexpr diagonal_reverse_iterator diagonal_rend() const noexcept { return diagonal_reverse_iterator(diagonal_begin()); }
		constexpr const_diagonal_reverse_iterator diagonal_crbegin() const noexcept { return const_diagonal_reverse_iterator(diagonal_cend()); }
		constexpr const_diagonal_reverse_iterator diagonal_crend() const noexcept { return const_diagonal_reverse_iterator(diagonal_cbegin()); }

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

};
// static_assert(std::ranges::view<const_Submatrix<detail::tCont>>);

template<typename Cont>
struct Submatrix : public const_Submatrix<Cont> {
	private:
		using base_t = typename const_Submatrix<Cont>;

	public:
		using typename base_t::value_type;
		using typename base_t::pointer;
		using typename base_t::const_pointer;
		using typename base_t::size_type;
		using typename base_t::difference_type;
		using typename base_t::reference;
		using typename base_t::const_reference;

	public:
#if 0
		using iterator = submatrix_default_iterator<Cont>;
		using const_iterator = submatrix_const_default_iterator<Cont>;
		using reverse_iterator = std::reverse_iterator<iterator>;
		using const_reverse_iterator = std::reverse_iterator<const_iterator>;
#endif

		using col_iterator = matrix_col_iterator<Cont>;
		using const_col_iterator = matrix_const_col_iterator<Cont>;
		using col_reverse_iterator = std::reverse_iterator<col_iterator>;
		using const_col_reverse_iterator = std::reverse_iterator<const_col_iterator>;

		using row_iterator = submatrix_row_iterator<Cont>;
		using const_row_iterator = submatrix_row_const_iterator<Cont>;
		using row_reverse_iterator = std::reverse_iterator<row_iterator>;
		using const_row_reverse_iterator = std::reverse_iterator<const_row_iterator>;

		using diagonal_iterator = matrix_diagonal_iterator<Cont>;
		using const_diagonal_iterator = matrix_const_diagonal_iterator<Cont>;
		using diagonal_reverse_iterator = std::reverse_iterator<diagonal_iterator>;
		using const_diagonal_reverse_iterator = std::reverse_iterator<const_diagonal_iterator>;

	public:
		using col_proxy = ColProxy<Cont>;
		using row_proxy = RowProxy<Cont>;
		using typename base_t::const_col_proxy;
		using typename base_t::const_row_proxy;

	private:
		using base_t::cont;
		using base_t::first;
		using base_t::last;

	public:
		/* todo ctors */
		constexpr Submatrix() noexcept = delete;
		constexpr Submatrix(Cont* p, pointer f, pointer l) noexcept : base_t(p, f, l) {}

	public:
		using base_t::rows;
		using base_t::cols;
		using base_t::size;
		using base_t::empty;
		using base_t::is_square;

	public:
		constexpr std::pair<size_type, size_type> get_indices(const_reference elem) const noexcept {
			return base_t::get_indices(elem);
		}

		[[nodiscard]] constexpr row_proxy get_elem_row(const_reference elem) noexcept {
			auto&& [_row, _] = get_indices(elem);
			return row(_row);
		}

		[[nodiscard]] constexpr col_proxy get_elem_col(const_reference elem) noexcept {
			auto&& [_, _col] = get_indices(elem);
			return col(_col);
		}

		[[nodiscard]] constexpr const_row_proxy get_elem_row(const_reference elem) const noexcept {
			return base_t::get_elem_row(elem);
		}

		[[nodiscard]] constexpr const_col_proxy get_elem_col(const_reference elem) const noexcept {
			return base_t::get_elem_col(elem);
		}

	public:
		constexpr col_proxy col(size_type c) noexcept {
			return col_proxy(first + c, rows(), (*cont).cols());
		}

		constexpr const_col_proxy col(size_type c) const noexcept {
			return base_t::col(c);
		}

		constexpr row_proxy row(size_type r) noexcept {
			return row_proxy(first + r * (*cont).cols(), cols());
		}

		constexpr const_row_proxy row(size_type r) const noexcept {
			return base_t::row(r);
		}

		constexpr row_proxy operator[](size_type r) noexcept {
			return row(r);
		}

		constexpr const_row_proxy operator[](size_type r) const noexcept {
			return base_t::row(r);
		}

	public:
		constexpr void to_submatrix(size_type fr, size_type fc, size_type lr, size_type lc) noexcept {
			first = std::addressof(row(fr)[fc]);
			last = std::addressof(row(lr)[lc]);
		}

		constexpr void to_submatrix(const_reference f, const_reference l) noexcept {
			auto&& [fr, fc] = get_indices(f);
			auto&& [lr, lc] = get_indices(l);
			to_submatrix(fr, fc, lr, lc);
		}

	public:
#if 0
		/* submatrix default iterator */
		constexpr iterator begin() noexcept { return iterator(); }
		constexpr iterator end() noexcept { return iterator(); }
		constexpr const_iterator begin() const noexcept { return const_iterator(); }
		constexpr const_iterator end() const noexcept { return const_iterator(); }
		constexpr const_iterator cbegin() const noexcept { return const_iterator(begin()); }
		constexpr const_iterator cend() const noexcept { return const_iterator(end()); }
		constexpr reverse_iterator rbegin() const noexcept { return reverse_iterator(end()); }
		constexpr reverse_iterator rend() const noexcept { return reverse_iterator(begin()); }
		constexpr const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }
		constexpr const_reverse_iterator crend() const noexcept { return const_reverse_iterator(cbegin()); }
#endif

		/* submatrix col iterator */
		constexpr col_iterator col_begin() noexcept { return col_iterator(first, rows(), cols()); }
		constexpr col_iterator col_end() noexcept { return col_iterator(first + cols(), rows(), cols()); }
		constexpr const_col_iterator col_begin() const noexcept { return const_col_iterator(first, rows(), cols()); }
		constexpr const_col_iterator col_end() const noexcept { return const_col_iterator(first + cols(), rows(), cols()); }
		constexpr const_col_iterator col_cbegin() const noexcept { return const_col_iterator(col_begin()); }
		constexpr const_col_iterator col_cend() const noexcept { return const_col_iterator(col_end()); }
		constexpr col_reverse_iterator col_rbegin() const noexcept { return col_reverse_iterator(col_end()); }
		constexpr col_reverse_iterator col_rend() const noexcept { return col_reverse_iterator(col_begin()); }
		constexpr const_col_reverse_iterator col_crbegin() const noexcept { return const_col_reverse_iterator(col_cend()); }
		constexpr const_col_reverse_iterator col_crend() const noexcept { return const_col_reverse_iterator(col_cbegin()); }

		/* submatrix row iterator */
		constexpr row_iterator row_begin() noexcept { return row_iterator(first, ((*cont).cols()), cols()); }
		constexpr row_iterator row_end() noexcept { return row_iterator(first + rows() * ((*cont).cols()), ((*cont).cols()), cols()); }
		constexpr const_row_iterator row_begin() const noexcept { return const_row_iterator(first, ((*cont).cols()), cols()); }
		constexpr const_row_iterator row_end() const noexcept { return const_row_iterator(first + rows() * ((*cont).cols()), ((*cont).cols()), cols()); }
		constexpr const_row_iterator row_cbegin() const noexcept { return const_row_iterator(row_begin()); }
		constexpr const_row_iterator row_cend() const noexcept { return const_row_iterator(row_end()); }
		constexpr row_reverse_iterator row_rbegin() noexcept { return row_reverse_iterator(row_end()); }
		constexpr row_reverse_iterator row_rend() noexcept { return row_reverse_iterator(row_begin()); }
		constexpr const_row_reverse_iterator row_crbegin() const noexcept { return const_row_reverse_iterator(row_cend()); }
		constexpr const_row_reverse_iterator row_crend() const noexcept { return const_row_reverse_iterator(row_cbegin()); }

		/* submatrix diagonal iterator */
		constexpr diagonal_iterator diagonal_begin() noexcept { return diagonal_iterator(first, ((*cont).cols() + 1)); }
		constexpr diagonal_iterator diagonal_end() noexcept { return diagonal_iterator(first + rows() * ((*cont).cols() + 1), ((*cont).cols() + 1)); }
		constexpr const_diagonal_iterator diagonal_begin() const noexcept { return const_diagonal_iterator(first, ((*cont).cols() + 1)); }
		constexpr const_diagonal_iterator diagonal_end() const noexcept { return const_diagonal_iterator(first + rows() * ((*cont).cols() + 1), ((*cont).cols() + 1)); }
		constexpr const_diagonal_iterator diagonal_cbegin() const noexcept { return const_diagonal_iterator(diagonal_begin()); }
		constexpr const_diagonal_iterator diagonal_cend() const noexcept { return const_diagonal_iterator(diagonal_end()); }
		constexpr diagonal_reverse_iterator diagonal_rbegin() const noexcept { return diagonal_reverse_iterator(diagonal_end()); }
		constexpr diagonal_reverse_iterator diagonal_rend() const noexcept { return diagonal_reverse_iterator(diagonal_begin()); }
		constexpr const_diagonal_reverse_iterator diagonal_crbegin() const noexcept { return const_diagonal_reverse_iterator(diagonal_cend()); }
		constexpr const_diagonal_reverse_iterator diagonal_crend() const noexcept { return const_diagonal_reverse_iterator(diagonal_cbegin()); }

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

};
// static_assert(std::ranges::view<Submatrix<detail::tCont>>);

template<typename T> constexpr auto maybe_null(T ptr) { return ptr ? std::addressof(*ptr) : nullptr; }

template<typename allocator_type, typename value_type> using rebound_allocator = typename std::allocator_traits<allocator_type>::template rebind_alloc<value_type>;

template<detail::object T, typename Alloc = std::allocator<T>>
struct Matrix {
	public:
		using allocator_type = Alloc;
		using allocator_traits = std::allocator_traits<rebound_allocator<Alloc, T>>;

	public:
		using value_type = typename allocator_traits::value_type;
		using pointer = typename allocator_traits::pointer;
		using const_pointer = typename allocator_traits::const_pointer;
		using size_type = typename allocator_traits::size_type;
		using difference_type = typename allocator_traits::difference_type;
		using reference = value_type&;
		using const_reference = const value_type&;

	private:
		using base = typename Matrix<value_type, allocator_type>;

	public:
		using submatrix = Submatrix<base>;
		using const_submatrix = const_Submatrix<base>;

	public:
		using col_proxy = ColProxy<base>;
		using const_col_proxy = const_ColProxy<base>;
		using row_proxy = RowProxy<base>;
		using const_row_proxy = const_RowProxy<base>;

	public:
		using iterator = matrix_default_iterator<base>;
		using const_iterator = matrix_const_default_iterator<base>;
		using reverse_iterator = std::reverse_iterator<iterator>;
		using const_reverse_iterator = std::reverse_iterator<const_iterator>;

		using row_iterator = matrix_row_iterator<base>;
		using row_const_iterator = matrix_const_row_iterator<base>;
		using row_const_reverse_iterator = std::reverse_iterator<row_const_iterator>;
		using row_reverse_iterator = std::reverse_iterator<row_iterator>;

		using col_iterator = matrix_col_iterator<base>;
		using col_const_iterator = matrix_const_col_iterator<base>;
		using col_const_reverse_iterator = std::reverse_iterator<col_const_iterator>;
		using col_reverse_iterator = std::reverse_iterator<col_iterator>;

		using diagonal_iterator = matrix_diagonal_iterator<base>;
		using diagonal_const_iterator = matrix_const_diagonal_iterator<base>;
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
			_data = allocator_traits::allocate(_allocator, n);
			for (size_type i = 0; i < n; ++i) {
				allocator_traits::construct(_allocator, std::addressof(_data[i]));
			}
		}

		// dtor
		~Matrix() {
			if (_data) {
				const size_type n = _rows * _cols;
				for (size_type i = 0; i < n; ++i) {
					allocator_traits::destroy(_allocator, std::addressof(_data[i]));
				}
				allocator_traits::deallocate(_allocator, _data, n);
				_data = nullptr;
			}
		}

		// copy
		Matrix(const Matrix& other)
			: _rows(other._rows), _cols(other._cols) {
			if (other._data) {
				const size_type n = _rows * _cols;
				_data = allocator_traits::allocate(_allocator, n);
				for (size_type i = 0; i < n; ++i) {
					allocator_traits::construct(_allocator, std::addressof(_data[i]), other._data[i]);
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

	private:
		constexpr pointer get_elem(size_type r, size_type c) const noexcept {
			return _data + (r * _cols + c);
		}

	public:
		constexpr size_type size() const noexcept { return _rows * _cols; }
		constexpr size_type rows() const noexcept { return _rows; }
		constexpr size_type cols() const noexcept { return _cols; }
		constexpr bool empty() const noexcept { return ((_rows == 0) || (_cols == 0)); }
		constexpr bool is_square() const noexcept { return _rows == _cols; }


		constexpr size_type get_elem_row_index(const value_type& elem) const noexcept {
			auto x = std::addressof(elem);
			// if (((x < _data) || (x >= _data + _rows * _cols))) { throw std::runtime_error("elem Out of bound"); };
			auto z = static_cast<size_type>(x - _data);
			return (z / _cols);
		}
		constexpr const_row_proxy get_elem_row(const value_type& elem) const noexcept {
			return row(get_elem_row_index(elem));
		}

		constexpr row_proxy get_elem_row(const value_type& elem) noexcept {
			return row(get_elem_row_index(elem));
		}

		constexpr size_type get_elem_col_index(const value_type& elem) const noexcept {
			auto x = std::addressof(elem);
			// if(((x < _data) || (x >= _data + _rows * _cols))) { throw std::runtime_error("elem Out of bound"); };
			auto z = static_cast<size_type>(x - _data);
			return (z % _cols);
		}

		constexpr const_col_proxy get_elem_col(const value_type& elem) const noexcept {
			return col(get_elem_col_index(elem));
		}

		constexpr col_proxy get_elem_col(const value_type& elem) noexcept {
			return col(get_elem_col_index(elem));
		}

		constexpr std::pair<size_type, size_type> get_elem_indices(const value_type& elem) const  noexcept {
			return { get_elem_row_index(elem), get_elem_col_index(elem) };
		}

		constexpr pointer data() noexcept { return maybe_null(_data); }
		constexpr const_pointer data() const noexcept { return maybe_null(_data); }

	public:
		constexpr submatrix get_submatrix(size_type fr, size_type fc, size_type lr, size_type lc) noexcept {
			return submatrix(this, get_elem(fr, fc), get_elem(lr, lc));
		}

		constexpr const_submatrix get_submatrix(size_type fr, size_type fc, size_type lr, size_type lc) const noexcept {			
			return const_submatrix(this, get_elem(fr, fc), get_elem(lr, lc));
		}

		constexpr submatrix get_submatrix(const_reference first, const_reference last) noexcept {
			auto&& [fr, fc] = get_elem_indices(first);
			auto&& [lr, lc] = get_elem_indices(first);
			return get_submatrix(fr, fc, lr, lc);
		}

		constexpr const_submatrix get_submatrix(const_reference first, const_reference last) const noexcept {
			auto&& [fr, fc] = get_elem_indices(first);
			auto&& [lr, lc] = get_elem_indices(first);
			return get_submatrix(fr, fc, lr, lc);
		}

		constexpr const_submatrix get_submatrix(const_reference first, size_type lr, size_type lc) const noexcept {
			auto&& [fr, fc] = get_elem_indices(first);
			return get_submatrix(fr, fc, lr, lc);
		}

		constexpr submatrix get_submatrix(const_reference first, size_type lr, size_type lc) noexcept {
			auto&& [fr, fc] = get_elem_indices(first);
			return get_submatrix(fr, fc, lr, lc);
		}

		constexpr const_submatrix get_submatrix(size_type fr, size_type fc, const_reference last) const noexcept {
			auto&& [lr, lc] = get_elem_indices(last);
			return get_submatrix(fr, fc, lr, lc);
		}

		constexpr submatrix get_submatrix(size_type fr, size_type fc, const_reference last) noexcept {
			auto&& [lr, lc] = get_elem_indices(last);
			return get_submatrix(fr, fc, lr, lc);
		}

	public:
		/* default */
		[[nodiscard]] constexpr iterator begin() noexcept {	return iterator(_data);	}
		[[nodiscard]] constexpr iterator end() noexcept { return iterator(_data + size()); }
		[[nodiscard]] constexpr const_iterator begin() const noexcept { return const_iterator(_data); }
		[[nodiscard]] constexpr const_iterator end() const noexcept { return const_iterator(_data + size()); }
		[[nodiscard]] constexpr const_iterator cbegin() const noexcept { return begin(); }
		[[nodiscard]] constexpr const_iterator cend() const noexcept { return end(); }
		[[nodiscard]] constexpr reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
		[[nodiscard]] constexpr reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
		[[nodiscard]] constexpr const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
		[[nodiscard]] constexpr const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
		[[nodiscard]] constexpr const_reverse_iterator crbegin() const noexcept { return rbegin(); }
		[[nodiscard]] constexpr const_reverse_iterator crend() const noexcept { return rend(); }

		/* row */
		[[nodiscard]] constexpr row_iterator row_begin() noexcept { return row_iterator(_data, _cols); }
		[[nodiscard]] constexpr row_iterator row_end() noexcept { return row_iterator(_data + size(), _cols); }
		[[nodiscard]] constexpr row_const_iterator row_cbegin() const noexcept { return row_begin(); }
		[[nodiscard]] constexpr row_const_iterator row_cend() const noexcept { return row_end(); }
		[[nodiscard]] constexpr row_const_iterator row_begin() const noexcept { return row_const_iterator(_data, _cols); }
		[[nodiscard]] constexpr row_const_iterator row_end() const noexcept { return row_const_iterator(_data + size(), _cols); }
		[[nodiscard]] constexpr row_reverse_iterator row_rbegin() noexcept { return row_reverse_iterator(row_end()); }
		[[nodiscard]] constexpr row_reverse_iterator row_rend() noexcept { return row_reverse_iterator(row_begin()); }
		[[nodiscard]] constexpr row_const_reverse_iterator row_rbegin() const noexcept { return row_const_reverse_iterator(row_end()); }
		[[nodiscard]] constexpr row_const_reverse_iterator row_rend() const noexcept { return row_const_reverse_iterator(row_begin()); }
		[[nodiscard]] constexpr row_const_reverse_iterator row_crbegin() const noexcept { return row_const_reverse_iterator(row_cend()); }
		[[nodiscard]] constexpr row_const_reverse_iterator row_crend() const noexcept { return row_const_reverse_iterator(row_cbegin()); }

		/* col */
		[[nodiscard]] constexpr col_iterator col_begin() noexcept { return col_iterator(_data, _rows, _cols);	}
		[[nodiscard]] constexpr col_const_iterator col_begin() const noexcept { return col_const_iterator(_data, _rows, _cols); }
		[[nodiscard]] constexpr col_iterator col_end() noexcept { return col_iterator(_data + _cols, _rows, _cols); }
		[[nodiscard]] constexpr col_const_iterator col_end() const noexcept { return col_const_iterator(_data + _cols, _rows, _cols); }
		[[nodiscard]] constexpr col_reverse_iterator col_rbegin() noexcept { return col_reverse_iterator(col_end()); }
		[[nodiscard]] constexpr col_const_reverse_iterator col_rbegin() const noexcept { return col_const_reverse_iterator(col_end()); }
		[[nodiscard]] constexpr col_reverse_iterator col_rend() noexcept { return col_reverse_iterator(col_begin()); }
		[[nodiscard]] constexpr col_const_reverse_iterator col_rend() const noexcept { return col_const_reverse_iterator(col_begin()); }
		[[nodiscard]] constexpr col_const_iterator col_cbegin() const noexcept { return col_begin(); }
		[[nodiscard]] constexpr col_const_iterator col_cend() const noexcept { return col_end(); }
		[[nodiscard]] constexpr col_const_reverse_iterator col_crbegin() const noexcept { return col_rbegin(); }
		[[nodiscard]] constexpr col_const_reverse_iterator col_crend() const noexcept { return col_rend(); }

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
			return diagonal_const_iterator(_data + (_rows * _cols + _rows), _rows + 1u);
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

		[[nodiscard]] constexpr diagonal_const_reverse_iterator diagonal_rend() const {
			if (!is_square()) { throw std::runtime_error("diagonal iterator usable only in square matrices"); }
			return diagonal_const_reverse_iterator(diagonal_begin());
		}

		[[nodiscard]] constexpr diagonal_const_iterator diagonal_cbegin() const {
			if (!is_square()) { throw std::runtime_error("diagonal iterator usable only in square matrices"); }
			return diagonal_const_iterator(diagonal_begin());
		}

		[[nodiscard]] constexpr diagonal_const_iterator diagonal_cend() const {
			if (!is_square()) { throw std::runtime_error("diagonal iterator usable only in square matrices"); }
			return diagonal_const_iterator(diagonal_end());
		}

		[[nodiscard]] constexpr diagonal_const_reverse_iterator diagonal_crbegin() const {
			if (!is_square()) { throw std::runtime_error("diagonal iterator usable only in square matrices"); }
			return diagonal_const_reverse_iterator(diagonal_rbegin());
		}

		[[nodiscard]] constexpr diagonal_const_reverse_iterator diagonal_crend() const {
			if (!is_square()) { throw std::runtime_error("diagonal iterator usable only in square matrices"); }
			return diagonal_const_reverse_iterator(diagonal_rend());
		}


	public:
		[[nodiscard]] constexpr col_proxy col(size_type j) noexcept {
			// if (j >= _cols) throw std::out_of_range("Column index out of range");
			return col_proxy{_data + j, _rows, _cols};
		}

		[[nodiscard]] constexpr const_col_proxy col(size_type j) const noexcept {
			// if (j >= _cols) throw std::out_of_range("Column index out of range");
			return const_col_proxy{_data + j, _rows, _cols};
		}

		[[nodiscard]] constexpr row_proxy row(size_type j) noexcept {
			return row_proxy(_data + j * _cols, _cols);
		}

		[[nodiscard]] constexpr const_row_proxy row(size_type j) const noexcept {
			return const_row_proxy(_data + j * _cols, _cols);
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
		constexpr row_proxy operator[](size_type r) noexcept {
			// if (r >= _rows) throw std::out_of_range("Row index out of range");
			return row_proxy{ _data + r * _cols, _cols };
		}

		constexpr const const_row_proxy operator[](size_type r) const noexcept {
			// if (r >= _rows) throw std::out_of_range("Row index out of range");
			return const_row_proxy{ _data + r * _cols, _cols };
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

#if 0	
		template<typename AllocR = allocator_type>
		auto get_row_as_vector(size_type x) const {
			auto&& it = *this;
			auto r = it[x];
			return std::vector<value_type, AllocR>(r.begin(), r.end());
		}

		template<typename AllocR = allocator_type>
		auto get_row_as_matrix_row(size_type x) const {
			auto&& it = *this;
			auto r = it[x];
			auto result = Matrix<value_type, AllocR>(1, it.cols());
			for (size_type i = 0; i < it.cols(); ++i) {
				result[0][i] = r[i];
			}
			return result;
		}

		template<typename AllocR = allocator_type>
		auto get_row_as_matrix_col(size_type x) const {
			auto&& it = *this;
			auto r = it[x];
			auto result = Matrix<value_type, AllocR>(it.cols(), 1);
			for (size_type i = 0; i < it.cols(); ++i) {
				result[i][0] = r[i];
			}
			return result;
		}

		template<typename AllocR = allocator_type>
		auto get_col_as_vector(size_type x) const {
			auto&& it = *this;
			auto c = it.col(x);
			return std::vector<value_type, AllocR>(c.begin(), c.end());
		}

		template<typename AllocR = allocator_type>
		auto get_col_as_matrix_row(size_type x) const {
			auto&& it = *this;
			auto c = it.col(x);
			auto result = Matrix<value_type, AllocR>(1, it.rows());
			for (size_type i = 0; i < it.rows(); ++i) {
				result[0][i] = c[i];
			}
			return result;
		}

		template<typename AllocR = allocator_type>
		auto get_col_as_matrix_col(size_type x) const {
			auto&& it = *this;
			auto c = it.col(x);
			auto result = Matrix<value_type, AllocR>(it.rows(), 1);
			for (size_type i = 0; i < it.rows(); ++i) {
				result[i][0] = c[i];
			}
			return result;
		}

		template<typename AllocR = allocator_type>
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

		template<typename AllocR = allocator_type>
		auto get_submatrix(const value_type& from_elem, const value_type& to_elem) const {
			auto&& [from_x, from_y] = get_indices(from_elem);
			auto&& [to_x, to_y] = get_indices(to_elem);
			return get_submatrix<AllocR>(from_x, from_y, to_x, to_y);
		}
#endif

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
		//	static_assert(to_x >= it.rows() || to_y >= it.cols());
		//	static_assert(to_x + A.rows() > it.rows() || to_y + A.cols() > it.cols());
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
				auto&& rci = result.col(i);
				auto&& iri = it.row(i);
				for (size_type j = 0; j < _cols; ++j) {
					rci[j] = iri[j];
				}
			}
			return result;
		}

	public:

		template<typename U>
		constexpr auto operator*(const Matrix<U>& other) const noexcept
			-> Matrix<std::common_type_t<value_type, U>>  {
			return mult(*this, other);
		}

		template<typename U, typename Alloc = std::allocator<std::common_type_t<value_type, U>>>
		constexpr Matrix& operator*=(const Matrix<U>& other) {
			auto&& it = *this;
			if (cols() != other.rows()) {
				throw std::runtime_error("Matrix dimensions do not match for multiplication");
			}
			using ResultType = std::common_type_t<value_type, U>;
			it = mult<value_type, U, ResultType, Alloc>(*this, other);
			return it;
		}

		// Scalar multiplication (matrix * scalar)
		template<typename U>
		auto operator*(U scalar) const noexcept
			-> Matrix<std::common_type_t<T, U>> {
			auto&& it = *this;
			using ResultType = std::common_type_t<T, U>;
			auto result = it;

			std::ranges::for_each(result, [&scalar](auto&& it) {
				it *= scalar;
			});

			return result;
		}

		template<typename U>
		constexpr Matrix& operator*=(const U& scalar) {
			auto&& it = *this;
			std::ranges::for_each(it, [&scalar](auto&& it) {
				it *= scalar;
			});
			return *this;
		}

		template<typename U>
		auto operator/(U scalar) const -> Matrix<std::common_type_t<T, U>> {
			using ResultType = std::common_type_t<T, U>;
			if (scalar == U{0}) { throw std::runtime_error("Division by zero"); }
			auto result = Matrix<ResultType>(_rows, _cols);
			auto rb = result.begin();
			auto re = result.end();
			auto ib = begin();

			for (; rb != re; ++rb, ++ib) {
				*rb = *ib / scalar;
			}
		
			return result;
		}
	
		template<typename U>
		Matrix& operator/=(U scalar) {
			if (scalar == U{0}) { throw std::runtime_error("Division by zero"); }
			auto&& it_ = *this;
			std::ranges::for_each(it_, [&scalar](auto&& it) {
				it /= scalar;
			});
			return *this;
		}

		template<typename U>
		auto operator+(const Matrix<U>& other) const -> Matrix<std::common_type_t<T, U>> {
			if (_rows != other.rows() || _cols != other.cols()) {
				throw std::runtime_error("Matrix dimensions do not match for addition");
			}
			auto&& it = *this;
			
			using ResultType = std::common_type_t<T, U>;
			auto result = Matrix<ResultType>(_rows, _cols);

			auto rb = result.begin();
			auto re = result.end();
			auto ib = it.cbegin();
			auto ob = other.cbegin();

			for (; rb != re; ++rb, ++ib, ++ob) {
				*rb = (*ib) + (*ob);
			}
			
			return result;
		}

		template<typename U>
		auto operator-(const Matrix<U>& other) const -> Matrix<std::common_type_t<T, U>> {
			if (_rows != other.rows() || _cols != other.cols()) {
				throw std::runtime_error("Matrix dimensions do not match for subtraction");
			}

			auto&& it = *this;

			using ResultType = std::common_type_t<T, U>;
			auto result = Matrix<ResultType>(_rows, _cols);

			auto rb = result.begin();
			auto re = result.end();
			auto ib = it.cbegin();
			auto ob = other.cbegin();

			for (; rb != re; ++rb, ++ib, ++ob) {
				*rb = (*ib) - (*ob);
			}

			return result;
		}

		Matrix operator+() const { return *this; }

		Matrix operator-() const {
			Matrix result(_rows, _cols);
			auto rb = result.begin();
			auto re = result.end();
			auto ib = cbegin();
		
			for (; rb != re; ++rb, ++ib) {
				*rb = -(*ib);
			}
		
			return result;
		}

		Matrix& operator+=(const Matrix& other) {
			if (_rows != other._rows || _cols != other._cols) {
				throw std::runtime_error("Matrix dimensions do not match for addition");
			}

			auto ib = begin();
			auto ie = end();
			auto ob = other.cbegin();
			for (; ib != ie; ++ib, ++ob) {
				*ib += *ob;
			}

			return *this;
		}

		Matrix& operator-=(const Matrix& other) {
			if (_rows != other._rows || _cols != other._cols) {
				throw std::runtime_error("Matrix dimensions do not match for subtraction");
			}

			auto ib = begin();
			auto ie = end();
			auto ob = other.cbegin();
			for (; ib != ie; ++ib, ++ob) {
				*ib -= *ob;
			}

			return *this;
		}

	private:
		size_type _rows = 0;
		size_type _cols = 0;
		allocator_type _allocator = allocator_type();
		pointer _data = nullptr;
};

template<typename T, typename U, typename allocator_type>
auto operator*(U scalar, const Matrix<T, allocator_type>& matrix) -> Matrix<std::common_type_t<T, U>> {
	return matrix * scalar;
}


#if 0
template<typename T, typename allocator_type>
Matrix<T> operator/(const Matrix<T, allocator_type>& B, const Matrix<T, allocator_type>& A) {
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

#if 0
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

template<typename T, typename allocator_type = std::allocator<T>>
struct OneColMatrix : public Matrix<T, allocator_type> {
	using base_t = typename Matrix<T, allocator_type>;
	OneColMatrix(std::ranges::range auto&& range) : base_t(to_Matrix_col(range)) {};
};

template<typename T, typename allocator_type = std::allocator<T>>
struct OneRowMatrix : public Matrix<T, allocator_type> {
	using base_t = typename Matrix<T, allocator_type>;
	OneRowMatrix(std::ranges::range auto&& range) : base_t(to_Matrix_row(range)) {};
};
#endif

template<typename T, typename U>
constexpr auto dummy_mult(const Matrix<T>& A, const Matrix<U>& B )
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

template<typename T, typename U, typename W>
constexpr void mult_(const Matrix<T>& A, const Matrix<U>& B, Matrix<W>& C) noexcept {
	using size_type = typename Matrix<T>::size_type;
	const auto M = A.rows(); // C.rows()
	const auto N = B.cols(); // C.cols()
	const auto K = A.cols();

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

template<
	typename T,
	typename U,
	typename W = typename std::common_type_t<T, U>,
	typename Alloc = std::allocator<W>
>
constexpr auto mult(const Matrix<T>& A, const Matrix<U>& B) noexcept
	-> Matrix<W, Alloc>
{
	const auto M = A.rows();
	const auto N = B.cols();
	auto C = Matrix<W, Alloc>(M, N);
	mult_<T, U, W>(A, B, C);
	return C;
}

#endif
