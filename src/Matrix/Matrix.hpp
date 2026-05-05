#ifndef MY_MATRIX
#define MY_MATRIX
#pragma once

#include <iostream>
#include <memory>
#include <memory_resource>
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

	using tCont = TestCont<float>;
}

template<typename, typename, typename> struct rebind_container;
template<template<typename, typename> class Container,
	typename OldT, typename OldAlloc,
	typename NewT, typename NewAlloc>
struct rebind_container<Container<OldT, OldAlloc>, NewT, NewAlloc> {
	using type = Container<NewT, NewAlloc>;
};

template<typename Old, typename NewT, typename NewAlloc = void>
using rebind_container_t = typename rebind_container<Old, NewT, NewAlloc>::type;

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
		constexpr const_reference operator[](difference_type c) const { return *(row_data + c); }

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
		constexpr reference operator[](difference_type c) { return row_data[c];	}
		constexpr const_reference operator[](difference_type c) const { return row_data[c]; }
		
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

		constexpr const_reference operator[](difference_type r) const {
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

		constexpr const_reference operator[](difference_type r) const {
			return *(col_data + r * stride);
		}

		constexpr reference operator[](difference_type r) {
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

#if 0
	public:
		std::ranges::subrange get_range() { return std::ranges::subrange(begin(), end()); }
		std::ranges::subrange get_range() const { return std::ranges::subrange(cbegin(), cend()); }
#endif
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
		//constexpr internal_reference operator->() const { return *ptr; }
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
		constexpr pointer operator->() const { /* return nullptr; */ }
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
		using base_t = matrix_const_diagonal_iterator<Cont>;

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

		using allocator_type = Cont::allocator_type;

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
#if 0
		using const_col_proxy = const_ColProxy<Cont>;
		using const_row_proxy = const_RowProxy<Cont>;
#else
		using const_col_proxy = const_ColProxy<const_Submatrix>;
		using const_row_proxy = const_RowProxy<const_Submatrix>;
#endif

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
			constexpr auto one = static_cast<size_type>(1);
			return to_y - from_y + one;
		}
		constexpr size_type cols() const noexcept {
			auto&& mtx = *cont;
			auto&& f = *first;
			auto&& l = *last;
			auto&& from_x = mtx.get_elem_col_index(f);
			auto&& to_x = mtx.get_elem_col_index(l);
			constexpr auto one = static_cast<size_type>(1);
			return to_x - from_x + one; 
		}
		constexpr size_type size() const noexcept { return rows() * cols(); }
		constexpr pointer data()  const noexcept { return first; }
		constexpr bool is_square() const noexcept { return rows() == cols(); }
		constexpr bool empty() const noexcept { return ((cols() == 0) || (rows() == 0)); }

		const_reference get_elem(size_type row, size_type col) const {
			auto&& it = *this;
			auto&& cnt = *cont;
			auto&& [x, y] = cnt.get_elem_indices(it[row][col]);
			return cnt.get_elem(x, y);
		}

		void print() const {
			size_t precision = 8;
			const auto m = rows();
			const auto n = cols();

			std::cout << std::fixed << std::setprecision(precision);
			auto&& it = *this;
			for (size_type i = 0; i < m; ++i) {
				std::cout << "[ ";
				for (size_type j = 0; j < n; ++j) {
					std::cout << std::setw(10) << it[i][j];
					if (j + 1 < n) std::cout << ' ';
				}
				std::cout << " ]" << std::endl;
			}
			std::cout << std::endl;
		}

	public:
		constexpr std::pair<size_type, size_type> get_indices(const_reference elem) const noexcept {
			auto&& mtx = *cont;
			auto&& elem_ptr = std::addressof(elem);
#if 1
			if (elem_ptr < first || elem_ptr > last) {
				throw std::out_of_range("Element not in submatrix");
			}
#endif
			auto&& [elem_row, elem_col] = mtx.get_elem_indices(elem);
			auto&& fst = *first;
			auto&& [first_row, first_col] = mtx.get_elem_indices(fst);
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
#if 0
	public:
		template<typename value_type_, typename allocator_type_ = std::allocator<value_type>>
		auto get_matrix(const allocator_type_& allocator = {}) const {
			using new_cont = rebind_container_t<Cont, value_type_, allocator_type_>;
			auto&& it = *this;
			const auto r = it.rows();
			const auto c = it.cols();
			new_cont mtx(r, c, allocator);
			for (size_type i = 0; i < r; ++i) {
				auto&& it_row = it.row(i);
				auto&& mtx_row = mtx.row(i);
				for (size_type j = 0; j < c; ++j) {
					mtx_row[j] = it_row[j];
				}
			}
			return std::move(mtx);
		}
#endif
	public:
		const_Submatrix<Cont> get_submatrix(size_type fr, size_type fc, size_type lr, size_type lc) const {
			auto&& cnt = *cont;
			auto&& it = *this;
			return cnt.get_submatrix(it[fr][fc], it[lr][lc]);
		}

		const_Submatrix<Cont> get_submatrix(const value_type& fe, const value_type& le) const {
			auto&& cnt = *cont;
			return cnt.get_submatrix(fe, le);
		}

		const_Submatrix<Cont> get_submatrix(size_type fr, size_type fc, const value_type& le) const {
			auto&& cnt = *cont;
			auto&& it = *this;
			return cnt.get_submatrix(it[fr][fc], le);
		}

		const_Submatrix<Cont> get_submatrix(const value_type& fe, size_type lr, size_type lc) const {
			auto&& cnt = *cont;
			auto&& it = *this;
			return cnt.get_submatrix(fe, it[lr][lc]);
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
		using base_t = const_Submatrix<Cont>;

	public:
		using typename base_t::value_type;
		using typename base_t::pointer;
		using typename base_t::const_pointer;
		using typename base_t::size_type;
		using typename base_t::difference_type;
		using typename base_t::reference;
		using typename base_t::const_reference;

		using typename base_t::allocator_type;

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
		
		template<typename value_type_, typename allocator_type_ = std::allocator<value_type>>
		auto get_matrix(const allocator_type_& allocator = {}) {
			using new_cont = rebind_container_t<Cont, value_type_, allocator_type_>;
			auto&& it = *this;
			const auto r = it.rows();
			const auto c = it.cols();
			new_cont mtx(r, c, allocator);
			for (size_type i = 0; i < r; ++i) {
				auto&& it_row = it.row(i);
				auto&& mtx_row = mtx.row(i);
				for (size_type j = 0; j < c; ++j) {
					mtx_row[j] = it_row[j];
				}
			}
			return std::move(mtx);
		}
	public:
		Submatrix<Cont> get_submatrix(size_type fr, size_type fc, size_type lr, size_type lc) {
			auto&& cnt = *cont;
			auto&& it = *this;
			return cnt.get_submatrix(it[fr][fc], it[lr][lc]);
		}

		Submatrix<Cont> get_submatrix(const value_type& fe, const value_type& le) {
			auto&& cnt = *cont;
			return cnt.get_submatrix(fe, le);
		}

		Submatrix<Cont> get_submatrix(size_type fr, size_type fc, const value_type& le) {
			auto&& cnt = *cont;
			auto&& it = *this;
			return cnt.get_submatrix(it[fr][fc], le);
		}

		Submatrix<Cont> get_submatrix(const value_type& fe, size_type lr, size_type lc) {
			auto&& cnt = *cont;
			auto&& it = *this;
			return cnt.get_submatrix(fe, it[lr][lc]);
		}

	public:
		reference get_elem(size_type row, size_type col) {
			auto&& it = *this;
			auto&& cnt = *cont;
			auto&& [x, y] = cnt.get_elem_indices(it[row][col]);
			return cnt.get_elem(x, y);
		}

		const_reference get_elem(size_type row, size_type col) const {
			auto&& it = *this;
			auto&& cnt = *cont;
			auto&& [x, y] = cnt.get_elem_indices(it[row][col]);
			return cnt.get_elem(x, y);
		}

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

template<typename allocator_type, typename value_type> using rebind_allocator = typename std::allocator_traits<allocator_type>::template rebind_alloc<value_type>;

template<typename T, typename Alloc = std::pmr::polymorphic_allocator<T>>
struct Matrix {
	public:
		using allocator_type = rebind_allocator<Alloc, T>;
		using allocator_traits = std::allocator_traits<allocator_type>;

	public:
		using value_type = T;
		using pointer = typename allocator_traits::pointer;
		using const_pointer = typename allocator_traits::const_pointer;
		using size_type = typename allocator_traits::size_type;
		using difference_type = typename allocator_traits::difference_type;
		using reference = value_type&;
		using const_reference = const value_type&;

	protected:
		using base = Matrix<value_type, allocator_type>;

	public:
		using submatrix = Submatrix<base>;
		using const_submatrix = const_Submatrix<const base>;

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

	private:
		constexpr Matrix(const allocator_type& allocator) : _allocator(allocator) {}
	public:
		constexpr Matrix(size_type rows, size_type cols, const allocator_type& allocator = {})
			: Matrix(allocator) 
		{
			_rows = rows;
			_cols = cols;
			auto n = rows * cols;
			if (n > 0) {
				_data = allocator_traits::allocate(_allocator, n);
				for (size_type i = 0; i < n; ++i) {
					allocator_traits::construct(_allocator, (_data + i));
				}
			}	else {
				_data = nullptr;
			}
		}

		template<typename MatrixType>
		Matrix(const const_Submatrix<MatrixType>& submatrix, const allocator_type& allocator = {}) : Matrix(submatrix.rows(), submatrix.cols(), allocator) {
			auto&& it = *this;
			for (size_type i = 0; i < _rows; ++i) {
				for (size_type j = 0; j < _cols; ++j) {
					it.get_elem(i, j) = submatrix[i][j];
				}
			}
		}

		template<typename MatrixType>
		Matrix(const Submatrix<MatrixType>& submatrix, const allocator_type& allocator = {}) : Matrix(submatrix.rows(), submatrix.cols(), allocator) {
			auto&& it = *this;
			for (size_type i = 0; i < _rows; ++i) {
				for (size_type j = 0; j < _cols; ++j) {
					it.get_elem(i, j) = submatrix[i][j];
				}
			}
		}

#if 1
		Matrix(const Matrix& rhs) : Matrix(allocator_type{}) {
			_rows = rhs._rows;
			_cols = rhs._cols;
			const auto n = _rows * _cols;
			if (n > 0) {
				_data = allocator_traits::allocate(_allocator, n);
				for (size_type i = 0; i < n; ++i) {
					allocator_traits::construct(_allocator, &_data[i], rhs._data[i]);
				}
			} else {
				_data = nullptr;
			}
		}

		Matrix(const Matrix& rhs, const allocator_type& allocator) : Matrix(allocator) {
			_rows = rhs._rows;
			_cols = rhs._cols;
			const auto n = _rows * _cols;
			if (n > 0) {
				_data = allocator_traits::allocate(_allocator, n);
				for (size_type i = 0; i < n; ++i) {
					allocator_traits::construct(_allocator, &_data[i], rhs._data[i]);
				}
			}	else {
				_data = nullptr;
			}
		}

		Matrix(Matrix&& rhs) noexcept : Matrix(rhs.get_allocator()) {
			std::swap(rhs._rows, _rows);
			std::swap(rhs._cols, _cols);
			std::swap(rhs._data, _data);
		}

		Matrix(Matrix&& rhs, const auto& allocator)
			: Matrix(rhs._rows, rhs._cols, allocator) {
			if (_data) {
				std::copy(rhs._data, rhs._data + _rows * _cols, _data);
			}
		}

		Matrix& operator=(const Matrix& rhs) {
			if (&rhs == this) { return *this; }
			if (rhs.rows() == rows() && rhs.cols() == cols()) {
				const auto n = _rows * _cols;
				for (size_type i = 0; i < n; ++i) {
					_data[i] = rhs._data[i];
				}
			}	else {
				{
					if (_data) {
						const auto n = _rows * _cols;
						for (size_type i = 0; i < n; ++i) {
							allocator_traits::destroy(_allocator, &_data[i]);
						}
						allocator_traits::deallocate(_allocator, _data, n);
						_data = nullptr;
					}
					_cols = 0;
					_rows = 0;
				}

				const auto n = rhs.rows() * rhs.cols();
				if (n > 0) {
					_data = allocator_traits::allocate(_allocator, n);
					for (size_type i = 0; i < n; ++i) {
						allocator_traits::construct(_allocator, &_data[i], rhs._data[i]);
					}
				} else {
					_data = nullptr;
				}
				_rows = rhs._rows;
				_cols = rhs._cols;
			}
			return *this;
		}

		template<typename val_type, typename alloc_type>
		Matrix& operator=(const Matrix<val_type, alloc_type>& rhs) {
			auto&& it = *this;
			if (&rhs == this) { return *this; }
			if (rhs.rows() == rows() && rhs.cols() == cols()) {
				const auto n = _rows * _cols;
				for (size_type i = 0; i < n; ++i) {
					_data[i] = static_cast<value_type>(rhs._data[i]);
				}
			}	else {
				{
					if (_data) {
						const auto n = _rows * _cols;
						for (size_type i = 0; i < n; ++i) {
							allocator_traits::destroy(_allocator, &_data[i]);
						}
						allocator_traits::deallocate(_allocator, _data, n);
						_data = nullptr;
					}
					_cols = 0;
					_rows = 0;
				}

				const auto n = rhs.rows() * rhs.cols();
				if (n > 0) {
					_data = allocator_traits::allocate(_allocator, n);
					for (size_type i = 0; i < n; ++i) {
						allocator_traits::construct(_allocator, &_data[i], static_cast<value_type>(rhs._data[i]));
					}
				}	else {
					_data = nullptr;
				}
				_rows = rhs._rows;
				_cols = rhs._cols;
			}
			return it;
		}

		Matrix& operator=(Matrix&& rhs) {
			if (&rhs == this) { return *this; }
			if (get_allocator() == rhs.get_allocator()) {
				std::swap(_data, rhs._data);
				std::swap(_rows, rhs._rows);
				std::swap(_cols, rhs._cols);
			}	else {
				operator=(rhs);
			}
			return *this;
		}
#endif

		constexpr ~Matrix() {
			if (_data) {
				const auto n = _rows * _cols;
				for (size_type i = 0; i < n; ++i) {
					allocator_traits::destroy(_allocator, _data + i);
				}
				allocator_traits::deallocate(_allocator, _data, n);
				_data = nullptr;
			}
			_cols = 0;
			_rows = 0;
		}

	private:
		constexpr pointer get_elem_ptr(size_type r, size_type c) const noexcept {
			return _data + (r * _cols + c);
		}

	private:
		constexpr allocator_type& _get_allocator() noexcept { return _allocator; }
		constexpr const allocator_type& _get_allocator() const noexcept {	return _allocator; }

	public:
		constexpr size_type size() const noexcept { return _rows * _cols; }
		constexpr size_type rows() const noexcept { return _rows; }
		constexpr size_type cols() const noexcept { return _cols; }
		constexpr bool empty() const noexcept { return ((_rows == 0) || (_cols == 0)); }
		constexpr bool is_square() const noexcept { return _rows == _cols; }
		constexpr allocator_type get_allocator() const noexcept { return _get_allocator(); }

	public:
		constexpr const auto& get_elem(size_type r, size_type c) const {
			auto ptr = get_elem_ptr(r, c);
			return *ptr;
		}

		constexpr auto& get_elem(size_type r, size_type c) {
			auto ptr = get_elem_ptr(r, c);
			return *ptr;
		}

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
			return submatrix(this, get_elem_ptr(fr, fc), get_elem_ptr(lr, lc));
		}

		constexpr const_submatrix get_submatrix(size_type fr, size_type fc, size_type lr, size_type lc) const noexcept {			
			return const_submatrix(this, get_elem_ptr(fr, fc), get_elem_ptr(lr, lc));
		}

		constexpr submatrix get_submatrix(const_reference first, const_reference last) noexcept {
			auto&& [fr, fc] = get_elem_indices(first);
			auto&& [lr, lc] = get_elem_indices(last);
			return get_submatrix(fr, fc, lr, lc);
		}

		constexpr const_submatrix get_submatrix(const_reference first, const_reference last) const noexcept {
			auto&& [fr, fc] = get_elem_indices(first);
			auto&& [lr, lc] = get_elem_indices(last);
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
		constexpr void set_submatrix(const value_type& elem, const auto& mtx) {
			auto&& it = *this;
			const auto r = mtx.rows();
			const auto c = mtx.cols();

			const auto [x, y] = get_elem_indices(elem);

#if 0
			if (x + r > this->rows() || y + c > this->cols()) {
				throw std::out_of_range("set_submatrix: submatrix does not fit");
			}
#endif
#if 0
			for (size_t i = 0; i < r; ++i) {
				for (size_t j = 0; j < c; ++j) {
					it[x + i][y + j] = mtx[i][j];
				}
			}
#endif

			for (size_t i = 0; i < r; ++i) {
				auto&& mtx_i_col = mtx.row(i);
				auto&& it_xpi_col = it.row(x + i);
				for (size_t j = 0; j < c; ++j) {
					it_xpi_col[y + j] = mtx_i_col[j];
				}
			}
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

		/* def */
		[[nodiscard]] constexpr auto def_range() noexcept {
			return std::ranges::subrange(begin(), end());
		}

		[[nodiscard]] constexpr auto def_range() const noexcept {
			return std::ranges::subrange(cbegin(), cend());
		}

		[[nodiscard]] constexpr auto const_def_range() const noexcept {
			return std::ranges::subrange(cbegin(), cend());
		}

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

	private:
		allocator_type _allocator = {};
		size_type _rows = 0;
		size_type _cols = 0;
		pointer _data = nullptr;
};

#endif
