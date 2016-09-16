// Copyright 2015 Markus Ilmola
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef UUID_98EC23264CC64D72B996E6FC39002D48
#define UUID_98EC23264CC64D72B996E6FC39002D48


#include <assert.h>
#include <iostream>
#include <cmath>
#include <utility>
#include <limits>
#include <sstream>
#include <string>


namespace gml {


/**
 * A vector with N components.
 * @tparam T Type of a single component
 * @tparam N Number of components. Must not be zero!
 */
template <typename T, std::size_t N>
class vec {
public:

	static_assert(N > 0, "N is zero!");

	/// The type of a single component.
	using value_type = T;

	/// Initializes all components to zero
	vec() : vec{T{0}} { }

	/// Initializes all components to the given value
	explicit vec(const T& a) {
		for (std::size_t i = 0; i < N; ++i) data_[i] = a;
	}

	/// Initializes components from a C-array.
	/// The data MUST have at least N components or behaviour is undefined.
	/// If data is null assertion error will occur.
	explicit vec(const T* data) {
		assert( data != nullptr );
		for (std::size_t i = 0; i < N; ++i) data_[i] = data[i];
	}

	/// Initializes components from N values directly.
	template <
		typename... Args,
		typename std::enable_if<N == sizeof...(Args), int>::type = 0
	>
	vec(const Args&... args) :
		data_{ args... }
	{
		static_assert(sizeof...(args) == N, "Invalid number of arguments!");
	}

	/// Initializes from a smaller vector by padding with given value.
	template <
		std::size_t M,
		typename std::enable_if<(M < N), int>::type = 0
	>
	explicit vec(const vec<T, M>& v, const T& a = T{0}) {
		for (std::size_t i = 0; i < M; ++i) data_[i] = v[i];
		for (std::size_t i = M; i < N; ++i) data_[i] = a;
	}

	/// Initializes from a from a bigger vector by dropping trailing components.
	template <
		std::size_t M,
		typename std::enable_if<(M > N), int>::type = 0
	>
	explicit vec(const vec<T, M>& v) {
		for (std::size_t i = 0; i < N; ++i) data_[i] = v[i];
	}

	/// Initializes from a vector by dropping component with index i.
	/// If i is larger than N assertion error will occure.
	vec(const vec<T, N+1>& v, std::size_t i) {
		assert(i <= N);
		for (std::size_t j = 0; j < N; ++j) {
			if (j < i) data_[j] = v[j];
			else data_[j] = v[j+1];
		}
	}

	vec(const vec<T, N>&) = default;

	vec(vec<T, N>&&) = default;

	vec<T, N>& operator=(const vec<T, N>&) = default;

	vec<T, N>& operator=(vec<T, N>&&) = default;

	/// Returns a reference to the i:th component of the vector
	T& operator[](std::size_t i) noexcept {
		assert(i < N);
		return data_[i];
	}

	/// Returns a reference to the i:th component of the vector
	const T& operator[](std::size_t i) const noexcept {
		assert(i < N);
		return data_[i];
	}

	/// Component-wise sum
	vec<T, N> operator+(const vec<T, N>& v) const {
		vec<T, N> temp{*this};
		temp += v;
		return temp;
	}

	/// Component-wise sum
	vec<T, N> operator+(const T& a) const {
		vec<T, N> temp{*this};
		temp += a;
		return temp;
	}

	/// Component-wise subtraction
	vec<T, N> operator-(const vec<T, N>& v) const {
		vec<T, N> temp{*this};
		temp -= v;
		return temp;
	}

	/// Component-wise subtraction
	vec<T, N> operator-(const T& a) const {
		vec<T, N> temp{*this};
		temp -= a;
		return temp;
	}

	/// Component-wise multiplication
	vec<T, N> operator*(const vec<T, N>& v) const {
		vec<T, N> temp{*this};
		temp *= v;
		return temp;
	}

	/// Component-wise multiplication
	vec<T, N> operator*(const T& a) const {
		vec<T, N> temp{*this};
		temp *= a;
		return temp;
	}

	/// Component-wise division
	vec<T, N> operator/(const vec<T, N>& v) const {
		vec<T, N> temp{*this};
		temp /= v;
		return temp;
	}

	/// Component-wise division
	vec<T, N> operator/(const T& a) const {
		vec<T, N> temp{*this};
		temp /= a;
		return temp;
	}

	/// Component-wise sum
	vec<T, N>& operator+=(const vec<T, N>& v) {
		for (std::size_t i = 0; i < N; ++i) {
			data_[i] += v.data_[i];
		}
		return *this;
	}

	/// Component-wise sum
	vec<T, N>& operator+=(const T& a) {
		for (std::size_t i = 0; i < N; ++i) {
			data_[i] += a;
		}
		return *this;
	}

	/// Component-wise subtraction
	vec<T, N>& operator-=(const vec<T, N>& v) {
		for (std::size_t i = 0; i < N; ++i) {
			data_[i] -= v.data_[i];
		}
		return *this;
	}

	/// Component-wise subtraction
	vec<T, N>& operator-=(const T& a) {
		for (std::size_t i = 0; i < N; ++i) {
			data_[i] -= a;
		}
		return *this;
	}

	/// Component-wise multiplication
	vec<T, N>& operator*=(const vec<T, N>& v) {
		for (std::size_t i = 0; i < N; ++i) {
			data_[i] *= v.data_[i];
		}
		return *this;
	}

	/// Component-wise multiplication
	vec<T, N>& operator*=(const T& a) {
		for (std::size_t i = 0; i < N; ++i) {
			data_[i] *= a;
		}
		return *this;
	}

	/// Component-wise division
	vec<T, N>& operator/=(const vec<T, N>& v) {
		for (std::size_t i = 0; i < N; ++i) {
			data_[i] /= v.data_[i];
		}
		return *this;
	}

	/// Component-wise division
	vec<T, N>& operator/=(const T& a) {
		for (std::size_t i = 0; i < N; ++i) {
			data_[i] /= a;
		}
		return *this;
	}

	/// Vectors are equal if all of the corresponding components are equal
	bool operator==(const vec<T, N>& v) const {
		for (std::size_t i = 0; i < N; ++i) {
			if (data_[i] != v.data_[i]) return false;
		}
		return true;
	}

	/// Vectors are not equal if any of the corresponding components are not equal
	bool operator!=(const vec<T, N>& v) const {
		for (std::size_t i = 0; i < N; ++i) {
			if (data_[i] != v.data_[i]) return true;
		}
		return false;
	}

	/// Lexicographical less-than comparison.
	/// Used to make vec function as map key
	bool operator<(const vec<T, N>& v) const {
		for (std::size_t i = 0; i < N; ++i) {
			if (data_[i] < v.data_[i]) return true;
			if (v.data_[i] < data_[i]) return false;
		}
		return false;
	}

	/// Returns a pointer to the first component
	const T* data() const noexcept { return data_; }

	/// Returns the number of components in the vector.
	/// The maximum value that can be given to []-operator.
	static std::size_t size() noexcept { return N; }

	/// Iterator to the first component.
	T* begin() noexcept { return data_; }
	const T* begin() const noexcept { return data_; }

	/// Iterator to one past last component.
	T* end() noexcept { return data_ + N; }
	const T* end() const noexcept { return data_ + N; }

private:

	T data_[N];

};


/// Applies the given function fn to each component
template <typename T, std::size_t N, typename F>
vec<T, N> transform(F fn, const vec<T, N>& v) {
	vec<T, N> temp;
	for (std::size_t i = 0; i < N; ++i) {
		temp[i] = fn(v[i]);
	}
	return temp;
}


/// Negates all components
template <typename T, std::size_t N>
vec<T, N> operator-(const vec<T, N>& v) {
	vec<T, N> temp;
	for (std::size_t i = 0; i < N; ++i) {
		temp[i] = -v[i];
	}
	return temp;
}


/// Component-wise sum
template <typename T, std::size_t N>
vec<T, N> operator+(const T& a, const vec<T, N>& v) {
	vec<T, N> temp;
	for (std::size_t i = 0; i < N; ++i) {
		temp[i] = a + v[i];
	}
	return temp;
}


/// Component-wise subtraction
template <typename T, std::size_t N>
vec<T, N> operator-(const T& a, const vec<T, N>& v) {
	vec<T, N> temp;
	for (std::size_t i = 0; i < N; ++i) {
		temp[i] = a - v[i];
	}
	return temp;
}


/// Component-wise multiplication
template <typename T, std::size_t N>
vec<T, N> operator*(const T& a, const vec<T, N>& v) {
	vec<T, N> temp;
	for (std::size_t i = 0; i < N; ++i) {
		temp[i] = a * v[i];
	}
	return temp;
}


/// Component-wise division
template <typename T, std::size_t N>
vec<T, N> operator/(const T& a, const vec<T, N>& v) {
	vec<T, N> temp;
	for (std::size_t i = 0; i < N; ++i) {
		temp[i] = a / v[i];
	}
	return temp;
}


/// Prints the vector to a stream inside brackets components separated by a comma.
template <typename T, std::size_t N>
std::ostream& operator<<(std::ostream& os, const vec<T, N>& v) {
	os << '(';
	for (std::size_t i = 0; i < N; ++i) {
		if (i > 0) os << ',';
		os << v[i];
	}
	os << ')';
	return os;
}


/// Read a vector from a stream.
/// The vector must be inside brackets components separated by a comma.
template <typename T, std::size_t N>
std::istream& operator>>(std::istream& is, vec<T, N>& v) {
	char tmp;
	is >> tmp;
	for (std::size_t i = 0; i < N; ++i) {
		is >> v[i];
		is >> tmp;
	}
	return is;
}


/// Converts a vec to std::string.
template <typename T, std::size_t N>
std::string to_string(const vec<T, N>& v) {
	std::stringstream ss{};
	ss << v;
	return ss.str();
}


/// Dot products of vectors v1 and v2
template <typename T, std::size_t N>
T dot(const vec<T, N>& v1, const vec<T, N>& v2) {
	T temp = v1[0] * v2[0];
	for (std::size_t i = 1; i < N; ++i) {
		temp += v1[i] * v2[i];
	}
	return temp;
}


/// Dot product where v1 is replaced by vector rotated 90 degrees counter clockwise
template <typename T>
T perpDot(const vec<T, 2>& v1, const vec<T, 2>& v2) {
	return v1[0] * v2[1] - v1[1] * v2[0];
}


/// Cross product of two 3 component vectors
template <typename T>
vec<T, 3> cross(const vec<T, 3>& v1, const vec<T, 3>& v2) {
	return vec<T, 3>{
		v1[1] * v2[2] - v1[2] * v2[1],
		v1[2] * v2[0] - v1[0] * v2[2],
		v1[0] * v2[1] - v1[1] * v2[0]
	};
}


/// Returns a perpendicular vector (counter clockwise rotation by 90 degrees)
/// See: http://reference.wolfram.com/language/ref/Cross.html
template <typename T>
vec<T, 2> cross(const vec<T, 2>& v) {
	return vec<T, 2>{-v[1], v[0]};
}


/// Returns the geometric length of the vector.
template <typename T, std::size_t N>
T length(const vec<T, N>& v) {
	using std::sqrt;
	return sqrt(dot(v, v));
}


/// Returns a unit length copy of the vector
/// The vector must not have a zero length!
template <typename T, std::size_t N>
vec<T, N> normalize(const vec<T, N>& v) {
	return v / length(v);
}


/// Returns the length of the vector from p1 to p2
template <typename T, std::size_t N>
T distance(const vec<T, N>& p1, const vec<T, N>& p2) {
	return length(p2 - p1);
}


/// Computes the area between the 3 points
template <typename T>
T area(const vec<T, 3>& p1, const vec<T, 3>& p2, const vec<T, 3>& p3) {
	return length(cross(p2 - p1, p3 - p1)) / T{2};
}


/// Calculate the reflection direction for an incident vector.
/// n must be normalized
template <typename T, std::size_t N>
vec<T, N> reflect(const vec<T, N>& v, const vec<T, N>& n) {
	return v - T{2} * dot(v, n) * n;
}


/// Projects one vector on to another.
template <typename T, std::size_t N>
vec<T, N> project(const vec<T, N>& v, const vec<T, N>& u) {
	return dot(v, u) / dot(u, u) * u;
}


/// Calculate the refraction direction for an incident vector
/// @param eta Index of refraction
template <typename T, std::size_t N>
vec<T, N> refract(const vec<T, N>& v, const vec<T, N>& n, const T& eta) {
	using std::sqrt;
	T d = dot(n, v);
	T k = T{1} - eta * eta * (T{1} - d * d);
	if (k < T{0}) return vec<T, N>{T{0}};
	return eta * v - (eta * d + sqrt(k)) * n;
}


/// Linear interpolation between v1 and v2 using a as factor
template <typename T, std::size_t N>
vec<T, N> mix(const vec<T, N>& v1, const vec<T, N>& v2, const T& a) {
	return (T{1} - a) * v1 + a * v2;
}


/// Linear interpolation between v1 and v2 using a as factor
template <typename T, std::size_t N>
vec<T, N> mix(const vec<T, N>& v1, const vec<T, N>& v2, const vec<T, N>& a) {
	return (vec<T, N>{1} - a) * v1 + a * v2;
}


/// Linear spherical interpolation between v1 and v2 using a as factor
template <typename T, std::size_t N>
vec<T, N> slerp(const vec<T, N>& v1, const vec<T, N>& v2, const T& a) {
	using std::sin;
	const T theta = angle(v1, v2);
	const T sine = sin(theta);
	return sin((T{1} - a) * theta) / sine * v1 + sin(a * theta) / sine * v2;
}


/// Returns the angle (in radians) between vectors v1 and v2
/// If v1 or v2 is zero length zero is returned.
template <typename T, std::size_t N>
T angle(const vec<T, N>& v1, const vec<T, N>& v2) {
	using std::sqrt;
	using std::acos;
	using std::numeric_limits;

	const T len = sqrt(dot(v1, v1) * dot(v2, v2));
	if (len <= numeric_limits<T>::epsilon()) return T{0};
	return acos(clamp(dot(v1, v2) / len, T{-1}, T{1}));
}


/// Component-wise min
template <typename T, std::size_t N>
vec<T, N> min(const vec<T, N>& v1, const vec<T, N>& v2) {
	vec<T, N> temp;
	for (std::size_t i = 0; i < N; ++i) {
		temp[i] = v1[i] < v2[i] ? v1[i] : v2[i];
	}
	return temp;
}


/// Component-wise min
template <typename T, std::size_t N>
vec<T, N> min(const vec<T, N>& v, const T& a) {
	vec<T, N> temp;
	for (std::size_t i = 0; i < N; ++i) {
		temp[i] = v[i]<a ? v[i] : a;
	}
	return temp;
}


/// Component-wise max
template <typename T, std::size_t N>
vec<T, N> max(const vec<T, N>& v1, const vec<T, N>& v2) {
	vec<T, N> temp;
	for (std::size_t i = 0; i < N; ++i) {
		temp[i] = v1[i] > v2[i] ? v1[i] : v2[i];
	}
	return temp;
}


/// Component-wise max
template <typename T, std::size_t N>
vec<T, N> max(const vec<T, N>& v, const T& a) {
	vec<T, N> temp;
	for (std::size_t i = 0; i < N; ++i) {
		temp[i] = v[i] > a ? v[i] : a;
	}
	return temp;
}


/// Constrain components to lie between given values
template <typename T, std::size_t N>
vec<T, N> clamp(const vec<T, N>& v, const vec<T, N>& minVal, const vec<T, N>& maxVal) {
	return min(max(v, minVal), maxVal);
}


/// Constrain components to lie between given values
template <typename T, std::size_t N>
vec<T, N> clamp(const vec<T, N>& v, const T& minVal, const T& maxVal) {
	return min(max(v, minVal), maxVal);
}


/// Computes the unit length normal of a triangle with vertices p1, p2 and p3
/// The points must not lie on the same line!
template <typename T>
vec<T, 3> normal(const vec<T, 3>& p1, const vec<T, 3>& p2, const vec<T, 3>& p3) {
	return normalize(cross(p2 - p1, p3 - p1));
}


/// Component-wise sin
template <typename T, std::size_t N>
vec<T, N> sin(const vec<T, N>& v) {
	using std::sin;
	vec<T, N> temp;
	for (std::size_t i = 0; i < N; ++i) temp[i] = sin(v[i]);
	return temp;
}


/// Component-wise cos
template <typename T, std::size_t N>
vec<T, N> cos(const vec<T, N>& v) {
	using std::cos;
	vec<T, N> temp;
	for (std::size_t i = 0; i < N; ++i) temp[i] = cos(v[i]);
	return temp;
}


/// Component-wise tan
template <typename T, std::size_t N>
vec<T, N> tan(const vec<T, N>& v) {
	using std::tan;
	vec<T, N> temp;
	for (std::size_t i = 0; i < N; ++i) temp[i] = tan(v[i]);
	return temp;
}


/// Component-wise asin
template <typename T, std::size_t N>
vec<T, N> asin(const vec<T, N>& v) {
	using std::asin;
	vec<T, N> temp;
	for (std::size_t i = 0; i < N; ++i) temp[i] = asin(v[i]);
	return temp;
}


/// Component-wise acos
template <typename T, std::size_t N>
vec<T, N> acos(const vec<T, N>& v) {
	using std::acos;
	vec<T, N> temp;
	for (std::size_t i = 0; i < N; ++i) temp[i] = acos(v[i]);
	return temp;
}


/// Component-wise atan
template <typename T, std::size_t N>
vec<T, N> atan(const vec<T, N>& v) {
	using std::atan;
	vec<T, N> temp;
	for (std::size_t i = 0; i < N; ++i) temp[i] = atan(v[i]);
	return temp;
}


/// Component-wise sqrt
template <typename T, std::size_t N>
vec<T, N> sqrt(const vec<T, N>& v) {
	using std::sqrt;
	vec<T, N> temp;
	for (std::size_t i = 0; i < N; ++i) temp[i] = sqrt(v[i]);
	return temp;
}


/// Component-wise abs
template <typename T, std::size_t N>
vec<T, N> abs(const vec<T, N>& v) {
	using std::abs;
	vec<T, N> temp;
	for (std::size_t i = 0; i < N; ++i) temp[i] = abs(v[i]);
	return temp;
}


/// Component-wise pow
template <typename T, std::size_t N>
vec<T, N> pow(const vec<T, N>& v1, const vec<T, N>& v2) {
	using std::pow;
	vec<T, N> temp;
	for (std::size_t i = 0; i < N; ++i) temp[i] = pow(v1[i], v2[i]);
	return temp;
}


/// Component-wise exp
template <typename T, std::size_t N>
vec<T, N> exp(const vec<T, N>& v) {
	using std::exp;
	vec<T, N> temp;
	for (std::size_t i = 0; i < N; ++i) temp[i] = exp(v[i]);
	return temp;
}


/// Component-wise log
template <typename T, std::size_t N>
vec<T, N> log(const vec<T, N>& v) {
	using std::log;
	vec<T, N> temp;
	for (std::size_t i = 0; i < N; ++i) temp[i] = log(v[i]);
	return temp;
}


/// Component-wise radians
template <typename T, std::size_t N>
vec<T, N> radians(const vec<T, N>& v) {
	vec<T, N> temp;
	for (std::size_t i = 0; i < N; ++i) temp[i] = radians(v[i]);
	return temp;
}


/// Component-wise degrees
template <typename T, std::size_t N>
vec<T, N> degrees(const vec<T, N>& v) {
	vec<T, N> temp;
	for (std::size_t i = 0; i < N; ++i) temp[i] = degrees(v[i]);
	return temp;
}


/// Static cast each component from T2 to T1.
template <typename T1, typename T2, std::size_t N>
vec<T1, N> staticVecCast(const vec<T2, N>& v) {
	vec<T1, N> temp;
	for (std::size_t i = 0; i < N; ++i)
		temp[i] = static_cast<T1>(v[i]);
	return temp;
}


/// Component-wise unpackUnorm
template <typename TF, typename TI, std::size_t N>
vec<TF, N> unpackUnorm(const vec<TI, N>& v) {
	vec<TF, N> temp;
	for (std::size_t i = 0; i < N; ++i) temp[i] = unpackUnorm<TF, TI>(v[i]);
	return temp;
}


/// Component-wise packUnorm
template <typename TI, typename TF, std::size_t N>
vec<TI, N> packUnorm(const vec<TF, N>& v) {
	vec<TI, N> temp;
	for (std::size_t i = 0; i < N; ++i) temp[i] = packUnorm<TI, TF>(v[i]);
	return temp;
}


/// Component-wise unpackSnorm
template <typename TF, typename TI, std::size_t N>
vec<TF, N> unpackSnorm(const vec<TI, N>& v) {
	vec<TF, N> temp;
	for (std::size_t i = 0; i < N; ++i) temp[i] = unpackSnorm<TF, TI>(v[i]);
	return temp;
}


/// Component-wise packSnorm
template <typename TI, typename TF, std::size_t N>
vec<TI, N> packSnorm(const vec<TF, N>& v) {
	vec<TI, N> temp;
	for (std::size_t i = 0; i < N; ++i) temp[i] = packSnorm<TI, TF>(v[i]);
	return temp;
}



typedef vec<float, 2> vec2;
typedef vec<float, 3> vec3;
typedef vec<float, 4> vec4;

typedef vec<double, 2> dvec2;
typedef vec<double, 3> dvec3;
typedef vec<double, 4> dvec4;

typedef vec<int, 2> ivec2;
typedef vec<int, 3> ivec3;
typedef vec<int, 4> ivec4;

typedef vec<unsigned int, 2> uvec2;
typedef vec<unsigned int, 3> uvec3;
typedef vec<unsigned int, 4> uvec4;

typedef vec<bool, 2> bvec2;
typedef vec<bool, 3> bvec3;
typedef vec<bool, 4> bvec4;

typedef vec<std::size_t, 2> zvec2;
typedef vec<std::size_t, 3> zvec3;
typedef vec<std::size_t, 4> zvec4;

typedef vec<char, 2> cvec2;
typedef vec<char, 3> cvec3;
typedef vec<char, 4> cvec4;

typedef vec<signed char, 2> scvec2;
typedef vec<signed char, 3> scvec3;
typedef vec<signed char, 4> scvec4;

typedef vec<unsigned char, 2> ucvec2;
typedef vec<unsigned char, 3> ucvec3;
typedef vec<unsigned char, 4> ucvec4;

}


#endif
