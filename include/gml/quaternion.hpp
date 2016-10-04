// Copyright 2015 Markus Ilmola
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef UUID_B15DF4F752A94088BD3A77BC2D628831
#define UUID_B15DF4F752A94088BD3A77BC2D628831

#include <assert.h>
#include <iostream>
#include <cmath>
#include <sstream>
#include <string>

namespace gml {


/**
 * Complex number with scalar as real part and 3-vector as imaginary part.
 * Used for rotations and for storing orientation in 3d space.
 */
template <typename T>
class quaternion {
public:

	/// Type of the real part and a single component of the imaginary part.
	using value_type = T;

	/// Quaternion with zero real an imaginary parts.
	quaternion() : real{0}, imag{T{0}} { }

	/// A quaternion with given real part and zero imaginary part
	explicit quaternion(const T& real) : real{real}, imag{T{0}} { }

	/// A quaternion with zero real part and given imaginary part
	explicit quaternion(const vec<T, 3>& imag) : real{0}, imag{imag} { }

	/// A quaternion with given real and imaginary parts.
	quaternion(T real, const vec<T, 3>& imag) : real{real}, imag{imag} { }

	quaternion(const quaternion&) = default;

	quaternion(quaternion&&) = default;

	quaternion& operator=(const quaternion&) = default;

	quaternion& operator=(quaternion&&) = default;

	/// Component-wise addition
	quaternion<T> operator+(const quaternion<T>& q) const {
		return quaternion<T>{real + q.real, imag + q.imag};
	}

	/// Component-wise addition by interpreting v as imaginary part of a quaternion.
	quaternion<T> operator+(const vec<T, 3>& v) const {
		return quaternion<T>{real, imag + v};
	}

	/// Component-wise addition.
	quaternion<T> operator+(const T& a) const {
		return quaternion<T>{real + a, imag};
	}

	/// Components wise subtraction
	quaternion<T> operator-(const quaternion<T>& q) const {
		return quaternion<T>{real - q.real, imag - q.imag};
	}

	/// Components wise subtraction
	quaternion<T> operator-(const vec<T, 3>& v) const {
		return quaternion<T>{real, imag - v};
	}

	/// Components wise subtraction
	quaternion<T> operator-(const T& a) const {
		return quaternion<T>{real - a, imag};
	}

	/// Quaternion multiplication.
	/// Not done component wise.
	quaternion<T> operator*(const quaternion<T>& q) const {
		return quaternion<T>{
			real * q.real - dot(imag, q.imag),
			real * q.imag + q.real * imag + cross(imag, q.imag)
		};
	}

	/// Multiply quaternion and 3-vector (aka quaternion with zero real part).
	/// Note that vector v is not rotated by quaternion q by q*v but q*v*conj(q)
	/// (or use qtransform).
	quaternion<T> operator*(const vec<T, 3>& v) const {
		return quaternion<T>{-dot(imag, v), real*v + cross(imag, v)};
	}

	/// Multiply with scalar as it were the real part of a quaternion.
	quaternion<T> operator*(const T& a) const {
		return quaternion<T>{real * a, imag * a};
	}


	quaternion<T> operator/(const T& a) const {
		return quaternion<T>{real / a, imag / a};
	}


	quaternion<T>& operator+=(const quaternion<T>& q) {
		real += q.real;
		imag += q.imag;
		return *this;
	}


	quaternion<T>& operator-=(const quaternion<T>& q) {
		real -= q.real;
		imag -= q.imag;
		return *this;
	}


	quaternion<T>& operator*=(const quaternion<T>& q) {
		*this = *this * q;
		return *this;
	}

	/// Quaternions are equal if both parts are equal.
	bool operator==(const quaternion<T>& q) const {
		if (real != q.real) return false;
		if (imag != q.imag) return false;
		return true;
	}

	bool operator==(const T& a) const {
		if (real != a) return false;
		if (imag != vec<T, 3>{0}) return false;
		return true;
	}


	bool operator==(const vec<T, 3>& v) const {
		if (real != T{0}) return false;
		if (imag != v) return false;
		return true;
	}

	/// Quaternions are not equal if either of the parts are not equal.
	bool operator!=(const quaternion<T>& q) const {
		if (real != q.real) return true;
		if (imag != q.imag) return true;
		return false;
	}


	/// The real part of the quaternion
	T real;

	/// The imaginary part of the quaternion
	vec<T, 3> imag;

};


/// Multiplies scalar (aka quaternion with zero imaginary part) and quaternion
template <typename T>
quaternion<T> operator*(const T& a, const quaternion<T>& q) {
	return quaternion<T>{a * q.real, a * q.imag};
}


/// Multiply quaternion and vector as it were the imaginary part of a quaternion.
template <typename T>
quaternion<T> operator*(const vec<T, 3>& v, const quaternion<T>& q) {
	return quaternion<T>{-dot(v, q.imag), q.real * v + cross(v, q.imag)};
}


/// Add quaternion and vector as it were the imaginary part of a quaternion.
template <typename T>
quaternion<T> operator+(vec<T, 3> v, const quaternion<T>& q) {
	return quaternion<T>{q.real, v+q.imag};
}


template <typename T>
quaternion<T> operator+(const T& a, const quaternion<T>& q) {
	return quaternion<T>{a + q.real, q.imag};
}


template <typename T>
quaternion<T> operator-(const quaternion<T>& q) {
	return quaternion<T>{-q.real, -q.imag};
}


template <typename T>
quaternion<T> operator-(const T& a, const quaternion<T>& q) {
	return quaternion<T>{a - q.real, -q.imag};
}


template <typename T>
quaternion<T> operator-(const vec<T, 3>& v, const quaternion<T>& q) {
	return quaternion<T>{-q.real, v - q.imag};
}


/// Write a quaternion to a stream inside brackets parts separated by a comma.
template <typename T>
std::ostream& operator<<(std::ostream& os, const quaternion<T>& q) {
	os << '(' << q.real << ',' << q.imag << ')';
	return os;
}


/// Reads a quaternion from a stream.
/// The parts must be inside brackets separated be a comma.
template <typename T>
std::istream& operator>>(std::istream& is, quaternion<T>& q) {
	char ch = 0;
	is >> ch >> q.real >> ch >> q.imag >> ch;
	return is;
}


/// Converts a quaternion to std::string.
template <typename T>
std::string to_string(const quaternion<T>& q) {
	std::stringstream ss{};
	ss << q;
	return ss.str();
}


/// Absolute value.
template <typename T>
T abs(const quaternion<T>& q) {
	using std::sqrt;
	return sqrt(q.real * q.real + dot(q.imag, q.imag));
}


/// Makes the quaternion a unit quaternion.
template <typename T>
quaternion<T> normalize(const quaternion<T>& q) {
	return q / abs(q);
}


/// Quaternion conjugate (negates imaginary part)
template <typename T>
quaternion<T> conj(const quaternion<T>& q) {
	return quaternion<T>{q.real, -q.imag};
}


/// Generates rotation quaternion from axis and angle
/// @param angle Rotation angle in radians
/// @param axis Unit length rotation axis.
template <typename T>
quaternion<T> qrotate(const T& angle, const vec<T, 3>& axis) {
	using std::sin;
	using std::cos;

	const T a = angle / T{2};
	return quaternion<T>{cos(a), sin(a) * axis};
}


/// Generates rotation quaternion from Euler angles
/// @param angle Euler angles in radians.
template <typename T>
quaternion<T> qrotate(const vec<T, 3>& angle) {
	using std::sin;
	using std::cos;

	const T a1 = angle[0] / T{2};
	const T a2 = angle[1] / T{2};
	const T a3 = angle[2] / T{2};

	const T sx = sin(a1);
	const T cx = cos(a1);
	const T sy = sin(a2);
	const T cy = cos(a2);
	const T sz = sin(a3);
	const T cz = cos(a3);

	return quaternion<T>{
		cx * cy * cz + sx * sy * sz,
		vec<T, 3>{
			cy * cz * sx - cx * sy * sz,
			cx * cz * sy + cy * sx * sz,
			-cz * sx * sy + cx * cy * sz
		}
	};
}


/// Linear interpolation between quaternions.
/// the resulting quaternion is NOT normalized
template <typename T>
quaternion<T> mix(const quaternion<T>& q1, const quaternion<T>& q2, const T& a) {
	return (T{1} - a) * q1 + a * q2;
}


/// Rotates 3-vector v with quaternion q (computes q*v*conj(q))
/// Note q has to be a unit quaternion.
template <typename T>
vec<T, 3> transform(const quaternion<T>& q, const vec<T, 3>& v) {
	const vec<T, 3> temp = T{2} * cross(q.imag, v);
	return v + q.real * temp + cross(q.imag, temp);
}


/// Inverse of quaternion.
template <typename T>
quaternion<T> inverse(const quaternion<T>& q) {
	return conj(q) / (q.real*q.real + dot(q.imag, q.imag));
}


/// Static cast each component from T2 to T1.
template <typename T1, typename T2>
quaternion<T1> static_quaternion_cast(const quaternion<T2>& q) {
	return quaternion<T1>{static_cast<T1>(q.real), static_vec_cast<T1>(q.imag)};
}


typedef quaternion<float> quat;

typedef quaternion<double> dquat;

typedef quaternion<int> iquat;

typedef quaternion<unsigned int> uquat;

typedef quaternion<bool> bquat;

}

#endif
