// Copyright 2015 Markus Ilmola
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef UUID_8B39B0365617488895EAC6FEC2A32C6E
#define UUID_8B39B0365617488895EAC6FEC2A32C6E


#include <assert.h>
#include <iostream>
#include <cmath>
#include <sstream>
#include <string>
#include <tuple>

#include "vec.hpp"
#include "quaternion.hpp"

namespace gml {


/**
 * A matrix with C Columns and R rows
 * @tparam C Number of columns
 * @tparam R Number of rows (length of column vectors)
 */
template <typename T, int C, int R>
class mat {
public:

	static_assert(C > 0, "Columns must be greater than zero!");
	static_assert(R > 0, "Rows must be greater than zero!");

	/// The type of a single component (Not the type of single column!)
	using value_type = T;

	/// Initialize all components to zero.
	mat() { }

	/// Initialize the diagonal to given value and 0 elsewhere
	explicit mat(const T& a) {
		for (int i = 0; i < std::min(C, R); ++i) data_[i][i] = a;
	}

	/// Initialize all columns to v.
	explicit mat(const vec<T, R>& v) {
		for (int i = 0; i < C; ++i) data_[i] = v;
	}

	/// Initialize from C column vectors with R components each.
	template <
		typename... Args,
		typename std::enable_if<C == sizeof...(Args), int>::type = 0
	>
	mat(const Args&... args) :
		data_{ args... }
	{
		static_assert(sizeof...(args) == C, "Invalid number of arguments!");
	}

	/// Initialize from a c-array.
	/// The array must have at least C*R components or behaviour is undefined.
	/// If data is null assertion error will occur.
	/// @param data Pointer to the first element.
	/// @param columnMajor Are the components in the input in column major order
	mat(const T* data, bool columnMajor) {
		assert( data != nullptr );
		if (columnMajor) {
			for (int i = 0; i < C; ++i) {
				data_[i] = vec<T, R>(&data[i * R]);
			}
		}
		else {
			for (int i = 0; i < C; ++i) {
				for (int j = 0; j < R; ++j) {
					data_[i][j] = data[j * C + i];
				}
			}
		}
	}

	/// Initialize from a smaller matrix by adding given value to the diagonal
	/// and zero elsewhere.
	template <
		int CC, int RR,
		typename std::enable_if<(CC < C || RR < R), int>::type = 0
	>
	explicit mat(const mat<T, CC, RR>& m, const T& a = T{0}) {
		for (int i = 0; i < std::min(C, CC); ++i) {
			data_[i] = vec<T, R>{m[i]};
		}
		for (int i = std::min(CC, RR); i < std::min(C, R); ++i) {
			data_[i][i] = a;
		}
	}

	/// Initialize from bigger matrix by dropping trailing rows and columns.
	template <
		int CC, int RR,
		typename std::enable_if<((CC != C || RR != R) && CC >= C && RR >= R), int>::type = 0
	>
	explicit mat(const mat<T, CC, RR>& m) {
		for (int i = 0; i < C; ++i) data_[i] = vec<T, R>{m[i]};
	}

	/// Creates a sub matrix by removing a row and a column
	/// @param m Input matrix.
	/// @param col Zero based index of the column to remove
	/// @param row Zero based index of the row to remove
	mat(const mat<T, C+1, R+1>& m, int col, int row) {
		assert(col <= C && row <= R);
		for (int i = 0; i < C; ++i) {
			if (i < col) data_[i] = vec<T, R>{m[i], row};
			else data_[i] = vec<T, R>{m[i+1], row};
		}
	}

	mat(const mat&) = default;

	mat(mat&&) = default;

	mat& operator=(const mat&) = default;

	mat& operator=(mat&&) = default;

	/// Returns a reference to the i:th column vector.
	/// If i is not in the range [0, C) and assertion failure will occur.
	const vec<T, R>& operator[](int i) const noexcept {
		assert(i >= 0 && i < C);
		return data_[i];
	}

	/// Returns a reference to the i:th column vector.
	/// If i is not in the range [0, C) and assertion failure will occur.
	vec<T, R>& operator[](int i) noexcept {
		assert(i >= 0 && i < C);
		return data_[i];
	}

	/// Component-wise sum
	mat<T, C, R> operator+(const mat<T, C, R>& m) const {
		mat<T, C, R> temp{*this};
		temp += m;
		return temp;
	}

	/// Component-wise subtraction
	mat<T, C, R> operator-(const mat<T, C, R>& m) const {
		mat<T, C, R> temp{*this};
		temp -= m;
		return temp;
	}

	/// Returns the product of matrices
	template<int N>
	mat<T, N, R> operator*(const mat<T, N, C>& m) const {
		mat<T, N, R> temp;
		for (int i = 0; i < N; ++i) {
			for (int r = 0; r < R; ++r) {
				for (int c = 0; c < C; ++c) {
					temp[i][r] += data_[c][r] * m[i][c];
				}
			}
		}
		return temp;
	}

	/// The product of a matrix and a vector as if the vector was a column of a matrix.
	/// Note: you can't multiply 4x4-matrix and 3-vector. Use transform instead.
	vec<T, R> operator*(const vec<T, C>& v) const {
		vec<T, R> temp;
		for (int r = 0; r < R; ++r) {
			for (int c = 0; c < C; ++c) {
				temp[r] += data_[c][r] * v[c];
			}
		}
		return temp;
	}

	/// Returns the product of a matrix and a scalar
	mat<T, C, R> operator*(const T& a) const {
		mat<T, C, R> temp{*this};
		temp *= a;
		return temp;
	}

	/// Component-wise sum
	mat<T, C, R>& operator+=(const mat<T, C, R>& m) {
		for (int i = 0; i < C; ++i) {
			data_[i] += m.data_[i];
		}
		return *this;
	}

	/// Component wise subtraction
	mat<T, C, R>& operator-=(const mat<T, C, R>& m) {
		for (int i = 0; i < C; ++i) {
			data_[i] -= m.data_[i];
		}
		return *this;
	}

	/// Same as M = M * a
	mat<T, C, R>& operator*=(const T& a) {
		for (int i = 0; i < C; ++i) {
			data_[i] *= a;
		}
		return *this;
	}

	/// Same as M = M * m
	mat<T, C, R>& operator*=(const mat<T, C, R>& m) {
		*this = *this * m;
		return *this;
	}

	/// Matrices are equal if all corresponding components are equal.
	bool operator==(const mat<T, C, R>& m) const {
		for (int i = 0; i < C; ++i) {
			if (data_[i] != m.data_[i]) return false;
		}
		return true;
	}

	/// Matrices are not equal if any of the corresponding components are not equal
	bool operator!=(const mat<T, C, R>& m) const {
		for (int i = 0; i < C; ++i) {
			if (data_[i] != m.data_[i]) return true;
		}
		return false;
	}

	/// Returns pointer to the first component.
	/// The matrix data is in column major order
	const T* data() const noexcept { return data_[0].data(); }

	/// Returns the number of columns in the matrix
	/// (NOT the number of components)
	/// This is the largest value that can be given to the [] -operator.
	static int size() noexcept { return C; }

	/// Iterator to the first column
	vec<T, R>* begin() noexcept { return data_; }
	const vec<T, R>* begin() const noexcept { return data_; }

	/// Iterator to the one past the last column
	vec<T, R>* end() noexcept { return data_ + C; }
	const vec<T, R>* end() const noexcept { return data_ + C; }

private:

	vec<T, R> data_[C];

};


/// Multiplies all components of the matrix with a scalar
template <typename T, int C, int R>
mat<T, C, R> operator*(const T& a, const mat<T, C, R>& m) {
	mat<T, C, R> temp;
	for (int i = 0; i < C; ++i) {
		temp[i] = a * m[i];
	}
	return temp;
}


/// Prints the matrix to a stream inside brackets columns separated by a comma.
template <typename T, int C, int R>
std::ostream& operator<<(std::ostream& os, const mat<T, C, R>& m) {
	os << '(';
	for (int i = 0; i < C; ++i) {
		if (i > 0) os << ',';
		os << m[i];
	}
	os << ')';
	return os;
}


/// Read matrix from a stream.
/// The matrix must be inside brackets columns separeted by a comma.
template <typename T, int C, int R>
std::istream& operator>>(std::istream& is, mat<T, C, R>& m) {
	char tmp;
	is >> tmp;
	for (int i = 0; i < C; ++i) {
		is >> m[i];
		is >> tmp;
	}
	return is;
}


/// Converts a mat to std::string.
template <typename T, int C, int R>
std::string to_string(const mat<T, C, R>& m) {
	std::stringstream ss{};
	ss << m;
	return ss.str();
}


/// Returns the transpose of the matrix.
template <typename T, int C, int R>
mat<T, R, C> transpose(const mat<T, C, R>& m) {
	mat<T, R, C> temp;
	for (int i = 0; i < C; ++i) { // C
		for (int j = 0; j < R; ++j) {  // R
			temp[j][i] = m[i][j];
		}
	}
	return temp;
}


/// Component wise multiplication of matrices
template <typename T, int C, int R>
mat<T, C, R> matrixCompMult(const mat<T, C, R>& m1, const mat<T, C, R>& m2) {
	mat<T, C, R> temp;
	for (int i = 0; i < C; ++i) {
		temp[i] = m1[i] * m2[i];
	}
	return temp;
}


/// Treats the first vector as matrix with one column and second as matrix with one row
template <typename T, int C, int R>
mat<T, C, R> outerProduct(const vec<T, R>& v1, const vec<T, C>& v2) {
	mat<T, C, R> temp;
	for (int i = 0; i < C; ++i) {
		for (int j = 0; j < R; ++j) {
			temp[i][j] = v1[j] * v2[i];
		}
	}
	return temp;
}


/// Calculates the first minor (determinant of a submatrix)
/// @param col Index of the column to remove (must be [0, N - 1])
/// @param row Index of the row to remove (must be [0, N - 1])
template <typename T, int N>
T firstMinor(const mat<T, N, N>& m, int col, int row) {
	static_assert(N > 1, "N must be greater than 1.");

	return determinant(mat<T, N-1, N-1>{m, col, row});
}


template <typename T>
T firstMinor(const mat<T, 3, 3>& m, int col, int row) {
	assert(col >= 0 && col < 3);
	assert(row >= 0 && row < 3);

	const int c0 = (col == 0 ? 1 : 0);
	const int c1 = (col == 2 ? 1 : 2);
	const int r0 = (row == 0 ? 1 : 0);
	const int r1 = (row == 2 ? 1 : 2);

	return m[c0][r0] * m[c1][r1] - m[c1][r0] * m[c0][r1];
}


template <typename T>
T firstMinor(const mat<T, 2, 2>& m, int col, int row) {
	assert(col >= 0 && col < 2);
	assert(row >= 0 && row < 2);

	return m[!col][!row];
}


/// Computes the determinant of a matrix
template <typename T, int N>
T determinant(const mat<T, N, N>& m) {
	T det = 0;
	for (int i = 0; i < N; ++i) {
		if (i % 2 == 0)
			det += m[i][0] * firstMinor(m, i, 0);
		else
			det -= m[i][0] * firstMinor(m, i, 0);
	}
	return det;
}


/// Computes the determinant of a 1x1 matrix
template <typename T>
T determinant(const mat<T, 1, 1>& m) {
	return m[0][0];
}


/// Computes the determinant of a 2x2 matrix
template <typename T>
T determinant(const mat<T, 2, 2>& m) {
	return m[0][0] * m[1][1] - m[1][0] * m[0][1];
}


/// Computes the inverse of a matrix
template <typename T, int N>
mat<T, N, N> inverse(const mat<T, N, N>& m) {
	mat<T, N, N> temp;
	const T a = determinant(m);
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			if ((i + j) % 2 == 0)
				temp[j][i] = firstMinor(m, i, j) / a;
			else
				temp[j][i] = -firstMinor(m, i, j) / a;
		}
	}
	return temp;
}


/// Computes the inverse of a 1x1 matrix
template <typename T>
mat<T, 1, 1> inverse(const mat<T, 1, 1>& m) {
	return mat<T, 1, 1>{T{1} / m[0][0]};
}


/// Generates a translation matrix so that transform(translate(v), p) = p + v
/// This the same matrix that would generated by glTranslate
template <typename T, int N>
mat<T, N+1, N+1> translate(const vec<T, N>& v) {
	mat<T, N+1, N+1> result{T{1}};
	for (int i = 0; i < N; ++i) {
		result[N][i] = v[i];
	}
	return result;
}


/// Generates a scaling matrix so that transform(scale(v), p) = v * p
/// This the same matrix that would generated by glScale
template <typename T, int N>
mat<T, N+1, N+1> scale(const vec<T, N>& v) {
	mat<T, N+1, N+1> result{T{1}};
	for (int i = 0; i < N; ++i) {
		result[i][i] = v[i];
	}
	return result;
}


/// Generates rotations matrix so that
template <typename T>
mat<T, 3, 3> rotate(const T& angle) {
	using std::sin;
	using std::cos;

	const T s = sin(angle);
	const T c = cos(angle);

	const T data[9] = {
		c,    -s,    T{0},
		s,     c,    T{0},
		T{0},  T{0}, T{1}
	};
	return mat<T, 3, 3>{data, false};
}


/// Generates rotation matrix from Eular angles
template <typename T>
mat<T, 4, 4> rotate(const vec<T, 3>& angle) {
	using std::sin;
	using std::cos;

	const T sy = sin(angle[2]);
	const T cy = cos(angle[2]);
	const T sp = sin(angle[1]);
	const T cp = cos(angle[1]);
	const T sr = sin(angle[0]);
	const T cr = cos(angle[0]);

	const T data[16] = {
		cp * cy, sr * sp * cy + cr * -sy, cr * sp * cy + -sr * -sy, T{0},
		cp * sy, sr * sp * sy + cr * cy,  cr * sp * sy + -sr * cy,  T{0},
		-sp,     sr * cp,                 cr * cp,                  T{0},
		T{0},    T{0},                    T{0},                     T{1}
	};
	return mat<T, 4, 4>{data, false};
}


/// Generates rotation matrix from angle and axis
template <typename T>
mat<T, 4, 4> rotate(const T& angle, const vec<T, 3>& axis) {
	using std::cos;
	using std::sin;

	const T rcos = cos(angle);
	const T rsin = sin(angle);
	const T rcos_1 = T{1} - rcos;
	const T xz = axis[0] * axis[2];
	const T yz = axis[1] * axis[2];
	const T xy = axis[0] * axis[1];
	const T xx = axis[0] * axis[0];
	const T yy = axis[1] * axis[1];
	const T zz = axis[2] * axis[2];

	const T data[16] = {
		rcos + xx * rcos_1,            -axis[2] * rsin + xy * rcos_1, axis[1] * rsin + xz * rcos_1,  T{0},
		axis[2] * rsin + xy * rcos_1,  rcos + yy * rcos_1,            -axis[0] * rsin + yz * rcos_1, T{0},
		-axis[1] * rsin + xz * rcos_1, axis[0] * rsin + yz * rcos_1,  rcos + zz * rcos_1,            T{0},
		T{0},                          T{0},                          T{0},                          T{1}
	};
	return mat<T, 4, 4>{data, false};
}


/// Generates rotation matrix from a rotation quaternion
template <typename T>
mat<T, 4, 4> rotate(const quaternion<T>& q) {

	const T xx = q.imag[0] * q.imag[0];
	const T xy = q.imag[0] * q.imag[1];
	const T xz = q.imag[0] * q.imag[2];
	const T xw = q.imag[0] * q.real;
	const T yy = q.imag[1] * q.imag[1];
	const T yz = q.imag[1] * q.imag[2];
	const T yw = q.imag[1] * q.real;
	const T zz = q.imag[2] * q.imag[2];
	const T zw = q.imag[2] * q.real;

	const T data[16] = {
		T{1} - T{2} * (yy + zz), T{2} * (xy - zw),        T{2} * (xz + yw),        T{0},
		T{2} * (xy + zw),        T{1} - T{2} * (xx + zz), T{2} * (yz - xw),        T{0},
		T{2} * (xz - yw),        T{2} * (yz + xw),        T{1} - T{2} * (xx + yy), T{0},
		T{0},                    T{0},                    T{0},                    T{1}
	};
	return mat<T, 4, 4>{data, false};
}


/// Generates a TRS matrix. Same as translate(translation) *
/// rotate(angle, axis) * scale(scaling), but potentially faster.
template <typename T>
mat<T, 4, 4> translateRotateScale(
	const vec<T, 3>& translation,
	T angle, const vec<T, 3>& axis,
	const gml::vec<T, 3>& scaling
) {
	return translateRotateScale(
		translation, qrotate(angle, axis), scaling
	);
}


/// Generates a TRS matrix. Same as translate(translation) *
/// rotate(rotation) * scale(scaling), but potentially faster.
template <typename T>
mat<T, 4, 4> translateRotateScale(
	const vec<T, 3>& translation,
	const quaternion<T>& rotation,
	const vec<T, 3>& scaling
) {
	mat<T, 4, 4> m = rotate(rotation);

	m[0] *= scaling[0];
	m[1] *= scaling[1];
	m[2] *= scaling[2];

	m[3][0] = translation[0];
	m[3][1] = translation[1];
	m[3][2] = translation[2];

	return m;
}


/// Decomposes a rotation matrix to a rotation quaternion.
/// The input matrix is assumed to be a valid rotation matrix.
template <typename T>
quaternion<T> qdecomposeRotate(const mat<T, 4, 4>& m)
{
	using std::sqrt;

	const T one = static_cast<T>(1);
	const T two = static_cast<T>(2);

	const T trace = m[0][0] + m[1][1] + m[2][2];

	if (trace > static_cast<T>(0)) {
		const T s = static_cast<T>(0.5) / sqrt(trace + one);

		return quaternion<T>{
			static_cast<T>(0.25) / s,
			vec<T, 3>{
				(m[1][2] - m[2][1]) * s,
				(m[2][0] - m[0][2]) * s,
				(m[0][1] - m[1][0]) * s
			}
		};
	}
	else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
		const T s = two * sqrt(one + m[0][0] - m[1][1] - m[2][2]);

		return quaternion<T>{
			(m[1][2] - m[2][1]) / s,
			vec<T, 3>{
				static_cast<T>(0.25) * s,
				(m[1][0] + m[0][1]) / s,
				(m[2][0] + m[0][2]) / s
			}
		};
	}
	else if (m[1][1] > m[2][2]) {
		const T s = two * sqrt(one + m[1][1] - m[0][0] - m[2][2]);

		return quaternion<T>{
			(m[2][0] - m[0][2]) / s,
			vec<T, 3>{
				(m[1][0] + m[0][1]) / s,
				static_cast<T>(0.25) * s,
				(m[2][1] + m[1][2]) / s
			}
		};
	}
	else {
		const T s = two * sqrt(one + m[2][2] - m[0][0] - m[1][1]);

		return quaternion<T>{
			(m[0][1] - m[1][0]) / s,
			vec<T, 3>{
				(m[2][0] + m[0][2]) / s,
				(m[2][1] + m[1][2]) / s,
				static_cast<T>(0.25) * s
			}
		};
	}
}


/// Decomposes a rotation matrix to an angle and a axis.
/// The input matrix is assumed to be a valid rotation matrix.
template <typename T>
std::tuple<T, vec<T, 3>> decomposeRotate(const mat<T, 4, 4>& m) {
	return decomposeRotate(qdecomposeRotate(m));
}


/// Decomposes translate, rotate, scale -matrix to translation, angle, axis and scale.
template <typename T>
std::tuple<vec<T, 3>, T, vec<T, 3>, vec<T, 3>> decomposeTrs(
	const gml::mat<T, 4, 4>& m
) {
	std::tuple<vec<T, 3>, T, vec<T, 3>, vec<T, 3>> temp{};

	quaternion<T> q{};
	std::tie(std::get<0>(temp), q, std::get<3>(temp)) = qdecomposeTrs(m);

	std::tie(std::get<1>(temp), std::get<2>(temp)) = decomposeRotate(q);

	return temp;
}


/// Decomposes translate, rotate, scale -matrix to translation, rotation quaternion and scale.
template <typename T>
std::tuple<vec<T, 3>, quaternion<T>, vec<T, 3>> qdecomposeTrs(
	const gml::mat<T, 4, 4>& m
) {
	gml::mat<T, 4, 4> temp = m;

	const T s = firstMinor(m, 3, 3) < 0 ? static_cast<T>(-1) : static_cast<T>(1);

	vec<T, 3> scaling{s * length(m[0]), length(m[1]), length(m[2])};

	temp[0] /= scaling[0];
	temp[1] /= scaling[1];
	temp[2] /= scaling[2];

	return std::make_tuple(
		vec<T, 3>{m[3]},
		qdecomposeRotate(temp),
		scaling
	);
}


/// Generates a perspective projection matrix.
/// This is the same matrix that glFrustum would generate.
template <typename T>
mat<T, 4, 4> frustum(
	const T& left, const T& right, const T& bottom, const T& top,
	const T& zNear, const T& zFar
) {
	const T dX = right - left;
	const T dY = top - bottom;
	const T dZ = zFar - zNear;
	const T data[16] = {
		(T{2} * zNear) / dX, T{0},                (right + left) / dX,  T{0},
		T{0},                (T{2} * zNear) / dY, (top + bottom) / dY,  T{0},
		T{0},                T{0},                -(zFar + zNear) / dZ, (T{-2} * zFar * zNear) / dZ,
		T{0},                T{0},                T{-1},                T{0}
	};
	return mat<T, 4, 4>{data, false};
}


/// Returns a perspective projection matrix.
/// This is the same matrix that gluPerspective would generate.
template <typename T>
mat<T, 4, 4> perspective(
	const T& fovy, const T& aspect, const T& zNear, const T& zFar
) {
	using std::tan;
	const T ymax = zNear * tan(fovy / T{2});
	const T ymin = -ymax;
	const T xmin = ymin * aspect;
	const T xmax = ymax * aspect;
	return frustum(xmin, xmax, ymin, ymax, zNear, zFar);
}


/// Returns an orthographic projection matrix
/// This is the same matrix that glOrtho would generate.
template <typename T>
mat<T, 4, 4> ortho(
	const T& left, const T& right, const T& bottom, const T& top,
	const T& zNear, const T& zFar
) {
	const T dX = right - left;
	const T dY = top - bottom;
	const T dZ = zFar - zNear;
	const T data[16] = {
		T{2} / dX, T{0},      T{0},       -(right + left) / dX,
		T{0},      T{2} / dY, T{0},       -(top + bottom) / dY,
		T{0},      T{0},      T{-2} / dZ, -(zFar + zNear) / dZ,
		T{0},      T{0},      T{0},       T{1}
	};
	return mat<T, 4, 4>{data, false};
}


/// Returns an orthographic projection matrix
template <typename T>
mat<T, 4, 4> ortho2D(
	const T& left, const T& right, const T& bottom, const T& top
) {
	return ortho(left, right, bottom, top, T{-1}, T{1});
}


/// Returns same matrix as gluLookAt would generate
template <typename T>
mat<T, 4, 4> lookAt(
	const vec<T, 3>& eye, const vec<T, 3>& center, const vec<T, 3>& up
) {
	const vec<T, 3> z = normalize(center - eye);
	const vec<T, 3> x = normalize(cross(z, up));
	const vec<T, 3> y = cross(x, z);

	const T data[16] = {
		x[0],  x[1],  x[2],  T{0},
		y[0],  y[1],  y[2],  T{0},
		-z[0], -z[1], -z[2], T{0},
		T{0},  T{0},  T{0},  T{1}
	};
	return mat<T, 4, 4>{data, false} * translate(-eye);
}


/// Map object coordinates to window coordinates (same as gluProject).
/// @param v Object coordinates to project.
/// @param modelView The model view matrix
/// @param proj The projection matrix
/// @param viewportOrigin Lower left corner of the viewport.
/// @param viewportSize Size of the viewport
template <typename T, typename TI, typename TS>
vec<T, 3> project(
	const vec<T, 3>& v,
	const mat<T, 4, 4>& modelView, const mat<T, 4, 4>& proj,
	const vec<TI, 2>& viewportOrigin, const vec<TS, 2>& viewportSize
) {
	return project(v, proj * modelView, viewportOrigin, viewportSize);
}


/// Map object coordinates to window coordinates (same as gluProject).
/// @param v Object coordinates to project.
/// @param modelViewProj The model view projection matrix (proj * modelView).
/// @param viewportOrigin Lower left corner of the viewport.
/// @param viewportSize Size of the viewport
template <typename T, typename TI, typename TS>
vec<T, 3> project(
	const vec<T, 3>& v,
	const mat<T, 4, 4>& modelViewProj,
	const vec<TI, 2>& viewportOrigin, const vec<TS, 2>& viewportSize
) {
	vec<T, 4> in = modelViewProj * vec<T, 4>{v, static_cast<T>(1)};

	in[0] /= in[3];
	in[1] /= in[3];
	in[2] /= in[3];

	const T half = static_cast<T>(0.5);

	in[0] = in[0] * half + half;
	in[1] = in[1] * half + half;
	in[2] = in[2] * half + half;

	in[0] = in[0] * static_cast<T>(viewportSize[0]) + static_cast<T>(viewportOrigin[0]);
	in[1] = in[1] * static_cast<T>(viewportSize[1]) + static_cast<T>(viewportOrigin[1]);

	return vec<T, 3>{in, 3u};
}


/// Map window coordinates to object coordinates (same as gluUnProject).
/// @param v Window coordinates to map.
/// @param modelView The model view matrix.
/// @param proj The projection matrix.
/// @param viewportOrigin Lower left corner of the viewport.
/// @param viewportSize Size of the viewport
template <typename T, typename TI, typename TS>
vec<T, 3> unProject(
	const vec<T, 3>& v,
	const mat<T, 4, 4>& modelView, const mat<T, 4, 4>& proj,
	const vec<TI, 2>& viewportOrigin, const vec<TS, 2>& viewportSize
) {
	return unProject(v, inverse(proj * modelView), viewportOrigin, viewportSize);
}


/// Map window coordinates to object coordinates (same as gluUnProject).
/// @param v Window coordinates to map.
/// @param invModelViewProj Inverse model view projection matrix (proj * modelView)^-1.
/// @param viewportOrigin Lower left corner of the viewport.
/// @param viewportSize Size of the viewport
template <typename T, typename TI, typename TS>
vec<T, 3> unProject(
	const vec<T, 3>& v,
	const mat<T, 4, 4>& invModelViewProj,
	const vec<TI, 2>& viewportOrigin, const vec<TS, 2>& viewportSize
) {
	vec<T, 4> in{v, static_cast<T>(1)};

	in[0] = (in[0] - static_cast<T>(viewportOrigin[0])) / static_cast<T>(viewportSize[0]);
	in[1] = (in[1] - static_cast<T>(viewportOrigin[1])) / static_cast<T>(viewportSize[1]);

	const T one = static_cast<T>(1);
	const T two = static_cast<T>(2);

	in[0] = in[0] * two - one;
	in[1] = in[1] * two - one;
	in[2] = in[2] * two - one;

	vec<T, 4> out = invModelViewProj * in;

	out[0] /= out[3];
	out[1] /= out[3];
	out[2] /= out[3];

	return vec<T, 3>{out, 3u};
}


/// Multiply 4x4 matrix by 3-vector by adding 1 as last component to the vector.
template <typename T, int N>
vec<T, N-1> transform(const mat<T, N, N>& m, const vec<T, N-1>& v) {
	vec<T, N-1> temp{m[N-1]};
	for (int c = 0; c < N-1; ++c) {
		for (int r = 0; r < N-1; ++r) {
			temp[r] += m[c][r] * v[c];
		}
	}
	return temp;
}


/// Static cast each component from T2 to T1.
template <typename T1, typename T2, int C, int R>
mat<T1, C, R> static_mat_cast(const mat<T2, C, R>& m) {
	mat<T1, C, R> temp;
	for (int i = 0; i < C; ++i)
		temp[i] = static_vec_cast<T1>(m[i]);
	return temp;
}



/// Returns the trace of a matrix (sum of the diagonal elements).
template <typename T, int N>
T trace(const mat<T, N, N>& m) {
	T temp = m[0][0];
	for (int i = 1; i < N; ++i) {
		temp += m[i][i];
	}
	return temp;
}


/// Returns the transpose of the inverse of the upper leftmost 3x3 of the matrix
template <typename T>
mat<T, 3, 3> normalMatrix(const mat<T, 4, 4>& m)
{
	return transpose(inverse(mat<T, 3, 3>{m}));
}



using mat2x2 = mat<float, 2, 2>;
using mat2x3 = mat<float, 2, 3>;
using mat2x4 = mat<float, 2, 4>;
using mat3x2 = mat<float, 3, 2>;
using mat3x3 = mat<float, 3, 3>;
using mat3x4 = mat<float, 3, 4>;
using mat4x2 = mat<float, 4, 2>;
using mat4x3 = mat<float, 4, 3>;
using mat4x4 = mat<float, 4, 4>;

using mat2 = mat<float, 2, 2>;
using mat3 = mat<float, 3, 3>;
using mat4 = mat<float, 4, 4>;

using dmat2x2 = mat<double, 2, 2>;
using dmat2x3 = mat<double, 2, 3>;
using dmat2x4 = mat<double, 2, 4>;
using dmat3x2 = mat<double, 3, 2>;
using dmat3x3 = mat<double, 3, 3>;
using dmat3x4 = mat<double, 3, 4>;
using dmat4x2 = mat<double, 4, 2>;
using dmat4x3 = mat<double, 4, 3>;
using dmat4x4 = mat<double, 4, 4>;

using dmat2 = mat<double, 2, 2>;
using dmat3 = mat<double, 3, 3>;
using dmat4 = mat<double, 4, 4>;

using imat2x2 = mat<int, 2, 2>;
using imat2x3 = mat<int, 2, 3>;
using imat2x4 = mat<int, 2, 4>;
using imat3x2 = mat<int, 3, 2>;
using imat3x3 = mat<int, 3, 3>;
using imat3x4 = mat<int, 3, 4>;
using imat4x2 = mat<int, 4, 2>;
using imat4x3 = mat<int, 4, 3>;
using imat4x4 = mat<int, 4, 4>;

using imat2 = mat<int, 2, 2>;
using imat3 = mat<int, 3, 3>;
using imat4 = mat<int, 4, 4>;

using umat2x2 = mat<unsigned, 2, 2>;
using umat2x3 = mat<unsigned, 2, 3>;
using umat2x4 = mat<unsigned, 2, 4>;
using umat3x2 = mat<unsigned, 3, 2>;
using umat3x3 = mat<unsigned, 3, 3>;
using umat3x4 = mat<unsigned, 3, 4>;
using umat4x2 = mat<unsigned, 4, 2>;
using umat4x3 = mat<unsigned, 4, 3>;
using umat4x4 = mat<unsigned, 4, 4>;

using umat2 = mat<unsigned, 2, 2>;
using umat3 = mat<unsigned, 3, 3>;
using umat4 = mat<unsigned, 4, 4>;

using bmat2x2 = mat<bool, 2, 2>;
using bmat2x3 = mat<bool, 2, 3>;
using bmat2x4 = mat<bool, 2, 4>;
using bmat3x2 = mat<bool, 3, 2>;
using bmat3x3 = mat<bool, 3, 3>;
using bmat3x4 = mat<bool, 3, 4>;
using bmat4x2 = mat<bool, 4, 2>;
using bmat4x3 = mat<bool, 4, 3>;
using bmat4x4 = mat<bool, 4, 4>;

using bmat2 = mat<bool, 2, 2>;
using bmat3 = mat<bool, 3, 3>;
using bmat4 = mat<bool, 4, 4>;


}

#endif
