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


namespace gml {


/**
 * A matrix with C Columns and R rows
 * @tparam C Number of columns
 * @tparam R Number of rows (length of column vectors)
 */
template <typename T, std::size_t C, std::size_t R>
class mat {
public:

	static_assert(C > 0, "Columns is zero.");
	static_assert(R > 0, "Rows is zero.");

	/// The type of a single component (Not the type of single column!)
	using value_type = T;

	/// Initialize all components to zero.
	mat() { }

	/// Initialize the diagonal to given value and 0 elsewhere
	explicit mat(const T& a) {
		for (std::size_t i = 0; i < std::min(C, R); ++i) data_[i][i] = a;
	}

	/// Initialize all columns to v.
	explicit mat(const vec<T, R>& v) {
		for (std::size_t i = 0; i < C; ++i) data_[i] = v;
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
	/// @param columnMajor Are the components in the input in column major order
	mat(const T* data, bool columnMajor) {
		assert( data != nullptr );
		if (columnMajor) {
			for (std::size_t i = 0; i < C; ++i) {
				data_[i] = vec<T, R>(&data[i * R]);
			}
		}
		else {
			for (std::size_t i = 0; i < C; ++i) {
				for (std::size_t j = 0; j < R; ++j) {
					data_[i][j] = data[j * C + i];
				}
			}
		}
	}

	/// Initialize from a smaller matrix by adding given value to the diagonal
	/// and zero elsewhere.
	template <
		std::size_t CC, std::size_t RR,
		typename std::enable_if<(CC < C || RR < R), int>::type = 0
	>
	explicit mat(const mat<T, CC, RR>& m, const T& a = T{0}) {
		for (std::size_t i = 0; i < std::min(C, CC); ++i) {
			data_[i] = vec<T, R>{m[i]};
		}
		for (std::size_t i = std::min(CC, RR); i < std::min(C, R); ++i) {
			data_[i][i] = a;
		}
	}

	/// Initialize from bigger matrix by dropping trailing rows and columns.
	template <
		std::size_t CC, std::size_t RR,
		typename std::enable_if<((CC != C || RR != R) && CC >= C && RR >= R), int>::type = 0
	>
	explicit mat(const mat<T, CC, RR>& m) {
		for (std::size_t i = 0; i < C; ++i) data_[i] = vec<T, R>{m[i]};
	}

	/// Creates a sub matrix by removing a row and a column
	/// @param col zero based index of the column to remove
	/// @param row zero based index of the row to remove
	mat(const mat<T, C+1, R+1>& m, std::size_t col, std::size_t row) {
		assert(col <= C && row <= R);
		for (std::size_t i = 0; i < C; ++i) {
			if (i < col) data_[i] = vec<T, R>{m[i], row};
			else data_[i] = vec<T, R>{m[i+1], row};
		}
	}

	mat(const mat&) = default;

	mat(mat&&) = default;

	mat& operator=(const mat&) = default;

	mat& operator=(mat&&) = default;

	/// Returns a reference to the i:th column vector.
	/// If i is not smaller than C assertion error will occur.
	const vec<T, R>& operator[](std::size_t i) const noexcept {
		assert(i < C);
		return data_[i];
	}

	/// Returns a reference to the i:th column vector.
	/// If i is not smaller than C assertion error will occur.
	vec<T, R>& operator[](std::size_t i) noexcept {
		assert(i < C);
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
	template<std::size_t N>
	mat<T, N, R> operator*(const mat<T, N, C>& m) const {
		mat<T, N, R> temp;
		for (std::size_t i = 0; i < N; ++i) {
			for (std::size_t r = 0; r < R; ++r) {
				for (std::size_t c = 0; c < C; ++c) {
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
		for (std::size_t r = 0; r < R; ++r) {
			for (std::size_t c = 0; c < C; ++c) {
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
		for (std::size_t i = 0; i < C; ++i) {
			data_[i] += m.data_[i];
		}
		return *this;
	}

	/// Component wise subtraction
	mat<T, C, R>& operator-=(const mat<T, C, R>& m) {
		for (std::size_t i = 0; i < C; ++i) {
			data_[i] -= m.data_[i];
		}
		return *this;
	}

	/// Same as M = M * a
	mat<T, C, R>& operator*=(const T& a) {
		for (std::size_t i = 0; i < C; ++i) {
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
		for (std::size_t i = 0; i < C; ++i) {
			if (data_[i] != m.data_[i]) return false;
		}
		return true;
	}

	/// Matrices are not equal if any of the corresponding components are not equal
	bool operator!=(const mat<T, C, R>& m) const {
		for (std::size_t i = 0; i < C; ++i) {
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
	static std::size_t size() noexcept { return C; }

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
template <typename T, std::size_t C, std::size_t R>
mat<T, C, R> operator*(const T& a, const mat<T, C, R>& m) {
	mat<T, C, R> temp;
	for (std::size_t i = 0; i < C; ++i) {
		temp[i] = a * m[i];
	}
	return temp;
}


/// Prints the matrix to a stream inside brackets columns separated by a comma.
template <typename T, std::size_t C, std::size_t R>
std::ostream& operator<<(std::ostream& os, const mat<T, C, R>& m) {
	os << '(';
	for (std::size_t i = 0; i < C; ++i) {
		if (i > 0) os << ',';
		os << m[i];
	}
	os << ')';
	return os;
}


/// Read matrix from a stream.
/// The matrix must be inside brackets columns separeted by a comma.
template <typename T, std::size_t C, std::size_t R>
std::istream& operator>>(std::istream& is, mat<T, C, R>& m) {
	char tmp;
	is >> tmp;
	for (std::size_t i = 0; i < C; ++i) {
		is >> m[i];
		is >> tmp;
	}
	return is;
}


/// Converts a mat to std::string.
template <typename T, std::size_t C, std::size_t R>
std::string to_string(const mat<T, C, R>& m) {
	std::stringstream ss{};
	ss << m;
	return ss.str();
}


/// Returns the transpose of the matrix.
template <typename T, std::size_t C, std::size_t R>
mat<T, R, C> transpose(const mat<T, C, R>& m) {
	mat<T, R, C> temp;
	for (std::size_t i = 0; i < C; ++i) { // C
		for (std::size_t j = 0; j < R; ++j) {  // R
			temp[j][i] = m[i][j];
		}
	}
	return temp;
}


/// Component wise multiplication of matrices
template <typename T, std::size_t C, std::size_t R>
mat<T, C, R> matrixCompMult(const mat<T, C, R>& m1, const mat<T, C, R>& m2) {
	mat<T, C, R> temp;
	for (std::size_t i = 0; i < C; ++i) {
		temp[i] = m1[i] * m2[i];
	}
	return temp;
}


/// Treats the first vector as matrix with one column and second as matrix with one row
template <typename T, std::size_t C, std::size_t R>
mat<T, C, R> outerProduct(const vec<T, R>& v1, const vec<T, C>& v2) {
	mat<T, C, R> temp;
	for (std::size_t i = 0; i < C; ++i) {
		for (std::size_t j = 0; j < R; ++j) {
			temp[i][j] = v1[j] * v2[i];
		}
	}
	return temp;
}


/// Computes the determinant of a matrix
template <typename T, std::size_t N>
T determinant(const mat<T, N, N>& m) {
	T det = 0;
	for (std::size_t i = 0; i < N; ++i) {
		if (i % 2 == 0)
			det += m[i][0] * determinant(mat<T, N-1, N-1>{m, i, std::size_t{0}});
		else
			det -= m[i][0] * determinant(mat<T, N-1, N-1>{m, i, std::size_t{0}});
	}
	return det;
}


/// Computes the determinant of a 1x1 matrix
template <typename T>
T determinant(const mat<T, 1, 1>& m) {
	return m[0][0];
}


/// Computes the inverse of a matrix
template <typename T, std::size_t N>
mat<T, N, N> inverse(const mat<T, N, N>& m) {
	mat<T, N, N> temp;
	const T a = determinant(m);
	for (std::size_t i = 0; i < N; ++i) {
		for (std::size_t j = 0; j < N; ++j) {
			if ((i + j) % 2 == 0)
				temp[j][i] = determinant(mat<T, N-1, N-1>{m, i, j}) / a;
			else
				temp[j][i] = -determinant(mat<T, N-1, N-1>{m, i, j}) / a;
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
template <typename T, std::size_t N>
mat<T, N+1, N+1> translate(const vec<T, N>& v) {
	mat<T, N+1, N+1> result{T{1}};
	for (std::size_t i = 0; i < N; ++i) {
		result[N][i] = v[i];
	}
	return result;
}


/// Generates a scaling matrix so that transform(scale(v), p) = v * p
/// This the same matrix that would generated by glScale
template <typename T, std::size_t N>
mat<T, N+1, N+1> scale(const vec<T, N>& v) {
	mat<T, N+1, N+1> result{T{1}};
	for (std::size_t i = 0; i < N; ++i) {
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


/// Generates a rotation quaternion from a rotation matrix.
/// The input matrix is assumed to be a valid rotation matrix.
template <typename T>
quaternion<T> qrotate(const mat<T, 4, 4>& m) {
	using std::sqrt;

	const T t = trace(m);
	const T S = static_cast<T>(0.5) / sqrt(t);

	return quaternion<T>{
		static_cast<T>(0.25) / S,
		vec<T, 3>{
			S * (m[1][2] - m[2][1]),
			S * (m[2][0] - m[0][2]),
			S * (m[0][1] - m[1][0])
		}
	};

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


/// Map object coordinates to window coordinates.
/// Same as gluProject
template <typename T>
vec<T, 3> project(
	const vec<T, 3>& v, const mat<T, 4, 4>& modelView,
	const mat<T, 4, 4>& proj, const ivec4& viewport
) {
	return project(v, proj * modelView, viewport);
}


/// Map object coordinates to window coordinates.
/// Same as gluProject.
template <typename T>
vec<T, 3> project(
	const vec<T, 3>& v, const mat<T, 4, 4>& modelViewNroj,
	const ivec4& viewport
) {
	vec<T, 4> in = modelViewNroj * vec<T, 4>{v, T{1}};

	in[0] /= in[3];
	in[1] /= in[3];
	in[2] /= in[3];

	in[0] = in[0] * T{0.5} + T{0.5};
	in[1] = in[1] * T{0.5} + T{0.5};
	in[2] = in[2] * T{0.5} + T{0.5};

	in[0] = in[0] * viewport[2] + viewport[0];
	in[1] = in[1] * viewport[3] + viewport[1];

	return vec<T, 3>{in, 3};
}


/// Same as gluUnProject.
template <typename T>
vec<T, 3> unProject(
	const vec<T, 3>& v, const mat<T, 4, 4>& modelView,
	const mat<T, 4, 4>& proj, const ivec4& viewport
) {
	return unProject(v, inverse(proj * modelView), viewport);
}


/// unProject with known inverse of ModelViewProj
template <typename T>
vec<T, 3> unProject(
	const vec<T, 3>& v, const mat<T, 4, 4>& invModelViewProj, const ivec4& viewport
) {
	vec<T, 4> in{v, T{1}};

	in[0] = (in[0] - viewport[0]) / viewport[2];
	in[1] = (in[1] - viewport[1]) / viewport[3];

	in[0] = in[0] * T{2} - T{1};
	in[1] = in[1] * T{2} - T{1};
	in[2] = in[2] * T{2} - T{1};

	vec<T, 4> out = invModelViewProj * in;

	out[0] /= out[3];
	out[1] /= out[3];
	out[2] /= out[3];

	return vec<T, 3>{out, 3};
}


/// Multiply 4x4 matrix by 3-vector by adding 1 as last component to the vector.
template <typename T, std::size_t N>
vec<T, N-1> transform(const mat<T, N, N>& m, const vec<T, N-1>& v) {
	vec<T, N-1> temp{m[N-1]};
	for (std::size_t c = 0; c < N-1; ++c) {
		for (std::size_t r = 0; r < N-1; ++r) {
			temp[r] += m[c][r] * v[c];
		}
	}
	return temp;
}


/// Static cast each component from T2 to T1.
template <typename T1, typename T2, std::size_t C, std::size_t R>
mat<T1, C, R> staticMatCast(const mat<T2, C, R>& m) {
	mat<T1, C, R> temp;
	for (std::size_t i = 0; i < C; ++i)
		temp[i] = staticVecCast<T1>(m[i]);
	return temp;
}



/// Returns the trace of a matrix (sum of the diagonal elements).
template <typename T, std::size_t N>
T trace(const mat<T, N, N>& m) {
	T temp = m[0][0];
	for (std::size_t i = 1u; i < N; ++i) {
		temp += m[i][i];
	}
	return temp;
}


typedef mat<float, 2, 2> mat2x2;
typedef mat<float, 2, 3> mat2x3;
typedef mat<float, 2, 4> mat2x4;
typedef mat<float, 3, 2> mat3x2;
typedef mat<float, 3, 3> mat3x3;
typedef mat<float, 3, 4> mat3x4;
typedef mat<float, 4, 2> mat4x2;
typedef mat<float, 4, 3> mat4x3;
typedef mat<float, 4, 4> mat4x4;

typedef mat<float, 2, 2> mat2;
typedef mat<float, 3, 3> mat3;
typedef mat<float, 4, 4> mat4;

typedef mat<double, 2, 2> dmat2x2;
typedef mat<double, 2, 3> dmat2x3;
typedef mat<double, 2, 4> dmat2x4;
typedef mat<double, 3, 2> dmat3x2;
typedef mat<double, 3, 3> dmat3x3;
typedef mat<double, 3, 4> dmat3x4;
typedef mat<double, 4, 2> dmat4x2;
typedef mat<double, 4, 3> dmat4x3;
typedef mat<double, 4, 4> dmat4x4;

typedef mat<double, 2, 2> dmat2;
typedef mat<double, 3, 3> dmat3;
typedef mat<double, 4, 4> dmat4;

typedef mat<int, 2, 2> imat2x2;
typedef mat<int, 2, 3> imat2x3;
typedef mat<int, 2, 4> imat2x4;
typedef mat<int, 3, 2> imat3x2;
typedef mat<int, 3, 3> imat3x3;
typedef mat<int, 3, 4> imat3x4;
typedef mat<int, 4, 2> imat4x2;
typedef mat<int, 4, 3> imat4x3;
typedef mat<int, 4, 4> imat4x4;

typedef mat<int, 2, 2> imat2;
typedef mat<int, 3, 3> imat3;
typedef mat<int, 4, 4> imat4;

typedef mat<unsigned int, 2, 2> umat2x2;
typedef mat<unsigned int, 2, 3> umat2x3;
typedef mat<unsigned int, 2, 4> umat2x4;
typedef mat<unsigned int, 3, 2> umat3x2;
typedef mat<unsigned int, 3, 3> umat3x3;
typedef mat<unsigned int, 3, 4> umat3x4;
typedef mat<unsigned int, 4, 2> umat4x2;
typedef mat<unsigned int, 4, 3> umat4x3;
typedef mat<unsigned int, 4, 4> umat4x4;

typedef mat<unsigned int, 2, 2> umat2;
typedef mat<unsigned int, 3, 3> umat3;
typedef mat<unsigned int, 4, 4> umat4;

typedef mat<bool, 2, 2> bmat2x2;
typedef mat<bool, 2, 3> bmat2x3;
typedef mat<bool, 2, 4> bmat2x4;
typedef mat<bool, 3, 2> bmat3x2;
typedef mat<bool, 3, 3> bmat3x3;
typedef mat<bool, 3, 4> bmat3x4;
typedef mat<bool, 4, 2> bmat4x2;
typedef mat<bool, 4, 3> bmat4x3;
typedef mat<bool, 4, 4> bmat4x4;

typedef mat<bool, 2, 2> bmat2;
typedef mat<bool, 3, 3> bmat3;
typedef mat<bool, 4, 4> bmat4;


}

#endif
