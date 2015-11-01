// Copyright Markus Ilmola
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

/*
 * This file contains a basic fuzz tester for the library.
 */

#include <sstream>
#include <random>

#include "gml.hpp"

using namespace gml;


// Holds context information for a test case.
// Use macro SC to create one
class SourceContext {
public:
	SourceContext(const char* file, std::size_t line) : file(file), line(line) { }
	const char* file;
	const std::size_t line;
};

#define SC SourceContext(__FILE__, __LINE__)


inline std::ostream& operator<<(std::ostream& os, const SourceContext& sc) {
	os << sc.file << ":" << sc.line;
	return os;
}


// Check if floats are close enough to pass test.
inline bool fcmp(float a, float b) {
	const float EPSILON = 0.01f;
	return std::abs(a - b) <= EPSILON * std::max(std::abs(a), std::abs(b));
}


inline void EQ(const SourceContext& sc, int a, int b) {
	if (a != b) { std::cout << sc << ": " << a << " != " << b << "\n"; }
}


inline void EQ(const SourceContext& sc, float a, float b) {
	if (!fcmp(a, b)) {
		std::cout << sc << ": " << a << " != " << b << "\n";
	}
}


template <std::size_t N>
void EQ(const SourceContext& sc, const gml::vec<int, N>& a, const gml::vec<int, N>& b) {
	if (a != b) { std::cout << sc << ": " << a << " != " << b << "\n"; }
}


template <std::size_t N>
void EQ(const SourceContext& sc, const gml::vec<float, N>& a, const gml::vec<float, N>& b) {
	for (std::size_t i = 0; i < N; ++i) {
		if (!fcmp(a[i], b[i])) {
			std::cout << sc << ": " << a << " != " << b << "\n";
			return;
		}
	}
}


template <std::size_t C, std::size_t R>
void EQ(const SourceContext& sc, const gml::mat<int, C, R>& a, const gml::mat<int, C, R>& b) {
	if (a != b) { std::cout << sc << ": " << a << " != " << b << "\n"; }
}


template <std::size_t C, std::size_t R>
void EQ(const SourceContext& sc, const gml::mat<float, C, R>& a, const gml::mat<float, C, R>& b) {
	for (std::size_t c = 0; c < C; ++c) {
		for (std::size_t r = 0; r < R; ++r) {
			if (!fcmp(a[c][r], b[c][r])) {
				std::cout << sc << ": " << a << " != " << b << "\n";
				return;
			}
		}

	}
}


inline void EQ(const SourceContext& sc, const gml::quaternion<int>& a, const gml::quaternion<int>& b) {
	if (a != b) { std::cout << sc << ": " << a << " != " << b << "\n"; }
}


inline void EQ(const SourceContext& sc, const gml::quaternion<float>& a, const gml::quaternion<float>& b) {
	if (
		!fcmp(a.real, b.real) ||
		!fcmp(a.imag[0], b.imag[0]) ||
		!fcmp(a.imag[1], b.imag[1]) ||
		!fcmp(a.imag[2], b.imag[2])
	) {
		std::cout << sc << ": " << a << " != " << b << "\n";
	}
}


inline float rand(float min, float max) {
	static std::mt19937 generator{};
	std::uniform_real_distribution<float> distribution(min, max);
	return distribution(generator);
}


template <std::size_t N>
vec<float, N> randVec(float min, float max) {
	vec<float, N> result{};
	for (auto& value : result) {
		value = rand(min, max);
	}
	return result;
}


template <std::size_t C, std::size_t R>
mat<float, C, R> randMat(float min, float max) {
	mat<float, C, R> result{};
	for (auto& value : result) {
		value = randVec<R>(min, max);
	}
	return result;
}


inline quat randQ(float min, float max) {
	return quat{rand(min, max), randVec<3>(min, max)};
}



int main() {

	const size_t testRuns = 1000;
	const int min = -20;
	const int max = 20;

	vec3 temp;
	mat4 TEMP;
	quat tempq;

	const float zero = 0.0f;
	const float one = 1.0f;
	const vec3 zeros;
	const vec3 ones(1.0f);
	const mat4 I(1.0f);
	const mat4 ZEROS;
	const quat iq(1);

	for (size_t i=0; i<testRuns; i++) {

		//-Test data-----------------------------------------------------------
		const float scalar = rand(min, max);
		const vec3 v1 = randVec<3>(min, max);
		const vec3 v2 = randVec<3>(min, max);
		const vec3 v3 = randVec<3>(min, max);
		const mat4 M1 = randMat<4, 4>(min, max);
		const mat4 M2 = randMat<4, 4>(min, max);
		const mat4 M3 = randMat<4, 4>(min, max);
		const quat q1 = randQ(min, max);
		const quat q2 = randQ(min, max);
		const quat q3 = randQ(min, max);


		//-Utils---------------------------------------------------------------

		EQ(SC, radians(degrees(scalar)), scalar );
		EQ(SC, std::sin(radians(90.0f)), one );


		//-Vectors-------------------------------------------------------------

		// Contructors
		EQ(SC, vec3{}, vec3{zero});
		EQ(SC, vec3{scalar}, vec3{scalar, scalar, scalar});
		EQ(SC, vec3{v1.data()}, v1);
		EQ(SC, vec3{v1[0], v1[1], v1[2]}, v1);
		EQ(SC, vec3{vec2{v1}, v1[2]}, v1);
		EQ(SC, vec3{v1}, v1);
		EQ(SC, vec3{v1, v2}, v2 - v1);
		EQ(SC, vec2{v1}, vec2{v1, std::size_t{2}});

		// Cast
		EQ(SC,
			static_vec_cast<int>(v1),
			ivec3{static_cast<int>(v1[0]), static_cast<int>(v1[1]), static_cast<int>(v1[2])}
		);
		EQ(SC, static_vec_cast<float>(static_vec_cast<double>(v1)), v1);

		// Iteration
		std::size_t index = 0;
		for (auto value : v1) {
			EQ(SC, value, v1[index]);
			++index;
		}

		// Operators
		EQ(SC, v1, v1);
		EQ(SC, v1[0], v1.data()[0]);
		EQ(SC, v1 + v2, v2 + v1);
		EQ(SC, v1 + v1, 2.0f * v1);
		EQ(SC, v1 + zero, v1);
		EQ(SC, v1 + scalar, v1 + vec3{scalar});
		EQ(SC, scalar + v1, vec3{scalar} + v1);
		EQ(SC, v1 - scalar, v1 - vec3{scalar});
		EQ(SC, scalar - v1, vec3{scalar} - v1);
		EQ(SC, v1 - v1, zeros);
		EQ(SC, v1 - zeros, v1);
		EQ(SC, v1 - v2, (-v2) + v1);
		EQ(SC, v1 * v2, v2 * v1);
		EQ(SC, zeros * v1, zeros);
		EQ(SC, scalar * v1, v1 *scalar);
		EQ(SC, v1 * scalar, v1 * vec3{scalar});
		EQ(SC, v1 / ones, v1);
		EQ(SC, v1 / one, v1);
		EQ(SC, scalar / ones, vec3{scalar} / ones);
		EQ(SC, -(-v1), v1);

		EQ(SC, temp = v1, v1);
		EQ(SC, temp += v2, v1+v2);
		EQ(SC, temp *= v3, (v1+v2)*v3);

		// Streaming
		std::stringstream vstream;
		vstream << v1;
		vstream >> temp;
		EQ(SC, temp, v1);

		// vector functions
		EQ(SC, static_cast<int>(v1.size()), 3);
		EQ(SC, dot(v1, v2), dot(v2, v1));
		EQ(SC, dot(v1, v2 + v3), dot(v1, v2) + dot(v1, v3));
		EQ(SC, dot(v1, scalar * v2 + v3), scalar * dot(v1, v2) + dot(v1, v3));
		EQ(SC, perpDot(vec2{v1}, vec2{v1}), zero);
		EQ(SC, perpDot(vec2{v1}, vec2{v2}), dot(cross(vec2{v1}), vec2{v2}));
		EQ(SC, cross(v1, v2) , -cross(v2, v1));
		EQ(SC, cross(v1, cross(v2, v3)), v2 * dot(v1, v3) - v3 * dot(v1, v2));
		EQ(SC, dot(cross(vec2{v1}), vec2{v1}), zero);
		EQ(SC, length(zeros), zero);
		EQ(SC, length(normalize(ones)), one);
		EQ(SC, area(v1, v1, v1), zero);
		EQ(SC, mix(v1, v2, one), v2);
		EQ(SC, mix(v1, v2, zero), v1);
		EQ(SC, gml::min(zeros, ones), zeros);
		EQ(SC, gml::max(zeros, ones), ones);
		EQ(SC, gml::min(clamp(v1, zeros, ones), zeros), zeros);
		EQ(SC, gml::max(clamp(v1, zeros, ones), ones), ones);
		EQ(SC, normal(v1, v2, v3), -normal(v3, v2, v1));
		EQ(SC, pow(sin(v1), vec3{2.0f}) + pow(cos(v1), vec3{2.0f}), ones);
		EQ(SC, sin(asin(sin(v1))), sin(v1));
		EQ(SC, cos(acos(cos(v1))), cos(v1));
		EQ(SC, tan(atan(tan(v1))), tan(v1));
		EQ(SC, distance(zeros, v1), length(v1));
		EQ(SC, angle(v1, v1), zero);
		EQ(SC, project(ones, ones), ones);


		//-Matrices-------------------------------------------------------------

		// Constructors
		EQ(SC, mat4{zero}, ZEROS);
		EQ(SC, mat4{M1.data(), true}, M1);
		EQ(SC, mat4{M1}, M1);
		EQ(SC, mat4{M1[0], M1[1], M1[2], M1[3]}, M1);
		EQ(SC, mat3{v1}, mat3{v1, v1, v1});
		EQ(SC, mat3{M1}, mat3{M1, std::size_t{3}, std::size_t{3}});
		EQ(SC, mat4{mat3{I}, 1.0f}, I);
		EQ(SC, mat2{mat4x2{mat2x4{M1}}}, mat2{M1});

		// Casts
		EQ(SC, static_mat_cast<float>(static_mat_cast<double>(M1)), M1);
		EQ(SC,
			static_mat_cast<int>(M1),
			imat4{
				static_vec_cast<int>(M1[0]),
				static_vec_cast<int>(M1[1]),
				static_vec_cast<int>(M1[2]),
				static_vec_cast<int>(M1[3])
			}
		);

		// Iteration
		std::size_t cIndex = 0;
		for (const auto& column : M1) {
			EQ(SC, column, M1[cIndex]);
			++cIndex;
		}

		// Matrix operators
		EQ(SC, M1, M1);
		EQ(SC, M1[0][0], M1.data()[0]);
		EQ(SC, M1 + M2, M2 + M1);
		EQ(SC, M1 + ZEROS, M1);
		EQ(SC, M1 - M1, ZEROS);
		EQ(SC, (M1 * M2) * M3, M1 * (M2 * M3));
		EQ(SC, I * M1, M1);
		EQ(SC, M1 * I, M1);
		EQ(SC, ZEROS * M1, ZEROS);
		EQ(SC, 2.0f * M1, M1 + M1);
		EQ(SC, scalar * M1, M1 * scalar);

		EQ(SC, TEMP = M1, M1);
		EQ(SC, TEMP += M2, M1 + M2);
		EQ(SC, TEMP *= M3, (M1 + M2) * M3);

		// Streaming
		std::stringstream mstream;
		mstream << M1;
		mstream >> TEMP;
		EQ(SC, TEMP, M1);

		EQ(SC, static_cast<int>(M1.size()), 4 );

		// Matrix functions
		EQ(SC, M1, transpose(transpose(M1)));
		EQ(SC, mat4{M1.data(), false}, transpose(M1));
		EQ(SC, determinant(I), one);
		EQ(SC, determinant(M1), determinant(transpose(M1)));
		EQ(SC, determinant(M1 * M2), determinant(M1) * determinant(M2));
		EQ(SC, inverse(I), I);
		//if (determinant(M1) != 0.0f) EQ(SC, inverse(inverse(M1)), M1 );
		EQ(SC, outerProduct(v1, v2), mat<float, 1, 3>{v1} * transpose(mat<float, 1, 3>{v2}));
		EQ(SC, matrixCompMult(M1, M2), mat4{M1[0] * M2[0], M1[1] * M2[1], M1[2] * M2[2], M1[3] * M2[3]});

		// Matrix and vector
		EQ(SC, I * vec4{v1, scalar}, vec4{v1, scalar});
		EQ(SC, (M1 * M2) * vec4{v1, scalar}, M1 * (M2 * vec4{v1, scalar}));
		EQ(SC, transform(M1, v1), vec3{M1 * vec4{v1, one}});

		// Transformation matrices
		EQ(SC, translate(zeros), I);
		EQ(SC, inverse(translate(v1)), translate(-v1));
		EQ(SC, rotate(zeros), I);
		EQ(SC, scale(ones), I);
		EQ(SC, transform(translate(v1), v2), v1 + v2);
		EQ(SC, transform(scale(v1), v2), v1 * v2);
		EQ(SC, transform(rotate(radians(90.0f)), vec2{v1}), cross(vec2{v1}));
		EQ(SC, rotate(vec3{zero, zero, scalar}), rotate(scalar, vec3{zero, zero, one}));
		EQ(SC, rotate(vec3{zero, scalar, zero}), rotate(scalar, vec3{zero, one, zero}));
		EQ(SC, rotate(vec3{scalar, zero, zero}), rotate(scalar, vec3{one, zero, zero}));
		EQ(SC, ortho(-one, one, -one, one, one, -one), I);
		EQ(SC, lookAt(zeros, vec3{zero, zero, -one}, vec3{zero, one, zero}), I);
		EQ(SC, project(vec3{zero, zero, -one}, I, perspective(one, one, one, 2.0f), ivec4{-1, -1, 2, 2}), zeros);


		//-Quaternions----------------------------------------------------------

		// Constructors
		EQ(SC, quat{q1.real}.real, q1.real);
		EQ(SC, quat{q1.imag}.imag, q1.imag);
		EQ(SC, quat{q1.real, q1.imag}, q1);
		EQ(SC, quat{q1}, q1);

		// Casts
		EQ(SC, static_quaternion_cast<float>(static_quaternion_cast<double>(q1)), q1);
		EQ(SC,
			static_quaternion_cast<int>(q1),
			iquat{static_cast<int>(q1.real), static_vec_cast<int>(q1.imag)}
		);

		// Quaternion operators
		EQ(SC, -(-q1), q1);
		EQ(SC, (-q1) + q1, quat{0.0f});
		EQ(SC, q1 + q2, q2 + q1);
		EQ(SC, q1 + v1, q1 + quat{v1});
		EQ(SC, v1 + q1, quat{v1} + q1);
		EQ(SC, scalar + q1, quat{scalar} + q1);
		EQ(SC, q1 + scalar, q1 + quat{scalar});
		EQ(SC, q1 - q1, quat{0.0f});
		EQ(SC, q1 - v1, q1 - quat{v1});
		EQ(SC, v1 - q1, quat{v1} - q1);
		EQ(SC, q1 - scalar, q1 - quat{scalar});
		EQ(SC, scalar - q1, quat{scalar} - q1);
		EQ(SC, iq * q1, q1);
		EQ(SC, q1 * iq, q1);
		EQ(SC, scalar * q1, quat{scalar} * q1);
		EQ(SC, q1 * scalar, q1 * quat{scalar});
		EQ(SC, q1 * v1, q1 * quat{v1});
		EQ(SC, v1 * q1, quat{v1} * q1);

		EQ(SC, (tempq = q1), q1);
		EQ(SC, (tempq += q2), q1 + q2);
		EQ(SC, (tempq *= q3), (q1 + q2) * q3);

		// Streaming
		std::stringstream qstream;
		qstream << q1;
		qstream >> tempq;
		EQ(SC, tempq, q1);

		// Quaternion functions
		EQ(SC, conj(q1 * q2), conj(q2) * conj(q1));
		EQ(SC, conj(conj(q1)), q1);
		EQ(SC, mix(q1, q2, zero), q1);
		EQ(SC, mix(q1, q2, one), q2);
		EQ(SC, inverse(iq), iq);
		EQ(SC, inverse(inverse(q1)), q1);
		EQ(SC, abs(iq), one);

		// Transformation
		EQ(SC, qrotate(zeros), iq);
		EQ(SC, qrotate(vec3{scalar, zero, zero}), qrotate(scalar, vec3{one, zero, zero}));
		EQ(SC, qrotate(vec3{zero, scalar, zero}), qrotate(scalar, vec3{zero, one, zero}));
		EQ(SC, qrotate(vec3{zero, zero, scalar}), qrotate(scalar, vec3{zero, zero, one}));
		EQ(SC, transform(qrotate(zeros), v1), v1);
		EQ(SC, transform(qrotate(vec3{radians(180.0f), zero, zero}), v1), vec3{v1[0], -v1[1], -v1[2]});
		EQ(SC, transform(normalize(q1), v1), (normalize(q1) * v1 * conj(normalize(q1))).imag);

	}

	std::cout << "All tests done.\n";

	return 0;
}
