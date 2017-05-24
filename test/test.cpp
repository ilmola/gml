// Copyright Markus Ilmola
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

/*
 * This file contains a basic fuzz tester for the library.
 */

#include <sstream>
#include <iomanip>
#include <random>

#include <gml/gml.hpp>

using namespace gml;

template <typename T, int N>
using array_t = T[N];

template <typename T, int N1, int N2>
using array2_t = T[N1][N2];



// Holds context information for a test case.
// Use macro SC to create one
class SourceContext {
public:
	SourceContext(const char* file, int line) : file(file), line(line) { }
	const char* file;
	const int line;
};

#define SC SourceContext(__FILE__, __LINE__)


inline std::ostream& operator<<(std::ostream& os, const SourceContext& sc) {
	os << sc.file << ":" << sc.line;
	return os;
}


// Check if doubles are close enough to pass test.
inline bool fcmp(double a, double b) {
	const double EPSILON = 0.01;
	return std::abs(a - b) <= EPSILON * std::max(std::abs(a), std::abs(b));
}


template <typename T>
inline void EQ(const SourceContext& sc, const T& a, const T& b) {
	if (a != b) { std::cout << sc << ": " << a << " != " << b << "\n"; }
}


inline void EQ(const SourceContext& sc, double a, double b) {
	if (!fcmp(a, b)) {
		std::cout << sc << ": " << a << " != " << b << "\n";
	}
}


template <int N>
void EQ(const SourceContext& sc, const gml::vec<unsigned, N>& a, const gml::vec<unsigned, N>& b) {
	if (a != b) { std::cout << sc << ": " << a << " != " << b << "\n"; }
}


template <int N>
void EQ(const SourceContext& sc, const gml::vec<int, N>& a, const gml::vec<int, N>& b) {
	if (a != b) { std::cout << sc << ": " << a << " != " << b << "\n"; }
}


template <int N>
void EQ(const SourceContext& sc, const gml::vec<double, N>& a, const gml::vec<double, N>& b) {
	for (int i = 0; i < N; ++i) {
		if (!fcmp(a[i], b[i])) {
			std::cout << sc << ": " << a << " != " << b << "\n";
			return;
		}
	}
}


template <int C, int R>
void EQ(const SourceContext& sc, const gml::mat<int, C, R>& a, const gml::mat<int, C, R>& b) {
	if (a != b) { std::cout << sc << ": " << a << " != " << b << "\n"; }
}


template <int C, int R>
void EQ(const SourceContext& sc, const gml::mat<double, C, R>& a, const gml::mat<double, C, R>& b) {
	for (int c = 0; c < C; ++c) {
		for (int r = 0; r < R; ++r) {
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


inline void EQ(const SourceContext& sc, const gml::quaternion<double>& a, const gml::quaternion<double>& b) {
	if (
		!fcmp(a.real, b.real) ||
		!fcmp(a.imag[0], b.imag[0]) ||
		!fcmp(a.imag[1], b.imag[1]) ||
		!fcmp(a.imag[2], b.imag[2])
	) {
		std::cout << sc << ": " << a << " != " << b << "\n";
	}
}


inline double rand(double min, double max) {
	static std::mt19937 generator{};
	std::uniform_real_distribution<double> distribution(min, max);
	return distribution(generator);
}


template <int N>
vec<double, N> randVec(double min, double max) {
	vec<double, N> result{};
	for (auto& value : result) {
		value = rand(min, max);
	}
	return result;
}


template <int C, int R>
mat<double, C, R> randMat(double min, double max) {
	mat<double, C, R> result{};
	for (auto& value : result) {
		value = randVec<R>(min, max);
	}
	return result;
}


inline dquat randQ(double min, double max) {
	return dquat{rand(min, max), randVec<3>(min, max)};
}




int main() {

	const int testRuns = 1000;
	const int min = -20;
	const int max = 20;

	dvec3 temp;
	dmat4 TEMP;
	dquat tempq;

	const double zero = 0.0;
	const double one = 1.0;
	const dvec3 zeros;
	const dvec3 ones(1.0);
	const dmat4 I(1.0);
	const dmat4 ZEROS;
	const dquat iq(1);


	for (int i=0; i<testRuns; i++) {

		//-Test data-----------------------------------------------------------
		const double t = rand(0.0, 1.0);
		const double scalar = rand(min, max);
		const dvec3 v1 = randVec<3>(min, max);
		const dvec3 v2 = randVec<3>(min, max);
		const dvec3 v3 = randVec<3>(min, max);
		const dvec3 v4 = randVec<3>(min, max);
		const dmat4 M1 = randMat<4, 4>(min, max);
		const dmat4 M2 = randMat<4, 4>(min, max);
		const dmat4 M3 = randMat<4, 4>(min, max);
		const dquat q1 = randQ(min, max);
		const dquat q2 = randQ(min, max);
		const dquat q3 = randQ(min, max);

		//-Utils---------------------------------------------------------------

		EQ(SC, radians(degrees(scalar)), scalar );
		EQ(SC, std::sin(radians(90.0)), one );
		EQ(SC, repeat(scalar, scalar, scalar+1.0), scalar);
		EQ(SC, repeat(scalar, scalar-1.0, scalar), scalar-1.0);
		EQ(SC, repeat(scalar, scalar-1.0, scalar+1.0), scalar);


		//-Vectors-------------------------------------------------------------

		EQ(SC, std::is_same<gml::dvec3::value_type, double>::value, true);
		EQ(SC, std::is_same<gml::ivec2::value_type, int>::value, true);

		// Contructors
		EQ(SC, dvec3{}, dvec3{zero});
		EQ(SC, dvec3{scalar}, dvec3{scalar, scalar, scalar});
		EQ(SC, dvec3{v1.data()}, v1);
		EQ(SC, dvec3{v1[0], v1[1], v1[2]}, v1);
		EQ(SC, dvec3{dvec2{v1}, v1[2]}, v1);
		EQ(SC, dvec3{v1}, v1);
		EQ(SC, dvec2{v1}, dvec2{v1, int{2}});

		// Cast
		EQ(SC,
			static_vec_cast<int>(v1),
			ivec3{static_cast<int>(v1[0]), static_cast<int>(v1[1]), static_cast<int>(v1[2])}
		);
		EQ(SC, static_vec_cast<double>(v1), v1);

		// Iteration
		int index = 0;
		for (auto value : v1) {
			EQ(SC, value, v1[index]);
			++index;
		}

		// Operators
		EQ(SC, v1, v1);
		EQ(SC, v1[0], v1.data()[0]);
		EQ(SC, v1 + v2, v2 + v1);
		EQ(SC, v1 + v1, 2.0 * v1);
		EQ(SC, v1 + zero, v1);
		EQ(SC, v1 + scalar, v1 + dvec3{scalar});
		EQ(SC, scalar + v1, dvec3{scalar} + v1);
		EQ(SC, v1 - scalar, v1 - dvec3{scalar});
		EQ(SC, scalar - v1, dvec3{scalar} - v1);
		EQ(SC, v1 - v1, zeros);
		EQ(SC, v1 - zeros, v1);
		EQ(SC, v1 - v2, (-v2) + v1);
		EQ(SC, v1 * v2, v2 * v1);
		EQ(SC, zeros * v1, zeros);
		EQ(SC, scalar * v1, v1 *scalar);
		EQ(SC, v1 * scalar, v1 * dvec3{scalar});
		EQ(SC, v1 / ones, v1);
		EQ(SC, v1 / one, v1);
		EQ(SC, scalar / ones, dvec3{scalar} / ones);
		EQ(SC, -(-v1), v1);

		EQ(SC, temp = v1, v1);
		EQ(SC, temp += v2, v1+v2);
		EQ(SC, temp *= v3, (v1+v2)*v3);

		// Streaming
		std::stringstream vstream;
		vstream << v1;
		vstream >> temp;
		EQ(SC, temp, v1);
		EQ(SC, to_string(v1), vstream.str());

		// vector functions
		EQ(SC, static_cast<int>(v1.size()), 3);
		EQ(SC, dot(v1, v2), dot(v2, v1));
		EQ(SC, dot(v1, v2 + v3), dot(v1, v2) + dot(v1, v3));
		EQ(SC, dot(v1, scalar * v2 + v3), scalar * dot(v1, v2) + dot(v1, v3));
		EQ(SC, perpDot(dvec2{v1}, dvec2{v1}), zero);
		EQ(SC, perpDot(dvec2{v1}, dvec2{v2}), dot(cross(dvec2{v1}), dvec2{v2}));
		EQ(SC, cross(v1, v2) , -cross(v2, v1));
		EQ(SC, cross(v1, cross(v2, v3)), v2 * dot(v1, v3) - v3 * dot(v1, v2));
		EQ(SC, dot(cross(dvec2{v1}), dvec2{v1}), zero);
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
		EQ(SC, pow(sin(v1), dvec3{2.0}) + pow(cos(v1), dvec3{2.0}), ones);
		EQ(SC, sin(asin(sin(v1))), sin(v1));
		EQ(SC, cos(acos(cos(v1))), cos(v1));
		EQ(SC, tan(atan(tan(v1))), tan(v1));
		EQ(SC, distance(zeros, v1), length(v1));
		EQ(SC, angle(v1, v1), zero);
		EQ(SC, project(ones, ones), ones);
		EQ(SC, transform([] (double x) { return x * x; }, v1), v1 * v1);
		EQ(SC, unpackUnorm<double>(gml::uvec3{}), zeros);
		EQ(SC, unpackUnorm<double>(gml::ucvec3{}), zeros);
		EQ(SC, unpackUnorm<double>(gml::uvec3{std::numeric_limits<unsigned>::max()}), ones);
		EQ(SC, unpackUnorm<double>(gml::ucvec3{std::numeric_limits<unsigned char>::max()}), ones);
		EQ(SC, packUnorm<unsigned>(zeros), gml::uvec3{});
		EQ(SC, packUnorm<unsigned>(ones), gml::uvec3{std::numeric_limits<unsigned>::max()});
		EQ(SC, unpackUnorm<double>(packUnorm<unsigned>(normalize(v1))), gml::clamp(normalize(v1), zeros, ones));
		EQ(SC, unpackSnorm<double>(gml::ivec3{}), zeros);
		EQ(SC, unpackSnorm<double>(gml::ivec3{std::numeric_limits<int>::max()}), ones);
		EQ(SC, unpackSnorm<double>(gml::ivec3{std::numeric_limits<int>::min()}), -ones);
		EQ(SC, unpackSnorm<double>(packSnorm<int>(normalize(v1))), normalize(v1));
		EQ(SC, packSnorm<int>(zeros), gml::ivec3{});


		//-Matrices-------------------------------------------------------------

		EQ(SC, std::is_same<gml::dmat3::value_type, double>::value, true);
		EQ(SC, std::is_same<gml::imat2::value_type, int>::value, true);

		// Constructors
		EQ(SC, dmat4{zero}, ZEROS);
		EQ(SC, dmat4{M1.data(), true}, M1);
		EQ(SC, dmat4{M1}, M1);
		EQ(SC, dmat4{M1[0], M1[1], M1[2], M1[3]}, M1);
		EQ(SC, dmat3{v1}, dmat3{v1, v1, v1});
		EQ(SC, dmat3{M1}, dmat3{M1, int{3}, int{3}});
		EQ(SC, dmat4{dmat3{I}, 1.0}, I);
		EQ(SC, dmat2{dmat4x2{dmat2x4{M1}}}, dmat2{M1});

		// Casts
		EQ(SC, static_mat_cast<double>(M1), M1);
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
		int cIndex = 0;
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
		EQ(SC, 2.0 * M1, M1 + M1);
		EQ(SC, scalar * M1, M1 * scalar);

		EQ(SC, TEMP = M1, M1);
		EQ(SC, TEMP += M2, M1 + M2);
		EQ(SC, TEMP *= M3, (M1 + M2) * M3);

		// Streaming
		std::stringstream mstream;
		mstream << M1;
		mstream >> TEMP;
		EQ(SC, TEMP, M1);
		EQ(SC, to_string(M1), mstream.str());

		EQ(SC, static_cast<int>(M1.size()), 4 );

		// Matrix functions
		EQ(SC, M1, transpose(transpose(M1)));
		EQ(SC, dmat4{M1.data(), false}, transpose(M1));
		EQ(SC, determinant(I), one);
		EQ(SC, determinant(M1), determinant(transpose(M1)));
		EQ(SC, determinant(M1 * M2), determinant(M1) * determinant(M2));
		EQ(SC, inverse(I), I);
		//if (determinant(M1) != 0.0) EQ(SC, inverse(inverse(M1)), M1 );
		EQ(SC, outerProduct(v1, v2), mat<double, 1, 3>{v1} * transpose(mat<double, 1, 3>{v2}));
		EQ(SC, matrixCompMult(M1, M2), dmat4{M1[0] * M2[0], M1[1] * M2[1], M1[2] * M2[2], M1[3] * M2[3]});
		EQ(SC, trace(M1), M1[0][0] + M1[1][1] + M1[2][2] + M1[3][3]);
		EQ(SC, trace(M1), trace(transpose(M1)));

		// Matrix and vector
		EQ(SC, I * dvec4{v1, scalar}, dvec4{v1, scalar});
		EQ(SC, (M1 * M2) * dvec4{v1, scalar}, M1 * (M2 * dvec4{v1, scalar}));
		EQ(SC, transform(M1, v1), dvec3{M1 * dvec4{v1, one}});

		// Transformation matrices
		EQ(SC, translate(zeros), I);
		EQ(SC, inverse(translate(v1)), translate(-v1));
		EQ(SC, rotate(zeros), I);
		EQ(SC, scale(ones), I);
		EQ(SC, transform(translate(v1), v2), v1 + v2);
		EQ(SC, transform(scale(v1), v2), v1 * v2);
		EQ(SC, transform(rotate(radians(90.0)), dvec2{v1}), cross(dvec2{v1}));
		EQ(SC, rotate(dvec3{zero, zero, scalar}), rotate(scalar, dvec3{zero, zero, one}));
		EQ(SC, rotate(dvec3{zero, scalar, zero}), rotate(scalar, dvec3{zero, one, zero}));
		EQ(SC, rotate(dvec3{scalar, zero, zero}), rotate(scalar, dvec3{one, zero, zero}));
		EQ(SC, ortho(-one, one, -one, one, one, -one), I);
		EQ(SC, lookAt(zeros, dvec3{zero, zero, -one}, dvec3{zero, one, zero}), I);

		const auto tempProj = perspective(one, one, one, 2.0);
		EQ(SC, project(dvec3{zero, zero, -one}, I, tempProj, ivec2{-1, -1}, ivec2{2, 2}), zeros);
		EQ(SC, unProject(zeros, I, tempProj, ivec2{-1, -1}, ivec2{2, 2}), dvec3{zero, zero, -one});


		//-Quaternions----------------------------------------------------------

		EQ(SC, std::is_same<gml::dquat::value_type, double>::value, true);
		EQ(SC, std::is_same<gml::iquat::value_type, int>::value, true);

		// Constructors
		EQ(SC, dquat{q1.real}.real, q1.real);
		EQ(SC, dquat{q1.imag}.imag, q1.imag);
		EQ(SC, dquat{q1.real, q1.imag}, q1);
		EQ(SC, dquat{q1}, q1);

		// Casts
		EQ(SC, static_quaternion_cast<double>(q1), q1);
		EQ(SC,
			static_quaternion_cast<int>(q1),
			iquat{static_cast<int>(q1.real), static_vec_cast<int>(q1.imag)}
		);

		// Quaternion operators
		EQ(SC, -(-q1), q1);
		EQ(SC, (-q1) + q1, dquat{0.0});
		EQ(SC, q1 + q2, q2 + q1);
		EQ(SC, q1 + v1, q1 + dquat{v1});
		EQ(SC, v1 + q1, dquat{v1} + q1);
		EQ(SC, scalar + q1, dquat{scalar} + q1);
		EQ(SC, q1 + scalar, q1 + dquat{scalar});
		EQ(SC, q1 - q1, dquat{0.0});
		EQ(SC, q1 - v1, q1 - dquat{v1});
		EQ(SC, v1 - q1, dquat{v1} - q1);
		EQ(SC, q1 - scalar, q1 - dquat{scalar});
		EQ(SC, scalar - q1, dquat{scalar} - q1);
		EQ(SC, iq * q1, q1);
		EQ(SC, q1 * iq, q1);
		EQ(SC, scalar * q1, dquat{scalar} * q1);
		EQ(SC, q1 * scalar, q1 * dquat{scalar});
		EQ(SC, q1 * v1, q1 * dquat{v1});
		EQ(SC, v1 * q1, dquat{v1} * q1);

		EQ(SC, (tempq = q1), q1);
		EQ(SC, (tempq += q2), q1 + q2);
		EQ(SC, (tempq *= q3), (q1 + q2) * q3);

		// Streaming
		std::stringstream qstream;
		qstream << q1;
		qstream >> tempq;
		EQ(SC, tempq, q1);
		EQ(SC, to_string(q1), qstream.str());

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
		EQ(SC, qrotate(dvec3{scalar, zero, zero}), qrotate(scalar, dvec3{one, zero, zero}));
		EQ(SC, qrotate(dvec3{zero, scalar, zero}), qrotate(scalar, dvec3{zero, one, zero}));
		EQ(SC, qrotate(dvec3{zero, zero, scalar}), qrotate(scalar, dvec3{zero, zero, one}));
		EQ(SC, rotate(iq), I);
		EQ(SC, rotate(qrotate(v1)), rotate(v1));
		EQ(SC, rotate(qrotate(rotate(qrotate(v1)))), rotate(v1));
		EQ(SC, transform(qrotate(zeros), v1), v1);
		EQ(SC, transform(qrotate(dvec3{radians(180.0), zero, zero}), v1), dvec3{v1[0], -v1[1], -v1[2]});
		EQ(SC, transform(normalize(q1), v1), (normalize(q1) * v1 * conj(normalize(q1))).imag);


		//-Splines--------------------------------------------------------------

		auto B3 = [] (gml::dvec3 P0, gml::dvec3 P1, gml::dvec3 P2, gml::dvec3 P3, double t)
		{
			return pow(1.0-t,3.0)*P0+3.0*pow(1.0-t,2.0)*t*P1+3.0*(1.0-t)*t*t*P2+pow(t,3.0)*P3;
		};

		auto dB3 = [] (gml::dvec3 P0, gml::dvec3 P1, gml::dvec3 P2, gml::dvec3 P3, double t)
		{
			return 3.0*pow(1.0-t,2.0)*(P1-P0)+6.0*(1.0-t)*t*(P2-P1)+3.0*t*t*(P3-P2);
		};

		auto ddB3 = [] (gml::dvec3 P0, gml::dvec3 P1, gml::dvec3 P2, gml::dvec3 P3, double t)
		{
			return 6.0*(1.0-t)*(P2-2.0*P1+P0)+6.0*t*(P3-2.0*P2+P1);
		};

		EQ(SC, bezier(array_t<dvec3, 1>{v1}, t), v1);
		EQ(SC, bezier(array_t<dvec3, 4>{v1, v2, v3, v4}, zero), v1);
		EQ(SC, bezier(array_t<dvec3, 4>{v1, v2, v3, v4}, one), v4);
		EQ(SC, bezier(array_t<dvec3, 2>{v1, v2}, 0.5), 0.5 * (v1 + v2));
		EQ(SC, bezier(array_t<dvec3, 2>{v1, v2}, t), mix(v1, v2, t));
		EQ(SC, bezier(array_t<dvec3, 4>{v1, v2, v3, v4}, t), B3(v1, v2, v3, v4, t));

		EQ(SC, bezier2(array2_t<dvec3, 1, 1>{{v1}}, gml::dvec2{t, t}), v1);
		EQ(SC, bezier2(array2_t<dvec3, 1, 4>{{v1, v2, v3, v4}}, gml::dvec2{t, 0.0}), B3(v1, v2, v3, v4, t));
		EQ(SC, bezier2(array2_t<dvec3, 4, 1>{{v1}, {v2}, {v3}, {v4}}, gml::dvec2{0.0, t}), B3(v1, v2, v3, v4, t));
		EQ(SC, bezier2(array2_t<dvec3, 2, 2>{{v1, v2}, {v3, v4}}, gml::dvec2{0.0, 0.0}), v1);
		EQ(SC, bezier2(array2_t<dvec3, 2, 2>{{v1, v2}, {v3, v4}}, gml::dvec2{1.0, 0.0}), v2);
		EQ(SC, bezier2(array2_t<dvec3, 2, 2>{{v1, v2}, {v3, v4}}, gml::dvec2{0.0, 1.0}), v3);
		EQ(SC, bezier2(array2_t<dvec3, 2, 2>{{v1, v2}, {v3, v4}}, gml::dvec2{1.0, 1.0}), v4);
		EQ(SC, bezier2(array2_t<dvec3, 2, 2>{{v1, v2}, {v3, v4}}, gml::dvec2{0.5, 0.5}), 0.25 * (v1 + v2 + v3 + v4));

		EQ(SC, bezierDerivative<1>(array_t<dvec3, 1>{v1}, t), zeros);
		EQ(SC, bezierDerivative<1>(array_t<dvec3, 2>{v1, v2}, t), v2 - v1);
		EQ(SC, bezierDerivative<2>(array_t<dvec3, 2>{v1, v2}, t), zeros);
		EQ(SC, bezierDerivative<3>(array_t<dvec3, 2>{v1, v2}, t), zeros);
		EQ(SC, bezierDerivative<1>(array_t<dvec3, 4>{v1, v2, v3, v4}, t), dB3(v1, v2, v3, v4, t));
		EQ(SC, bezierDerivative<2>(array_t<dvec3, 4>{v1, v2, v3, v4}, t), ddB3(v1, v2, v3, v4, t));

		const auto J = bezier2Jacobian<1>(
			array2_t<dvec3, 2, 2>{
				{gml::dvec3{0.0, 0.0, 0.0}, gml::dvec3{1.0, 0.0, 0.0}},
				{gml::dvec3{0.0, 1.0, 0.0}, gml::dvec3{1.0, 1.0, 0.0}}
			},
			gml::dvec2{t, 1.0 - t}
		);
		EQ(SC, cross(J[0], J[1]), gml::dvec3{0.0, 0.0, 1.0});


		//-Texture--------------------------------------------------------------

		EQ(SC, texelCenter<double>(gml::ivec2{0, 0}, gml::zvec2{1u, 2u}), gml::dvec2{0.5, 0.25});

		EQ(SC, nearestTexel<int>(ones, gml::zvec3{1u, 2u, 3u}), gml::ivec3{1,2,3});

		const auto size = abs(static_vec_cast<int>(v2)) + 1;
		EQ(SC,
			nearestTexel<int>(texelCenter<double>(static_vec_cast<int>(v1), size), size),
			static_vec_cast<int>(v1)
		);

		EQ(SC, faceAt(gml::dvec3{ 1.0,  0.0,  0.0} + 0.01 * v1), 0u);
		EQ(SC, faceAt(gml::dvec3{-1.0,  0.0,  0.0} + 0.01 * v1), 1u);
		EQ(SC, faceAt(gml::dvec3{ 0.0,  1.0,  0.0} + 0.01 * v1), 2u);
		EQ(SC, faceAt(gml::dvec3{ 0.0, -1.0,  0.0} + 0.01 * v1), 3u);
		EQ(SC, faceAt(gml::dvec3{ 0.0,  0.0,  1.0} + 0.01 * v1), 4u);
		EQ(SC, faceAt(gml::dvec3{ 0.0,  0.0, -1.0} + 0.01 * v1), 5u);

		EQ(SC, cubeTexCoord(v1).first, cubeTexCoord(normalize(v1)).first);
		EQ(SC, cubeTexCoord(v1).second, cubeTexCoord(normalize(v1)).second);

		const auto cubeCoord = cubeTexCoord(v1);
		EQ(SC, normalize(cubeDirection(cubeCoord.first, cubeCoord.second)), normalize(v1));



		//-Intersect------------------------------------------------------------
		{
			const ivec2 mpos{static_vec_cast<int>(v1)};
			const auto proj = ortho2D(-1.0, 1.0, -1.0, 1.0);
			const auto ray = pickRay(mpos, I, proj, mpos, ivec2{2});
			EQ(SC, std::get<0>(ray), dvec3{-0.5, -0.5, 1.0});
			EQ(SC, std::get<1>(ray), dvec3{0.0, 0.0, -1.0});
		}

		{
			const auto intersection = intersectRayPlane(v1, dvec3{0.0, 0.0, -1.0}, v2, dvec3{0.0, 0.0, 1.0});
			EQ(SC, std::get<0>(intersection), true);
			EQ(SC, std::get<1>(intersection), v1[2] - v2[2]);
		}

		{
			const auto intersection = intersectRayPlane(v1, dvec3{1.0, 0.0, 0.0}, v2, dvec3{0.0, 0.0, 1.0});
			EQ(SC, std::get<0>(intersection), false);
		}

		{
			const auto intersection = intersectRaySphere(
				dvec3{0.0, 0.0, v1[0]}, dvec3{0.0, 0.0, -1.0}, dvec3{0.0, 0.0, v2[0]}, std::abs(scalar)
			);
			EQ(SC, std::get<0>(intersection), true);
			EQ(SC, std::get<1>(intersection), v1[0] - (v2[0] + std::abs(scalar)));
			EQ(SC, std::get<2>(intersection), v1[0] - (v2[0] - std::abs(scalar)));
		}
	}

	std::cout << "All tests done.\n";

	return 0;
}
