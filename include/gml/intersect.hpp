// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef UUID_72960EDDD32341DBBCADBFA777E98A81
#define UUID_72960EDDD32341DBBCADBFA777E98A81

#include <tuple>

#include "vec.hpp"


namespace gml {


/// Computes the intersection of a ray and a N-plane.
/// A plane if N == 3, a line if N == 2.
/// @param origin Origin of the ray
/// @param direction Unit length direction of the ray.
/// @param center Any point on the plane
/// @param normal Unit length normal vector of the plane
/// @return Returns a tuple consisting of a bool denoting whether there was an
/// intersection (true) or not (false, when the ray is parallel to the plane)
/// and a scalar denoting the distance to the intersection point (positive or
/// negative depending if the point is in front of or behind the origin).
template <typename T, std::size_t N>
std::tuple<bool, T> intersectRayPlane(
	const vec<T, N>& origin, const vec<T, N>& direction,
	const vec<T, N>& center, const vec<T, N>& normal
) {
	using std::abs;

	const T divisor = dot(direction, normal);

	if (abs(divisor) < std::numeric_limits<T>::epsilon()) {
		return std::make_tuple(false, static_cast<T>(0));
	}

	const T d = dot(center, normal);

	return std::make_tuple(true, (d - dot(origin, normal)) / divisor);
}



/// Computes the intersection of a ray and N-sphere.
/// A sphere if N == 3, circle if N == 2.
/// @param origin Origin of the ray
/// @param direction Unit length direction of the ray.
/// @param center Position of the center of the sphere.
/// @param radius Radius of the sphere.
/// @return Returns a tuple consisting of a bool denoting whether there was an
/// intersection (true) or not (false) and two scalars denoting the distances to
/// the front and back intersection points (positive or negative depending if
/// the point is in front of or behind the origin).
template <typename T, std::size_t N>
std::tuple<bool, T, T> intersectRaySphere(
	const vec<T, N>& origin, const vec<T, N>& direction,
	const vec<T, N>& center, T radius
) {
	using std::sqrt;

	const vec<T, N> L = center - origin;
	const T tca = dot(L, direction);
	const T d2 = dot(L, L) - tca * tca;

	const T radius2 = radius * radius;
	if (d2 > radius2) {
		return std::make_tuple(false, static_cast<T>(0), static_cast<T>(0));
	}

	const T thc = sqrt(radius2 - d2);

	return std::make_tuple(true, tca - thc, tca + thc);
}


}


#endif

