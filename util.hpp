// Copyright Markus Ilmola
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef __gml_util_hpp__
#define __gml_util_hpp__


namespace gml {


/// Converts degrees to radians
template <typename T> 
T radians(T degrees) {
	return degrees * static_cast<T>(0.017453292519943295769236907684886);
}


/// Converts radians to degrees
template <typename T>
T degrees(T radians) {
	return radians * static_cast<T>(57.295779513082320876798154814105);
}


template <typename T>
T clamp(T val, T minVal, T maxVal) {
	return std::min<T>(std::max<T>(val, minVal), maxVal);
}

}

#endif
