// Copyright 2015 Markus Ilmola
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef UUID_BD878441A4634595B03E6859030E9A7D
#define UUID_BD878441A4634595B03E6859030E9A7D

#include <algorithm>


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


/// Wraps a value around to the given interval
/// @param min start point of the interval inclusive
/// @param max end point of the interval exclusive
template <
	typename T,
	typename std::enable_if<!std::is_integral<T>::value, int>::type = 0
>
T repeat(T val, T min, T max)
{
	using std::fmod;
	T temp = fmod(val - min, max - min);
	if (temp < T{0}) temp += max - min;
	return temp + min;
}


/// Wraps a value around to the given interval
/// @param min start point of the interval inclusive
/// @param max end point of the interval exclusive
template <
	typename T,
	typename std::enable_if<std::is_integral<T>::value, int>::type = 0
>
T repeat(T val, T min, T max)
{
	T temp = (val - min) % (max - min);
	if (temp < T{0}) temp += max - min;
	return temp + min;
}



}

#endif
