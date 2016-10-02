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
/// @param val Value to wrap to the given interval.
/// @param min Start point of the interval (inclusive)
/// @param max End point of the interval (exclusive)
template <
	typename T,
	typename std::enable_if<!std::is_integral<T>::value, int>::type = 0
>
T repeat(T val, T min, T max)
{
	using std::fmod;
	T temp = fmod(val - min, max - min);
	if (temp < static_cast<T>(0)) temp += max - min;
	return temp + min;
}


/// Wraps a value around to the given interval
/// @param val Value to wrap to the given interval.
/// @param min Start point of the interval (inclusive)
/// @param max End point of the interval (exclusive)
template <
	typename T,
	typename std::enable_if<std::is_integral<T>::value, int>::type = 0
>
T repeat(T val, T min, T max)
{
	T temp = (val - min) % (max - min);
	if (temp < static_cast<T>(0)) temp += max - min;
	return temp + min;
}


/// Convert an unsigned integral type TI to floating point type TF by mapping it
/// linearly to range [0.0, 1.0].
/// @tparam TF Must be a floating point type.
/// @tparam TI Must be an unsigned integral type.
template <typename TF, typename TI>
TF unpackUnorm(TI val) {

	static_assert(
		std::is_integral<TI>::value && !std::is_signed<TI>::value,
		"TI must be an unsigned integral type."
	);

	static_assert(
		std::is_floating_point<TF>::value,
		"TF must be a floating point type!"
	);

	return static_cast<TF>(val) / static_cast<TF>(std::numeric_limits<TI>::max());
}


/// Convert a floating point type to an unsigned integral type by first clamping
/// it to range [0.0, 1.0] and then mapping it to the full range of the integer.
/// @tparam TI Must be an unsigned integral type.
/// @tparam TF Must be a floating point type.
template <typename TI, typename TF>
TI packUnorm(TF val) {

	static_assert(
		std::is_integral<TI>::value && !std::is_signed<TI>::value,
		"TI must be an unsigned integral type."
	);

	static_assert(
		std::is_floating_point<TF>::value,
		"TF must be a floating point type!"
	);

	return static_cast<TI>(
		clamp<TF>(val, static_cast<TF>(0.0), static_cast<TF>(1.0)) *
		static_cast<TF>(std::numeric_limits<TI>::max()) +
		static_cast<TF>(0.5)
	);
}


/// Convert a signed integral type TI to floating point type TF by mapping it
/// linearly to range [-1.0, 1.0].
/// @tparam TF Must be a floating point type.
/// @tparam TI Must be a signed integral type.
template <typename TF, typename TI>
TF unpackSnorm(TI val) {

	static_assert(
		std::is_integral<TI>::value && std::is_signed<TI>::value,
		"TI must be a signed integral type."
	);

	static_assert(
		std::is_floating_point<TF>::value,
		"TF must be a floating point type!"
	);

	return clamp(
		static_cast<TF>(val) /
		static_cast<TF>(std::numeric_limits<TI>::max()),
		static_cast<TF>(-1.0), static_cast<TF>(1.0)
	);
}


/// Convert a floating point type to a signed integral type by first clamping
/// it to range [-1.0, 1.0] and then mapping it to the full range of the integer.
/// @tparam TI Must be a signed integral type.
/// @tparam TF Must be a floating point type.
template <typename TI, typename TF>
TI packSnorm(TF val) {

	static_assert(
		std::is_integral<TI>::value && std::is_signed<TI>::value,
		"TI must be a signed integral type."
	);

	static_assert(
		std::is_floating_point<TF>::value,
		"TF must be a floating point type!"
	);

	return static_cast<TI>(
		clamp<TF>(val, static_cast<TF>(-1.0), static_cast<TF>(1.0)) *
		static_cast<TF>(std::numeric_limits<TI>::max()) +
		static_cast<TF>(0.5)
	);
}




}

#endif
