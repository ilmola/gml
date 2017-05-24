# GML #

GML (recursive acronym for "GML math library") is a GLSL like minimalist vector, matrix and quaternion math library for C++11.

It's API is modeled closely after GLSL and is thus easy to use in OpenGL programs. It provides most of the GLSL functionality plus some extra.

This library should not be confused with [glm](http://glm.g-truc.net) or with [Boost qvm](http://www.revergestudios.com/boost-qvm/).


## Setup ##

This is a header only template library.
Just make sure that the "gml" directory is in the include path and include `gml.hpp`.
~~~c++
#include <gml/gml.hpp>
~~~

Use Doxygen to generate full API documentation. Doxyfile is provided.


## Usage ##

There are only three class templates `vec` (a vector), `mat` (a matrix) and `quaternion` (a quaternion).

All gml types and functions are defined under name space `gml` (of which examples in this document omit).

Although vector and matrix sizes can be anything this library is meant to be used with small vectors and matrices (2-4 columns/rows).

For optimal performance you may wish to compile with `-O3 -ffast-math -DNODEBUG`.

## vec ##

The `vec` template looks a bit like `std::array`.

~~~c++
template <typename T, int N> class vec;
~~~

Where `T` is the type of a single component and `N` is the number of components.

GLSL-like type aliases for common types are provided. They are named `TvecN` where `T` is `i` for `int`, `d` for `double`, `u` for `unsigned`, `b` for `bool` and non-existent for `float`. `N` is 2, 3 or 4.

For example:
~~~c++
vec3 = vec<float, 3>
ivec4 = vec<int, 4>
uvec2 = vec<unsigned, 2>
~~~

Common constructors:
~~~c++
vec3 a{};  // All zeros
vec3 b{1.0f}; // All ones
vec3 c{1.0f, 2.0f, 3.0f}; // Vector (1, 2, 3)
vec4 e{a, 1.0f}; // Copy a and add 1 as the last component.
vec2 f{a}; // Copy a but drop the last component
vec3 g{1.0f, 2.0f}; // ERROR: Will not compile. Invalid number of arguments.
~~~

Use the `operator[]` to reference the components and `data()` to get a pointer to the first component. Vectors are also iterable.

Component-wise arithmetic operators `*`, `/`, `-` and `+` between vectors of same size and between scalars and vectors are provided.

Most vector operations from GLSL such as `dot` and `cross` are also provided.

There are no GLSL-like swizzle operations.


## mat ##

The `mat` template works a bit like nested `std::array`.

~~~c++
template <typename T, int C, int R> class mat;
~~~
Where `T` is the type of a single component, `C` is the number of columns and `R` is the number of rows.

GLSL-like type aliases for common matrix types are provided. They are named `TmatCxR` where `T` is same as for `vec`. `C` and `R` are 2, 3, or 4. When `C` equals `R` a shorter alias `TmatC` is also available.

For example:
~~~c++
imat4x3 = mat<int, 4, 3>
mat4 = mat<float, 4, 4>
~~~

Common constructors:
~~~c++
mat4 a{}; // All zeros
mat4 b{1.0f}; // 1 at the diagonal and 0 elsewhere. (Identity matrix)
const float data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
mat2x3 c{data, true}; // Matrix with columns (1, 2, 3) and (4, 5, 6).
imat2 d{ivec2{1, 2}, ivec2{3, 4}}; // Matrix with column vectors (1, 2) and (3, 4)
mat3 e{a}; // Copy a and drop last row and column.
~~~

Each column in the matrix is a vector with `R` components. Use `operator[]` to reference the columns. Use `data()` to get pointer to the first component of the first column. Method `size` returns the number of columns (the largest value that can be given to `operator[]`).

Note: Matrices store data in column major order!
Matrices can be iterated, but they iterate columns not components.

~~~c++
mat3 m{};
m[0] = vec3{1.0f, 2.0f, 3.0f}; // Set first column to (1,2,3)
m[2][0] = 10.0f;  // Set first component of last column to 10.
~~~

Arithmetic operators `*`, `/`, `-`  and `+` are provided.
Note that the multiplication is the matrix multiplication and not done component-wise. `matrixCompMult` can be used to do component wise multiplication.

Most matrix operation function from GLSL are provided such as `inverse`, `transpose` and `determinant`.

Replacements for the (deprecated) OpenGL and GLU matrix generation functions are provided. They have the same names as OpenGL counter parts, but without any prefix or suffix and they return the matrices they generate: `translate`, `rotate`, `scale`, `perspective`, `ortho`, `lookAt`.

To transform a 3-vector `v` with 4x4 matrix `M` you need to add 1 as the last component, multiply with the matrix and then drop the last component.
~~~c++
vec3 result{M * vec4{v, 1.0f}};
~~~

A more convenient (and more efficient) function `transform` is provided to do this.
~~~c++
vec3 result = transform(M, v);
~~~


## quaternion ##

Even though GLSL has no quaternions, they are provided for their usefulness. The `quaternion` template is modeled after `std::complex` except that the imaginary part is a 3-vector.

~~~c++
template <typename T> class quaternion;
~~~
Where `T` is the type of a single component.

GLSL-like type aliases are provided in the form of `Tquat`. Where `T` the same as for `vec` and `mat`.
~~~c++
quat = quaternion<float>
dquat = quaternion<double>
~~~

Common constructors:
~~~c++
quat a{}; // All zero
quat b{1.0f}; // 1 as real part and zero imaginary (Identity quaternion)
quat c{1.0f, vec3{2.0f, 3.0f, 4.0f}}; // Quaternion with 1 as real and (2,3,4) as imaginary part
~~~

Use public members `real` and `imag` to access data.
~~~c++
quat q{};
q.real = 5.0f; // Set real part to 5
q.imag = vec3{1.0f, 2.0f, 3.0f}; // Set imaginary part to (1,2,3)
~~~

All arithmetic operations between quaternions, quaternions and scalars and quaternions and 3-vectors are provided. Scalar is interpreted as the real part and 3-vector as the imaginary part.

`std::complex` like operations are provided such as `conj` and `inverse`.

Quaternions are used for presenting orientation or rotation in 3d space (just like rotation matrices). Use `qrotate` to generate a rotation quaternion.
~~~c++
// 90dec rotation around x axis.
auto rotation = qrotate(radians(90.0f), vec3{1.0f, 0.0f, 0.0f});
~~~

You can multiply quaternion and vector, but this does not rotate the vector (like in the case of a rotation matrix). The vector in this case is interpreted as the imaginary part of a quaternion having a zero as real part.

To rotate a vector `v` with quaternion `q` calculate `q*v*conj(q)` and use the imaginary part as the rotated vector. Overload for the `transform` function is provided to do this more efficiently.
~~~c++
vec3 result = transform(q, v);
~~~


## Testing ##

A fairly good fuzz tester is provided with the library. It's in the `test.cpp` and should be run when ever making changes to the library.

