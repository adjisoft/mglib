#ifndef MGLIB_MGCONST_H
#define MGLIB_MGCONST_H

/**
 * @file mgconst.h
 * @brief Shared constants, portability helpers, and version metadata for mglib.
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(_MSC_VER)
#define MG_COMPILER_MSVC 1
#else
#define MG_COMPILER_MSVC 0
#endif

#if defined(__clang__)
#define MG_COMPILER_CLANG 1
#else
#define MG_COMPILER_CLANG 0
#endif

#if defined(__GNUC__) && !defined(__clang__)
#define MG_COMPILER_GCC 1
#else
#define MG_COMPILER_GCC 0
#endif

#if defined(_WIN32)
#define MG_PLATFORM_WINDOWS 1
#else
#define MG_PLATFORM_WINDOWS 0
#endif

#if defined(__APPLE__)
#define MG_PLATFORM_MACOS 1
#else
#define MG_PLATFORM_MACOS 0
#endif

#if defined(__linux__)
#define MG_PLATFORM_LINUX 1
#else
#define MG_PLATFORM_LINUX 0
#endif

#if defined(__unix__) || defined(__unix)
#define MG_PLATFORM_UNIX 1
#else
#define MG_PLATFORM_UNIX 0
#endif

#define MG_INLINE inline
#define MG_CONSTEXPR constexpr
#define MG_NOEXCEPT noexcept
#define MG_NODISCARD [[nodiscard]]

#define MGLIB_VERSION_MAJOR 0
#define MGLIB_VERSION_MINOR 1
#define MGLIB_VERSION_PATCH 0

namespace mg {

/**
 * @brief mglib major version.
 */
MG_CONSTEXPR int version_major = MGLIB_VERSION_MAJOR;

/**
 * @brief mglib minor version.
 */
MG_CONSTEXPR int version_minor = MGLIB_VERSION_MINOR;

/**
 * @brief mglib patch version.
 */
MG_CONSTEXPR int version_patch = MGLIB_VERSION_PATCH;

/**
 * @brief Human-readable version string.
 */
MG_CONSTEXPR const char* version_string = "0.1.0";

/**
 * @brief Mathematical constant pi for floating-point types.
 */
template <typename T>
MG_CONSTEXPR T pi_v = static_cast<T>(3.141592653589793238462643383279502884L);

/**
 * @brief Mathematical constant tau for floating-point types.
 */
template <typename T>
MG_CONSTEXPR T tau_v = static_cast<T>(6.283185307179586476925286766559005768L);

/**
 * @brief Mathematical constant e for floating-point types.
 */
template <typename T>
MG_CONSTEXPR T e_v = static_cast<T>(2.718281828459045235360287471352662497L);

/**
 * @brief Default epsilon for approximate numeric comparisons.
 */
template <typename T>
MG_CONSTEXPR T epsilon_v = static_cast<T>(std::is_floating_point<T>::value
                                              ? std::numeric_limits<T>::epsilon() * static_cast<T>(64)
                                              : static_cast<T>(0));

}  // namespace mg

#endif  // MGLIB_MGCONST_H
