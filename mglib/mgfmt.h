#ifndef MGLIB_MGFMT_H
#define MGLIB_MGFMT_H

/**
 * @file mgfmt.h
 * @brief Header-only formatting utilities for readable console and stream output.
 *
 * Example:
 * @code
 * std::vector<int> values{1, 2, 3};
 * mg::fmt::println("values = ", values);
 * auto text = mg::fmt::format("joined: ", mg::fmt::join(values, ", "));
 * @endcode
 */

#include "mgconst.h"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace mg {
namespace fmt {

namespace detail {

template <typename T>
using decay_t = typename std::decay<T>::type;

template <typename...>
struct always_false : std::false_type {};

template <typename T>
struct is_string_like
    : std::integral_constant<
          bool,
          std::is_same<decay_t<T>, std::string>::value ||
              std::is_same<decay_t<T>, std::string_view>::value ||
              std::is_same<decay_t<T>, char*>::value ||
              std::is_same<decay_t<T>, const char*>::value> {};

template <typename T>
struct is_path : std::is_same<decay_t<T>, std::filesystem::path> {};

template <typename T, typename = void>
struct is_iterable : std::false_type {};

template <typename T>
struct is_iterable<T,
                   std::void_t<decltype(std::begin(std::declval<const T&>())),
                               decltype(std::end(std::declval<const T&>()))>>
    : std::true_type {};

template <typename T, typename = void>
struct is_streamable : std::false_type {};

template <typename T>
struct is_streamable<T,
                     std::void_t<decltype(std::declval<std::ostream&>()
                                          << std::declval<const T&>())>> : std::true_type {};

template <typename T>
struct is_pair : std::false_type {};

template <typename A, typename B>
struct is_pair<std::pair<A, B>> : std::true_type {};

template <typename T>
MG_INLINE void append_value(std::ostream& output, const T& value);

template <typename Range>
MG_INLINE void append_iterable(std::ostream& output, const Range& values) {
    output << '[';
    bool first = true;
    for (const auto& value : values) {
        if (!first) {
            output << ", ";
        }
        append_value(output, value);
        first = false;
    }
    output << ']';
}

template <typename A, typename B>
MG_INLINE void append_pair(std::ostream& output, const std::pair<A, B>& value) {
    output << '(';
    append_value(output, value.first);
    output << ", ";
    append_value(output, value.second);
    output << ')';
}

template <typename T>
MG_INLINE void append_value(std::ostream& output, const T& value) {
    if constexpr (std::is_same<decay_t<T>, bool>::value) {
        output << (value ? "true" : "false");
    } else if constexpr (is_string_like<T>::value) {
        output << value;
    } else if constexpr (is_pair<decay_t<T>>::value) {
        append_pair(output, value);
    } else if constexpr (is_iterable<T>::value && !is_string_like<T>::value && !is_path<T>::value) {
        append_iterable(output, value);
    } else if constexpr (is_streamable<T>::value) {
        output << value;
    } else {
        static_assert(always_false<T>::value, "mg::fmt::format cannot format this type");
    }
}

template <typename... Args>
MG_INLINE void append_all(std::ostream& output, const Args&... args) {
    (append_value(output, args), ...);
}

}  // namespace detail

/**
 * @brief Formats a sequence of arguments into a single string.
 */
template <typename... Args>
MG_NODISCARD MG_INLINE std::string format(const Args&... args) {
    std::ostringstream output;
    detail::append_all(output, args...);
    return output.str();
}

/**
 * @brief Writes a sequence of formatted arguments to the provided output stream.
 */
template <typename... Args>
MG_INLINE void fprint(std::ostream& output, const Args&... args) {
    detail::append_all(output, args...);
}

/**
 * @brief Writes formatted arguments to @c std::cout.
 */
template <typename... Args>
MG_INLINE void print(const Args&... args) {
    fprint(std::cout, args...);
}

/**
 * @brief Writes formatted arguments to @c std::cout and appends a newline.
 */
template <typename... Args>
MG_INLINE void println(const Args&... args) {
    fprint(std::cout, args...);
    std::cout << '\n';
}

/**
 * @brief Joins an iterable range using the provided separator.
 */
template <typename Range>
MG_NODISCARD MG_INLINE std::string join(const Range& range, std::string_view separator) {
    static_assert(detail::is_iterable<Range>::value, "mg::fmt::join requires an iterable type");
    std::ostringstream output;
    bool first = true;
    for (const auto& value : range) {
        if (!first) {
            output << separator;
        }
        detail::append_value(output, value);
        first = false;
    }
    return output.str();
}

/**
 * @brief Joins an iterator range using the provided separator.
 */
template <typename Iterator>
MG_NODISCARD MG_INLINE std::string join(Iterator begin, Iterator end, std::string_view separator) {
    std::ostringstream output;
    bool first = true;
    for (Iterator it = begin; it != end; ++it) {
        if (!first) {
            output << separator;
        }
        detail::append_value(output, *it);
        first = false;
    }
    return output.str();
}

}  // namespace fmt
}  // namespace mg

#endif  // MGLIB_MGFMT_H
