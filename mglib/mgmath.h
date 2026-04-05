#ifndef MGLIB_MGMATH_H
#define MGLIB_MGMATH_H

/**
 * @file mgmath.h
 * @brief Header-only linear algebra primitives for vectors, matrices, and tensors.
 *
 * Example:
 * @code
 * mg::math::Vec<double, 3> a{1.0, 2.0, 3.0};
 * mg::math::Vec<double, 3> b{4.0, 5.0, 6.0};
 * auto c = mg::math::cross(a, b);
 *
 * mg::math::Mat<double, 2, 2> m{{ {1.0, 2.0}, {3.0, 4.0} }};
 * auto inverse_m = mg::math::inverse(m);
 * @endcode
 */

#include "mgconst.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace mg {
namespace math {

namespace detail {

template <typename T>
using decay_t = typename std::decay<T>::type;

template <typename T>
using compute_type_t =
    typename std::conditional<std::is_floating_point<T>::value, T, double>::type;

template <std::size_t... Values>
struct static_product;

template <>
struct static_product<> {
    static MG_CONSTEXPR std::size_t value = 1;
};

template <std::size_t Head, std::size_t... Tail>
struct static_product<Head, Tail...> {
    static MG_CONSTEXPR std::size_t value = Head * static_product<Tail...>::value;
};

template <typename T>
MG_NODISCARD MG_INLINE compute_type_t<T> abs_value(T value) {
    using Compute = compute_type_t<T>;
    return static_cast<Compute>(std::abs(static_cast<Compute>(value)));
}

template <typename T>
MG_NODISCARD MG_INLINE bool near_zero(T value, compute_type_t<T> epsilon = mg::epsilon_v<compute_type_t<T>>) {
    return abs_value(value) <= epsilon;
}

template <typename Target, typename Source, std::size_t N>
MG_NODISCARD MG_INLINE std::array<Target, N> cast_array(const std::array<Source, N>& source) {
    std::array<Target, N> result{};
    for (std::size_t index = 0; index < N; ++index) {
        result[index] = static_cast<Target>(source[index]);
    }
    return result;
}

template <typename T, std::size_t Rows, std::size_t Cols>
class Mat;

}  // namespace detail

/**
 * @brief Fixed-size vector with contiguous storage.
 */
template <typename T, std::size_t N>
struct Vec {
    using value_type = T;
    static MG_CONSTEXPR std::size_t size_v = N;

    std::array<T, N> values{};

    MG_CONSTEXPR Vec() = default;

    explicit MG_CONSTEXPR Vec(const T& fill_value) {
        values.fill(fill_value);
    }

    Vec(std::initializer_list<T> items) {
        std::size_t index = 0;
        for (const auto& item : items) {
            if (index >= N) {
                break;
            }
            values[index++] = item;
        }
        for (; index < N; ++index) {
            values[index] = T{};
        }
    }

    MG_NODISCARD MG_CONSTEXPR T& operator[](std::size_t index) { return values[index]; }
    MG_NODISCARD MG_CONSTEXPR const T& operator[](std::size_t index) const { return values[index]; }
    MG_NODISCARD MG_CONSTEXPR T* data() MG_NOEXCEPT { return values.data(); }
    MG_NODISCARD MG_CONSTEXPR const T* data() const MG_NOEXCEPT { return values.data(); }
    MG_NODISCARD MG_CONSTEXPR std::size_t size() const MG_NOEXCEPT { return N; }
    MG_NODISCARD MG_CONSTEXPR auto begin() MG_NOEXCEPT { return values.begin(); }
    MG_NODISCARD MG_CONSTEXPR auto end() MG_NOEXCEPT { return values.end(); }
    MG_NODISCARD MG_CONSTEXPR auto begin() const MG_NOEXCEPT { return values.begin(); }
    MG_NODISCARD MG_CONSTEXPR auto end() const MG_NOEXCEPT { return values.end(); }

    MG_NODISCARD static MG_CONSTEXPR Vec zero() {
        return Vec{};
    }

    MG_NODISCARD static MG_CONSTEXPR Vec filled(const T& value) {
        return Vec(value);
    }

    MG_CONSTEXPR Vec& operator+=(const Vec& other) {
        for (std::size_t index = 0; index < N; ++index) {
            values[index] += other[index];
        }
        return *this;
    }

    MG_CONSTEXPR Vec& operator-=(const Vec& other) {
        for (std::size_t index = 0; index < N; ++index) {
            values[index] -= other[index];
        }
        return *this;
    }

    MG_CONSTEXPR Vec& operator*=(const T& scalar) {
        for (auto& value : values) {
            value *= scalar;
        }
        return *this;
    }

    MG_CONSTEXPR Vec& operator/=(const T& scalar) {
        for (auto& value : values) {
            value /= scalar;
        }
        return *this;
    }

    MG_NODISCARD MG_CONSTEXPR bool operator==(const Vec& other) const {
        return values == other.values;
    }
};

template <typename T, std::size_t N>
MG_NODISCARD MG_INLINE std::ostream& operator<<(std::ostream& output, const Vec<T, N>& vector) {
    output << '[';
    for (std::size_t index = 0; index < N; ++index) {
        if (index != 0) {
            output << ", ";
        }
        output << vector[index];
    }
    output << ']';
    return output;
}

template <typename T, typename U, std::size_t N>
MG_NODISCARD MG_INLINE auto operator+(const Vec<T, N>& left, const Vec<U, N>& right)
    -> Vec<typename std::common_type<T, U>::type, N> {
    using Result = typename std::common_type<T, U>::type;
    Vec<Result, N> value;
    for (std::size_t index = 0; index < N; ++index) {
        value[index] = static_cast<Result>(left[index]) + static_cast<Result>(right[index]);
    }
    return value;
}

template <typename T, typename U, std::size_t N>
MG_NODISCARD MG_INLINE auto operator-(const Vec<T, N>& left, const Vec<U, N>& right)
    -> Vec<typename std::common_type<T, U>::type, N> {
    using Result = typename std::common_type<T, U>::type;
    Vec<Result, N> value;
    for (std::size_t index = 0; index < N; ++index) {
        value[index] = static_cast<Result>(left[index]) - static_cast<Result>(right[index]);
    }
    return value;
}

template <typename T, std::size_t N>
MG_NODISCARD MG_INLINE Vec<T, N> operator-(const Vec<T, N>& vector) {
    Vec<T, N> value;
    for (std::size_t index = 0; index < N; ++index) {
        value[index] = -vector[index];
    }
    return value;
}

template <typename T, typename Scalar, std::size_t N>
MG_NODISCARD MG_INLINE auto operator*(const Vec<T, N>& vector, Scalar scalar)
    -> Vec<typename std::common_type<T, Scalar>::type, N> {
    using Result = typename std::common_type<T, Scalar>::type;
    Vec<Result, N> value;
    for (std::size_t index = 0; index < N; ++index) {
        value[index] = static_cast<Result>(vector[index]) * static_cast<Result>(scalar);
    }
    return value;
}

template <typename Scalar, typename T, std::size_t N>
MG_NODISCARD MG_INLINE auto operator*(Scalar scalar, const Vec<T, N>& vector)
    -> Vec<typename std::common_type<T, Scalar>::type, N> {
    return vector * scalar;
}

template <typename T, typename Scalar, std::size_t N>
MG_NODISCARD MG_INLINE auto operator/(const Vec<T, N>& vector, Scalar scalar)
    -> Vec<typename std::common_type<T, Scalar>::type, N> {
    using Result = typename std::common_type<T, Scalar>::type;
    Vec<Result, N> value;
    for (std::size_t index = 0; index < N; ++index) {
        value[index] = static_cast<Result>(vector[index]) / static_cast<Result>(scalar);
    }
    return value;
}

template <typename T, typename U, std::size_t N>
MG_NODISCARD MG_INLINE auto dot(const Vec<T, N>& left, const Vec<U, N>& right)
    -> typename std::common_type<T, U>::type {
    using Result = typename std::common_type<T, U>::type;
    Result value = Result{};
    for (std::size_t index = 0; index < N; ++index) {
        value += static_cast<Result>(left[index]) * static_cast<Result>(right[index]);
    }
    return value;
}

template <typename T>
MG_NODISCARD MG_INLINE auto cross(const Vec<T, 3>& left, const Vec<T, 3>& right) -> Vec<T, 3> {
    return Vec<T, 3>{left[1] * right[2] - left[2] * right[1],
                     left[2] * right[0] - left[0] * right[2],
                     left[0] * right[1] - left[1] * right[0]};
}

template <typename T, std::size_t N>
MG_NODISCARD MG_INLINE auto squared_norm(const Vec<T, N>& vector) -> detail::compute_type_t<T> {
    using Compute = detail::compute_type_t<T>;
    Compute value = Compute{};
    for (std::size_t index = 0; index < N; ++index) {
        value += static_cast<Compute>(vector[index]) * static_cast<Compute>(vector[index]);
    }
    return value;
}

template <typename T, std::size_t N>
MG_NODISCARD MG_INLINE auto norm(const Vec<T, N>& vector) -> detail::compute_type_t<T> {
    return static_cast<detail::compute_type_t<T>>(std::sqrt(squared_norm(vector)));
}

template <typename T, std::size_t N>
MG_NODISCARD MG_INLINE auto normalize(const Vec<T, N>& vector)
    -> Vec<detail::compute_type_t<T>, N> {
    using Compute = detail::compute_type_t<T>;
    const Compute length = norm(vector);
    if (detail::near_zero(length)) {
        throw std::runtime_error("mg::math::normalize cannot normalize a zero-length vector");
    }
    Vec<Compute, N> value;
    for (std::size_t index = 0; index < N; ++index) {
        value[index] = static_cast<Compute>(vector[index]) / length;
    }
    return value;
}

/**
 * @brief Fixed-size row-major matrix with contiguous storage.
 */
template <typename T, std::size_t Rows, std::size_t Cols>
struct Mat {
    using value_type = T;
    static MG_CONSTEXPR std::size_t rows_v = Rows;
    static MG_CONSTEXPR std::size_t cols_v = Cols;

    std::array<T, Rows * Cols> values{};

    MG_CONSTEXPR Mat() = default;

    explicit MG_CONSTEXPR Mat(const T& fill_value) {
        values.fill(fill_value);
    }

    Mat(std::initializer_list<std::initializer_list<T>> rows) {
        std::size_t row_index = 0;
        for (const auto& row : rows) {
            if (row_index >= Rows) {
                break;
            }
            std::size_t column_index = 0;
            for (const auto& value : row) {
                if (column_index >= Cols) {
                    break;
                }
                (*this)(row_index, column_index++) = value;
            }
            ++row_index;
        }
    }

    MG_NODISCARD MG_CONSTEXPR T& operator()(std::size_t row, std::size_t column) {
        return values[row * Cols + column];
    }

    MG_NODISCARD MG_CONSTEXPR const T& operator()(std::size_t row, std::size_t column) const {
        return values[row * Cols + column];
    }

    MG_NODISCARD MG_CONSTEXPR T* data() MG_NOEXCEPT { return values.data(); }
    MG_NODISCARD MG_CONSTEXPR const T* data() const MG_NOEXCEPT { return values.data(); }
    MG_NODISCARD MG_CONSTEXPR std::size_t rows() const MG_NOEXCEPT { return Rows; }
    MG_NODISCARD MG_CONSTEXPR std::size_t cols() const MG_NOEXCEPT { return Cols; }
    MG_NODISCARD MG_CONSTEXPR auto begin() MG_NOEXCEPT { return values.begin(); }
    MG_NODISCARD MG_CONSTEXPR auto end() MG_NOEXCEPT { return values.end(); }
    MG_NODISCARD MG_CONSTEXPR auto begin() const MG_NOEXCEPT { return values.begin(); }
    MG_NODISCARD MG_CONSTEXPR auto end() const MG_NOEXCEPT { return values.end(); }

    MG_NODISCARD static MG_CONSTEXPR Mat zero() {
        return Mat{};
    }

    MG_NODISCARD static MG_CONSTEXPR Mat filled(const T& value) {
        return Mat(value);
    }

    MG_NODISCARD static MG_INLINE Mat identity() {
        static_assert(Rows == Cols, "mg::math::Mat::identity requires a square matrix");
        Mat value;
        for (std::size_t index = 0; index < Rows; ++index) {
            value(index, index) = static_cast<T>(1);
        }
        return value;
    }

    MG_CONSTEXPR Mat& operator+=(const Mat& other) {
        for (std::size_t index = 0; index < Rows * Cols; ++index) {
            values[index] += other.values[index];
        }
        return *this;
    }

    MG_CONSTEXPR Mat& operator-=(const Mat& other) {
        for (std::size_t index = 0; index < Rows * Cols; ++index) {
            values[index] -= other.values[index];
        }
        return *this;
    }

    MG_CONSTEXPR Mat& operator*=(const T& scalar) {
        for (auto& value : values) {
            value *= scalar;
        }
        return *this;
    }

    MG_CONSTEXPR Mat& operator/=(const T& scalar) {
        for (auto& value : values) {
            value /= scalar;
        }
        return *this;
    }

    MG_NODISCARD MG_CONSTEXPR bool operator==(const Mat& other) const {
        return values == other.values;
    }
};

template <typename T, std::size_t Rows, std::size_t Cols>
MG_NODISCARD MG_INLINE std::ostream& operator<<(std::ostream& output, const Mat<T, Rows, Cols>& matrix) {
    output << '[';
    for (std::size_t row = 0; row < Rows; ++row) {
        if (row != 0) {
            output << ", ";
        }
        output << '[';
        for (std::size_t column = 0; column < Cols; ++column) {
            if (column != 0) {
                output << ", ";
            }
            output << matrix(row, column);
        }
        output << ']';
    }
    output << ']';
    return output;
}

template <typename T, typename U, std::size_t Rows, std::size_t Cols>
MG_NODISCARD MG_INLINE auto operator+(const Mat<T, Rows, Cols>& left,
                                      const Mat<U, Rows, Cols>& right)
    -> Mat<typename std::common_type<T, U>::type, Rows, Cols> {
    using Result = typename std::common_type<T, U>::type;
    Mat<Result, Rows, Cols> value;
    for (std::size_t row = 0; row < Rows; ++row) {
        for (std::size_t column = 0; column < Cols; ++column) {
            value(row, column) =
                static_cast<Result>(left(row, column)) + static_cast<Result>(right(row, column));
        }
    }
    return value;
}

template <typename T, typename U, std::size_t Rows, std::size_t Cols>
MG_NODISCARD MG_INLINE auto operator-(const Mat<T, Rows, Cols>& left,
                                      const Mat<U, Rows, Cols>& right)
    -> Mat<typename std::common_type<T, U>::type, Rows, Cols> {
    using Result = typename std::common_type<T, U>::type;
    Mat<Result, Rows, Cols> value;
    for (std::size_t row = 0; row < Rows; ++row) {
        for (std::size_t column = 0; column < Cols; ++column) {
            value(row, column) =
                static_cast<Result>(left(row, column)) - static_cast<Result>(right(row, column));
        }
    }
    return value;
}

template <typename T, typename Scalar, std::size_t Rows, std::size_t Cols>
MG_NODISCARD MG_INLINE auto operator*(const Mat<T, Rows, Cols>& matrix, Scalar scalar)
    -> Mat<typename std::common_type<T, Scalar>::type, Rows, Cols> {
    using Result = typename std::common_type<T, Scalar>::type;
    Mat<Result, Rows, Cols> value;
    for (std::size_t row = 0; row < Rows; ++row) {
        for (std::size_t column = 0; column < Cols; ++column) {
            value(row, column) = static_cast<Result>(matrix(row, column)) * static_cast<Result>(scalar);
        }
    }
    return value;
}

template <typename Scalar, typename T, std::size_t Rows, std::size_t Cols>
MG_NODISCARD MG_INLINE auto operator*(Scalar scalar, const Mat<T, Rows, Cols>& matrix)
    -> Mat<typename std::common_type<T, Scalar>::type, Rows, Cols> {
    return matrix * scalar;
}

template <typename T, typename Scalar, std::size_t Rows, std::size_t Cols>
MG_NODISCARD MG_INLINE auto operator/(const Mat<T, Rows, Cols>& matrix, Scalar scalar)
    -> Mat<typename std::common_type<T, Scalar>::type, Rows, Cols> {
    using Result = typename std::common_type<T, Scalar>::type;
    Mat<Result, Rows, Cols> value;
    for (std::size_t row = 0; row < Rows; ++row) {
        for (std::size_t column = 0; column < Cols; ++column) {
            value(row, column) = static_cast<Result>(matrix(row, column)) / static_cast<Result>(scalar);
        }
    }
    return value;
}

template <typename T, typename U, std::size_t Rows, std::size_t Inner, std::size_t Cols>
MG_NODISCARD MG_INLINE auto operator*(const Mat<T, Rows, Inner>& left, const Mat<U, Inner, Cols>& right)
    -> Mat<typename std::common_type<T, U>::type, Rows, Cols> {
    using Result = typename std::common_type<T, U>::type;
    Mat<Result, Rows, Cols> value;
    for (std::size_t row = 0; row < Rows; ++row) {
        for (std::size_t column = 0; column < Cols; ++column) {
            Result sum = Result{};
            for (std::size_t index = 0; index < Inner; ++index) {
                sum += static_cast<Result>(left(row, index)) * static_cast<Result>(right(index, column));
            }
            value(row, column) = sum;
        }
    }
    return value;
}

template <typename T, typename U, std::size_t Rows, std::size_t Cols>
MG_NODISCARD MG_INLINE auto operator*(const Mat<T, Rows, Cols>& matrix, const Vec<U, Cols>& vector)
    -> Vec<typename std::common_type<T, U>::type, Rows> {
    using Result = typename std::common_type<T, U>::type;
    Vec<Result, Rows> value;
    for (std::size_t row = 0; row < Rows; ++row) {
        Result sum = Result{};
        for (std::size_t column = 0; column < Cols; ++column) {
            sum += static_cast<Result>(matrix(row, column)) * static_cast<Result>(vector[column]);
        }
        value[row] = sum;
    }
    return value;
}

template <typename T, std::size_t Rows, std::size_t Cols>
MG_NODISCARD MG_INLINE auto transpose(const Mat<T, Rows, Cols>& matrix) -> Mat<T, Cols, Rows> {
    Mat<T, Cols, Rows> value;
    for (std::size_t row = 0; row < Rows; ++row) {
        for (std::size_t column = 0; column < Cols; ++column) {
            value(column, row) = matrix(row, column);
        }
    }
    return value;
}

template <typename T, std::size_t Rows, std::size_t Cols>
MG_NODISCARD MG_INLINE auto frobenius_norm(const Mat<T, Rows, Cols>& matrix) -> detail::compute_type_t<T> {
    using Compute = detail::compute_type_t<T>;
    Compute sum = Compute{};
    for (const auto& value : matrix.values) {
        sum += static_cast<Compute>(value) * static_cast<Compute>(value);
    }
    return static_cast<Compute>(std::sqrt(sum));
}

template <typename T, std::size_t N>
MG_NODISCARD MG_INLINE auto determinant(const Mat<T, N, N>& matrix) -> detail::compute_type_t<T> {
    using Compute = detail::compute_type_t<T>;
    Mat<Compute, N, N> work;
    for (std::size_t row = 0; row < N; ++row) {
        for (std::size_t column = 0; column < N; ++column) {
            work(row, column) = static_cast<Compute>(matrix(row, column));
        }
    }

    Compute det = static_cast<Compute>(1);
    int sign = 1;

    for (std::size_t pivot = 0; pivot < N; ++pivot) {
        std::size_t pivot_row = pivot;
        Compute pivot_value = detail::abs_value(work(pivot_row, pivot));
        for (std::size_t row = pivot + 1; row < N; ++row) {
            const Compute candidate = detail::abs_value(work(row, pivot));
            if (candidate > pivot_value) {
                pivot_value = candidate;
                pivot_row = row;
            }
        }

        if (detail::near_zero(pivot_value)) {
            return Compute{};
        }

        if (pivot_row != pivot) {
            for (std::size_t column = 0; column < N; ++column) {
                std::swap(work(pivot, column), work(pivot_row, column));
            }
            sign *= -1;
        }

        const Compute diagonal = work(pivot, pivot);
        det *= diagonal;
        for (std::size_t row = pivot + 1; row < N; ++row) {
            const Compute factor = work(row, pivot) / diagonal;
            work(row, pivot) = Compute{};
            for (std::size_t column = pivot + 1; column < N; ++column) {
                work(row, column) -= factor * work(pivot, column);
            }
        }
    }

    return sign > 0 ? det : -det;
}

template <typename T, std::size_t N>
MG_NODISCARD MG_INLINE auto inverse(const Mat<T, N, N>& matrix)
    -> Mat<detail::compute_type_t<T>, N, N> {
    using Compute = detail::compute_type_t<T>;
    Mat<Compute, N, N> work;
    Mat<Compute, N, N> inverse_matrix = Mat<Compute, N, N>::identity();

    for (std::size_t row = 0; row < N; ++row) {
        for (std::size_t column = 0; column < N; ++column) {
            work(row, column) = static_cast<Compute>(matrix(row, column));
        }
    }

    for (std::size_t pivot = 0; pivot < N; ++pivot) {
        std::size_t pivot_row = pivot;
        Compute pivot_value = detail::abs_value(work(pivot_row, pivot));
        for (std::size_t row = pivot + 1; row < N; ++row) {
            const Compute candidate = detail::abs_value(work(row, pivot));
            if (candidate > pivot_value) {
                pivot_value = candidate;
                pivot_row = row;
            }
        }

        if (detail::near_zero(pivot_value)) {
            throw std::runtime_error("mg::math::inverse cannot invert a singular matrix");
        }

        if (pivot_row != pivot) {
            for (std::size_t column = 0; column < N; ++column) {
                std::swap(work(pivot, column), work(pivot_row, column));
                std::swap(inverse_matrix(pivot, column), inverse_matrix(pivot_row, column));
            }
        }

        const Compute diagonal = work(pivot, pivot);
        for (std::size_t column = 0; column < N; ++column) {
            work(pivot, column) /= diagonal;
            inverse_matrix(pivot, column) /= diagonal;
        }

        for (std::size_t row = 0; row < N; ++row) {
            if (row == pivot) {
                continue;
            }
            const Compute factor = work(row, pivot);
            if (detail::near_zero(factor)) {
                continue;
            }
            for (std::size_t column = 0; column < N; ++column) {
                work(row, column) -= factor * work(pivot, column);
                inverse_matrix(row, column) -= factor * inverse_matrix(pivot, column);
            }
        }
    }

    return inverse_matrix;
}

template <typename T, std::size_t N>
struct LUResult {
    using value_type = detail::compute_type_t<T>;

    Mat<value_type, N, N> lower = Mat<value_type, N, N>::identity();
    Mat<value_type, N, N> upper{};
    Mat<value_type, N, N> permutation = Mat<value_type, N, N>::identity();
    std::array<std::size_t, N> pivot_indices{};
    int permutation_sign = 1;
    bool success = false;
};

template <typename T, std::size_t N>
MG_NODISCARD MG_INLINE auto lu_decompose(const Mat<T, N, N>& matrix) -> LUResult<T, N> {
    using Compute = detail::compute_type_t<T>;
    LUResult<T, N> result;

    for (std::size_t index = 0; index < N; ++index) {
        result.pivot_indices[index] = index;
        for (std::size_t column = 0; column < N; ++column) {
            result.upper(index, column) = static_cast<Compute>(matrix(index, column));
        }
    }

    for (std::size_t pivot = 0; pivot < N; ++pivot) {
        std::size_t pivot_row = pivot;
        Compute pivot_value = detail::abs_value(result.upper(pivot_row, pivot));
        for (std::size_t row = pivot + 1; row < N; ++row) {
            const Compute candidate = detail::abs_value(result.upper(row, pivot));
            if (candidate > pivot_value) {
                pivot_value = candidate;
                pivot_row = row;
            }
        }

        if (detail::near_zero(pivot_value)) {
            return result;
        }

        if (pivot_row != pivot) {
            for (std::size_t column = 0; column < N; ++column) {
                std::swap(result.upper(pivot, column), result.upper(pivot_row, column));
                std::swap(result.permutation(pivot, column), result.permutation(pivot_row, column));
            }
            for (std::size_t column = 0; column < pivot; ++column) {
                std::swap(result.lower(pivot, column), result.lower(pivot_row, column));
            }
            std::swap(result.pivot_indices[pivot], result.pivot_indices[pivot_row]);
            result.permutation_sign *= -1;
        }

        for (std::size_t row = pivot + 1; row < N; ++row) {
            const Compute factor = result.upper(row, pivot) / result.upper(pivot, pivot);
            result.lower(row, pivot) = factor;
            result.upper(row, pivot) = Compute{};
            for (std::size_t column = pivot + 1; column < N; ++column) {
                result.upper(row, column) -= factor * result.upper(pivot, column);
            }
        }
    }

    result.success = true;
    return result;
}

template <typename T, std::size_t Rows, std::size_t Cols>
struct QRResult {
    using value_type = detail::compute_type_t<T>;

    Mat<value_type, Rows, Cols> q{};
    Mat<value_type, Cols, Cols> r{};
    bool success = false;
};

template <typename T, std::size_t Rows, std::size_t Cols>
MG_NODISCARD MG_INLINE auto qr_decompose(const Mat<T, Rows, Cols>& matrix) -> QRResult<T, Rows, Cols> {
    using Compute = detail::compute_type_t<T>;
    QRResult<T, Rows, Cols> result;

    for (std::size_t column = 0; column < Cols; ++column) {
        Vec<Compute, Rows> working_column;
        for (std::size_t row = 0; row < Rows; ++row) {
            working_column[row] = static_cast<Compute>(matrix(row, column));
        }

        for (std::size_t previous = 0; previous < column; ++previous) {
            Vec<Compute, Rows> q_column;
            for (std::size_t row = 0; row < Rows; ++row) {
                q_column[row] = result.q(row, previous);
            }
            result.r(previous, column) = dot(q_column, working_column);
            working_column = working_column - q_column * result.r(previous, column);
        }

        const Compute column_norm = norm(working_column);
        if (detail::near_zero(column_norm)) {
            return result;
        }

        result.r(column, column) = column_norm;
        const auto normalized = working_column / column_norm;
        for (std::size_t row = 0; row < Rows; ++row) {
            result.q(row, column) = normalized[row];
        }
    }

    result.success = true;
    return result;
}

template <typename T, std::size_t N>
struct PowerIterationResult {
    using value_type = detail::compute_type_t<T>;

    value_type eigenvalue{};
    Vec<value_type, N> eigenvector{};
    std::size_t iterations = 0;
    bool success = false;
};

template <typename T, std::size_t N>
MG_NODISCARD MG_INLINE auto power_iteration(const Mat<T, N, N>& matrix,
                                            std::size_t max_iterations = 256,
                                            detail::compute_type_t<T> tolerance =
                                                mg::epsilon_v<detail::compute_type_t<T>>)
    -> PowerIterationResult<T, N> {
    using Compute = detail::compute_type_t<T>;
    PowerIterationResult<T, N> result;

    Vec<Compute, N> vector(static_cast<Compute>(1));
    vector = normalize(vector);

    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
        const auto next = matrix * vector;
        const Compute next_norm = norm(next);
        if (detail::near_zero(next_norm)) {
            return result;
        }

        const auto normalized = next / next_norm;
        if (norm(normalized - vector) <= tolerance) {
            vector = normalized;
            result.iterations = iteration + 1;
            result.success = true;
            break;
        }

        vector = normalized;
        result.iterations = iteration + 1;
    }

    const auto transformed = matrix * vector;
    result.eigenvalue = dot(vector, transformed) / dot(vector, vector);
    result.eigenvector = vector;
    return result;
}

/**
 * @brief Fixed-size tensor backed by contiguous row-major storage.
 */
template <typename T, std::size_t... Dims>
struct Tensor {
    static_assert(sizeof...(Dims) > 0, "mg::math::Tensor requires at least one dimension");

    using value_type = T;
    static MG_CONSTEXPR std::size_t order_v = sizeof...(Dims);
    static MG_CONSTEXPR std::size_t size_v = detail::static_product<Dims...>::value;

    std::array<T, size_v> values{};

    MG_CONSTEXPR Tensor() = default;

    explicit MG_CONSTEXPR Tensor(const T& fill_value) {
        values.fill(fill_value);
    }

    Tensor(std::initializer_list<T> items) {
        std::size_t index = 0;
        for (const auto& item : items) {
            if (index >= size_v) {
                break;
            }
            values[index++] = item;
        }
        for (; index < size_v; ++index) {
            values[index] = T{};
        }
    }

    MG_NODISCARD MG_CONSTEXPR std::size_t size() const MG_NOEXCEPT { return size_v; }
    MG_NODISCARD MG_CONSTEXPR auto begin() MG_NOEXCEPT { return values.begin(); }
    MG_NODISCARD MG_CONSTEXPR auto end() MG_NOEXCEPT { return values.end(); }
    MG_NODISCARD MG_CONSTEXPR auto begin() const MG_NOEXCEPT { return values.begin(); }
    MG_NODISCARD MG_CONSTEXPR auto end() const MG_NOEXCEPT { return values.end(); }
    MG_NODISCARD MG_CONSTEXPR T* data() MG_NOEXCEPT { return values.data(); }
    MG_NODISCARD MG_CONSTEXPR const T* data() const MG_NOEXCEPT { return values.data(); }

    MG_NODISCARD static MG_CONSTEXPR Tensor zero() {
        return Tensor{};
    }

    MG_NODISCARD static MG_CONSTEXPR Tensor filled(const T& value) {
        return Tensor(value);
    }

    MG_NODISCARD MG_CONSTEXPR std::array<std::size_t, order_v> shape() const MG_NOEXCEPT {
        return std::array<std::size_t, order_v>{Dims...};
    }

    template <typename... Index>
    MG_NODISCARD MG_INLINE T& operator()(Index... indices) {
        static_assert(sizeof...(Index) == order_v, "mg::math::Tensor index count mismatch");
        const auto flat = flat_index(static_cast<std::size_t>(indices)...);
        return values[flat];
    }

    template <typename... Index>
    MG_NODISCARD MG_INLINE const T& operator()(Index... indices) const {
        static_assert(sizeof...(Index) == order_v, "mg::math::Tensor index count mismatch");
        const auto flat = flat_index(static_cast<std::size_t>(indices)...);
        return values[flat];
    }

    MG_NODISCARD MG_CONSTEXPR const std::array<T, size_v>& flatten() const MG_NOEXCEPT { return values; }

    MG_NODISCARD MG_CONSTEXPR bool operator==(const Tensor& other) const {
        return values == other.values;
    }

   private:
    template <typename... Index>
    MG_NODISCARD MG_INLINE std::size_t flat_index(Index... indices) const {
        const std::array<std::size_t, order_v> shape_values{Dims...};
        const std::array<std::size_t, order_v> index_values{static_cast<std::size_t>(indices)...};
        std::size_t flat = 0;
        for (std::size_t index = 0; index < order_v; ++index) {
            if (index_values[index] >= shape_values[index]) {
                throw std::out_of_range("mg::math::Tensor index out of range");
            }
            flat *= shape_values[index];
            flat += index_values[index];
        }
        return flat;
    }
};

template <typename T, std::size_t... Dims>
MG_NODISCARD MG_INLINE std::ostream& operator<<(std::ostream& output, const Tensor<T, Dims...>& tensor) {
    output << '[';
    for (std::size_t index = 0; index < tensor.size(); ++index) {
        if (index != 0) {
            output << ", ";
        }
        output << tensor.values[index];
    }
    output << ']';
    return output;
}

template <typename T, typename U, std::size_t... Dims>
MG_NODISCARD MG_INLINE auto operator+(const Tensor<T, Dims...>& left, const Tensor<U, Dims...>& right)
    -> Tensor<typename std::common_type<T, U>::type, Dims...> {
    using Result = typename std::common_type<T, U>::type;
    Tensor<Result, Dims...> value;
    for (std::size_t index = 0; index < value.size(); ++index) {
        value.values[index] = static_cast<Result>(left.values[index]) + static_cast<Result>(right.values[index]);
    }
    return value;
}

template <typename T, typename U, std::size_t... Dims>
MG_NODISCARD MG_INLINE auto operator-(const Tensor<T, Dims...>& left, const Tensor<U, Dims...>& right)
    -> Tensor<typename std::common_type<T, U>::type, Dims...> {
    using Result = typename std::common_type<T, U>::type;
    Tensor<Result, Dims...> value;
    for (std::size_t index = 0; index < value.size(); ++index) {
        value.values[index] = static_cast<Result>(left.values[index]) - static_cast<Result>(right.values[index]);
    }
    return value;
}

template <typename T, typename Scalar, std::size_t... Dims>
MG_NODISCARD MG_INLINE auto operator*(const Tensor<T, Dims...>& tensor, Scalar scalar)
    -> Tensor<typename std::common_type<T, Scalar>::type, Dims...> {
    using Result = typename std::common_type<T, Scalar>::type;
    Tensor<Result, Dims...> value;
    for (std::size_t index = 0; index < value.size(); ++index) {
        value.values[index] = static_cast<Result>(tensor.values[index]) * static_cast<Result>(scalar);
    }
    return value;
}

template <typename Scalar, typename T, std::size_t... Dims>
MG_NODISCARD MG_INLINE auto operator*(Scalar scalar, const Tensor<T, Dims...>& tensor)
    -> Tensor<typename std::common_type<T, Scalar>::type, Dims...> {
    return tensor * scalar;
}

template <typename T, typename Scalar, std::size_t... Dims>
MG_NODISCARD MG_INLINE auto operator/(const Tensor<T, Dims...>& tensor, Scalar scalar)
    -> Tensor<typename std::common_type<T, Scalar>::type, Dims...> {
    using Result = typename std::common_type<T, Scalar>::type;
    Tensor<Result, Dims...> value;
    for (std::size_t index = 0; index < value.size(); ++index) {
        value.values[index] = static_cast<Result>(tensor.values[index]) / static_cast<Result>(scalar);
    }
    return value;
}

template <typename T, typename U>
MG_NODISCARD MG_INLINE bool approx_equal(T left,
                                         U right,
                                         detail::compute_type_t<typename std::common_type<T, U>::type> epsilon =
                                             mg::epsilon_v<detail::compute_type_t<typename std::common_type<T, U>::type>>) {
    using Compute = detail::compute_type_t<typename std::common_type<T, U>::type>;
    return detail::abs_value(static_cast<Compute>(left) - static_cast<Compute>(right)) <= epsilon;
}

template <typename T, typename U, std::size_t N>
MG_NODISCARD MG_INLINE bool approx_equal(const Vec<T, N>& left,
                                         const Vec<U, N>& right,
                                         detail::compute_type_t<typename std::common_type<T, U>::type> epsilon =
                                             mg::epsilon_v<detail::compute_type_t<typename std::common_type<T, U>::type>>) {
    for (std::size_t index = 0; index < N; ++index) {
        if (!approx_equal(left[index], right[index], epsilon)) {
            return false;
        }
    }
    return true;
}

template <typename T, typename U, std::size_t Rows, std::size_t Cols>
MG_NODISCARD MG_INLINE bool approx_equal(const Mat<T, Rows, Cols>& left,
                                         const Mat<U, Rows, Cols>& right,
                                         detail::compute_type_t<typename std::common_type<T, U>::type> epsilon =
                                             mg::epsilon_v<detail::compute_type_t<typename std::common_type<T, U>::type>>) {
    for (std::size_t row = 0; row < Rows; ++row) {
        for (std::size_t column = 0; column < Cols; ++column) {
            if (!approx_equal(left(row, column), right(row, column), epsilon)) {
                return false;
            }
        }
    }
    return true;
}

template <typename T, typename U, std::size_t... Dims>
MG_NODISCARD MG_INLINE bool approx_equal(
    const Tensor<T, Dims...>& left,
    const Tensor<U, Dims...>& right,
    detail::compute_type_t<typename std::common_type<T, U>::type> epsilon =
        mg::epsilon_v<detail::compute_type_t<typename std::common_type<T, U>::type>>) {
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (!approx_equal(left.values[index], right.values[index], epsilon)) {
            return false;
        }
    }
    return true;
}

}  // namespace math
}  // namespace mg

#endif  // MGLIB_MGMATH_H
