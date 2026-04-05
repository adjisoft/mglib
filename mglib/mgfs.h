#ifndef MGLIB_MGFS_H
#define MGLIB_MGFS_H

/**
 * @file mgfs.h
 * @brief Header-only filesystem and file stream helpers.
 *
 * Example:
 * @code
 * auto path = std::filesystem::path("example.txt");
 * mg::fs::write_text(path, "hello world");
 * auto text = mg::fs::read_text(path);
 * @endcode
 */

#include "mgconst.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mg {
namespace fs {

/**
 * @brief Controls whether writes may replace an existing file.
 */
enum class overwrite_mode {
    allow,
    prevent
};

/**
 * @brief Controls whether file streams use text or binary mode.
 */
enum class open_mode {
    text,
    binary
};

namespace detail {

MG_NODISCARD MG_INLINE std::filesystem::path make_path(const std::filesystem::path& path) {
    return path;
}

MG_NODISCARD MG_INLINE std::filesystem::path make_path(std::string_view path) {
    return std::filesystem::path(path.begin(), path.end());
}

MG_NODISCARD MG_INLINE std::filesystem::path make_path(const char* path) {
    return std::filesystem::path(path);
}

MG_INLINE void ensure_can_write(const std::filesystem::path& path, overwrite_mode mode) {
    if (mode == overwrite_mode::prevent && std::filesystem::exists(path)) {
        throw std::runtime_error("mg::fs write prevented because the target file already exists");
    }
}

MG_NODISCARD MG_INLINE std::ios::openmode to_open_mode(open_mode mode) {
    return mode == open_mode::binary ? std::ios::binary : std::ios::openmode{};
}

}  // namespace detail

/**
 * @brief Returns true when a path exists.
 */
MG_NODISCARD MG_INLINE bool exists(const std::filesystem::path& path) {
    return std::filesystem::exists(path);
}

MG_NODISCARD MG_INLINE bool exists(std::string_view path) {
    return mg::fs::exists(detail::make_path(path));
}

MG_NODISCARD MG_INLINE bool exists(const char* path) {
    return mg::fs::exists(detail::make_path(path));
}

/**
 * @brief Returns the size of a file in bytes.
 */
MG_NODISCARD MG_INLINE std::uintmax_t file_size(const std::filesystem::path& path) {
    return std::filesystem::file_size(path);
}

MG_NODISCARD MG_INLINE std::uintmax_t file_size(std::string_view path) {
    return mg::fs::file_size(detail::make_path(path));
}

MG_NODISCARD MG_INLINE std::uintmax_t file_size(const char* path) {
    return mg::fs::file_size(detail::make_path(path));
}

/**
 * @brief Reads an entire text file into a string.
 */
MG_NODISCARD MG_INLINE std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("mg::fs::read_text failed to open file: " + path.string());
    }
    return std::string((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
}

MG_NODISCARD MG_INLINE std::string read_text(std::string_view path) {
    return read_text(detail::make_path(path));
}

MG_NODISCARD MG_INLINE std::string read_text(const char* path) {
    return read_text(detail::make_path(path));
}

/**
 * @brief Reads an entire file into a byte buffer.
 */
MG_NODISCARD MG_INLINE std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("mg::fs::read_bytes failed to open file: " + path.string());
    }
    const auto size = std::filesystem::file_size(path);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    if (!bytes.empty()) {
        input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    return bytes;
}

MG_NODISCARD MG_INLINE std::vector<std::uint8_t> read_bytes(std::string_view path) {
    return read_bytes(detail::make_path(path));
}

MG_NODISCARD MG_INLINE std::vector<std::uint8_t> read_bytes(const char* path) {
    return read_bytes(detail::make_path(path));
}

/**
 * @brief Reads a text file and splits it into lines.
 */
MG_NODISCARD MG_INLINE std::vector<std::string> read_lines(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("mg::fs::read_lines failed to open file: " + path.string());
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
        lines.push_back(line);
    }
    return lines;
}

MG_NODISCARD MG_INLINE std::vector<std::string> read_lines(std::string_view path) {
    return read_lines(detail::make_path(path));
}

MG_NODISCARD MG_INLINE std::vector<std::string> read_lines(const char* path) {
    return read_lines(detail::make_path(path));
}

/**
 * @brief Writes a string to a file.
 */
MG_INLINE void write_text(const std::filesystem::path& path,
                          std::string_view content,
                          overwrite_mode mode = overwrite_mode::allow) {
    detail::ensure_can_write(path, mode);
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("mg::fs::write_text failed to open file: " + path.string());
    }
    output.write(content.data(), static_cast<std::streamsize>(content.size()));
}

MG_INLINE void write_text(std::string_view path,
                          std::string_view content,
                          overwrite_mode mode = overwrite_mode::allow) {
    write_text(detail::make_path(path), content, mode);
}

MG_INLINE void write_text(const char* path,
                          std::string_view content,
                          overwrite_mode mode = overwrite_mode::allow) {
    write_text(detail::make_path(path), content, mode);
}

/**
 * @brief Appends text to the end of a file.
 */
MG_INLINE void append_text(const std::filesystem::path& path, std::string_view content) {
    std::ofstream output(path, std::ios::app);
    if (!output) {
        throw std::runtime_error("mg::fs::append_text failed to open file: " + path.string());
    }
    output.write(content.data(), static_cast<std::streamsize>(content.size()));
}

MG_INLINE void append_text(std::string_view path, std::string_view content) {
    append_text(detail::make_path(path), content);
}

MG_INLINE void append_text(const char* path, std::string_view content) {
    append_text(detail::make_path(path), content);
}

/**
 * @brief Writes a byte buffer to a file.
 */
MG_INLINE void write_bytes(const std::filesystem::path& path,
                           const std::vector<std::uint8_t>& bytes,
                           overwrite_mode mode = overwrite_mode::allow) {
    detail::ensure_can_write(path, mode);
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("mg::fs::write_bytes failed to open file: " + path.string());
    }
    if (!bytes.empty()) {
        output.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
    }
}

MG_INLINE void write_bytes(std::string_view path,
                           const std::vector<std::uint8_t>& bytes,
                           overwrite_mode mode = overwrite_mode::allow) {
    write_bytes(detail::make_path(path), bytes, mode);
}

MG_INLINE void write_bytes(const char* path,
                           const std::vector<std::uint8_t>& bytes,
                           overwrite_mode mode = overwrite_mode::allow) {
    write_bytes(detail::make_path(path), bytes, mode);
}

/**
 * @brief Lightweight RAII input file wrapper.
 */
class input_file {
   public:
    input_file() = default;

    explicit input_file(const std::filesystem::path& path, open_mode mode = open_mode::text) {
        open(path, mode);
    }

    explicit input_file(std::string_view path, open_mode mode = open_mode::text) {
        open(detail::make_path(path), mode);
    }

    explicit input_file(const char* path, open_mode mode = open_mode::text) {
        open(detail::make_path(path), mode);
    }

    MG_INLINE void open(const std::filesystem::path& path, open_mode mode = open_mode::text) {
        stream_.open(path, std::ios::in | detail::to_open_mode(mode));
        if (!stream_) {
            throw std::runtime_error("mg::fs::input_file failed to open file: " + path.string());
        }
    }

    MG_NODISCARD MG_INLINE bool is_open() const MG_NOEXCEPT { return stream_.is_open(); }
    MG_NODISCARD MG_INLINE std::ifstream& stream() MG_NOEXCEPT { return stream_; }
    MG_NODISCARD MG_INLINE const std::ifstream& stream() const MG_NOEXCEPT { return stream_; }

    MG_NODISCARD MG_INLINE std::string read_all() {
        return std::string((std::istreambuf_iterator<char>(stream_)), std::istreambuf_iterator<char>());
    }

    MG_NODISCARD MG_INLINE std::vector<std::string> read_lines() {
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(stream_, line)) {
            lines.push_back(line);
        }
        return lines;
    }

    MG_NODISCARD MG_INLINE std::vector<std::uint8_t> read_all_bytes() {
        const auto start = stream_.tellg();
        stream_.seekg(0, std::ios::end);
        const auto end = stream_.tellg();
        stream_.seekg(0, std::ios::beg);
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end - start));
        if (!bytes.empty()) {
            stream_.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        }
        return bytes;
    }

   private:
    std::ifstream stream_;
};

/**
 * @brief Lightweight RAII output file wrapper.
 */
class output_file {
   public:
    output_file() = default;

    explicit output_file(const std::filesystem::path& path,
                         open_mode mode = open_mode::text,
                         overwrite_mode overwrite = overwrite_mode::allow) {
        open(path, mode, overwrite);
    }

    explicit output_file(std::string_view path,
                         open_mode mode = open_mode::text,
                         overwrite_mode overwrite = overwrite_mode::allow) {
        open(detail::make_path(path), mode, overwrite);
    }

    explicit output_file(const char* path,
                         open_mode mode = open_mode::text,
                         overwrite_mode overwrite = overwrite_mode::allow) {
        open(detail::make_path(path), mode, overwrite);
    }

    MG_INLINE void open(const std::filesystem::path& path,
                        open_mode mode = open_mode::text,
                        overwrite_mode overwrite = overwrite_mode::allow) {
        detail::ensure_can_write(path, overwrite);
        stream_.open(path, std::ios::out | detail::to_open_mode(mode));
        if (!stream_) {
            throw std::runtime_error("mg::fs::output_file failed to open file: " + path.string());
        }
    }

    MG_NODISCARD MG_INLINE bool is_open() const MG_NOEXCEPT { return stream_.is_open(); }
    MG_NODISCARD MG_INLINE std::ofstream& stream() MG_NOEXCEPT { return stream_; }
    MG_NODISCARD MG_INLINE const std::ofstream& stream() const MG_NOEXCEPT { return stream_; }

    MG_INLINE void write(std::string_view content) {
        stream_.write(content.data(), static_cast<std::streamsize>(content.size()));
    }

    MG_INLINE void write_line(std::string_view content) {
        write(content);
        stream_ << '\n';
    }

    MG_INLINE void write(const std::vector<std::uint8_t>& bytes) {
        if (!bytes.empty()) {
            stream_.write(reinterpret_cast<const char*>(bytes.data()),
                          static_cast<std::streamsize>(bytes.size()));
        }
    }

    MG_INLINE void flush() {
        stream_.flush();
    }

   private:
    std::ofstream stream_;
};

}  // namespace fs
}  // namespace mg

#endif  // MGLIB_MGFS_H
