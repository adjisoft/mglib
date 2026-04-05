#include "mglib/mglib.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Printable {
    int value = 0;
};

std::ostream& operator<<(std::ostream& output, const Printable& printable) {
    output << "Printable(" << printable.value << ")";
    return output;
}

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_mgfmt() {
    using namespace mg::fmt;

    expect(format(1, " + ", 2, " = ", 3) == "1 + 2 = 3", "format should concatenate values");
    expect(format(std::vector<int>{1, 2, 3}) == "[1, 2, 3]", "format should render containers");
    expect(join(std::vector<int>{1, 2, 3}, " | ") == "1 | 2 | 3", "join should use separator");

    std::ostringstream buffer;
    fprint(buffer, "item=", 42, " ", Printable{7});
    expect(buffer.str() == "item=42 Printable(7)", "fprint should write to provided stream");

    std::ostringstream captured;
    auto* previous = std::cout.rdbuf(captured.rdbuf());
    print("A", " ", 1);
    println(" B");
    std::cout.rdbuf(previous);
    expect(captured.str() == "A 1 B\n", "print and println should write to std::cout");
}

void test_mgmath() {
    using namespace mg::math;

    const Vec<double, 3> a{1.0, 2.0, 3.0};
    const Vec<double, 3> b{4.0, 5.0, 6.0};
    expect(approx_equal(dot(a, b), 32.0), "dot product should match expected value");
    expect(approx_equal(cross(a, b), Vec<double, 3>{-3.0, 6.0, -3.0}), "cross product should match");
    expect(approx_equal(norm(a), std::sqrt(14.0)), "vector norm should match");
    expect(approx_equal(normalize(a), Vec<double, 3>{1.0 / std::sqrt(14.0),
                                                    2.0 / std::sqrt(14.0),
                                                    3.0 / std::sqrt(14.0)}),
           "normalized vector should match");

    const Mat<double, 2, 2> left{{{4.0, 7.0}, {2.0, 6.0}}};
    const Mat<double, 2, 2> right{{{1.0, 2.0}, {3.0, 4.0}}};
    const auto product = left * right;
    expect(approx_equal(product, Mat<double, 2, 2>{{{25.0, 36.0}, {20.0, 28.0}}}),
           "matrix multiplication should match");
    expect(approx_equal(left * Vec<double, 2>{1.0, 2.0}, Vec<double, 2>{18.0, 14.0}),
           "matrix-vector multiplication should match");
    expect(approx_equal(determinant(left), 10.0), "determinant should match");
    expect(approx_equal(transpose(right), Mat<double, 2, 2>{{{1.0, 3.0}, {2.0, 4.0}}}),
           "transpose should swap rows and columns");

    const auto inverted = inverse(left);
    expect(approx_equal(inverted, Mat<double, 2, 2>{{{0.6, -0.7}, {-0.2, 0.4}}}, 1e-9),
           "inverse should match expected value");

    const auto lu = lu_decompose(left);
    expect(lu.success, "LU decomposition should succeed");
    expect(approx_equal(lu.lower * lu.upper, lu.permutation * left, 1e-9),
           "LU decomposition should satisfy P*A = L*U");

    const auto qr = qr_decompose(right);
    expect(qr.success, "QR decomposition should succeed");
    expect(approx_equal(qr.q * qr.r, right, 1e-8), "QR decomposition should reconstruct input");
    expect(approx_equal(transpose(qr.q) * qr.q, Mat<double, 2, 2>::identity(), 1e-8),
           "Q should be orthonormal");

    const Mat<double, 2, 2> diagonal{{{3.0, 0.0}, {0.0, 1.0}}};
    const auto dominant = power_iteration(diagonal, 256, 1e-10);
    expect(dominant.success, "power iteration should converge");
    expect(approx_equal(dominant.eigenvalue, 3.0, 1e-6), "dominant eigenvalue should match");
    expect(approx_equal(std::abs(dominant.eigenvector[0]), 1.0, 1e-5),
           "dominant eigenvector should align with first axis");

    const Tensor<int, 2, 2, 2> tensor{1, 2, 3, 4, 5, 6, 7, 8};
    expect(tensor(1, 0, 1) == 6, "tensor indexing should be row-major");
    const auto shifted = tensor + Tensor<int, 2, 2, 2>(1);
    expect(shifted(1, 1, 1) == 9, "tensor elementwise addition should match");
    expect(approx_equal(shifted / 2.0, Tensor<double, 2, 2, 2>{1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5}),
           "tensor scalar division should match");
}

void test_mgfs() {
    namespace fsys = std::filesystem;
    using namespace mg::fs;

    const fsys::path root = fsys::temp_directory_path() / "mglib_test_workspace";
    fsys::remove_all(root);
    fsys::create_directories(root);

    const fsys::path text_path = root / "sample.txt";
    const fsys::path binary_path = root / "sample.bin";
    const fsys::path stream_path = root / "stream.txt";

    write_text(text_path, "hello");
    append_text(text_path, "\nworld");
    expect(mg::fs::exists(text_path), "text file should exist");
    expect(read_text(text_path) == "hello\nworld", "text round-trip should match");
    expect(read_lines(text_path) == std::vector<std::string>({"hello", "world"}),
           "line reading should match");

    const auto bytes = std::vector<std::uint8_t>{1, 2, 3, 4};
    write_bytes(binary_path, bytes);
    expect(read_bytes(binary_path) == bytes, "binary round-trip should match");
    expect(mg::fs::file_size(binary_path) == 4, "binary file size should match");

    bool prevented = false;
    try {
        write_text(text_path, "again", overwrite_mode::prevent);
    } catch (const std::runtime_error&) {
        prevented = true;
    }
    expect(prevented, "overwrite protection should raise an error");

    output_file output(stream_path);
    output.write_line("stream");
    output.write("file");
    output.flush();

    input_file input(stream_path);
    expect(input.read_all() == "stream\nfile", "stream wrappers should round-trip text");

    const std::string binary_path_text = binary_path.string();
    const auto path_view = std::string_view(binary_path_text);
    expect(exists(binary_path_text.c_str()), "char* overload should work");
    expect(read_bytes(path_view) == bytes, "string_view overload should work");

    fsys::remove_all(root);
}

}  // namespace

int main() {
    try {
        test_mgfmt();
        test_mgmath();
        test_mgfs();
        std::cout << "All mglib tests passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
