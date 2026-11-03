/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// GPU-free unit test for the hip.dump_tensor runtime contract.
//
// Compiles lib/Runtime/tensor_dump.cpp natively against the MOCK runtime types
// and supplies hipdnn_ep_dump_copy_to_host as a memcpy, then checks:
//   * ONNX Runtime file naming and the NPY v1.0 header / payload
//   * strided views are packed to their logical row-major layout
//   * index.jsonl records name / file / dtype / shape
//   * the HIPDNN_EP_DUMP_START/END filter, the first-inference gate and the
//     per-file session ownership
//===----------------------------------------------------------------------===//

#include "hipdnn_ep_runtime.h"
#include "runtime_state_internal.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <stdlib.h>
#endif

extern "C" int hipdnn_ep_dump_copy_to_host(RuntimeState *state, void *dst,
                                           const void *src, size_t num_bytes) {
  if (!state || !dst || !src)
    return -1;
  std::memcpy(dst, src, num_bytes);
  return 0;
}

namespace {

int g_failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

void setEnv(const char *name, const char *value) {
#ifdef _WIN32
  SetEnvironmentVariableA(name, value);
#else
  setenv(name, value, 1);
#endif
}

std::string readFile(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

struct Npy {
  std::string dict; // header dict, padding included
  std::string data;
};

// Splits an NPY v1.0 file; `dict` is empty when the preamble is malformed or
// the data does not start on a 64-byte boundary.
Npy readNpy(const std::filesystem::path &p) {
  const std::string bytes = readFile(p);
  if (bytes.size() < 10 || bytes.compare(0, 8, "\x93NUMPY\x01\x00", 8) != 0)
    return {};
  const size_t len = static_cast<unsigned char>(bytes[8]) |
                     (static_cast<unsigned char>(bytes[9]) << 8);
  if (bytes.size() < 10 + len || (10 + len) % 64 != 0)
    return {};
  return {bytes.substr(10, len), bytes.substr(10 + len)};
}

template <typename T> std::vector<T> npyData(const std::filesystem::path &p) {
  const std::string data = readNpy(p).data;
  std::vector<T> out(data.size() / sizeof(T));
  std::memcpy(out.data(), data.data(), out.size() * sizeof(T));
  return out;
}

bool npyHas(const std::filesystem::path &p, const char *text) {
  return readNpy(p).dict.find(text) != std::string::npos;
}

void dump(RuntimeState &st, const void *data, std::vector<int64_t> shape,
          std::vector<int64_t> strides, int64_t elemBytes, const char *dtype,
          const char *name, int64_t index) {
  hipdnn_ep_dump_tensor(&st, data, shape.data(), strides.data(),
                        static_cast<int64_t>(shape.size()), elemBytes, dtype,
                        name, index);
}

} // namespace

int main() {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "hipdnn_ep_tensor_dump_test";
  fs::remove_all(dir);
  fs::create_directories(dir);
  setEnv("HIPDNN_EP_DUMP_DIR", dir.string().c_str());
  setEnv("HIPDNN_EP_DUMP_START", "0");
  setEnv("HIPDNN_EP_DUMP_END", "10");

  RuntimeState st{};
  st.completed_inferences = 0;

  // Contiguous 2x3 fp16-sized payload; name uses ORT sanitization.
  const uint16_t half[6] = {1, 2, 3, 4, 5, 6};
  dump(st, half, {2, 3}, {3, 1}, 2, "float16", "/model/layers.0/Add:0", 0);
  const fs::path addFile = dir / "_model_layers_0_Add_0.npy";
  CHECK(fs::exists(addFile));
  CHECK(npyHas(addFile, "'descr': '<f2'"));
  CHECK(npyHas(addFile, "'fortran_order': False"));
  CHECK(npyHas(addFile, "'shape': (2, 3)"));
  CHECK(npyData<uint16_t>(addFile) ==
        std::vector<uint16_t>(std::begin(half), std::end(half)));

  // Strided view: columns 1..2 of a 3x4 int32 buffer.
  const int32_t base[12] = {0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23};
  dump(st, base + 1, {3, 2}, {4, 1}, 4, "int32", "slice", 1);
  CHECK(npyHas(dir / "slice.npy", "'descr': '<i4'"));
  CHECK(npyData<int32_t>(dir / "slice.npy") ==
        (std::vector<int32_t>{1, 2, 11, 12, 21, 22}));

  // Rank-0 scalar and the rank-1 tuple spelling.
  const int64_t scalar = 42;
  dump(st, &scalar, {}, {}, 8, "int64", "scalar", 2);
  CHECK(npyHas(dir / "scalar.npy", "'shape': ()"));
  CHECK(npyData<int64_t>(dir / "scalar.npy") == std::vector<int64_t>{42});
  dump(st, half, {6}, {1}, 2, "bfloat16", "vector", 4);
  CHECK(npyHas(dir / "vector.npy", "'descr': '|V2'"));
  CHECK(npyHas(dir / "vector.npy", "'shape': (6,)"));

  // Outside [START, END]: not written.
  dump(st, half, {6}, {1}, 2, "float16", "filtered", 11);
  CHECK(!fs::exists(dir / "filtered.npy"));

  // A second session cannot overwrite a file owned by the first one.
  RuntimeState other{};
  other.completed_inferences = 0;
  const uint16_t zeros[6] = {};
  dump(other, zeros, {2, 3}, {3, 1}, 2, "float16", "/model/layers.0/Add:0", 0);
  CHECK(npyData<uint16_t>(addFile) ==
        std::vector<uint16_t>(std::begin(half), std::end(half)));

  // The owning session overwrites during its first inference (loop bodies)...
  const uint16_t sevens[6] = {7, 7, 7, 7, 7, 7};
  dump(st, sevens, {2, 3}, {3, 1}, 2, "float16", "/model/layers.0/Add:0", 0);
  CHECK(npyData<uint16_t>(addFile) ==
        std::vector<uint16_t>(std::begin(sevens), std::end(sevens)));

  // ...but not from its second inference on.
  st.completed_inferences = 1;
  dump(st, half, {2, 3}, {3, 1}, 2, "float16", "/model/layers.0/Add:0", 0);
  dump(st, half, {6}, {1}, 2, "float16", "late", 3);
  CHECK(npyData<uint16_t>(addFile) ==
        std::vector<uint16_t>(std::begin(sevens), std::end(sevens)));
  CHECK(!fs::exists(dir / "late.npy"));

  const std::string index = readFile(dir / "index.jsonl");
  CHECK(index.find("{\"index\":0,\"name\":\"/model/layers.0/Add:0\","
                   "\"file\":\"_model_layers_0_Add_0.npy\","
                   "\"dtype\":\"float16\",\"shape\":[2,3]}\n") == 0);
  CHECK(index.find("\"file\":\"slice.npy\",\"dtype\":\"int32\","
                   "\"shape\":[3,2]}") != std::string::npos);
  CHECK(index.find("\"file\":\"scalar.npy\",\"dtype\":\"int64\","
                   "\"shape\":[]}") != std::string::npos);
  CHECK(index.find("filtered") == std::string::npos);
  CHECK(index.find("late") == std::string::npos);

  fs::remove_all(dir);
  if (g_failures) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("TensorDumpUnitTest: all checks passed\n");
  return 0;
}
