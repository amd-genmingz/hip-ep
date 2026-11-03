/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Host half of hip.dump_tensor (docs/design/tensor-dump.md), shared by the
// real and mock runtimes; the device-to-host copy is the backend's
// hipdnn_ep_dump_copy_to_host. Debug-only: every call synchronizes the stream.

#include "debug_log.h"
#include "hipdnn_ep_runtime.h"
#include "runtime_state_internal.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

struct TensorDumpConfig {
  std::string dir;
  int64_t start = 0;
  int64_t end = INT64_MAX;
};

int64_t envInt(const char *name, int64_t fallback) {
  const std::string v = hipdnn_ep::env_string(name);
  if (v.empty())
    return fallback;
  char *endp = nullptr;
  const long long parsed = strtoll(v.c_str(), &endp, 10);
  return (endp && *endp == '\0') ? static_cast<int64_t>(parsed) : fallback;
}

// Latched on first use. Read at run time rather than baked in at compile time
// so a cached artifact that still contains dump ops is silent once the
// variable is unset.
const TensorDumpConfig &config() {
  static const TensorDumpConfig *cfg = [] {
    auto *c = new TensorDumpConfig;
    std::string dir = hipdnn_ep::env_string("HIPDNN_EP_DUMP_DIR");
    if (dir != "0")
      c->dir = dir;
    c->start = envInt("HIPDNN_EP_DUMP_START", 0);
    c->end = envInt("HIPDNN_EP_DUMP_END", INT64_MAX);
    return c;
  }();
  return *cfg;
}

// ONNX Runtime node-dump file naming: every character that is not an ASCII
// letter or digit becomes '_' ("/model/layers.0/Add" -> "_model_layers_0_Add").
std::string fileStem(const char *name) {
  std::string out;
  for (const char *p = name; *p; ++p) {
    const char c = *p;
    const bool alnum = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
                       (c >= 'A' && c <= 'Z');
    out.push_back(alnum ? c : '_');
  }
  return out;
}

// The first session to write a file owns it for the rest of the process, so a
// later compiled graph with the same tensor names (e.g. the decode
// specialization after prefill) cannot overwrite the first inference's dump.
// Repeated writes from the owning session (loop bodies) still overwrite.
bool claimFile(const RuntimeState *state, const std::string &stem) {
  static auto *owners = new std::map<std::string, const RuntimeState *>();
  auto it = owners->find(stem);
  if (it == owners->end()) {
    owners->emplace(stem, state);
    return true;
  }
  return it->second == state;
}

void appendJsonString(std::string &out, const char *s) {
  out.push_back('"');
  for (const char *p = s; *p; ++p) {
    const unsigned char c = static_cast<unsigned char>(*p);
    if (c == '"' || c == '\\') {
      out.push_back('\\');
      out.push_back(static_cast<char>(c));
    } else if (c < 0x20) {
      char esc[8];
      snprintf(esc, sizeof(esc), "\\u%04x", c);
      out += esc;
    } else {
      out.push_back(static_cast<char>(c));
    }
  }
  out.push_back('"');
}

// One JSON object per line, appended after each tensor so the index is
// complete up to the last dumped op even if the process crashes. The first
// append of the process truncates a stale index from an earlier run.
void appendIndexLine(const std::string &dir, const std::string &line) {
  static bool truncated = false;
  const std::string path = dir + "/index.jsonl";
  FILE *f = fopen(path.c_str(), truncated ? "ab" : "wb");
  truncated = true;
  if (!f) {
    fprintf(stderr, "[TensorDump] cannot open %s\n", path.c_str());
    return;
  }
  fwrite(line.data(), 1, line.size(), f);
  fclose(f);
}

// NumPy `descr` for a dump dtype name. bfloat16 and float8 have no NumPy
// type; they are stored as opaque void elements of the same width so the raw
// bits survive and nothing reinterprets them as another numeric type.
const char *npyDescr(const std::string &dtype, int64_t elemBytes) {
  static const struct {
    const char *dtype;
    const char *descr;
  } kTable[] = {
      {"float16", "<f2"}, {"float32", "<f4"}, {"float64", "<f8"},
      {"bool", "|b1"},    {"int8", "|i1"},    {"uint8", "|u1"},
      {"int16", "<i2"},   {"uint16", "<u2"},  {"int32", "<i4"},
      {"uint32", "<u4"},  {"int64", "<i8"},   {"uint64", "<u8"},
  };
  for (const auto &entry : kTable)
    if (dtype == entry.dtype)
      return entry.descr;
  switch (elemBytes) {
  case 1:
    return "|V1";
  case 2:
    return "|V2";
  case 4:
    return "|V4";
  default:
    return "|V8";
  }
}

// NPY format version 1.0 header: magic, version, little-endian u16 header
// length, then a Python dict literal space-padded and '\n'-terminated so the
// data starts on a 64-byte boundary.
std::string npyHeader(const char *descr, const int64_t *shape, int64_t rank) {
  std::string dict = "{'descr': '";
  dict += descr;
  dict += "', 'fortran_order': False, 'shape': (";
  for (int64_t d = 0; d < rank; ++d) {
    if (d)
      dict += ", ";
    dict += std::to_string(shape[d]);
  }
  if (rank == 1)
    dict.push_back(',');
  dict += "), }";
  const size_t preamble = 10;
  const size_t unpadded = preamble + dict.size() + 1;
  dict.append((64 - unpadded % 64) % 64, ' ');
  dict.push_back('\n');

  std::string header("\x93NUMPY\x01\x00", 8);
  const uint16_t len = static_cast<uint16_t>(dict.size());
  header.push_back(static_cast<char>(len & 0xff));
  header.push_back(static_cast<char>(len >> 8));
  header += dict;
  return header;
}

bool isRowMajorContiguous(const int64_t *shape, const int64_t *strides,
                          int64_t rank) {
  int64_t expected = 1;
  for (int64_t d = rank - 1; d >= 0; --d) {
    if (shape[d] != 1 && strides[d] != expected)
      return false;
    expected *= shape[d];
  }
  return true;
}

} // namespace

extern "C" void hipdnn_ep_dump_tensor(RuntimeState *state, const void *data,
                                      const int64_t *shape,
                                      const int64_t *strides, int64_t rank,
                                      int64_t elem_bytes, const char *dtype,
                                      const char *name, int64_t index) {
  const TensorDumpConfig &cfg = config();
  if (cfg.dir.empty() || !state || !shape || !strides || !name || !dtype ||
      rank < 0 || elem_bytes <= 0)
    return;
  if (index < cfg.start || index > cfg.end || state->completed_inferences > 0)
    return;
  const std::string stem = fileStem(name);
  if (stem.empty() || !claimFile(state, stem))
    return;

  int64_t numel = 1;
  for (int64_t d = 0; d < rank; ++d)
    numel *= shape[d];
  if (numel < 0)
    return;

  // Copy the smallest span that covers every element, then pack it on the
  // host when the view is strided.
  const bool contiguous = isRowMajorContiguous(shape, strides, rank);
  int64_t spanElems = 0;
  if (numel > 0) {
    spanElems = contiguous ? numel : 1;
    if (!contiguous) {
      for (int64_t d = 0; d < rank; ++d) {
        if (strides[d] < 0) {
          fprintf(stderr, "[TensorDump] skip '%s': negative stride\n", name);
          return;
        }
        spanElems += (shape[d] - 1) * strides[d];
      }
    }
  }
  std::vector<unsigned char> span(static_cast<size_t>(spanElems * elem_bytes));
  if (!span.empty() &&
      (!data || hipdnn_ep_dump_copy_to_host(state, span.data(), data,
                                            span.size()) != 0)) {
    fprintf(stderr, "[TensorDump] skip '%s': copy to host failed\n", name);
    return;
  }

  const unsigned char *payload = span.data();
  std::vector<unsigned char> packed;
  if (!contiguous && numel > 0) {
    packed.resize(static_cast<size_t>(numel * elem_bytes));
    std::vector<int64_t> pos(static_cast<size_t>(rank), 0);
    for (int64_t i = 0; i < numel; ++i) {
      int64_t offset = 0;
      for (int64_t d = 0; d < rank; ++d)
        offset += pos[d] * strides[d];
      memcpy(packed.data() + i * elem_bytes, span.data() + offset * elem_bytes,
             static_cast<size_t>(elem_bytes));
      for (int64_t d = rank - 1; d >= 0; --d) {
        if (++pos[d] < shape[d])
          break;
        pos[d] = 0;
      }
    }
    payload = packed.data();
  }

  const std::string file = stem + ".npy";
  const std::string path = cfg.dir + "/" + file;
  FILE *f = fopen(path.c_str(), "wb");
  if (!f) {
    fprintf(stderr, "[TensorDump] cannot open %s\n", path.c_str());
    return;
  }
  const std::string header =
      npyHeader(npyDescr(dtype, elem_bytes), shape, rank);
  const size_t bytes = static_cast<size_t>(numel * elem_bytes);
  const bool ok = fwrite(header.data(), 1, header.size(), f) == header.size() &&
                  (bytes == 0 || fwrite(payload, 1, bytes, f) == bytes);
  fclose(f);
  if (!ok) {
    fprintf(stderr, "[TensorDump] short write to %s\n", path.c_str());
    return;
  }

  std::string line = "{\"index\":" + std::to_string(index) + ",\"name\":";
  appendJsonString(line, name);
  line += ",\"file\":";
  appendJsonString(line, file.c_str());
  line += ",\"dtype\":";
  appendJsonString(line, dtype);
  line += ",\"shape\":[";
  for (int64_t d = 0; d < rank; ++d) {
    if (d)
      line.push_back(',');
    line += std::to_string(shape[d]);
  }
  line += "]}\n";
  appendIndexLine(cfg.dir, line);

  RUNTIME_DEBUG_LOG("[TensorDump] #%lld %s -> %s (%lld elements, %s)\n",
                    (long long)index, name, path.c_str(), (long long)numel,
                    dtype);
}
