/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_COMPILER_API_H
#define HIP_COMPILER_API_H

#include "compiler_types.h"

/* Linkage macro.
 *
 * This API only ever ships inside the HipCInterface static archive, which is
 * absorbed into whatever binary needs it. On PE that means undecorated: a
 * dllimport declaration would make callers bind through an __imp_ thunk that no
 * import library provides, and dllexport would add the entry points to the host
 * DLL's export table for nothing, since callers reach them through the
 * in-process plugin registry rather than by name. Define HIP_COMPILER_EXPORTS
 * only when building a shared library that must publish the API.
 *
 * ELF keeps default visibility: it costs nothing because the EP's version
 * script and the tools' --exclude-libs already keep these names out of .dynsym.
 */
#if defined(_WIN32) && defined(HIP_COMPILER_EXPORTS)
#define COMPILER_API __declspec(dllexport)
#elif defined(_WIN32)
#define COMPILER_API
#else
#define COMPILER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compile MLIR input to DLL/object/IR file.
 *
 * onnx.Constant data is written to "constants.bin" via the provided
 * FileSystem. External constants are identified by the `location`
 * attribute on onnx.Constant ops in the MLIR bytecode.
 *
 * @param input_mlir      Input MLIR data (text or bytecode)
 * @param input_size      Size of input data in bytes
 * @param output_path     Output DLL/object/IR file path
 * @param options_json    Compilation options as JSON string (can be NULL)
 * @param error           Error information output (can be NULL)
 * @param fs              morphizen::FileSystem* (cast to void* for C ABI)
 * @return                COMPILER_SUCCESS or error code
 */
COMPILER_API CompilerErrorCode hip_compile_with_fs(
    const void *input_mlir, size_t input_size, const char *output_path,
    const char *options_json, CompilerError *error, void *fs);

/**
 * Get compiler version string.
 *
 * @return Static version string (e.g., "1.0.0")
 */
COMPILER_API const char *hip_get_version(void);

#ifdef __cplusplus
}
#endif

#endif /* HIP_COMPILER_API_H */
