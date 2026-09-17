##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# ============================================================================
# HIP-level PDLL Pattern Compilation
# ============================================================================
# Independent parallel of the OnnxToHip PDL fusion (lib/Conversion/OnnxToHip/
# CMakeLists.txt). Compiles pdl/*.pdll into a byte array embedded in the
# HipToLLVM library, with its own accessors (hip_llvm_pdl_fused_pattern_*).
# See pdl/hip_llvm_fusion_pass.hpp for the run()/native-function side.
set(HIPLLVM_PDL_TOOL_HINTS
    "${CMAKE_BINARY_DIR}/_deps/llvm-project-build/bin"
    "${LLVM_TOOLS_BINARY_DIR}"
    "${LLVM_BINARY_DIR}/bin"
    "${MLIR_BINARY_DIR}/bin"
)

if(TARGET mlir-pdll)
    set(HIPLLVM_MLIR_PDLL_EXE mlir-pdll)
else()
    find_program(HIPLLVM_MLIR_PDLL_EXE NAMES mlir-pdll HINTS ${HIPLLVM_PDL_TOOL_HINTS})
endif()

if(TARGET mlir-opt)
    set(HIPLLVM_MLIR_OPT_EXE mlir-opt)
else()
    find_program(HIPLLVM_MLIR_OPT_EXE NAMES mlir-opt HINTS ${HIPLLVM_PDL_TOOL_HINTS})
endif()

set(HIPLLVM_PDL_EMBED_CPP "${CMAKE_CURRENT_BINARY_DIR}/hip_llvm_pdl_fused_pattern_data.cpp")

if(HIPLLVM_MLIR_PDLL_EXE)
    find_package(Python3 COMPONENTS Interpreter)
    if(NOT Python3_FOUND)
        find_program(Python3_EXECUTABLE python)
    endif()
    if(NOT Python3_EXECUTABLE)
        message(FATAL_ERROR
            "HipToLLVM: Python not found, so the PDL fusion patterns cannot be "
            "embedded into the HipToLLVM library.")
    endif()

    # HipLLVMFusionPatterns.pdll #includes every pattern file, so mlir-pdll runs
    # once and emits one PDL module. Every .pdll is still globbed as a
    # dependency so editing an included pattern rebuilds it.
    set(HIPLLVM_PDL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pdl")
    set(HIPLLVM_PDL_ENTRY "${HIPLLVM_PDL_SOURCE_DIR}/HipLLVMFusionPatterns.pdll")
    file(GLOB HIPLLVM_PDL_SOURCES CONFIGURE_DEPENDS "${HIPLLVM_PDL_SOURCE_DIR}/*.pdll")
    set(HIPLLVM_PDL_TEXT "${CMAKE_CURRENT_BINARY_DIR}/HipLLVMFusionPatterns.pdl.mlir")
    if(TARGET mlir-pdll)
        set(HIPLLVM_MLIR_PDLL_COMMAND "$<TARGET_FILE:mlir-pdll>")
        set(HIPLLVM_MLIR_PDLL_DEPENDS mlir-pdll)
    else()
        set(HIPLLVM_MLIR_PDLL_COMMAND "${HIPLLVM_MLIR_PDLL_EXE}")
        set(HIPLLVM_MLIR_PDLL_DEPENDS "${HIPLLVM_MLIR_PDLL_EXE}")
    endif()
    add_custom_command(OUTPUT "${HIPLLVM_PDL_TEXT}"
        COMMAND "${HIPLLVM_MLIR_PDLL_COMMAND}" -x mlir "-I${HIPLLVM_PDL_SOURCE_DIR}"
                "${HIPLLVM_PDL_ENTRY}" -o "${HIPLLVM_PDL_TEXT}"
        DEPENDS ${HIPLLVM_PDL_SOURCES} ${HIPLLVM_MLIR_PDLL_DEPENDS}
        COMMENT "Compiling HIP-level PDLL patterns -> HipLLVMFusionPatterns.pdl.mlir"
        VERBATIM)

    if(HIPLLVM_MLIR_OPT_EXE)
        if(TARGET mlir-opt)
            set(HIPLLVM_MLIR_OPT_COMMAND "$<TARGET_FILE:mlir-opt>")
            set(HIPLLVM_MLIR_OPT_DEPENDS mlir-opt)
        else()
            set(HIPLLVM_MLIR_OPT_COMMAND "${HIPLLVM_MLIR_OPT_EXE}")
            set(HIPLLVM_MLIR_OPT_DEPENDS "${HIPLLVM_MLIR_OPT_EXE}")
        endif()
        # Strip debug info (absolute source paths) and emit bytecode before
        # embedding.
        set(HIPLLVM_PDL_EMBED_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/HipLLVMFusionPatterns.pdl.mlirbc")
        add_custom_command(OUTPUT "${HIPLLVM_PDL_EMBED_SOURCE}"
            COMMAND "${HIPLLVM_MLIR_OPT_COMMAND}" --strip-debuginfo --emit-bytecode
                    "${HIPLLVM_PDL_TEXT}" -o "${HIPLLVM_PDL_EMBED_SOURCE}"
            DEPENDS "${HIPLLVM_PDL_TEXT}" ${HIPLLVM_MLIR_OPT_DEPENDS}
            COMMENT "Condensing HIP-level PDL patterns -> HipLLVMFusionPatterns.pdl.mlirbc"
            VERBATIM)
    else()
        message(STATUS
            "HipToLLVM: mlir-opt not found; embedding the textual PDL module.")
        set(HIPLLVM_PDL_EMBED_SOURCE "${HIPLLVM_PDL_TEXT}")
    endif()

    # xxd.py terminates the array with an extra 0x00, so the accessor reports
    # the real file size instead of sizeof().
    add_custom_command(OUTPUT "${HIPLLVM_PDL_EMBED_CPP}"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/cmake/xxd.py"
                --var hip_llvm_pdl_fused_pattern
                --output "${HIPLLVM_PDL_EMBED_CPP}.tmp"
                "${HIPLLVM_PDL_EMBED_SOURCE}"
        COMMAND "${Python3_EXECUTABLE}" -c
                "import os,sys; tmp,out,blob = sys.argv[1:4]; n = os.path.getsize(blob); open(out,'w').write('#include <cstddef>\\n' + open(tmp).read() + f'\\nextern \"C\" const unsigned char *hip_llvm_pdl_fused_pattern_data(void) {{ return hip_llvm_pdl_fused_pattern; }}\\nextern \"C\" size_t hip_llvm_pdl_fused_pattern_size(void) {{ return {n}; }}\\n')"
                "${HIPLLVM_PDL_EMBED_CPP}.tmp" "${HIPLLVM_PDL_EMBED_CPP}" "${HIPLLVM_PDL_EMBED_SOURCE}"
        COMMAND "${CMAKE_COMMAND}" -E remove "${HIPLLVM_PDL_EMBED_CPP}.tmp"
        DEPENDS "${HIPLLVM_PDL_EMBED_SOURCE}"
        COMMENT "Embedding HIP-level PDL patterns via hip_llvm_pdl_fused_pattern_data.cpp"
        VERBATIM)
    add_custom_target(HipLLVMPdllPatterns DEPENDS "${HIPLLVM_PDL_EMBED_CPP}")
    message(STATUS "HipToLLVM: PDLL fusion enabled — embedding ${HIPLLVM_PDL_EMBED_SOURCE}")
else()
    # No mlir-pdll -> empty pattern blob. hip::llvmfusion::run treats size 0 as
    # a no-op, so unrelated targets still build.
    file(WRITE "${HIPLLVM_PDL_EMBED_CPP}"
        "#include <cstddef>\n"
        "static const unsigned char hip_llvm_pdl_fused_pattern[] = {0};\n"
        "extern \"C\" const unsigned char *hip_llvm_pdl_fused_pattern_data(void) {\n"
        "  return hip_llvm_pdl_fused_pattern;\n"
        "}\n"
        "extern \"C\" size_t hip_llvm_pdl_fused_pattern_size(void) { return 0; }\n"
    )
    message(WARNING
        "HipToLLVM: mlir-pdll not found; HIP-level PDL fusion disabled "
        "(empty blob). Install mlir-pdll or put it on PATH to enable it.")
endif()

# ============================================================================
# HipToLLVM Library
# ============================================================================

add_library(HipToLLVM STATIC
  "${HIPLLVM_PDL_EMBED_CPP}"
  HipToLLVM.cpp
  MemoryLowering.cpp
  ConvLowering.cpp
  ConvTransposeLowering.cpp
  MatmulLowering.cpp
  RocMlirLowering.cpp
  ElementwiseLowering.cpp
  PowerLowering.cpp
  ActivationLowering.cpp
  BiasGeluLowering.cpp
  FastGeluLowering.cpp
  NormLowering.cpp
  GatherLowering.cpp
  GatherElementsLowering.cpp
  TopKLowering.cpp
  ScatterElementsLowering.cpp
  CompressLowering.cpp
  OneHotLowering.cpp
  RangeLowering.cpp
  CastLowering.cpp
  ReduceLowering.cpp
  TransposeLowering.cpp
  RopeLowering.cpp
  GqaLowering.cpp
  MultiHeadAttentionLowering.cpp
  MatMulNBitsLowering.cpp
  QMoELowering.cpp
  QMoEAmdLowering.cpp
  GatherBlockQuantizedLowering.cpp
  QdqLowering.cpp
  GraphLowering.cpp
  CausalConvWithStateLowering.cpp
  GemmLowering.cpp
  WhereLowering.cpp
  EqualLowering.cpp
  OrLowering.cpp
  AndLowering.cpp
  DivLowering.cpp
  LinearAttentionLowering.cpp
  UnaryElementwiseLowering.cpp
  CumSumLowering.cpp
  PadLowering.cpp
  TileLowering.cpp
  ExpandLowering.cpp
  LessLowering.cpp
  GatherNDLowering.cpp
  ModLowering.cpp
  SliceLowering.cpp
  ScatterNDLowering.cpp
  NonZeroLowering.cpp
  ReadbackDimLowering.cpp
  ReadbackScalarLowering.cpp
  SizeLowering.cpp
  LoopLowering.cpp
  IfLowering.cpp
  PoolLowering.cpp
  ResizeLowering.cpp
  GridSampleLowering.cpp
  GlobalPoolLowering.cpp
  QElementwiseLowering.cpp
  QMatMulLowering.cpp
  QGemmLowering.cpp
  QConvLowering.cpp
  QLpNormalizationLowering.cpp
  QSigmoidLowering.cpp
)

if(TARGET HipLLVMPdllPatterns)
    add_dependencies(HipToLLVM HipLLVMPdllPatterns)
endif()

add_dependencies(HipToLLVM
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
  HipPassesIncGen
)

target_link_libraries(HipToLLVM PUBLIC
  HipDialectIR
  MLIRPass
  MLIRLLVMDialect
  MLIRLLVMCommonConversion
  MLIRMemRefDialect
  MLIRTransforms
  MLIRArithTransforms
  # convert-hip-to-llvm populates the standard dialect->LLVM lowering patterns
  # (func/memref/arith/cf) directly in HipToLLVM.cpp. Link them here so every
  # consumer inherits them transitively in the correct order instead of each
  # executable re-listing them (and getting the link order right by luck).
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  # ConvertToLLVMPatternInterface: convert-hip-to-llvm walks the module and
  # collects HIP->LLVM lowerings from out-of-tree (plugin) dialects that
  # implement this interface. See the guarded walk in HipToLLVM.cpp.
  MLIRConvertToLLVMInterface
  # HIP-level PDL fusion slot (pdl/hip_llvm_fusion_pass.hpp): parse + apply the
  # embedded PDL patterns before the LLVM lowering.
  MLIRPDLDialect
  MLIRPDLInterpDialect
  MLIRParser
)
