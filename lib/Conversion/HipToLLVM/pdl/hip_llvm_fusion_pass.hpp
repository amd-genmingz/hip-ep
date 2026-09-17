/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// hip_llvm_fusion_pass.hpp — HIP-level PDL fusion framework for
// convert-hip-to-llvm.
//
// Standalone copy of the convert-onnx-to-hip PDL usage logic (see
// lib/Conversion/OnnxToHip/pdl/qdq_fusion_pass.hpp), specialized for hip.* op
// chains rather than onnx.* ones. The native helpers and run() live in
// namespace hip::llvmfusion so the two frameworks stay fully independent.
#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/MemoryBufferRef.h"

#include <optional>

namespace hip {
namespace llvmfusion {

// Per-tensor quantization only: a non-splat scale is per-axis and would not
// fold into a single coefficient. Only an inline dense `value` splat (e.g. a
// hip.constant carrier) is foldable; anything else leaves the chain to per-op
// lowering.
inline std::optional<float> trySplatScale(mlir::Value value) {
  if (!value)
    return std::nullopt;
  mlir::Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return std::nullopt;
  auto denseAttr =
      mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(defOp->getAttr("value"));
  if (!denseAttr || !denseAttr.isSplat())
    return std::nullopt;
  if (!mlir::isa<mlir::FloatType>(denseAttr.getType().getElementType()))
    return std::nullopt;
  return static_cast<float>(
      denseAttr.getSplatValue<mlir::FloatAttr>().getValueAsDouble());
}

// Companion for the zero point: signedness comes from the constant's own
// element type, which ONNX guarantees matches the quantized storage type.
inline std::optional<int64_t> trySplatZeropoint(mlir::Value value) {
  if (!value)
    return static_cast<int64_t>(0);
  mlir::Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return std::nullopt;
  auto denseAttr =
      mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(defOp->getAttr("value"));
  if (!denseAttr || !denseAttr.isSplat())
    return std::nullopt;
  auto intType =
      mlir::dyn_cast<mlir::IntegerType>(denseAttr.getType().getElementType());
  if (!intType)
    return std::nullopt;
  llvm::APInt raw = denseAttr.getSplatValue<llvm::APInt>();
  return intType.isUnsigned() ? static_cast<int64_t>(raw.getZExtValue())
                              : raw.getSExtValue();
}

//===----------------------------------------------------------------------===//
// Match constraints -- result-free, so several patterns may share them.
//===----------------------------------------------------------------------===//

inline mlir::LogicalResult
isSplatConstant(mlir::PatternRewriter &, mlir::PDLResultList &,
                llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 1)
    return mlir::failure();
  return mlir::success(
      trySplatScale(args[0].dyn_cast<mlir::Value>()).has_value());
}

//===----------------------------------------------------------------------===//
// Rewrite functions -- reached only after the constraints above accepted.
//===----------------------------------------------------------------------===//

inline mlir::LogicalResult
extractScale(mlir::PatternRewriter &rewriter, mlir::PDLResultList &results,
             llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 1)
    return mlir::failure();
  std::optional<float> scale = trySplatScale(args[0].dyn_cast<mlir::Value>());
  if (!scale)
    return mlir::failure();
  results.push_back(rewriter.getF32FloatAttr(*scale));
  return mlir::success();
}

inline mlir::LogicalResult
extractZeroPoint(mlir::PatternRewriter &rewriter, mlir::PDLResultList &results,
                 llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 1)
    return mlir::failure();
  std::optional<int64_t> zp =
      trySplatZeropoint(args[0].dyn_cast<mlir::Value>());
  if (!zp)
    return mlir::failure();
  results.push_back(rewriter.getI64IntegerAttr(*zp));
  return mlir::success();
}

// Apply the embedded PDL patterns. Mirrors hip::pdl::run: parse the buffer,
// register the native helpers, freeze, and greedily apply per FuncOp. An empty
// buffer is a no-op (the CMake fallback ships one when mlir-pdll is absent).
inline bool run(mlir::ModuleOp mlirModule, llvm::MemoryBufferRef pdlBuffer) {
  if (pdlBuffer.getBufferSize() == 0)
    return true;

  mlir::MLIRContext *ctx = mlirModule.getContext();

  mlir::ParserConfig parseConfig(ctx);
  // parseSourceString dispatches on the bytecode magic bytes, so textual IR
  // and bytecode both work.
  mlir::OwningOpRef<mlir::ModuleOp> pdlModule =
      mlir::parseSourceString<mlir::ModuleOp>(
          pdlBuffer.getBuffer(), parseConfig, pdlBuffer.getBufferIdentifier());
  if (!pdlModule)
    return false;

  // A module holding no pdl.pattern would make FrozenRewritePatternSet ask the
  // bytecode generator for a @matcher that was never produced; bail first.
  if (pdlModule->getOps<mlir::pdl::PatternOp>().empty())
    return true;

  mlir::PDLPatternModule pdlPatterns(std::move(pdlModule));
  pdlPatterns.registerConstraintFunction("IsSplatConstant", isSplatConstant);
  pdlPatterns.registerRewriteFunction("ExtractScale", extractScale);
  pdlPatterns.registerRewriteFunction("ExtractZeroPoint", extractZeroPoint);

  mlir::RewritePatternSet patterns(ctx);
  patterns.add(std::move(pdlPatterns));

  mlir::FrozenRewritePatternSet frozen(std::move(patterns));

  bool ok = true;
  mlirModule.walk([&](mlir::func::FuncOp funcOp) {
    if (mlir::failed(mlir::applyPatternsGreedily(funcOp, frozen)))
      ok = false;
  });
  return ok;
}

} // namespace llvmfusion
} // namespace hip
