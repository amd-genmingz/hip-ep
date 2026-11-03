/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- TensorDump.cpp - Tensor-dump instrumentation passes ----------------===//
//
// Two passes that implement HIPDNN_EP_DUMP_DIR (docs/design/tensor-dump.md):
//
//   --hip-tag-onnx-output-locs  (before ONNX-to-HIP)
//     onnx.* ops carry their ONNX output tensor names in the `node.outputs`
//     attribute, which conversion does not copy onto the hip.* ops it creates.
//     Conversion does forward the Location, so the names are moved there:
//
//       loc(unknown)
//     ->
//       loc(fused<{hip.onnx_outputs = ["/m/Add_out"]}>[unknown])
//
//   --hip-insert-tensor-dump  (after memory planning, before HIP-to-LLVM)
//       hip.add(%ctx) ins(...) outs(%buf)  loc(fused<{hip.onnx_outputs = ...}>)
//     ->
//       hip.add(%ctx) ins(...) outs(%buf)
//       hip.dump_tensor(%ctx) %buf {tensor_name = "/m/Add_out", ...}
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Statistic.h"

#include <optional>
#include <string>

#define DEBUG_TYPE "hip-insert-tensor-dump"

STATISTIC(NumTaggedOps, "Number of onnx ops tagged with output names");
STATISTIC(NumDumpsInserted, "Number of hip.dump_tensor ops inserted");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_TAGONNXOUTPUTLOCSPASS
#define GEN_PASS_DEF_INSERTTENSORDUMPPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kOnnxOutputsKey = "hip.onnx_outputs";

/// The output-name array recorded by --hip-tag-onnx-output-locs, searched
/// through the Location tree because later rewrites may nest the tagged
/// location inside other fused / name / call-site locations.
ArrayAttr findOnnxOutputNames(Location loc) {
  if (auto fused = dyn_cast<FusedLoc>(loc)) {
    if (auto dict = dyn_cast_or_null<DictionaryAttr>(fused.getMetadata()))
      if (auto names = dict.getAs<ArrayAttr>(kOnnxOutputsKey))
        return names;
    for (Location inner : fused.getLocations())
      if (ArrayAttr names = findOnnxOutputNames(inner))
        return names;
    return {};
  }
  if (auto name = dyn_cast<NameLoc>(loc))
    return findOnnxOutputNames(name.getChildLoc());
  if (auto callSite = dyn_cast<CallSiteLoc>(loc)) {
    if (ArrayAttr names = findOnnxOutputNames(callSite.getCallee()))
      return names;
    return findOnnxOutputNames(callSite.getCaller());
  }
  return {};
}

/// NumPy-style name of a dumpable element type; std::nullopt for types the
/// dump index cannot describe (index, complex, ...), which are not dumped.
std::optional<llvm::StringRef> dumpDtypeName(Type type) {
  if (type.isF16())
    return llvm::StringRef("float16");
  if (type.isF32())
    return llvm::StringRef("float32");
  if (type.isBF16())
    return llvm::StringRef("bfloat16");
  if (type.isF64())
    return llvm::StringRef("float64");
  if (isa<Float8E4M3FNType>(type))
    return llvm::StringRef("float8_e4m3fn");
  auto intTy = dyn_cast<IntegerType>(type);
  if (!intTy)
    return std::nullopt;
  const bool isUnsigned = intTy.isUnsigned();
  switch (intTy.getWidth()) {
  case 1:
    return llvm::StringRef("bool");
  case 8:
    return llvm::StringRef(isUnsigned ? "uint8" : "int8");
  case 16:
    return llvm::StringRef(isUnsigned ? "uint16" : "int16");
  case 32:
    return llvm::StringRef(isUnsigned ? "uint32" : "int32");
  case 64:
    return llvm::StringRef(isUnsigned ? "uint64" : "int64");
  default:
    return std::nullopt;
  }
}

/// Tensor name for init `idx` of an op with `numInits` inits. Inits map to
/// ONNX outputs positionally when the counts agree; otherwise the first named
/// output gets an `_out<idx>` suffix (paired with the unsuffixed name by
/// accuracy_dirs.py). Empty when the ONNX output at `idx` is unnamed.
std::string dumpTensorName(ArrayAttr names, size_t idx, size_t numInits) {
  if (names.size() == numInits) {
    auto name = dyn_cast<StringAttr>(names[idx]);
    return name ? name.getValue().str() : std::string();
  }
  std::string base;
  for (Attribute attr : names) {
    auto name = dyn_cast<StringAttr>(attr);
    if (name && !name.getValue().empty()) {
      base = name.getValue().str();
      break;
    }
  }
  if (base.empty() || numInits == 1)
    return base;
  return base + "_out" + std::to_string(idx);
}

struct TagOnnxOutputLocsPass
    : public impl::TagOnnxOutputLocsPassBase<TagOnnxOutputLocsPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    getOperation().walk([&](Operation *op) {
      if (!op->getName().getStringRef().starts_with("onnx."))
        return;
      auto outputs = op->getAttrOfType<ArrayAttr>("node.outputs");
      if (!outputs)
        return;
      bool anyNamed = llvm::any_of(outputs, [](Attribute attr) {
        auto name = dyn_cast<StringAttr>(attr);
        return name && !name.getValue().empty();
      });
      if (!anyNamed)
        return;
      auto metadata = DictionaryAttr::get(
          ctx,
          {NamedAttribute(StringAttr::get(ctx, kOnnxOutputsKey), outputs)});
      op->setLoc(FusedLoc::get(ctx, {op->getLoc()}, metadata));
      ++NumTaggedOps;
    });
  }
};

struct InsertTensorDumpPass
    : public impl::InsertTensorDumpPassBase<InsertTensorDumpPass> {
  void runOnOperation() override {
    int64_t nextIndex = 0;
    getOperation().walk(
        [&](func::FuncOp func) { instrumentFunction(func, nextIndex); });
  }

private:
  struct Candidate {
    Operation *op;
    Value ctx;
    ArrayAttr names;
    SmallVector<Value> inits;
  };

  static void instrumentFunction(func::FuncOp func, int64_t &nextIndex) {
    auto *hipDialect = func->getContext()->getLoadedDialect<HipDialect>();
    SmallVector<Candidate> candidates;
    llvm::DenseMap<Attribute, size_t> lastForNames;
    for (Block &block : func.getBody()) {
      for (Operation &op : block) {
        if (op.getDialect() != hipDialect || isa<DumpTensorOp>(op))
          continue;
        auto dps = dyn_cast<DestinationStyleOpInterface>(&op);
        if (!dps || op.getNumOperands() == 0 ||
            !isa<ContextType>(op.getOperand(0).getType()))
          continue;
        ArrayAttr names = findOnnxOutputNames(op.getLoc());
        if (!names)
          continue;
        Candidate candidate{&op, op.getOperand(0), names, {}};
        for (OpOperand &init : dps.getDpsInitsMutable())
          if (isa<MemRefType>(init.get().getType()))
            candidate.inits.push_back(init.get());
        if (candidate.inits.empty())
          continue;
        lastForNames[names] = candidates.size();
        candidates.push_back(std::move(candidate));
      }
    }

    for (auto [pos, candidate] : llvm::enumerate(candidates)) {
      if (lastForNames.lookup(candidate.names) != pos)
        continue;
      OpBuilder builder(candidate.op->getContext());
      builder.setInsertionPointAfter(candidate.op);
      const size_t numInits = candidate.inits.size();
      for (auto [idx, init] : llvm::enumerate(candidate.inits)) {
        std::string name = dumpTensorName(candidate.names, idx, numInits);
        auto memrefTy = cast<MemRefType>(init.getType());
        std::optional<llvm::StringRef> dtype =
            dumpDtypeName(memrefTy.getElementType());
        if (name.empty() || !dtype)
          continue;
        DumpTensorOp::create(builder, candidate.op->getLoc(), candidate.ctx,
                             init, builder.getStringAttr(name),
                             builder.getStringAttr(*dtype),
                             builder.getI64IntegerAttr(nextIndex++));
        ++NumDumpsInserted;
      }
    }
  }
};

} // namespace
} // namespace hip
} // namespace mlir
