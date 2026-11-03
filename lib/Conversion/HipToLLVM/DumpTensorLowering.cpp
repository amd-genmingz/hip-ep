/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.dump_tensor(%ctx) %buf {tensor_name, dtype, index} : memref<...>
//   -> %shape   = llvm.alloca rank x i64     // descriptor sizes
//      %strides = llvm.alloca rank x i64     // descriptor strides (elements)
//      llvm.call @hipdnn_ep_dump_tensor(state, data_ptr, %shape, %strides,
//                                       rank, elem_bytes, dtype, name, index)
//
// The data pointer includes the descriptor offset and the strides are passed
// through, so a subview or a pool-backed view is dumped as its logical tensor
// rather than as the base of the parent buffer.
struct DumpTensorOpLowering : public ConvertOpToLLVMPattern<DumpTensorOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  /// Pointer to a NUL-terminated module-level string. Globals are keyed by
  /// `symName`, so callers that pass the same symbol share one global.
  static Value globalCString(ModuleOp module,
                             ConversionPatternRewriter &rewriter, Location loc,
                             StringRef symName, StringRef value) {
    if (module.lookupSymbol<LLVM::GlobalOp>(symName))
      return LLVM::AddressOfOp::create(
          rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
          symName);
    std::string terminated = value.str();
    terminated.push_back('\0');
    return LLVM::createGlobalString(loc, rewriter, symName, terminated,
                                    LLVM::Linkage::Internal);
  }

  LogicalResult
  matchAndRewrite(DumpTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();

    auto memrefType = cast<MemRefType>(op.getInput().getType());
    Type elemType = memrefType.getElementType();
    if (!elemType.isIntOrFloat())
      return rewriter.notifyMatchFailure(op, "unsupported dump element type");
    const int64_t elemBytes = (elemType.getIntOrFloatBitWidth() + 7) / 8;
    const int64_t rank = memrefType.getRank();

    Value dataPtr = extractMemRefDataPtr(adaptor.getInput(), memrefType,
                                         typeConverter, rewriter, loc);
    if (!dataPtr)
      return rewriter.notifyMatchFailure(op, "failed to compute data ptr");

    auto i64Const = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    // A rank-0 buffer still gets a one-slot array so both pointers are valid.
    Value slots = i64Const(rank > 0 ? rank : 1);
    Value shapeArray =
        LLVM::AllocaOp::create(rewriter, loc, ptrType, i64Type, slots, 8);
    Value strideArray =
        LLVM::AllocaOp::create(rewriter, loc, ptrType, i64Type, slots, 8);
    MemRefDescriptor desc(adaptor.getInput());
    for (int64_t d : llvm::seq<int64_t>(rank)) {
      Value idx = i64Const(d);
      Value shapeSlot = LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type,
                                            shapeArray, ValueRange{idx});
      LLVM::StoreOp::create(rewriter, loc, desc.size(rewriter, loc, d),
                            shapeSlot);
      Value strideSlot = LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type,
                                             strideArray, ValueRange{idx});
      LLVM::StoreOp::create(rewriter, loc, desc.stride(rewriter, loc, d),
                            strideSlot);
    }

    const int64_t index = op.getIndex();
    Value namePtr = globalCString(module, rewriter, loc,
                                  ("__hip_dump_name_" + Twine(index)).str(),
                                  op.getTensorName());
    Value dtypePtr = globalCString(module, rewriter, loc,
                                   ("__hip_dump_dtype_" + op.getDtype()).str(),
                                   op.getDtype());

    // void hipdnn_ep_dump_tensor(RuntimeState*, const void* data,
    //                            const int64_t* shape, const int64_t* strides,
    //                            int64_t rank, int64_t elem_bytes,
    //                            const char* dtype, const char* name,
    //                            int64_t index)
    Type voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
    SmallVector<Type, 9> paramTypes = {ptrType, ptrType, ptrType,
                                       ptrType, i64Type, i64Type,
                                       ptrType, ptrType, i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipDumpTensor, paramTypes, voidTy);
    if (failed(funcOp))
      return failure();

    LLVM::CallOp::create(rewriter, loc, *funcOp,
                         ValueRange{adaptor.getCtx(), dataPtr, shapeArray,
                                    strideArray, i64Const(rank),
                                    i64Const(elemBytes), dtypePtr, namePtr,
                                    i64Const(index)});
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateDumpTensorLoweringPatterns(const LLVMTypeConverter &converter,
                                        RewritePatternSet &patterns) {
  patterns.add<DumpTensorOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
