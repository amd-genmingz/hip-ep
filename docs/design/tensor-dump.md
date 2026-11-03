<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Tensor dump

Debug instrumentation for accuracy triage: write the value of every ONNX tensor
the HIP graph computes during the first inference, named so that each file
pairs with ONNX Runtime's CPU node dump of the same model.

## Usage

| Variable | Read at | Meaning |
|---|---|---|
| `HIPDNN_EP_DUMP_DIR` | compile and run time | Dump directory. Unset, empty, or `0` disables the feature. |
| `HIPDNN_EP_DUMP_START` | run time | First dump index written (inclusive, default 0). |
| `HIPDNN_EP_DUMP_END` | run time | Last dump index written (inclusive, default unbounded). |

At compile time a set `HIPDNN_EP_DUMP_DIR` creates the directory if needed and
adds the instrumentation passes. If the directory cannot be created, the
compiler prints a warning and compiles without dumps.

At run time the dump ops write only when `HIPDNN_EP_DUMP_DIR` is set. An
artifact compiled with dumps therefore stays silent after the variable is
unset. Conversely, a cached artifact compiled without dumps never writes;
invalidate the model cache when turning the feature on.

The CPU reference comes from ONNX Runtime's own node dump
(`ORT_DEBUG_NODE_IO_DUMP_OUTPUT_DATA=1`,
`ORT_DEBUG_NODE_IO_DUMP_DATA_DESTINATION=files`). Both sides use the same file
stem for a tensor, so a script can pair the two directories by stem.

Every dump synchronizes the stream and copies to the host. Never benchmark with
the feature enabled.

## Output

For each dumped tensor:

- `<stem>.npy`: the logical tensor as an NPY v1.0 file (little-endian,
  row-major, strided views packed), so dtype and shape travel with the data
  and a comparison can type a headerless CPU-side `.bin` from it. `bfloat16`
  and `float8_e4m3fn` have no NumPy type and are stored as opaque `|V2` /
  `|V1` elements carrying the raw bits.
- One line appended to `index.jsonl`:

  ```json
  {"index":12,"name":"/model/layers.0/Add_output_0","file":"_model_layers_0_Add_output_0.npy","dtype":"float16","shape":[1,8,2048]}
  ```

`<stem>` is the ONNX tensor name with every character that is not an ASCII
letter or digit replaced by `_`, the ONNX Runtime node-dump rule. `index` is
the static order in which the compiler inserted the dump, which is the value
`HIPDNN_EP_DUMP_START`/`END` filter on. The index is rewritten by the first
dump of each process and is complete up to the last dump written, even if the
process crashes afterwards.

Only the first inference of each session is written. Finished inferences are
counted in `hipdnn_ep_stream_sync`, which `inference_compute` calls exactly
once on exit. `hipdnn_ep_runtime_begin_compute` cannot serve as the counter:
both the EP and `inference_compute` call it. Within that inference a tensor written
repeatedly, for example inside an outlined loop body, keeps its last value.
The first session to write a file owns it for the rest of the process, so a
later compiled graph with the same tensor names (such as a decode
specialization after prefill) does not overwrite it.

## Compiler side

`node.outputs`, the attribute the importer puts on each `onnx.*` op, is not
copied onto the `hip.*` ops that conversion creates, but the Location is.
Two passes, both added by `CompilerDriver` only when the feature is on, carry
the names across:

1. `hip-tag-onnx-output-locs` runs before `simplify-onnx`. It wraps each
   tagged op's Location in a `FusedLoc` whose metadata is
   `{hip.onnx_outputs = [...]}`.
2. `hip-insert-tensor-dump` runs after the ONNX→HIP tail (after pool
   planning, alloc lowering and extern-constant resolution) and before the
   HIP→LLVM pipeline. For each HIP destination-style op directly in a
   function body whose Location carries the names, it inserts one
   `hip.dump_tensor` per memref init, right after the op.

```mlir
hip.add(%ctx) ins(%a, %b : ...) outs(%buf : memref<1x8x2048xf16>)
hip.dump_tensor(%ctx) %buf {dtype = "float16", index = 12 : i64,
    tensor_name = "/model/layers.0/Add_output_0"} : memref<1x8x2048xf16>
```

Inits map to ONNX outputs positionally when their counts agree. Otherwise the
first named output is used, with an `_out<N>` suffix when the op has more than
one init. When one ONNX node lowered to several HIP ops, only the last of them
in program order is dumped, since it leaves the node's final value in the
buffer.

Inserting after memory planning keeps the instrumented program's buffer layout
identical to production and adds no allocation. The dump reads its buffer
before any later op can reuse that pool slot.

`hip.dump_tensor` lowers to `hipdnn_ep_dump_tensor(state, data, shape, strides,
rank, elem_bytes, dtype, name, index)`. The data pointer includes the
descriptor offset. The runtime copies the smallest span that covers the view
and packs it on the host.

## Limitations

- Ops nested in the region of another op (for example `hip.if` branches) are
  not instrumented; outlined loop bodies are, because they are functions.
- An ONNX tensor whose final value is assembled by several HIP ops writing
  slices of one buffer is dumped as the slice the last op writes.
- Tensors produced entirely by non-HIP code (folded constants, pure views such
  as `Reshape`) have no dump; the comparison simply has no file for them.
- Element types without a NumPy-style name (`index`, complex) are skipped.

## Code

| Piece | Location |
|---|---|
| Op | `include/hip/Dialect/IR/HipOps.td` (`hip.dump_tensor`) |
| Passes | `lib/Dialect/Transforms/TensorDump.cpp` |
| Pipeline wiring | `lib/Compiler/CompilerDriver.cpp` |
| Lowering | `lib/Conversion/HipToLLVM/DumpTensorLowering.cpp` |
| Runtime | `lib/Runtime/tensor_dump.cpp`, `hipdnn_ep_dump_copy_to_host` in `lib/Runtime/{real,mock}/memory.cpp` |
| Unit test | `test/runtime/test_tensor_dump.cpp` (GPU-free) |
