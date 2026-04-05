/**
 * TurboForward — C++ block dispatch for VRAMancer inference.
 *
 * Drives the transformer block loop in compiled C++, eliminating
 * per-block Python overhead:
 *   - No Python dict lookups for device resolution
 *   - No Python branch for device comparison
 *   - No Python exception handler setup per block
 *   - No per-block position_ids recomputation
 *   - No per-block rotary embedding recomputation
 *
 * All shared state (position_ids, position_embeddings, mask) is
 * pre-computed once in Python and passed in.  The C++ loop just
 * calls block.forward() with pre-computed arguments.
 *
 * Build (JIT):
 *   from torch.utils.cpp_extension import load
 *   turbo = load("turbo_forward", ["csrc/turbo_forward.cpp"])
 *
 * Usage:
 *   hidden = turbo.block_dispatch(hidden, blocks, cache, True, mask, pos_ids, pos_emb)
 */

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

/**
 * Fast block dispatch loop.
 *
 * Iterates over transformer blocks in compiled C++.
 * Each block is called as:
 *   (hidden_states, presents) = block(
 *       hidden_states,
 *       past_key_values=cache,
 *       use_cache=use_cache,
 *       attention_mask=mask,
 *       position_ids=position_ids,
 *       position_embeddings=position_embeddings,
 *   )
 *
 * For single-GPU: no device checks needed (all on same device).
 * For multi-GPU: caller handles device placement of hidden_states
 * before calling this function for each device's block group.
 */
torch::Tensor block_dispatch(
    torch::Tensor hidden_states,
    py::list blocks,
    py::object cache,
    bool use_cache,
    py::object attention_mask,
    torch::Tensor position_ids,
    py::object position_embeddings
) {
    const auto n_blocks = static_cast<int64_t>(py::len(blocks));

    for (int64_t i = 0; i < n_blocks; ++i) {
        py::object block = blocks[i];

        // Call block.forward() with keyword arguments
        py::object result = block(
            hidden_states,
            py::arg("past_key_values") = cache,
            py::arg("use_cache") = use_cache,
            py::arg("attention_mask") = attention_mask,
            py::arg("position_ids") = position_ids,
            py::arg("position_embeddings") = position_embeddings
        );

        // Unpack: block returns (hidden_states, presents) or just tensor
        if (py::isinstance<py::tuple>(result)) {
            auto tup = result.cast<py::tuple>();
            hidden_states = tup[0].cast<torch::Tensor>();
            // presents stored in-place in DynamicCache — no collection needed
        } else {
            hidden_states = result.cast<torch::Tensor>();
        }
    }

    return hidden_states;
}

/**
 * Multi-GPU block dispatch with device-aware transfers.
 *
 * Takes an additional vector of device indices (one per block).
 * When the current block's device differs from hidden_states.device,
 * moves hidden_states to the target device using non_blocking=True
 * on a dedicated CUDA stream, then synchronizes before compute.
 */
torch::Tensor block_dispatch_multi_gpu(
    torch::Tensor hidden_states,
    py::list blocks,
    py::object cache,
    bool use_cache,
    py::object attention_mask,
    torch::Tensor position_ids,
    py::object position_embeddings,
    std::vector<int64_t> block_devices
) {
    const auto n_blocks = static_cast<int64_t>(py::len(blocks));

    for (int64_t i = 0; i < n_blocks; ++i) {
        py::object block = blocks[i];

        // Check if we need a device transfer
        int64_t target_dev = block_devices[static_cast<size_t>(i)];
        auto current_dev = hidden_states.device();
        if (current_dev.is_cuda() && current_dev.index() != target_dev) {
            auto target = torch::Device(torch::kCUDA, static_cast<int16_t>(target_dev));
            hidden_states = hidden_states.to(target, /*non_blocking=*/true);
        }

        py::object result = block(
            hidden_states,
            py::arg("past_key_values") = cache,
            py::arg("use_cache") = use_cache,
            py::arg("attention_mask") = attention_mask,
            py::arg("position_ids") = position_ids,
            py::arg("position_embeddings") = position_embeddings
        );

        if (py::isinstance<py::tuple>(result)) {
            auto tup = result.cast<py::tuple>();
            hidden_states = tup[0].cast<torch::Tensor>();
        } else {
            hidden_states = result.cast<torch::Tensor>();
        }
    }

    return hidden_states;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "TurboForward — C++ block dispatch for VRAMancer inference";

    m.def("block_dispatch", &block_dispatch,
          "Fast single-GPU block dispatch (zero per-block Python overhead)",
          py::arg("hidden_states"),
          py::arg("blocks"),
          py::arg("cache"),
          py::arg("use_cache"),
          py::arg("attention_mask"),
          py::arg("position_ids"),
          py::arg("position_embeddings"));

    m.def("block_dispatch_multi_gpu", &block_dispatch_multi_gpu,
          "Multi-GPU block dispatch with device-aware transfers",
          py::arg("hidden_states"),
          py::arg("blocks"),
          py::arg("cache"),
          py::arg("use_cache"),
          py::arg("attention_mask"),
          py::arg("position_ids"),
          py::arg("position_embeddings"),
          py::arg("block_devices"));
}
