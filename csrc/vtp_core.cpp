#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <unordered_map>
#include <string>

namespace py = pybind11;

// Forward declaration of the CUDA implementation
torch::Tensor fast_p2p_transfer_cuda(torch::Tensor src, int dst_device);

// C++ wrapper that releases the Python GIL
torch::Tensor fast_p2p_transfer(torch::Tensor src, int dst_device) {
    // Release the Global Interpreter Lock (GIL) so Python threads can continue
    // while the GPU transfer happens asynchronously.
    pybind11::gil_scoped_release no_gil;
    
    return fast_p2p_transfer_cuda(src, dst_device);
}

// Hyper-fast C++ LRU/LFU Cache Scorer for HierarchicalMemoryManager
// Bypasses Python dictionary iteration overhead for millions of KV cache pages
py::dict compute_hotness_scores(
    const py::dict& access_counts,
    const py::dict& last_access_times,
    double current_time,
    double half_life
) {
    py::dict scores;
    double decay_constant = M_LN2 / half_life;

    for (auto item : access_counts) {
        auto key = item.first;
        double count = item.second.cast<double>();
        
        // If the key doesn't exist in last_access_times, default to current_time
        double last_time = current_time;
        if (last_access_times.contains(key)) {
            last_time = last_access_times[key].cast<double>();
        }
        
        double dt = current_time - last_time;
        if (dt < 0) dt = 0; // Prevent negative time delta
        
        double score = count * std::exp(-decay_constant * dt);
        scores[key] = score;
    }
    return scores;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fast_p2p_transfer", &fast_p2p_transfer, "Fast P2P Transfer releasing GIL");
    m.def("compute_hotness_scores", &compute_hotness_scores, "Fast LRU/LFU Cache Scorer");
}
