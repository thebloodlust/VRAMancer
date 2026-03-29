// VRAMancer File Offload — Native Extension (formerly "software_cxl")
// GIL-free binary dump/load between RAM pointers and NVMe files.
// Multi-OS compatible: compiles transparently on Windows, macOS, Linux.
//
// NOTE: Despite the legacy function names containing "cxl", this is plain
// file I/O (std::ofstream / std::ifstream). It does NOT use CXL (Compute
// Express Link) hardware. The names are kept for backward compatibility
// with the Rust crate (vramancer_rust) which exposes the same API.

#include <pybind11/pybind11.h>
#include <fstream>
#include <stdexcept>
#include <cstdint>

namespace py = pybind11;

// File offload: dump raw memory pointer to disk file
// GIL released so Python can continue while NVMe I/O completes
void cxl_direct_memory_dump(const std::string& path, uintptr_t ptr, size_t num_bytes) {
    py::gil_scoped_release release;
    
    std::ofstream dest(path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!dest.is_open()) {
        throw std::runtime_error("File offload error: cannot open " + path);
    }
    dest.write(reinterpret_cast<const char*>(ptr), num_bytes);
    dest.close();
}

// File onload: read disk file into raw memory pointer
void cxl_direct_memory_load(const std::string& path, uintptr_t ptr, size_t num_bytes) {
    py::gil_scoped_release release;
    
    std::ifstream src(path, std::ios::binary | std::ios::in);
    if (!src.is_open()) {
        throw std::runtime_error("File offload error: cannot read " + path);
    }
    src.read(reinterpret_cast<char*>(ptr), num_bytes);
    src.close();
}

PYBIND11_MODULE(software_cxl, m) {
    m.doc() = "VRAMancer File Offload (GIL-free RAM<->NVMe binary dump/load)";
    m.def("cxl_direct_memory_dump", &cxl_direct_memory_dump, "Dump RAM pointer to file (GIL-free)");
    m.def("cxl_direct_memory_load", &cxl_direct_memory_load, "Load file into RAM pointer (GIL-free)");
}
