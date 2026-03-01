/**
 * @file cuda_memory.h
 * @brief RAII wrapper for CUDA device memory
 *
 * Replaces raw cudaMalloc/cudaFree patterns with a type-safe,
 * exception-safe buffer that manages its own lifetime.
 */

#ifndef LBM_CUDA_MEMORY_H
#define LBM_CUDA_MEMORY_H

#include <cuda_runtime.h>
#include <cstddef>
#include "utils/cuda_check.h"

namespace lbm {
namespace utils {

/**
 * @brief RAII wrapper for a contiguous block of CUDA device memory.
 *
 * Owns a single allocation of `n` elements of type `T`.
 * Not copyable; move-only semantics prevent accidental double-free.
 *
 * Example:
 *   CudaBuffer<float> buf(nx * ny * nz);
 *   buf.zero();
 *   myKernel<<<grid, block>>>(buf.get(), nx, ny, nz);
 */
template <typename T>
class CudaBuffer {
public:
    /// Default constructor: empty buffer, no allocation.
    CudaBuffer() : ptr_(nullptr), size_(0) {}

    /**
     * @brief Allocate device memory for `n` elements of type T.
     * @param n Number of elements (must be >= 0; 0 produces an empty buffer).
     * @throws std::runtime_error on CUDA allocation failure.
     */
    explicit CudaBuffer(int n) : ptr_(nullptr), size_(n) {
        if (n > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, bytes()));
        }
    }

    /// Non-copyable.
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    /// Move constructor: transfers ownership.
    CudaBuffer(CudaBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    /// Move assignment: transfers ownership, releases existing allocation.
    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            free();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    /// Destructor: frees device memory.
    ~CudaBuffer() { free(); }

    // -------------------------------------------------------------------------
    // Access
    // -------------------------------------------------------------------------

    /// Raw device pointer.
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }

    /// Const raw device pointer (explicit name for clarity in kernel launches).
    const T* const_get() const { return ptr_; }

    /// Implicit conversion to T* for ergonomic kernel argument passing.
    operator T*() { return ptr_; }
    operator const T*() const { return ptr_; }

    /// Number of elements.
    int size() const { return size_; }

    /// Allocation size in bytes.
    std::size_t bytes() const { return static_cast<std::size_t>(size_) * sizeof(T); }

    // -------------------------------------------------------------------------
    // Operations
    // -------------------------------------------------------------------------

    /**
     * @brief Zero-fill the buffer with cudaMemset.
     * @throws std::runtime_error on CUDA error.
     */
    void zero() {
        if (ptr_) {
            CUDA_CHECK(cudaMemset(ptr_, 0, bytes()));
        }
    }

    /**
     * @brief Reallocate the buffer for a new number of elements.
     *
     * Frees the current allocation (if any) and allocates fresh memory.
     * The new buffer contents are uninitialized.
     *
     * @param new_size Number of elements for the new allocation.
     * @throws std::runtime_error on CUDA error.
     */
    void reset(int new_size) {
        free();
        size_ = new_size;
        if (new_size > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, bytes()));
        }
    }

    // -------------------------------------------------------------------------
    // Factory
    // -------------------------------------------------------------------------

    /**
     * @brief Factory method: equivalent to `CudaBuffer<T>(n)`.
     *
     * Useful when the template argument must be stated explicitly:
     *   auto buf = CudaBuffer<float>::allocate(n);
     */
    static CudaBuffer<T> allocate(int n) { return CudaBuffer<T>(n); }

private:
    void free() {
        if (ptr_) {
            cudaFree(ptr_);  // Intentionally no CUDA_CHECK in destructor.
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    T* ptr_;
    int size_;
};

} // namespace utils
} // namespace lbm

#endif // LBM_CUDA_MEMORY_H
