#pragma once

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "utils/utils.h"


/*  Linear arena memory pool allocator.
    This object is operated in Singleton, i.e., global unique.
*/
class MemoryPoolAllocator {
    public:
        size_t total_  = 0;
        std::string name_;
        // TODO: currently under development
        // Constructor: Preallocates 1GB
        explicit MemoryPoolAllocator(std::string name, size_t total_bytes = 1ULL << 30)
            : name_(std::move(name))
        {
            constexpr size_t alignment = 64;
            static_assert((alignment & (alignment - 1)) == 0, "alignment must be power of 2");

            if (total_bytes > SIZE_MAX - (alignment - 1))
                throw std::bad_alloc();
            total_bytes = (total_bytes + (alignment - 1)) & ~(alignment - 1);

            base_ = static_cast<uint8_t*>(::operator new(total_bytes, std::align_val_t(alignment)));
            assert(reinterpret_cast<uintptr_t>(base_) % alignment == 0);

            total_ = total_bytes;
            used_  = 0;

            std::memset(base_, 0, total_bytes);

            DEBUG("Initialized " << name_
                << " memory-pool of size: " << total_bytes
                << " bytes, aligned " << alignment);
        }

        ~MemoryPoolAllocator() {
            if (base_) {
                ::operator delete(base_, std::align_val_t(64));
                base_ = nullptr;
            }
        }

        void* allocate(size_t bytes, size_t alignment = 64) {
            std::lock_guard<std::mutex> lock(mutex_);

            // Overflow guard for alignment
            if (used_ > SIZE_MAX - (alignment - 1))
                throw std::bad_alloc();

            // Align up safely
            size_t aligned_offset = (used_ + (alignment - 1)) & ~(alignment - 1);

            // Overflow guard for addition
            if (bytes > total_ - aligned_offset)
                throw std::bad_alloc();

            void* ptr = base_ + aligned_offset;

            // Runtime alignment sanity check (debug only)
            assert(reinterpret_cast<uintptr_t>(ptr) % alignment == 0);

            used_ = aligned_offset + bytes;
            return ptr;
        }

        // TODO: Deallocation request is not viable in this design.
        // void deallocate(void*) noexcept {}

        void reset() {
            std::lock_guard<std::mutex> lock(mutex_);
            used_ = 0;
            DEBUG("Resetted " << name_ << " memory-pool");
        }

        size_t used() { return this->used_; }

    private:
        uint8_t* base_ = nullptr;
        size_t used_   = 0;
        std::mutex mutex_;
    };


extern MemoryPoolAllocator mallocator_temp;
extern MemoryPoolAllocator mallocator_pers;  // reset() here should never be used during program
