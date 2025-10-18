#include <allocator/pool.h>

constexpr std::size_t KiB = 1ull << 10;
constexpr std::size_t MiB = 1ull << 20;
constexpr std::size_t GiB = 1ull << 30;

MemoryPoolAllocator mallocator_temp("Temporary", 1 * GiB);
MemoryPoolAllocator mallocator_pers("Persistent", 0.5 * GiB);
