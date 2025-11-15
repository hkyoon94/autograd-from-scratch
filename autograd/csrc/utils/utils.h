#pragma once

#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <math.h>
#include <string.h>


inline bool is_integer(float x, double eps) {
    float r = std::round(x);
    return std::fabs(x - r) < eps;
}


std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& v);


template <typename KeyT>
struct Counter {
    std::unordered_map<KeyT, int> data;
    int& operator[](const KeyT& key) {
        return data[key];
    }

    void print() const {
        for (const auto& [k, v] : data) {
            std::cout << k << ": " << v << "\n";
        }
    }
};


/* ------------------------------------ MACROS ------------------------------------ */


#define PTR_ID(ptr) (reinterpret_cast<uintptr_t>(ptr))


#define THROWF(...) \
    do { \
        throw std::runtime_error(fmt::format(__VA_ARGS__)); \
    } while (0)


#ifndef NDEBUG
// When debugging, all console logs are redirected to "debug.log"
struct DebugRedirect {
    std::ofstream log_file;
    std::streambuf* cout_buf = nullptr;
    std::streambuf* cerr_buf = nullptr;

    DebugRedirect(const std::string& filename = "debug.log") {
        log_file.open(filename, std::ios::out | std::ios::trunc);
        cout_buf = std::cout.rdbuf(log_file.rdbuf());
        cerr_buf = std::cerr.rdbuf(log_file.rdbuf());
    }

    ~DebugRedirect() {
        std::cout.rdbuf(cout_buf);
        std::cerr.rdbuf(cerr_buf);
        log_file.close();
    }
};
static DebugRedirect __debug_redirect;
#endif


#ifndef NDEBUG
    /*  Defining DEBUG MODE ONLY scope.
        Usage:
            DEBUG_SCOPE("Checking NaN");
            {
                tensor.check_nan();
            }
        * Must be built with -DNDEBUG when use.
    */
    struct DebugScope {
        const char* name;
        DebugScope(const char* n) : name(n) {
            std::cerr << "Enter " << name << "\n";
        }
        ~DebugScope() {
            std::cerr << "Exit " << name << "\n";
        }
    };
    #define DEBUG_SCOPE(name) DebugScope debugScope(name);
#else
    #define DEBUG_SCOPE(name)
#endif


#ifndef NDEBUG
    /*  Debug mode only message.
        Supports continuous streaming, such as std::cout.
        Usage: DEBUG("Variable a: " << a << " says 'Hello world!'" << ");
        * Must be built with -DNDEBUG when use.
    */
    inline const char* _crop_fp(const char* path) {
        const char* pos = strstr(path, "/csrc/");
        return pos ? pos + 1 : path;  // only after /csrc/...
    }
    #define DEBUG(...) \
        do { \
            std::cerr << "[C++Backend] " << _crop_fp(__FILE__) << ":" << __LINE__ \
                      << " (" << __func__ << ") " \
                      << __VA_ARGS__ << std::endl; \
        } while (0)  // __VA_ARGS__ unpacks varargs from '...', in DEBUG(...).
#else
    #define DEBUG(...) ((void)0)
#endif


#ifndef NDEBUG
    /*  ASSERTION statement, that only compiled in debug build.
        * Must be built with -DNDEBUG when use.
    */
    #define DEBUG_ASSERT(expr) \
        do { \
            if (!(expr)) { \
                throw std::runtime_error( \
                    std::string("Assertion failed: (") + #expr + ")" \
                    " at " + _crop_fp(__FILE__) + ":" + std::to_string(__LINE__) + \
                    " in " + __func__ \
                ); \
            } \
        } while (0)
#else
    #define DEBUG_ASSERT(expr) ((void)0)
#endif


#ifndef NDEBUG
    /*  Function in-out scope tracer, that only compiled in debug build.
        Usage: TRACE_SCOPE(f());
        * Must be built with -DNDEBUG when use.
    */
    struct TraceScope {
        const char* func;
        TraceScope(const char* f) : func(f) {
            std::cerr << ">> Enter " << func << std::endl;
        }
        ~TraceScope() {
            std::cerr << "<< Exit " << func << std::endl;
        }
    };
    #define TRACE_SCOPE() TraceScope __trace(__func__)
#else
    #define TRACE_SCOPE()
#endif


#ifndef NDEBUG
    #define DUMP(var) \
        std::cerr << "[DUMP] " << #var << " = " << (var) << std::endl
    #else
    #define DUMP(var) ((void)0)
#endif
