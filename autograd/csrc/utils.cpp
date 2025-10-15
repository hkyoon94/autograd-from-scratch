#include <iostream>
#include <vector>


std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& v) {
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        os << v[i];
        if (i + 1 < v.size()) os << ", ";
    }
    os << "]";
    return os;
}
