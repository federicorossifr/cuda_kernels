#include <chrono>
using  ns = std::chrono::nanoseconds;
using  ms = std::chrono::milliseconds;
using get_time = std::chrono::steady_clock ;

#define chronoIt(...) { \
auto start = get_time::now(); \
__VA_ARGS__ \
auto end = get_time::now(); \
std::cout << std::chrono::duration_cast<ns>(end - start).count() << std::endl;}

