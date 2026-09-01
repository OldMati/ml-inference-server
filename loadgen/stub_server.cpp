// stub_server.cpp — a fake inference server for validating the load generator BEFORE
// the real server exists. Serves GET /infer with ~5 ms of fake "inference", and
// periodically FREEZES the whole server for a spell.
//
// Build:  g++ -std=c++17 -O2 -pthread stub_server.cpp -o stub_server   (httplib.h in same dir)
// Run:    ./stub_server [port] [stall_every_s] [stall_ms]
//
// The freeze is the point: a coordinated-omission-SAFE generator will show a p99/p99.9
// spike when the stall hits (requests scheduled during the freeze pile up and complete
// late). A broken (closed-loop / actual-send-timed) generator will hide it entirely.
// That contrast is your acceptance test for the generator.

#include "httplib.h"
#include <thread>
#include <chrono>
#include <mutex>
#include <cstdio>
#include <string>

using namespace std::chrono;

int main(int argc, char** argv) {
    int    port          = argc > 1 ? std::stoi(argv[1]) : 8080;
    double stall_every_s = argc > 2 ? std::stod(argv[2]) : 5.0;
    int    stall_ms      = argc > 3 ? std::stoi(argv[3]) : 50;

    std::mutex world;   // handlers touch this briefly; the stall thread HOLDS it to freeze everyone

    // Background stall injector: every stall_every_s, seize `world` for stall_ms.
    std::thread injector([&]{
        for (;;) {
            std::this_thread::sleep_for(duration<double>(stall_every_s));
            std::lock_guard<std::mutex> lk(world);          // freeze: all handlers block here
            std::this_thread::sleep_for(milliseconds(stall_ms));
            fprintf(stderr, "[stub] injected %d ms stall\n", stall_ms);
        }
    });
    injector.detach();

    httplib::Server svr;
    svr.Get("/infer", [&](const httplib::Request&, httplib::Response& res) {
        { std::lock_guard<std::mutex> lk(world); }          // blocks fully during a freeze
        std::this_thread::sleep_for(milliseconds(5));       // baseline fake inference cost
        res.set_content("ok", "text/plain");
    });

    printf("stub inference server on http://127.0.0.1:%d/infer  (stall %d ms every %.0f s)\n",
           port, stall_ms, stall_every_s);
    svr.listen("127.0.0.1", port);
    return 0;
}