// Build:  g++ -std=c++17 -O2 -pthread loadgen.cpp -o loadgen   (httplib.h in same dir)
// Out:    results.csv  (raw per-request rows for pandas/matplotlib) + a summary line.

// Architecture: one DISPATCHER walks a fixed Poisson schedule and pushes intended
// send-times into a queue; a POOL of workers pull jobs, issue the request over a
// reused keep-alive connection, and record latency measured from the INTENDED time.

#include "httplib.h"
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <random>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <cstdio>
#include <string>
#include <stdexcept>

using namespace std::chrono;

struct Job {
    steady_clock::time_point intended;
    bool stop = false;                 // sentinel that tells a worker to exit
};

struct Record {
    double intended_ms;                // when it was SUPPOSED to go (relative to run start)
    double send_lag_ms;                // actual_send - intended: how late the CLIENT was
    double latency_ms;                 // completion - intended  (the CO-correct number)
    int    status;                     // HTTP status, or -1 on connection failure
    bool   warmup;
};

// A minimal thread-safe blocking queue.
class JobQueue {
    std::queue<Job> q_;
    std::mutex m_;
    std::condition_variable cv_;
public:
    void push(Job j) {
        { std::lock_guard<std::mutex> lk(m_); q_.push(j); }
        cv_.notify_one();
    }
    Job pop() {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&]{ return !q_.empty(); });
        Job j = q_.front(); q_.pop();
        return j;
    }
};

constexpr size_t IMAGE_BYTES = 3 * 224 * 224 * 4; 

// std::vector<char> payload(50, 0x7f);  // TO REMOVE

std::vector<char> load_fixture(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("can't open fixture: " + path);
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (buf.size() != IMAGE_BYTES) {
        throw std::runtime_error("fixture is " + std::to_string(buf.size()) +
                                " bytes, expected " + std::to_string(IMAGE_BYTES));
    }
    return buf;
}

int main(int argc, char** argv) {
    // ---- config (all overridable on the command line) ----
    double rate       = argc > 1 ? std::stod(argv[1]) : 500.0;   // requests/sec
    int    workers    = argc > 2 ? std::stoi(argv[2]) : 128;      // pool size
    double duration_s = argc > 3 ? std::stod(argv[3]) : 10.0;    // measurement window
    std::string host  = argc > 4 ? argv[4] : "127.0.0.1";
    int    port       = argc > 5 ? std::stoi(argv[5]) : 8080;
    std::string img_path = argc > 6 ? argv[6] : "fixtures/sample_input.bin";
    std::string path = argc > 7 ? argv[7] : "/predict";
    std::string out = argc > 8 ? argv[8] : "results.csv";
    const double warmup_s = 5.0;                                 // discard first 5 s

    std::vector<char> image;
    try {
        image = load_fixture(img_path);
    } catch (const std::exception& e) {
        std::cerr << "fixture load failed: " << e.what() << "\n";
        return 1;
    }

    JobQueue queue;
    std::vector<Record> records;
    std::mutex rec_m;
    records.reserve((size_t)(rate * duration_s * 1.2));

    const auto run_start = steady_clock::now();

    // ---- worker: owns one keep-alive client, reused for every request ----
    auto worker_fn = [&]() {
        httplib::Client cli(host, port);
        cli.set_keep_alive(true);
        cli.set_tcp_nodelay(true);
        for (;;) {
            Job job = queue.pop();
            if (job.stop) break;

            auto actual_send = steady_clock::now();
            auto res = cli.Post(path, image.data(), image.size(), "application/octet-stream");
            // auto res = cli.Post("/echo", image.data(), image.size(), "application/octet-stream");
            // auto res  = cli.Get("/healthz");
            auto done = steady_clock::now();

            double latency_ms  = duration<double, std::milli>(done - job.intended).count();
            double intended_ms = duration<double, std::milli>(job.intended - run_start).count();
            double send_lag_ms = duration<double, std::milli>(actual_send - job.intended).count();
            int    status      = res ? res->status : -1;
            bool   warm        = intended_ms < warmup_s * 1000.0;

            std::lock_guard<std::mutex> lk(rec_m);
            records.push_back({intended_ms, send_lag_ms, latency_ms, status, warm});
        }
    };

    std::vector<std::thread> pool;
    for (int i = 0; i < workers; ++i) pool.emplace_back(worker_fn);

    // ---- dispatcher: fixed Poisson schedule, advanced independent of responses ----
    std::mt19937 rng{std::random_device{}()};
    std::exponential_distribution<double> gap(rate);            // exp inter-arrivals = Poisson
    auto next   = run_start;
    auto end_tp = run_start + duration_cast<steady_clock::duration>(duration<double>(duration_s));
    long long fired = 0;
    while (next < end_tp) {
        next += duration_cast<steady_clock::duration>(duration<double>(gap(rng)));
        std::this_thread::sleep_until(next);   // if we're behind, returns instantly -> lateness shows up
        Job j; j.intended = next; j.stop = false;
        queue.push(j);
        ++fired;
    }

    // ---- drain: one stop-sentinel per worker, then join ----
    for (int i = 0; i < workers; ++i) { Job s; s.stop = true; queue.push(s); }
    for (auto& t : pool) t.join();

    // ---- write raw per-request CSV (this is what your analysis scripts consume) ----
    std::ofstream csv(out);
    csv << "intended_ms,send_lag_ms,latency_ms,status,phase\n";
    for (auto& r : records)
        csv << r.intended_ms << "," << r.send_lag_ms << "," << r.latency_ms << ","
            << r.status << "," << (r.warmup ? "warmup" : "measure") << "\n";

    // ---- steady-state summary (excludes warmup + failures) ----
    std::vector<double> lat, lag, srv;
    for (auto& r : records)
        if (!r.warmup && r.status == 200) {
            lat.push_back(r.latency_ms);
            lag.push_back(r.send_lag_ms);
            srv.push_back(r.latency_ms - r.send_lag_ms);
        }
    std::sort(lat.begin(), lat.end());
    std::sort(lag.begin(), lag.end());
    std::sort(srv.begin(), srv.end());
    auto pct = [](const std::vector<double>& v, double p){
        return v.empty() ? 0.0 : v[(size_t)(p * (v.size() - 1))];
    };

    printf("fired=%lld  measured=%zu\n", fired, lat.size());
    printf("  latency         p50=%.2f  p99=%.2f  p99.9=%.2f ms\n", pct(lat,0.50), pct(lat,0.99), pct(lat,0.999));
    printf("  send_lag        p50=%.2f  p99=%.2f  max=%.2f ms\n",   pct(lag,0.50), pct(lag,0.99), pct(lag,1.0));
    printf("  server_response p50=%.2f  p99=%.2f  p99.9=%.2f ms\n", pct(srv,0.50), pct(srv,0.99), pct(srv,0.999));
    return 0;
}
