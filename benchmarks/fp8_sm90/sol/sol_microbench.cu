// Latency microbenchmarks for the SM90 FP8 Gemma decode speed-of-light study.
//
//   nvcc -O3 -std=c++17 -arch=sm_90a -o sol_microbench sol_microbench.cu \
//        -L/usr/local/cuda/lib64/stubs -lcuda
//   ./sol_microbench [chase] [bulk] [burst] [const] [dsmem] [launch] > microbench.jsonl
//
// Every result is one JSON object per line on stdout (progress goes to stderr).
//   chase  : dependent-load latency, L2-warm and DRAM-cold over regions of 2 MB .. 4 GB (TLB reach)
//   bulk   : one CTA, one 16 KB tile as 1x16 KB bulk copy / 4 paged TMA boxes, cold and warm
//   burst  : N CTAs each issue the kernel's first loads (K0 4 boxes, Q 2 KB, V0, K1, V1) as TMA
//            paged boxes from a 115 MB cache, with or without (1,6,1)-style clusters; per-CTA
//            landing times of every load (queueing under load)
//   const  : constant-bank (kernel parameter) miss cost, adjacent vs scattered words
//   dsmem  : st.async + mbarrier ping-pong between cluster ranks, cluster barrier cost
//   launch : empty-kernel floor (grid 4x6x4, 256 threads, 88 KB smem) with / without cluster,
//            back-to-back and single launches, and CTA dispatch skew
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define CK(x)                                                                              \
  do {                                                                                     \
    cudaError_t e_ = (x);                                                                  \
    if (e_ != cudaSuccess) {                                                               \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));   \
      exit(1);                                                                             \
    }                                                                                      \
  } while (0)
#define CKD(x)                                                                             \
  do {                                                                                     \
    CUresult r_ = (x);                                                                     \
    if (r_ != CUDA_SUCCESS) {                                                              \
      const char* s_ = nullptr;                                                            \
      cuGetErrorString(r_, &s_);                                                           \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, s_ ? s_ : "?");            \
      exit(1);                                                                             \
    }                                                                                      \
  } while (0)

// ------------------------------------------------------------------------------ device helpers

__device__ __forceinline__ uint64_t gtimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ uint32_t clk() {
  uint32_t t;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));
  return t;
}
__device__ __forceinline__ uint32_t smid() {
  uint32_t s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void fence_mbar_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ bool mbar_test(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n\t.reg .pred p;\n\tmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
      "selp.u32 %0, 1, 0, p;\n\t}"
      : "=r"(ok)
      : "r"(bar), "r"(parity)
      : "memory");
  return ok != 0;
}
__device__ __forceinline__ bool mbar_try(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
      "selp.u32 %0, 1, 0, p;\n\t}"
      : "=r"(ok)
      : "r"(bar), "r"(parity)
      : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  while (!mbar_try(bar, parity)) {
  }
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes,
                                         uint32_t bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::
          "r"(dst),
      "l"(src), "r"(bytes), "r"(bar)
      : "memory");
}
__device__ __forceinline__ void tma_4d(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2,
                                       int c3, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4, %5}], [%6];" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(bar)
      : "memory");
}
__device__ __forceinline__ void prefetch_tmap(const CUtensorMap* map) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(map)) : "memory");
}
__device__ __forceinline__ uint32_t mapa(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
__device__ __forceinline__ void st_async_b32(uint32_t raddr, uint32_t v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(
                   raddr),
               "r"(v), "r"(rbar)
               : "memory");
}
__device__ __forceinline__ void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::
                   : "memory");
}

// ------------------------------------------------------------------------------ host helpers

static std::mt19937_64 g_rng(1234);
static void* g_flush = nullptr;
static const size_t kFlushBytes = size_t(512) << 20;

static void l2_flush(cudaStream_t s = 0) {
  static int v = 0;
  CK(cudaMemsetAsync(g_flush, (++v) & 0xff, kFlushBytes, s));
}

template <typename T>
static double median(std::vector<T> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return n % 2 ? double(v[n / 2]) : 0.5 * (double(v[n / 2 - 1]) + double(v[n / 2]));
}
template <typename T>
static std::string jarr(const std::vector<T>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) s += ",";
    s += std::to_string(v[i]);
  }
  return s + "]";
}

static int g_sm_clock_khz = 0;

// ------------------------------------------------------------------------------ chase

__global__ void scatter_u64(unsigned long long* buf, const unsigned long long* pos,
                            const unsigned long long* val, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) buf[pos[i]] = val[i];
}

__global__ void chase_kernel(const unsigned long long* __restrict__ buf, unsigned long long start,
                             int hops, int passes, unsigned long long* out) {
  unsigned long long p = start;
  for (int pass = 0; pass < passes; ++pass) {
    p = start;
    uint64_t t0 = gtimer();
    uint32_t c0 = clk();
    for (int i = 0; i < hops; ++i) p = __ldcg(buf + p);
    uint32_t c1 = clk();
    uint64_t t1 = gtimer();
    out[pass * 3 + 0] = t1 - t0;
    out[pass * 3 + 1] = c1 - c0;
    out[pass * 3 + 2] = p;
  }
}

static void run_chase() {
  const size_t total = size_t(4) << 30;
  unsigned long long* buf;
  CK(cudaMalloc(&buf, total));
  unsigned long long* out;
  CK(cudaMalloc(&out, 64 * sizeof(unsigned long long)));
  struct Cfg {
    const char* name;
    size_t region;
    int hops;
    bool warm;
  };
  const Cfg cfgs[] = {
      {"l2_warm_256k", size_t(256) << 10, 2048, true},
      {"l2_warm_8m", size_t(8) << 20, 2048, true},
      {"dram_2m", size_t(2) << 20, 2048, false},
      {"dram_32m", size_t(32) << 20, 2048, false},
      {"dram_256m", size_t(256) << 20, 2048, false},
      {"dram_1g", size_t(1) << 30, 2048, false},
      {"dram_4g", size_t(4) << 30, 2048, false},
  };
  for (const Cfg& c : cfgs) {
    for (int rep = 0; rep < 3; ++rep) {
      size_t lines = c.region / 128;
      size_t spare = total / 128 - lines;
      size_t base_line = spare ? (g_rng() % spare) & ~size_t(16383) : 0;  // 2 MB aligned
      std::vector<size_t> pick;
      pick.reserve(c.hops + 1);
      std::vector<char> used;
      if (lines <= (size_t(1) << 24)) used.assign(lines, 0);
      while ((int)pick.size() < c.hops + 1) {
        size_t l = g_rng() % lines;
        if (!used.empty()) {
          if (used[l]) continue;
          used[l] = 1;
        }
        pick.push_back(base_line + l);
      }
      std::vector<unsigned long long> pos(c.hops), val(c.hops);
      for (int i = 0; i < c.hops; ++i) {
        pos[i] = pick[i] * 16;
        val[i] = pick[i + 1] * 16;
      }
      unsigned long long *dpos, *dval;
      CK(cudaMalloc(&dpos, c.hops * 8));
      CK(cudaMalloc(&dval, c.hops * 8));
      CK(cudaMemcpy(dpos, pos.data(), c.hops * 8, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dval, val.data(), c.hops * 8, cudaMemcpyHostToDevice));
      scatter_u64<<<(c.hops + 255) / 256, 256>>>(buf, dpos, dval, c.hops);
      CK(cudaDeviceSynchronize());
      l2_flush();
      CK(cudaDeviceSynchronize());
      int passes = c.warm ? 2 : 1;
      chase_kernel<<<1, 1>>>(buf, pos[0], c.hops, passes, out);
      CK(cudaDeviceSynchronize());
      unsigned long long h[6];
      CK(cudaMemcpy(h, out, sizeof(h), cudaMemcpyDeviceToHost));
      int k = passes - 1;
      printf(
          "{\"test\":\"chase\",\"cfg\":\"%s\",\"region_bytes\":%zu,\"hops\":%d,\"rep\":%d,"
          "\"ns_per_hop\":%.2f,\"cycles_per_hop\":%.2f,\"cold_pass_ns_per_hop\":%.2f}\n",
          c.name, c.region, c.hops, rep, double(h[3 * k]) / c.hops,
          double(h[3 * k + 1]) / c.hops, double(h[0]) / c.hops);
      fflush(stdout);
      CK(cudaFree(dpos));
      CK(cudaFree(dval));
    }
  }
  CK(cudaFree(buf));
  CK(cudaFree(out));
}

// ------------------------------------------------------------------------------ paged cache + TMA

// The kernel's paged FP8 K (or V) cache: (nblk, page=16, hk=4, d=256) bytes; one TMA box is one
// page of one head (16 rows of 256 B at a 1 KB stride, 4 KB).
struct PagedCache {
  uint8_t* ptr = nullptr;
  int nblk = 0;
  CUtensorMap map;
};

static PagedCache make_cache(int nblk) {
  PagedCache c;
  c.nblk = nblk;
  size_t bytes = size_t(nblk) * 16 * 4 * 256;
  CK(cudaMalloc(&c.ptr, bytes));
  CK(cudaMemset(c.ptr, 1, bytes));
  cuuint64_t dims[4] = {256, 4, 16, cuuint64_t(nblk)};
  cuuint64_t strides[3] = {256, 1024, 16384};
  cuuint32_t box[4] = {256, 1, 16, 1};
  cuuint32_t estr[4] = {1, 1, 1, 1};
  CKD(cuTensorMapEncodeTiled(&c.map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, c.ptr, dims, strides, box,
                             estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                             CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return c;
}

// ------------------------------------------------------------------------------ bulk (1 CTA)

// mode 0: one 16 KB bulk copy (contiguous); mode 1: 4 TMA page boxes (4 KB each, random pages).
__global__ void __launch_bounds__(128) bulk_lat_kernel(const __grid_constant__ CUtensorMap tk,
                                                       const uint8_t* src, const int* pages,
                                                       int mode, unsigned long long* out) {
  extern __shared__ __align__(1024) uint8_t smem[];
  __shared__ __align__(8) uint64_t bar;
  __shared__ int sink;
  if (threadIdx.x != 0) return;
  uint32_t b = smem_u32(&bar);
  mbar_init(b, 1);
  fence_mbar_init();
  prefetch_tmap(&tk);
  int pg[4];
  for (int i = 0; i < 4; ++i) pg[i] = pages[i];
  sink = pg[0] + pg[1] + pg[2] + pg[3];  // the loads complete before the timed region
  uint64_t t0 = gtimer();
  uint32_t c0 = clk();
  mbar_expect_tx(b, 16384);
  if (mode == 0) {
    bulk_g2s(smem_u32(smem), src, 16384, b);
  } else {
    for (int i = 0; i < 4; ++i) tma_4d(smem_u32(smem) + 4096 * i, &tk, 0, 0, 0, pg[i], b);
  }
  uint32_t c1 = clk();
  mbar_wait(b, 0);
  uint32_t c2 = clk();
  uint64_t t2 = gtimer();
  out[0] = t2 - t0;
  out[1] = c1 - c0;
  out[2] = c2 - c0;
}

static void run_bulk(PagedCache& kc) {
  int* dpages;
  CK(cudaMalloc(&dpages, 4 * sizeof(int)));
  unsigned long long* out;
  CK(cudaMalloc(&out, 8 * sizeof(unsigned long long)));
  uint8_t* src = kc.ptr;
  CK(cudaFuncSetAttribute(bulk_lat_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768));
  for (int mode = 0; mode < 2; ++mode) {
    for (int warm = 0; warm < 2; ++warm) {
      std::vector<double> ns, issue_cyc, total_cyc;
      for (int rep = 0; rep < 20; ++rep) {
        int pg[4];
        for (int i = 0; i < 4; ++i) pg[i] = int(g_rng() % kc.nblk);
        CK(cudaMemcpy(dpages, pg, sizeof(pg), cudaMemcpyHostToDevice));
        const uint8_t* s = src + (size_t(pg[0]) * 16384);
        if (warm) {
          bulk_lat_kernel<<<1, 128, 32768>>>(kc.map, s, dpages, mode, out);
        } else {
          l2_flush();
        }
        bulk_lat_kernel<<<1, 128, 32768>>>(kc.map, s, dpages, mode, out);
        CK(cudaDeviceSynchronize());
        unsigned long long h[3];
        CK(cudaMemcpy(h, out, sizeof(h), cudaMemcpyDeviceToHost));
        ns.push_back(double(h[0]));
        issue_cyc.push_back(double(h[1]));
        total_cyc.push_back(double(h[2]));
      }
      printf(
          "{\"test\":\"bulk\",\"mode\":\"%s\",\"l2\":\"%s\",\"bytes\":16384,"
          "\"median_ns\":%.1f,\"min_ns\":%.1f,\"median_issue_cycles\":%.1f,"
          "\"median_total_cycles\":%.1f,\"ns\":%s}\n",
          mode == 0 ? "bulk_1x16k" : "tma_4x4k_pages", warm ? "warm" : "cold", median(ns),
          *std::min_element(ns.begin(), ns.end()), median(issue_cyc), median(total_cyc),
          jarr(std::vector<long long>(ns.begin(), ns.end())).c_str());
      fflush(stdout);
    }
  }
  CK(cudaFree(dpages));
  CK(cudaFree(out));
}

// ------------------------------------------------------------------------------ burst (N CTAs)

enum { EV_ENTRY = 0, EV_ARGS, EV_ISSUED, EV_K0, EV_Q, EV_V0, EV_K1, EV_V1, EV_SM, EV_N = 10 };

// Each CTA's thread 0 issues what the attention kernel's producer issues first: K0 (4 page
// boxes), Q (2 KB bulk), V0, K1, V1 (mode 0); K0 + Q only (mode 1); K0 only (mode 2). It then
// polls the five full barriers and records when each lands (globaltimer ns).
__global__ void __launch_bounds__(256) burst_kernel(const __grid_constant__ CUtensorMap tk,
                                                    const __grid_constant__ CUtensorMap tv,
                                                    const uint8_t* q, const int* pages, int mode,
                                                    unsigned long long* out) {
  extern __shared__ __align__(1024) uint8_t smem[];
  __shared__ __align__(8) uint64_t bars[5];
  uint64_t t_entry = gtimer();
  int cta = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  if (threadIdx.x == 0) {
    for (int i = 0; i < 5; ++i) mbar_init(smem_u32(&bars[i]), 1);
    fence_mbar_init();
  }
  __syncthreads();
  if (threadIdx.x != 0) return;
  prefetch_tmap(&tk);
  prefetch_tmap(&tv);
  int pg[16];
  for (int i = 0; i < 16; ++i) pg[i] = pages[cta * 16 + i];
  int head = cta % 4;
  uint64_t t_args = gtimer();
  uint32_t s0 = smem_u32(smem);
  uint32_t b[5];
  for (int i = 0; i < 5; ++i) b[i] = smem_u32(&bars[i]);
  // K0
  mbar_expect_tx(b[0], 16384);
  for (int i = 0; i < 4; ++i) tma_4d(s0 + 4096 * i, &tk, 0, head, 0, pg[i], b[0]);
  uint32_t pending = 1;
  if (mode <= 1) {  // Q
    mbar_expect_tx(b[1], 2048);
    bulk_g2s(s0 + 65536, q + size_t(cta) * 2048, 2048, b[1]);
    pending |= 2;
  }
  if (mode == 0) {  // V0, K1, V1
    mbar_expect_tx(b[2], 16384);
    for (int i = 0; i < 4; ++i) tma_4d(s0 + 16384 + 4096 * i, &tv, 0, head, 0, pg[4 + i], b[2]);
    mbar_expect_tx(b[3], 16384);
    for (int i = 0; i < 4; ++i) tma_4d(s0 + 32768 + 4096 * i, &tk, 0, head, 0, pg[8 + i], b[3]);
    mbar_expect_tx(b[4], 16384);
    for (int i = 0; i < 4; ++i) tma_4d(s0 + 49152 + 4096 * i, &tv, 0, head, 0, pg[12 + i], b[4]);
    pending |= 4 | 8 | 16;
  }
  uint64_t t_issued = gtimer();
  uint64_t land[5] = {0, 0, 0, 0, 0};
  int guard = 0;
  while (pending && ++guard < (1 << 24)) {
    for (int i = 0; i < 5; ++i) {
      if ((pending >> i & 1) && mbar_test(b[i], 0)) {
        land[i] = gtimer();
        pending &= ~(1u << i);
      }
    }
  }
  unsigned long long* o = out + size_t(cta) * EV_N;
  o[EV_ENTRY] = t_entry;
  o[EV_ARGS] = t_args;
  o[EV_ISSUED] = t_issued;
  for (int i = 0; i < 5; ++i) o[EV_K0 + i] = land[i];
  o[EV_SM] = smid();
}

static void run_burst(PagedCache& kc, PagedCache& vc) {
  const int max_cta = 264;
  uint8_t* q;
  CK(cudaMalloc(&q, size_t(max_cta) * 2048));
  CK(cudaMemset(q, 2, size_t(max_cta) * 2048));
  int* dpages;
  CK(cudaMalloc(&dpages, max_cta * 16 * sizeof(int)));
  unsigned long long* out;
  CK(cudaMalloc(&out, size_t(max_cta) * EV_N * 8));
  const int smem = 88 * 1024;
  CK(cudaFuncSetAttribute(burst_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  const int ns[] = {1, 6, 24, 48, 96, 132};
  const char* modes[] = {"full", "k0q", "k0"};
  for (int cluster = 0; cluster < 2; ++cluster) {
    for (int mode = 0; mode < 3; ++mode) {
      for (int n : ns) {
        if (cluster && n % 6) continue;
        for (int rep = 0; rep < 5; ++rep) {
          std::vector<int> pg(n * 16);
          for (auto& p : pg) p = int(g_rng() % kc.nblk);
          CK(cudaMemcpy(dpages, pg.data(), pg.size() * sizeof(int), cudaMemcpyHostToDevice));
          CK(cudaMemset(out, 0, size_t(n) * EV_N * 8));
          l2_flush();
          cudaLaunchConfig_t cfg = {};
          cudaLaunchAttribute attr[1];
          cfg.blockDim = dim3(256, 1, 1);
          cfg.dynamicSmemBytes = smem;
          cfg.stream = 0;
          if (cluster) {
            cfg.gridDim = dim3(n, 1, 1);
            attr[0].id = cudaLaunchAttributeClusterDimension;
            attr[0].val.clusterDim.x = 6;
            attr[0].val.clusterDim.y = 1;
            attr[0].val.clusterDim.z = 1;
            cfg.attrs = attr;
            cfg.numAttrs = 1;
          } else {
            cfg.gridDim = dim3(n, 1, 1);
            cfg.numAttrs = 0;
          }
          CK(cudaLaunchKernelEx(&cfg, burst_kernel, kc.map, vc.map, (const uint8_t*)q,
                                (const int*)dpages, mode, out));
          CK(cudaDeviceSynchronize());
          std::vector<unsigned long long> h(size_t(n) * EV_N);
          CK(cudaMemcpy(h.data(), out, h.size() * 8, cudaMemcpyDeviceToHost));
          unsigned long long t0 = ~0ull;
          for (int i = 0; i < n; ++i) t0 = std::min(t0, h[i * EV_N + EV_ENTRY]);
          std::vector<long long> ev[EV_N];
          for (int i = 0; i < n; ++i) {
            for (int e = 0; e < EV_N; ++e) {
              unsigned long long v = h[i * EV_N + e];
              if (e == EV_SM)
                ev[e].push_back((long long)v);
              else
                ev[e].push_back(v ? (long long)(v - t0) : -1);
            }
          }
          printf(
              "{\"test\":\"burst\",\"mode\":\"%s\",\"cluster\":%d,\"n_cta\":%d,\"rep\":%d,"
              "\"entry\":%s,\"args\":%s,\"issued\":%s,\"k0\":%s,\"q\":%s,\"v0\":%s,\"k1\":%s,"
              "\"v1\":%s,\"smid\":%s}\n",
              modes[mode], cluster ? 6 : 0, n, rep, jarr(ev[EV_ENTRY]).c_str(),
              jarr(ev[EV_ARGS]).c_str(), jarr(ev[EV_ISSUED]).c_str(), jarr(ev[EV_K0]).c_str(),
              jarr(ev[EV_Q]).c_str(), jarr(ev[EV_V0]).c_str(), jarr(ev[EV_K1]).c_str(),
              jarr(ev[EV_V1]).c_str(), jarr(ev[EV_SM]).c_str());
          fflush(stdout);
        }
      }
    }
  }
  CK(cudaFree(q));
  CK(cudaFree(dpages));
  CK(cudaFree(out));
}

// ------------------------------------------------------------------------------ constant bank

struct ParamBlob {
  uint32_t v[960];
};

// Chase through the kernel's parameter words (constant bank 0); every launch gets fresh
// parameters, so the first pass sees the same cold misses as a kernel prologue.
__global__ void const_chase_kernel(const __grid_constant__ ParamBlob p, int hops, int start,
                                   unsigned long long* out) {
  uint32_t idx = start;
  uint32_t c0 = clk();
  for (int i = 0; i < hops; ++i) idx = p.v[idx];
  uint32_t c1 = clk();
  uint32_t idx2 = start;
  for (int i = 0; i < hops; ++i) idx2 = p.v[idx2];
  uint32_t c2 = clk();
  out[0] = c1 - c0;
  out[1] = c2 - c1;
  out[2] = idx + idx2;
}

static void run_const() {
  unsigned long long* out;
  CK(cudaMalloc(&out, 4 * 8));
  struct Cfg {
    const char* name;
    int stride_words;
    int hops;
  };
  const Cfg cfgs[] = {{"adjacent_4B", 1, 48}, {"stride_64B", 16, 48}, {"stride_256B", 64, 14}};
  for (const Cfg& c : cfgs) {
    std::vector<double> cold, warm;
    for (int rep = 0; rep < 20; ++rep) {
      ParamBlob p;
      for (int i = 0; i < 960; ++i) p.v[i] = (i + c.stride_words) % 960;
      const_chase_kernel<<<1, 1>>>(p, c.hops, 0, out);
      CK(cudaDeviceSynchronize());
      unsigned long long h[3];
      CK(cudaMemcpy(h, out, sizeof(h), cudaMemcpyDeviceToHost));
      cold.push_back(double(h[0]) / c.hops);
      warm.push_back(double(h[1]) / c.hops);
    }
    printf(
        "{\"test\":\"const\",\"cfg\":\"%s\",\"hops\":%d,\"cold_cycles_per_hop\":%.1f,"
        "\"warm_cycles_per_hop\":%.1f,\"cold_ns_per_hop\":%.1f}\n",
        c.name, c.hops, median(cold), median(warm),
        g_sm_clock_khz ? median(cold) / (g_sm_clock_khz * 1e-6) : 0.0);
    fflush(stdout);
  }
  CK(cudaFree(out));
}

// ------------------------------------------------------------------------------ DSMEM

// Rank 0 and rank `peer` ping-pong one word through st.async + mbarrier complete_tx; rank 0
// times `rounds` round trips. Then every CTA times `rounds` cluster barriers.
__global__ void __launch_bounds__(32) dsmem_kernel(int peer, int rounds, unsigned long long* out) {
  __shared__ __align__(8) uint64_t bar;
  __shared__ uint32_t slot;
  uint32_t rank = cluster_rank();
  uint32_t b = smem_u32(&bar);
  if (threadIdx.x == 0) {
    mbar_init(b, 1);
    fence_mbar_init();
  }
  cluster_sync_all();
  if (threadIdx.x == 0) {
    uint32_t other = rank == 0 ? peer : 0;
    uint32_t rslot = mapa(smem_u32(&slot), other);
    uint32_t rbar = mapa(b, other);
    if (rank == 0) {
      uint64_t t0 = gtimer();
      uint32_t c0 = clk();
      for (int i = 0; i < rounds; ++i) {
        mbar_expect_tx(b, 4);
        st_async_b32(rslot, i, rbar);
        mbar_wait(b, i & 1);
      }
      uint32_t c1 = clk();
      uint64_t t1 = gtimer();
      out[0] = t1 - t0;
      out[1] = c1 - c0;
    } else if (rank == (uint32_t)peer) {
      for (int i = 0; i < rounds; ++i) {
        mbar_expect_tx(b, 4);
        mbar_wait(b, i & 1);
        st_async_b32(rslot, i, rbar);
      }
    }
  }
  __syncwarp();
  cluster_sync_all();
  uint64_t t2 = gtimer();
  for (int i = 0; i < rounds; ++i) cluster_sync_all();
  uint64_t t3 = gtimer();
  if (threadIdx.x == 0 && rank == 0) out[2] = t3 - t2;
}

static void run_dsmem() {
  unsigned long long* out;
  CK(cudaMalloc(&out, 4 * 8));
  const int rounds = 256;
  for (int peer = 1; peer < 6; ++peer) {
    std::vector<double> rtt, bar;
    for (int rep = 0; rep < 10; ++rep) {
      cudaLaunchConfig_t cfg = {};
      cudaLaunchAttribute attr[1];
      cfg.gridDim = dim3(6, 1, 1);
      cfg.blockDim = dim3(32, 1, 1);
      attr[0].id = cudaLaunchAttributeClusterDimension;
      attr[0].val.clusterDim.x = 6;
      attr[0].val.clusterDim.y = 1;
      attr[0].val.clusterDim.z = 1;
      cfg.attrs = attr;
      cfg.numAttrs = 1;
      CK(cudaLaunchKernelEx(&cfg, dsmem_kernel, peer, rounds, out));
      CK(cudaDeviceSynchronize());
      unsigned long long h[3];
      CK(cudaMemcpy(h, out, sizeof(h), cudaMemcpyDeviceToHost));
      rtt.push_back(double(h[0]) / rounds);
      bar.push_back(double(h[2]) / rounds);
    }
    printf(
        "{\"test\":\"dsmem\",\"peer\":%d,\"rounds\":%d,\"rtt_ns\":%.1f,"
        "\"cluster_barrier_ns\":%.1f}\n",
        peer, rounds, median(rtt), median(bar));
    fflush(stdout);
  }
  CK(cudaFree(out));
}

// ------------------------------------------------------------------------------ launch floor

__global__ void __launch_bounds__(256) empty_kernel() {}

__global__ void __launch_bounds__(256) empty_param_kernel(const __grid_constant__ ParamBlob p) {
  if (p.v[0] == 0xdeadbeef && threadIdx.x == 1000) asm volatile("trap;");
}

__global__ void __launch_bounds__(256) stamp_kernel(unsigned long long* out) {
  if (threadIdx.x == 0) {
    uint64_t t = gtimer();
    int b = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    out[b * 3 + 0] = t;
    out[b * 3 + 1] = smid();
    out[b * 3 + 2] = cluster_rank();
  }
}

static cudaLaunchConfig_t grid_cfg(bool cluster, cudaStream_t s, cudaLaunchAttribute* attr) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(4, 6, 4);
  cfg.blockDim = dim3(256, 1, 1);
  cfg.dynamicSmemBytes = 88 * 1024;
  cfg.stream = s;
  if (cluster) {
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 1;
    attr[0].val.clusterDim.y = 6;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
  }
  return cfg;
}

static void run_launch() {
  const int smem = 88 * 1024;
  CK(cudaFuncSetAttribute(empty_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  CK(cudaFuncSetAttribute(empty_param_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  CK(cudaFuncSetAttribute(stamp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  cudaStream_t s;
  CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  ParamBlob pb;
  memset(&pb, 0, sizeof(pb));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int which = 0; which < 2; ++which) {
    for (int cluster = 0; cluster < 2; ++cluster) {
      for (int batch : {1, 20}) {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        for (int i = 0; i < batch; ++i) {
          cudaLaunchAttribute attr[1];
          cudaLaunchConfig_t cfg = grid_cfg(cluster, s, attr);
          if (which == 0)
            CK(cudaLaunchKernelEx(&cfg, empty_kernel));
          else
            CK(cudaLaunchKernelEx(&cfg, empty_param_kernel, pb));
        }
        CK(cudaStreamEndCapture(s, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));
        for (int i = 0; i < 10; ++i) CK(cudaGraphLaunch(ge, s));
        std::vector<double> us;
        for (int it = 0; it < 200; ++it) {
          CK(cudaEventRecord(e0, s));
          CK(cudaGraphLaunch(ge, s));
          CK(cudaEventRecord(e1, s));
          CK(cudaEventSynchronize(e1));
          float ms;
          CK(cudaEventElapsedTime(&ms, e0, e1));
          us.push_back(ms * 1e3 / batch);
        }
        printf(
            "{\"test\":\"launch\",\"kernel\":\"%s\",\"cluster\":%d,\"launches_per_graph\":%d,"
            "\"median_us_per_launch\":%.3f,\"min_us_per_launch\":%.3f}\n",
            which ? "empty_param_3840B" : "empty", cluster, batch, median(us),
            *std::min_element(us.begin(), us.end()));
        fflush(stdout);
        CK(cudaGraphExecDestroy(ge));
        CK(cudaGraphDestroy(g));
      }
    }
  }
  // Dispatch skew: when does each of the 96 CTAs start?
  unsigned long long* out;
  CK(cudaMalloc(&out, 96 * 3 * 8));
  for (int cluster = 0; cluster < 2; ++cluster) {
    for (int rep = 0; rep < 5; ++rep) {
      cudaLaunchAttribute attr[1];
      cudaLaunchConfig_t cfg = grid_cfg(cluster, s, attr);
      CK(cudaLaunchKernelEx(&cfg, stamp_kernel, out));
      CK(cudaStreamSynchronize(s));
      std::vector<unsigned long long> h(96 * 3);
      CK(cudaMemcpy(h.data(), out, h.size() * 8, cudaMemcpyDeviceToHost));
      unsigned long long t0 = ~0ull;
      for (int i = 0; i < 96; ++i) t0 = std::min(t0, h[i * 3]);
      std::vector<long long> start, sm, rank;
      for (int i = 0; i < 96; ++i) {
        start.push_back((long long)(h[i * 3] - t0));
        sm.push_back((long long)h[i * 3 + 1]);
        rank.push_back((long long)h[i * 3 + 2]);
      }
      printf(
          "{\"test\":\"dispatch\",\"cluster\":%d,\"rep\":%d,\"start_ns\":%s,\"smid\":%s,"
          "\"rank\":%s}\n",
          cluster, rep, jarr(start).c_str(), jarr(sm).c_str(), jarr(rank).c_str());
      fflush(stdout);
    }
  }
  CK(cudaFree(out));
  CK(cudaStreamDestroy(s));
}

// ------------------------------------------------------------------------------ main

int main(int argc, char** argv) {
  std::vector<std::string> want;
  for (int i = 1; i < argc; ++i) want.push_back(argv[i]);
  auto on = [&](const char* s) {
    return want.empty() || std::find(want.begin(), want.end(), s) != want.end();
  };
  CK(cudaSetDevice(0));
  CKD(cuInit(0));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  int clk_khz = 0;
  cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0);
  g_sm_clock_khz = clk_khz;
  printf(
      "{\"test\":\"device\",\"name\":\"%s\",\"sms\":%d,\"l2_bytes\":%d,\"clock_khz\":%d,"
      "\"smem_per_sm\":%zu}\n",
      prop.name, prop.multiProcessorCount, prop.l2CacheSize, clk_khz,
      prop.sharedMemPerMultiprocessor);
  fflush(stdout);
  CK(cudaMalloc(&g_flush, kFlushBytes));
  if (on("chase")) {
    fprintf(stderr, "[microbench] chase\n");
    run_chase();
  }
  if (on("bulk") || on("burst")) {
    PagedCache kc = make_cache(7056), vc = make_cache(7056);
    if (on("bulk")) {
      fprintf(stderr, "[microbench] bulk\n");
      run_bulk(kc);
    }
    if (on("burst")) {
      fprintf(stderr, "[microbench] burst\n");
      run_burst(kc, vc);
    }
    CK(cudaFree(kc.ptr));
    CK(cudaFree(vc.ptr));
  }
  if (on("const")) {
    fprintf(stderr, "[microbench] const\n");
    run_const();
  }
  if (on("dsmem")) {
    fprintf(stderr, "[microbench] dsmem\n");
    run_dsmem();
  }
  if (on("launch")) {
    fprintf(stderr, "[microbench] launch\n");
    run_launch();
  }
  CK(cudaFree(g_flush));
  fprintf(stderr, "[microbench] done\n");
  return 0;
}
