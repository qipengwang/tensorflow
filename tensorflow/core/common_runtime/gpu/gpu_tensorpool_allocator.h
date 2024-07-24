#ifndef TENSORFLOW_COMMON_RUNTIME_TENSORPOOL_ALLOCATOR_GPU_H_
#define TENSORFLOW_COMMON_RUNTIME_TENSORPOOL_ALLOCATOR_GPU_H_

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/lib/core/spin_lock.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
// #include "tensorflow/core/common_runtime/tensorpool_allocator.h"
#include "tensorflow/core/common_runtime/size_class.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"

#include <atomic>
#include <map>
#include <stack>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <limits>
#include <signal.h>
#include <sys/time.h>

namespace tensorflow {

namespace {

const SizeMap kSmallSizeMap;

inline size_t RoundedBytes(size_t bytes, size_t alignment) {
  return alignment * ((bytes + alignment - 1) / alignment);
}
}

struct GPUAllocStats {
  double begin;
  double end;
  size_t size;
  bool IsOverlap(const GPUAllocStats* other);
  std::string DebugString();
};

class GPUAllocBlock {
 public:
  GPUAllocBlock(size_t size, size_t bin_index);
  virtual ~GPUAllocBlock() {}

  void Insert(GPUAllocStats* alloc_stats);
  bool CanInsert(GPUAllocStats* alloc_stats);
  size_t BinIndex() const { return bin_index_; }
  void ResetStats();

 private:
  std::vector<GPUAllocStats*> stats_;  // not owned
  size_t size_;
  size_t bin_index_;
};

class VirtualGPUAllocBlock {
 public:
  VirtualGPUAllocBlock(GPUAllocBlock* block, size_t s) :
    internal_block_(block), size_(s) {
    };

  size_t BinIndex() const {
    return internal_block_->BinIndex();
  }

 private:
  GPUAllocBlock* internal_block_;
  size_t size_;
};

class GPULifetimePolicy;
class GPULifetimeBin {
 public:
  GPULifetimeBin(size_t bin_index, size_t chunk_size);
  virtual ~GPULifetimeBin();

  void TrackAllocate(size_t alignment);
  void TrackDeallocate(GPUAllocStats* stats);
  void BeginStep();
  size_t TotalMem() const;
  void Dump() const;
  void BestFit(GPULifetimePolicy* policy);
  void SmallFit();
  void Cleanup();

  GPUAllocBlock* FindBlock(GPUAllocStats* stats); // need memory-planner

  size_t BlockSize() const;
  size_t ChunkSize() const;
  size_t Alignment() const;
  size_t BinIndex() const { return bin_index_; }
  std::vector<VirtualGPUAllocBlock*>& VBlocks() {
    return virtual_blocks_;
  }

  void ResetStats();

 private:
  mutable spin_lock stats_lock_;
  std::vector<GPUAllocStats*> stats_;  // not owned
  std::vector<GPUAllocBlock*> blocks_;
  std::vector<VirtualGPUAllocBlock*> virtual_blocks_;
  size_t bin_index_;
  size_t chunk_size_;
  int64_t max_alignment_;
};

class GPULifetimePolicy {
 public:
  GPULifetimePolicy(size_t interval, size_t interval_offset, size_t start);
  virtual ~GPULifetimePolicy() {};

  void TrackAllocate(size_t alignment, size_t num_bytes);
  void TrackDeallocate(GPUAllocStats* stats);
  size_t TotalMem() const;

  void Dump() const;
  void Cleanup();

  GPUAllocBlock* FindBlock(GPUAllocStats* stats, size_t bin_index);

  void BestFit();
  size_t Interval();

  std::vector<GPULifetimeBin*>& GetBins();
  std::map<size_t, GPULifetimeBin*>& GetLargeBins();

  size_t Alignment() const;
  size_t AlignmentOffset() const;

  void ResetStats();

 private:
  GPULifetimeBin* GetBin(size_t index);

 private:
  std::vector<GPULifetimeBin*> bins_;
  std::map<size_t, GPULifetimeBin*> large_bins_;
  mutable spin_lock large_bin_lock_;
  const size_t interval_;
  const size_t interval_offset_;
  const size_t start_;
  const size_t large_bin_index_;
};

class GPUTensorPoolAllocator;
class GPUMemoryPlannerBase {
 public:
  virtual void SetAllocator(GPUTensorPoolAllocator* allocator) = 0;
  virtual void SetThreadPool(thread::ThreadPool* thread_pool) = 0;
  virtual void StartCollect() = 0;
  virtual void StopCollect() = 0;
  virtual void TrackAllocate(size_t alignment, size_t num_bytes, void* ptr) = 0;
  virtual void TrackDeallocate(void* ptr) = 0;
  virtual GPULifetimePolicy* BestLifetimePolicy() = 0;
  virtual std::vector<GPULifetimeBin*>& GetSmallBins() = 0;

  virtual void Reset() = 0;
};

class NullableGPUMemoryPlanner : public GPUMemoryPlannerBase {
  void SetAllocator(GPUTensorPoolAllocator* allocator) override {}
  void SetThreadPool(thread::ThreadPool* thread_pool) override {}
  void StartCollect() override {}
  void StopCollect() override {}
  void TrackAllocate(size_t alignment, size_t num_bytes, void* ptr) override {}
  void TrackDeallocate(void* ptr) override {}

  GPULifetimePolicy* BestLifetimePolicy() override {
    LOG(ERROR) << "Memory Optimization is disable, shouldn't be here";
    return nullptr;
  }

  std::vector<GPULifetimeBin*>& GetSmallBins() override {
    std::vector<GPULifetimeBin*> tmp;
    LOG(ERROR) << "Memory Optimization is disable, shouldn't be here";
    return tmp;
  }

  void Reset() override {}
};

class GPUMemoryPlanner : public GPUMemoryPlannerBase {
 public:
  GPUMemoryPlanner();
  virtual ~GPUMemoryPlanner();

  void SetAllocator(GPUTensorPoolAllocator* allocator) override;
  void SetThreadPool(thread::ThreadPool* thread_pool) override;

  void StartCollect() override;
  void StopCollect() override;
  void TrackAllocate(size_t alignment, size_t num_bytes, void* ptr) override;
  void TrackDeallocate(void* ptr) override;

  GPULifetimePolicy* BestLifetimePolicy() override;
  std::vector<GPULifetimeBin*>& GetSmallBins() override;
  void Reset() override;

 private:
  void Schedule(std::function<void()> f);
  void InitPolicy();
  void InitStepInfo();
  void CollectDone();
  void Cleanup();
  void ResetStats();
  void BestFit();

  GPULifetimeBin* GetSmallBin(size_t size);

 private:
  // statistics
  std::atomic_bool is_stats_;
  std::vector<GPULifetimePolicy*> lifetime_stats_polices_;
  std::vector<GPULifetimeBin*> small_bins_;

  GPUTensorPoolAllocator* allocator_;
  thread::ThreadPool* thread_pool_;

  mutable spin_lock stats_lock_;
  mutable spin_lock allocate_lock_;
  std::unordered_map<void*, GPUAllocStats*> ptr_stats_;
  std::vector<GPUAllocStats*> alloc_stats_;

  // step information
  std::atomic<int64_t> counter_;
  std::atomic<int64_t> logic_step_;
  int64 start_step_;
  int64 stop_step_;
  std::atomic_bool inited_;
};

class GPUMemoryPlannerFactory {
 public:
  static GPUMemoryPlannerBase* GetMemoryPlanner() {
    static GPUMemoryPlannerFactory factory;
    return factory.memory_planner_;
  }

 private:
  GPUMemoryPlannerFactory();

 private:
  bool enable_memory_opt_;
  GPUMemoryPlannerBase* memory_planner_;
};

class GPUScopedMemoryCollector {
 public:
  GPUScopedMemoryCollector(std::string filename = "DEFAULT", int line = -1) {
    VLOG(1) << "GPUScopedMemoryCollector starts collect memory allocation info at " << filename << ":" << line;
    GPUMemoryPlannerFactory::GetMemoryPlanner()->StartCollect();
  }
  ~GPUScopedMemoryCollector() {
    VLOG(1) << "GPUScopedMemoryCollector stops collect memory allocation info";
    GPUMemoryPlannerFactory::GetMemoryPlanner()->StopCollect();
  }
};

class GPUTensorPoolAllocator : public Allocator {
 public:
  GPUTensorPoolAllocator(SubAllocator* sub_allocator, Allocator* fallback_allocator, string name,
                      size_t total_memory);
  ~GPUTensorPoolAllocator() override;

  GPUTensorPoolAllocator(const GPUTensorPoolAllocator&) = delete;
  GPUTensorPoolAllocator& operator=(const GPUTensorPoolAllocator&) = delete;

  void Init();
  void BeginStep();

  string Name() override { return name_; }
  void* AllocateRaw(size_t alignment, size_t num_bytes, const AllocationAttributes& allocation_attr) override;
  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void* ptr) override;

  absl::optional<AllocatorStats> GetStats() override;

  void DumpStats();

  class Bin;
  Bin* GetBin(size_t bin_index);

  class VirtualBuffer {
   public:
    VirtualBuffer(std::vector<VirtualGPUAllocBlock*>& vblocks,
        GPUTensorPoolAllocator* tp);
    virtual ~VirtualBuffer() {}

    void* Allocate();
    void BeginStep();

   private:
    std::vector<Bin*> internal_bins_;
    std::atomic<size_t> curr_index_;
  };

  class Buffer {
   public:
    Buffer(size_t len, size_t chunk_size,
        size_t alignment, void* begin);

    void* Allocate();
    void Deallocate(void* p);

   private:
    mutable spin_lock lock_;
    std::stack<void*> buffer_;
    void* begin_;
    void* end_;
  };

  class Bin {
   public:
    Bin(size_t len, size_t chunk_size, size_t alignment,
        std::vector<VirtualGPUAllocBlock*>& vblocks,
        GPUTensorPoolAllocator* tp, void* begin);
    virtual ~Bin(){}

    Bin(const Bin&) = delete;
    Bin& operator=(const Bin&) = delete;

    void* Allocate();
    void* AllocateRaw();
    void DeallocateRaw(void* p);

    void BeginStep();

   private:
    Buffer buffer_;
    VirtualBuffer virtual_buffer_;
  };

  class SmallBin {
   public:
    SmallBin(size_t len, size_t chunk_size, size_t alignment, void* begin);
    virtual ~SmallBin(){}

    SmallBin(const SmallBin&) = delete;
    Bin& operator=(const Bin&) = delete;

    void* AllocateRaw();
    void DeallocateRaw(void* p);

   private:
    mutable spin_lock lock_;
    std::stack<void*> buffer_;
    void* begin_;
    void* end_;
  };

 private:
  bool IsBigOwned(void *ptr);
  bool IsSmallOwned(void *ptr);
  void* BigAllocate(size_t alignment, size_t num_bytes);
  void* BigAllocateStatistic(size_t alignment, size_t num_bytes);
  void BigDeallocate(void* ptr);

  SmallBin* GetSmallBin(size_t size);
  void* SmallAllocate(size_t alignment, size_t num_bytes);
  void SmallDeallocate(void* ptr);

 private:
  mutable spin_lock free_lock_;
  mutable spin_lock allocate_lock_;
  std::vector<void*> async_free_list_;
  string name_;
  AllocatorStats alloc_stats_;

  bool stats_;
  std::atomic_bool inited_;
  std::atomic_bool initing_;

  std::atomic_int step_id_;

  std::unique_ptr<SubAllocator> sub_allocator_;
  std::unique_ptr<Allocator> fallback_allocator_;
  GPUMemoryPlannerBase* mem_planner_;

  size_t large_bin_index_;
  std::vector<Bin*> lifetime_bins_;
  std::vector<SmallBin*> small_bins_;
  std::map<size_t, Bin*> large_lifetime_bins_;

  size_t alignment_;
  size_t alignment_offset_;
  size_t big_bytes_;
  void *big_mem_begin_;
  void *big_mem_end_;
  std::map<size_t, Bin*> offset_to_bin_;

  size_t small_bytes_;
  void *small_mem_begin_;
  void *small_mem_end_;
  std::map<size_t, SmallBin*> offset_to_small_bin_;
  std::set<void*> fallback_allocations_;

  // Statistic
  std::atomic<int64_t> null_bin_counter_;
  std::atomic<int64_t> hit_counter_;
  std::atomic<int64_t> missed_counter_;
};

}

#endif // TENSORFLOW_COMMON_RUNTIME_TENSORPOOL_ALLOCATOR_GPU_H_
