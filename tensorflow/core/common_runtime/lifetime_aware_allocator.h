#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_LIFETIME_AWARE_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_LIFETIME_AWARE_ALLOCATOR_H_

#include <array>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <limits>
#include <cstdlib>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/common_runtime/allocator_retry.h"
#include "tensorflow/core/common_runtime/shared_counter.h"
#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class LifetimeAwareAllocator : public Allocator {
public:
    LifetimeAwareAllocator(SubAllocator* sub_allocator, 
                           BFCAllocator param_bfc_allocator, BFCAllocator act_bfc_allocator, 
                           const string& name, size_t total_memory = 0);
    ~LifetimeAwareAllocator() override;

    string Name() override { return name_; }

    void* AllocateRaw(size_t alignment, size_t num_bytes) override {
        return AllocateRaw(alignment, num_bytes, AllocationAttributes());
    }

    void* AllocateRaw(size_t alignment, size_t num_bytes,
                      const AllocationAttributes& allocation_attr) override;
    
    void DeallocateRaw(void* ptr) override;

    bool TracksAllocationSizes() const override;

    size_t RequestedSize(const void* ptr) const override;

    size_t AllocatedSize(const void* ptr) const override;

    int64_t AllocationId(const void* ptr) const override;

    absl::optional<AllocatorStats> GetStats() override;

    bool ClearStats() override;

    void SetTimingCounter(SharedCounter* sc) { 
        VLOG(2) << "Calling ClearStats";
        timing_counter_ = sc; 
    }

    bool ShouldRecordOpName() const { 
        VLOG(2) << "Calling ClearStats --> True";
        return true; 
    }

private:
    static size_t RoundedBytes(size_t bytes);

    void* AllocateRawInternal(size_t num_bytes, int64_t allocation_id=std::numeric_limits<int64_t>::min());



    void* base_ptr_ = nullptr;
    string name_ = "";
    size_t memory_limit_ = 0;
    std::unique_ptr<SubAllocator> sub_allocator_; // only for activations allocation
    std::unique_ptr<BFCAllocator> param_bfc_allocator_, activation_bfc_allocator_;
    SharedCounter* timing_counter_ = nullptr;

    size_t kMinAllocationSize = 256;

    mutable mutex lock_;

    AllocatorStats stats_ TF_GUARDED_BY(lock_);

    int64_t next_allocation_id_ TF_GUARDED_BY(lock_);

#ifdef TENSORFLOW_MEM_DEBUG
  int64 action_counter_ = 0 TF_GUARDED_BY(lock_);
#define MEM_DEBUG_SIZE_HISTORY_SIZE 4096
  int64 size_history_[MEM_DEBUG_SIZE_HISTORY_SIZE];
#endif
}

}

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_LIFETIME_AWARE_ALLOCATOR_H_
