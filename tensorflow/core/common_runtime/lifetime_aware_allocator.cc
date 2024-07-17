#include "tensorflow/core/common_runtime/lifetime_aware_allocator.h"

#include <atomic>

#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/allocator_retry.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#ifdef TENSORFLOW_MEM_DEBUG
    #include "tensorflow/core/platform/stacktrace.h"
#endif
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/bfc_memory_map.pb.h"

namespace tensorflow {

enum LifetimeAwareAllocatorStage {
    LIFETIME_AWARE_ALLOCATOR_STAGE_UNKNOWN,
    LIFETIME_AWARE_ALLOCATOR_STAGE_SETUP,
    LIFETIME_AWARE_ALLOCATOR_STAGE_WARMUP,
    LIFETIME_AWARE_ALLOCATOR_STAGE_TRAINING
};

LifetimeAwareAllocatorStage GetCurrentStageFromEnv() {
    string stage = getenv("TF_LIFETIME_AWARE_ALLOCATOR_STAGE");
    if(stage == "SETUP") {
        return LIFETIME_AWARE_ALLOCATOR_STAGE_SETUP;
    } else if(stage == "WARMUP") {
        return LIFETIME_AWARE_ALLOCATOR_STAGE_WARMUP;
    } else if(stage == "TRAINING") {
        return LIFETIME_AWARE_ALLOCATOR_STAGE_TRAINING;
    } else {
        return LIFETIME_AWARE_ALLOCATOR_STAGE_UNKNOWN;
    }
}

LifetimeAwareAllocator::LifetimeAwareAllocator(SubAllocator* sub_allocator, BFCAllocator param_bfc_allocator, BFCAllocator act_bfc_allocator, const string& name, size_t total_memory)    
   :sub_allocator_(sub_allocator), 
    name_(name),
    param_bfc_allocator_(param_bfc_allocator),
    activation_bfc_allocator_(act_bfc_allocator),
    next_allocation_id_(1),
    base_ptr(nullptr) {

    VLOG(2) << "LifetimeAwareAllocator::LifetimeAwareAllocator";
    memory_limit_ = total_memory;
    stats_.bytes_limit = static_cast<int64_t>(total_memory);
}

LifetimeAwareAllocator::~LifetimeAwareAllocator() {
    VLOG(2) << "LifetimeAwareAllocator::~LifetimeAwareAllocator";
    if (base_ptr != nullptr) {
        sub_allocator_->Free(base_ptr, memory_limit_);
    }
    param_bfc_allocator_.reset();
    activation_bfc_allocator_.reset();
}

bool LifetimeAwareAllocator::ClearStats() {
  VLOG(2) << "Calling ClearStats";
  mutex_lock l(lock_);
  param_bfc_allocator_->ClearStats();
  activation_bfc_allocator_->ClearStats();

  stats_.num_allocs = 0;
  stats_.peak_bytes_in_use = stats_.bytes_in_use;
  stats_.largest_alloc_size = 0;
  return true;
}

void* LifetimeAwareAllocator::AllocateRaw(size_t unused_alignment, size_t num_bytes,
                                const AllocationAttributes& allocation_attr) {
    VLOG(2) << "Calling AllocateRaw";
    switch (GetCurrentStageFromEnv()) {
    case LIFETIME_AWARE_ALLOCATOR_STAGE_WARMUP:
        return activation_bfc_allocator_->AllocateRaw(unused_alignment, num_bytes, allocation_attr);
    case LIFETIME_AWARE_ALLOCATOR_STAGE_TRAINING:
        return AllocateRawInternal(num_bytes);
    case LIFETIME_AWARE_ALLOCATOR_STAGE_SETUP:
    case LIFETIME_AWARE_ALLOCATOR_STAGE_UNKNOWN:
    default:
        return param_bfc_allocator_->AllocateRaw(unused_alignment, num_bytes, allocation_attr);
    }
}

void* LifetimeAwareAllocator::AllocateRawInternal(size_t num_bytes, int64_t allocation_id) {
    
}

// static
size_t LifetimeAwareAllocator::RoundedBytes(size_t bytes) {
  size_t rounded_bytes = kMinAllocationSize * ((bytes + kMinAllocationSize - 1) / kMinAllocationSize);
  DCHECK_EQ(size_t{0}, rounded_bytes % kMinAllocationSize);
  return rounded_bytes;
}

}
