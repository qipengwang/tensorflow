// #include "tensorflow/core/common_runtime/memory_planner.h"
#include "tensorflow/core/common_runtime/tensorpool_allocator.h"
#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/env_var.h"
#include <algorithm>
#include <limits>
#include <signal.h>
#include <sys/time.h>

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

namespace tensorflow {

namespace {
constexpr int64 DEFAULT_START_STATISTIC_STEP = 100;
constexpr int64 DEFAULT_STABLE_STATISTIC_STEP = 10;
constexpr int64 DEFAULT_MAX_STATISTIC_STEP = 100;
}

MemoryPlanner::MemoryPlanner() :
    is_stats_(false),
    allocator_(nullptr),
    thread_pool_(nullptr),
    counter_(0),
    start_step_(DEFAULT_START_STATISTIC_STEP),
    stable_step_(DEFAULT_STABLE_STATISTIC_STEP),
    max_stat_step_(DEFAULT_MAX_STATISTIC_STEP),
    current_stable_step_(0),
    current_stat_step_(0) {
  InitPolicy();
  InitStepInfo();
}

MemoryPlanner::~MemoryPlanner() {
}

void MemoryPlanner::InitPolicy() {
  lifetime_stats_polices_.emplace_back(
      new LifetimePolicy(_4KB, _4KB_OFFSET, _32KB));
  lifetime_stats_polices_.emplace_back(
      new LifetimePolicy(_8KB, _8KB_OFFSET, _32KB));
  lifetime_stats_polices_.emplace_back(
      new LifetimePolicy(_16KB, _16KB_OFFSET, _32KB));
}

void MemoryPlanner::InitStepInfo() {
  Status s = ReadInt64FromEnvVar("START_STATISTIC_STEP",
      DEFAULT_START_STATISTIC_STEP,
      &start_step_);
  if (!s.ok()) {
    LOG(FATAL) << "Read START_STATISTIC_STEP envrionment error. "
                << s.error_message();
  }
  s = ReadInt64FromEnvVar("STABLE_STATISTIC_STEP",
      DEFAULT_STABLE_STATISTIC_STEP,
      &stable_step_);
  if (!s.ok()) {
    LOG(FATAL) << "Read STABLE_STATISTIC_STEP envrionment error. "
                << s.error_message();
  }
  s = ReadInt64FromEnvVar("MAX_STATISTIC_STEP",
      DEFAULT_MAX_STATISTIC_STEP,
      &max_stat_step_);
  if (!s.ok()) {
    LOG(FATAL) << "Read MAX_STATISTIC_STEP envrionment error. "
                << s.error_message();
  }
}

// lifetime policy
LifetimePolicy* MemoryPlanner::BestLifetimePolicy() {
  LifetimePolicy* best_policy = nullptr;
  auto total_mem = std::numeric_limits<size_t>::max();
  for (auto policy : lifetime_stats_polices_) {
    auto policy_mem = policy->TotalMem();
    if (policy_mem < total_mem) {
      best_policy = policy;
      total_mem = policy_mem;
    }
  }
  // LOG(INFO_DEV) << "MemoryPlanner's best lifetime policy consume Memory:"
  //          << total_mem;
  return best_policy;
}

void MemoryPlanner::Reset() {
  counter_ = 0;
  Cleanup();
}

void MemoryPlanner::StartCollect() {
  auto current = counter_.fetch_add(1);
  if (current == start_step_) {
    is_stats_ = true;
  }
}

void MemoryPlanner::StopCollect() {
  bool tmp = true;
  // stop collecting stat when generating policy
  // and avoid multiple threads entering
  if (is_stats_.compare_exchange_strong(tmp, false)) {
    Schedule([this]() {
      ++current_stat_step_;
      bool stable = true;
      for (auto policy : lifetime_stats_polices_) {
        if (!policy->BestFit()) {
          stable = false;
        }
      }
      if (stable) {
        ++current_stable_step_;
      } else {
        current_stable_step_ = 0;
      }
      if (current_stable_step_ > stable_step_
          || current_stat_step_ > max_stat_step_) {
        VLOG(2) << "end planner: " << current_stat_step_;
        CollectDone();
      } else {
        is_stats_ = true;
      }
    });
  }
}

void MemoryPlanner::CollectDone() {
  if (allocator_ != nullptr) {
    allocator_->Init();
  }
  Cleanup();
}

void MemoryPlanner::Cleanup() {
  for (auto policy : lifetime_stats_polices_) {
    policy->Cleanup();
  }
}

void MemoryPlanner::SetAllocator(TensorPoolAllocator* allocator) {
  allocator_ = allocator;
}

void MemoryPlanner::SetThreadPool(thread::ThreadPool* thread_pool) {
  if (thread_pool_ == nullptr) {
    thread_pool_ = thread_pool;
  }
}
void MemoryPlanner::Schedule(std::function<void()> f) {
  if (thread_pool_ == nullptr) {
    f();
  } else {
    thread_pool_->Schedule(std::move(f));
  }
}

void MemoryPlanner::TrackAllocate(size_t alignment, size_t num_bytes) {
  if (!is_stats_.load()) {
    return;
  }
  for (auto lifetime_policy : lifetime_stats_polices_) {
    lifetime_policy->TrackAllocate(alignment, num_bytes);
  }
}

void MemoryPlanner::TrackDeallocate(Header* header) {
  if (!is_stats_.load()) {
    return;
  }
  for (auto lifetime_policy : lifetime_stats_polices_) {
    lifetime_policy->TrackDeallocate(header);
  }
}

LifetimePolicy::LifetimePolicy(size_t interval,
    size_t interval_offset, size_t start) :
    interval_(interval), interval_offset_(interval_offset), start_(start),
    large_bin_index_(Index(_32MB, interval_, interval_offset_) + 1) {
  auto cur = start_ + interval_;
  bins_.resize(large_bin_index_);
  for (auto i = 0; i < large_bin_index_; ++i) {
    bins_[i] = new LifetimeBin(i, cur);
    cur += interval_;
  }
}

void LifetimePolicy::TrackAllocate(size_t alignment, size_t num_bytes) {
  auto index = Index(num_bytes, interval_, interval_offset_);
  if (index < 0) {
    LOG(ERROR) << "TensorPoolAllocator Invalid Index:" << index
               << ", size:" << num_bytes;
    return;
  }
  GetBin(index)->TrackAllocate(alignment);
}

LifetimeBin* LifetimePolicy::GetBin(size_t index) {
  if (index >= large_bin_index_) {
    std::lock_guard<spin_lock> l(large_bin_lock_);
    auto bin = large_bins_.find(index);
    if (bin == large_bins_.end()) {
      auto chunk_size = start_ + interval_ * (index + 1);
      bin = large_bins_.emplace(index, new LifetimeBin(index, chunk_size)).first;
    }
    return bin->second;
  } else {
    return bins_[index];
  }
}

void LifetimePolicy::TrackDeallocate(Header* header) {
  timeval tmp;
  gettimeofday(&tmp, nullptr);
  header->end = Timeval2Double(tmp);

  auto alloc_stats = new AllocStats;
  alloc_stats->begin = header->begin;
  alloc_stats->end = header->end;
  alloc_stats->size = header->total_size;
  
  auto index = Index(alloc_stats->size, interval_, interval_offset_);
  if (index < 0) {
    LOG(ERROR) << "TensorPoolAllocator Invalid Index:" << index
               << ", size:" << alloc_stats->size;
    return;
  }
  GetBin(index)->TrackDeallocate(alloc_stats);
}

size_t LifetimePolicy::TotalMem() const {
  size_t total_mem = 0;
  for (auto bin : bins_) {
    total_mem += bin->TotalMem();
  }
  {
    std::lock_guard<spin_lock> l(large_bin_lock_);
    for (auto large_bin : large_bins_) {
      auto bin_info = large_bin.second;
      total_mem += bin_info->TotalMem();
    }
  }
  return total_mem;
}

void LifetimePolicy::Dump() const {
  // LOG(INFO_DEV) << "LifetimePolicy, start:" << start_
  //          << ", interval:" << interval_
  //          << ", Detail:";
  for (auto& b : bins_) {
    b->Dump();
  }
  {
    std::lock_guard<spin_lock> l(large_bin_lock_);
    for (auto& large_bin : large_bins_) {
      auto bin_info = large_bin.second;
      bin_info->Dump();
    }
  }
}

void LifetimePolicy::Cleanup() {
  for (auto bin : bins_) {
    bin->Cleanup();
  }
  {
    std::lock_guard<spin_lock> l(large_bin_lock_);
    for (auto bin : large_bins_) {
      auto bin_info = bin.second;
      bin_info->Cleanup();
    }
  }
}

LifetimeBin::LifetimeBin(size_t bin_index, size_t chunk_size)
    : bin_index_(bin_index),
      chunk_size_(chunk_size),
      max_alignment_(Allocator::kAllocatorAlignment) {
}

LifetimeBin::~LifetimeBin() {
}

void LifetimeBin::TrackAllocate(size_t alignment) {
  max_alignment_ = std::max<int64_t>(max_alignment_, alignment);
}

void LifetimeBin::TrackDeallocate(AllocStats* stats) {
  // multiple thread enter
  std::lock_guard<spin_lock> l(stats_lock_);
  stats_.emplace_back(stats);
}

bool LifetimePolicy::BestFit() {
  bool stable = true;
  std::lock_guard<spin_lock> l(large_bin_lock_);
  for (auto it = large_bins_.rbegin();
      it != large_bins_.rend(); ++it) {
    auto bin_info = it->second;
    bool ret = bin_info->BestFit(this);
    if (!ret) {
      stable = false;
    }
  }
  for (auto it = bins_.rbegin(); it != bins_.rend(); ++it) {
    bool ret = (*it)->BestFit(this);
    if (!ret) {
      stable = false;
    }
  }
  return stable;
}

std::vector<LifetimeBin*>& LifetimePolicy::GetBins() {
  return bins_;
}

std::map<size_t, LifetimeBin*>& LifetimePolicy::GetLargeBins() {
  return large_bins_;
}

size_t LifetimePolicy::Alignment() const {
  return interval_;
}

size_t LifetimePolicy::AlignmentOffset() const {
  return interval_offset_;
}

void LifetimeBin::Cleanup() {
  for (auto block : blocks_) {
    delete block;
  }
  blocks_.clear();

  for (auto vblock : virtual_blocks_) {
    delete vblock;
  }
  virtual_blocks_.clear();

  // stats_ pointer's memory would be clear by blocks.
  // protect stats_ could be touched by other thread.
  std::lock_guard<spin_lock> l(stats_lock_);
  stats_.clear();
}

size_t LifetimePolicy::Interval() {
  return interval_;
}

bool LifetimeBin::BestFit(LifetimePolicy* policy) {
  std::lock_guard<spin_lock> l(stats_lock_);
  if (stats_.empty()) {
    return true;
  }
  bool stable = true;
  for (auto s : stats_) {
    auto block = FindBlock(s);
    if (block != nullptr) {
      block->Insert(s);
      continue;
    }

    block = policy->FindBlock(s, bin_index_+1);
    if (block != nullptr) {
      block->Insert(s);
      auto vblock = new VirtualAllocBlock(block, chunk_size_); 
      virtual_blocks_.emplace_back(vblock);
      continue;
    }
    block = new AllocBlock(chunk_size_, bin_index_);
    block->Insert(s);
    blocks_.emplace_back(block);
    stable = false;
  }
  stats_.clear();
  return stable;
}

AllocBlock* LifetimeBin::FindBlock(AllocStats* stats) {
  for (auto block : blocks_) {
    if (block->CanInsert(stats)) {
      return block;
    }
  }
  return nullptr;
}

size_t LifetimeBin::BlockSize() const {
  return blocks_.size();
}

size_t LifetimeBin::ChunkSize() const {
  return chunk_size_;
}

size_t LifetimeBin::Alignment() const {
  return max_alignment_;
}

AllocBlock* LifetimePolicy::FindBlock(
    AllocStats* stats, size_t bindex) {
  for ( ; bindex < large_bin_index_; ++bindex) {
    auto block = bins_[bindex]->FindBlock(stats);
    if (block != nullptr) {
      return block;
    }
  }
  for (auto it = large_bins_.lower_bound(bindex);
      it != large_bins_.end(); ++it) {
    auto block = (it->second)->FindBlock(stats);
    if (block != nullptr) {
      return block;
    }
  }
  return nullptr;
}

AllocBlock::AllocBlock(size_t size, size_t bin_index)
    : size_(size), bin_index_(bin_index) {
}

AllocBlock::~AllocBlock() {
  for (auto stats : stats_) {
    delete stats;
  }
  stats_.clear();
}

bool AllocStats::IsOverlap(const AllocStats* other) {
  // We counter micro seconds, if equal mean probabaly is overlap.
  if (begin == other->begin || begin == other->end
      || end == other->begin || end == other->end) {
    return true;
  }

  return ((begin > other->begin &&
           begin < other->end) ||
          (begin < other->begin &&
           end > other->begin));
}

bool AllocBlock::CanInsert(AllocStats* alloc_stats) {
  for (auto s : stats_) {
    if (s->IsOverlap(alloc_stats)) {
      return false;
    }
  }
  return true;
}

void AllocBlock::Insert(AllocStats* alloc_stats) {
  // single thread enter
  stats_.emplace_back(alloc_stats);
}

size_t LifetimeBin::TotalMem() const {
  return blocks_.size() * chunk_size_;
}

void LifetimeBin::Dump() const {
  size_t stats_size = 0;
  {
    std::lock_guard<spin_lock> l(stats_lock_);
    if (stats_.empty()) {
      return;
    }
    stats_size = stats_.size();
  }
  // LOG(INFO_DEV) << "Bin index:" << bin_index_
  //          << ", chunk size:" << chunk_size_
  //          << ", stats counter:" << stats_size
  //          << ", blocks counter:" << blocks_.size()
  //          << ", vblocks counter:" << virtual_blocks_.size()
  //          << ", realsize:" << blocks_.size() * chunk_size_;
}

MemoryPlannerFactory::MemoryPlannerFactory() {
  // Enable Memory Optimization by default
  Status s = ReadBoolFromEnvVar("ENABLE_MEMORY_OPTIMIZATION",
      true,
      &enable_memory_opt_);
  if (enable_memory_opt_) {
    //LOG(INFO_DEV) << "Enable Memory Optimization!";
    memory_planner_ = new MemoryPlanner();
  } else {
    memory_planner_ = new NullableMemoryPlanner();
  }
}

namespace {
void signal_handler(int sig_num) {
    // kill -10 would output TensorPoolAllocator's statistic information
  if (sig_num == SIGUSR1) {
    TensorPoolAllocator* p =
      dynamic_cast<TensorPoolAllocator*>(cpu_allocator());
    p->DumpStats();
  }
}

size_t RoundedBytes(size_t bytes, size_t alignment) {
  return alignment * ((bytes + alignment - 1) / alignment);
}

void* SetLightHeader(void* p, size_t total_bytes, size_t header_size) {
  // LightHeader *KB max(sizeof(LightHeader)=8B, alignment)
  //   { | .....| checksum (4B) | header_size (4B)}
  auto user_ptr = (char*)p + header_size;
  auto header = new((char*)user_ptr - sizeof(LightHeader))
                    LightHeader(header_size);
  return user_ptr;
}

LightHeader* GetLightHeader(void* p) {
  auto light_header = (LightHeader*)((char*)p - sizeof(LightHeader));
  return (strcmp(light_header->checksum, CHECK_SUM.c_str()) == 0)
           ? light_header
           : nullptr;
}

Header* GetHeader(void* p) {
  auto header = (Header*)((char*) p - sizeof(Header));
  if (header->user_ptr != p) {
    auto light_header = GetLightHeader(p);
    LOG(FATAL) << "Memory corruption!"
               << ", p:" << p
               << ", p->header_size:" << light_header->header_size
               << ", p->checksum:" << light_header->checksum;
  }
  return header;
}

void* SetDefaultHeader(bool stats_time, void* p,
    size_t total_bytes, size_t header_size) {
  auto h = new((char*)p + header_size - sizeof(Header))Header();
  h->user_size = total_bytes - header_size;
  h->user_ptr = (char*)p + header_size;
  h->raw_ptr = p;
  h->total_size = total_bytes;

  if (stats_time) {
    timeval tmp;
    gettimeofday(&tmp, nullptr);
    h->begin = Timeval2Double(tmp);
  }
  return h->user_ptr;
}

void* SetHeader(void* p, size_t total_bytes, size_t header_size,
    void* bin, void* internal_bin) {
  // Header *KB max(sizeof(Header)=64B, alignment)
  // { |........| begin_time 16B | end_time 16B | total_size 8B
  //            |  user_size 8B | raw_ptr 8B | user_ptr 8B }
  auto h = new((char*)p + header_size - sizeof(Header))Header();
  h->user_size = total_bytes - header_size;
  h->user_ptr = (char*)p + header_size;
  h->raw_ptr = p;
  h->total_size = total_bytes;
  h->bin = bin;
  h->internal_bin = internal_bin;
  return h->user_ptr;
}

class DefaultCPUSubAllocator : public SubAllocator {
public:
  DefaultCPUSubAllocator() : SubAllocator({}, {}) {}
  ~DefaultCPUSubAllocator() override {}

  void* Alloc(size_t alignment, size_t num_bytes, size_t* bytes_received) override {
    return port::AlignedMalloc(num_bytes, alignment);
  }

  void Free(void* ptr, size_t num_bytes) override {
    port::AlignedFree(ptr);
  }

  virtual bool SupportsCoalescing() const override {
    return false;
  }
};
}

TensorPoolAllocator::TensorPoolAllocator() :
    stats_(false),
    inited_(false),
    initing_(false),
    sub_allocator_(new DefaultCPUSubAllocator()),
    mem_planner_(MemoryPlannerFactory::GetMemoryPlanner()),
    large_bin_index_(0),
    null_bin_counter_(0),
    hit_counter_(0),
    missed_counter_(0) {
  mem_planner_->SetAllocator(this);
}

void TensorPoolAllocator::Init() {
  bool tmp = false;
  if (initing_.compare_exchange_strong(tmp, true)) {
    signal(SIGUSR1, signal_handler);
    auto lifetime_policy = mem_planner_->BestLifetimePolicy();
    lifetime_policy->Dump();

    alignment_ = lifetime_policy->Alignment();
    alignment_offset_ = lifetime_policy->AlignmentOffset();

    auto policy_large_bins = lifetime_policy->GetLargeBins();
    for (auto rit = policy_large_bins.rbegin();
        rit != policy_large_bins.rend(); ++rit) {
      auto bin_info = rit->second;
      auto bin = new Bin(bin_info->BlockSize(), bin_info->ChunkSize(),
          bin_info->Alignment(), bin_info->VBlocks(), 
          sub_allocator_.get(), this);
      large_lifetime_bins_.emplace(rit->first, bin);
    }
    auto policy_bins = lifetime_policy->GetBins();
    large_bin_index_ = policy_bins.size();
    lifetime_bins_.resize(large_bin_index_);

    // create bigger bin first
    for (auto it = policy_bins.rbegin(); it != policy_bins.rend();
        ++it) {
      Bin* bin = nullptr;
      if ((*it)->BlockSize() > 0 || (*it)->VBlocks().size() > 0) {
        bin = new Bin((*it)->BlockSize(), (*it)->ChunkSize(),
            (*it)->Alignment(), (*it)->VBlocks(), 
            sub_allocator_.get(), this);
      }
      lifetime_bins_[(*it)->BinIndex()] = bin;
    }
    LOG(INFO) << "TensorPoolAllocator enabled";
    inited_ = true;
  }
}

void* TensorPoolAllocator::AllocateRaw(size_t alignment,
    size_t num_bytes) {
  if (SmallAlloc(num_bytes)) {
    auto header_size = std::max(sizeof(LightHeader), alignment);
    auto total = num_bytes + header_size;
    size_t bytes_recived;
    auto ptr = sub_allocator_->Alloc(alignment, total, &bytes_recived);
    return SetLightHeader(ptr, total, header_size);
  }

  if (unlikely(stats_)) {
    return BigAllocateStatistic(alignment, num_bytes);
  } else {
    return BigAllocate(alignment, num_bytes);
  }
}

void TensorPoolAllocator::DeallocateRaw(void* ptr) {
  auto light_header = GetLightHeader(ptr);
  if (light_header != nullptr) {
    auto header_size = light_header->header_size;
    auto raw_ptr = static_cast<char*>(ptr) - header_size;
    // LightHeader not record allocation size
    // Free interface ignore the freed num_bytes
    sub_allocator_->Free(raw_ptr, 0);
    return;
  }

  auto header = GetHeader(ptr);
  BigDeallocate(header);
}

TensorPoolAllocator::Bin* TensorPoolAllocator::GetBin(
    size_t bin_index) {
  if (unlikely(bin_index < 0)) {
    return nullptr;
  }

  if (unlikely(bin_index >= large_bin_index_)) {
    auto it = large_lifetime_bins_.find(bin_index);
    if (it == large_lifetime_bins_.end()) {
      return nullptr;
    } else {
      return it->second;
    }
  }
  return lifetime_bins_[bin_index];
}

TensorPoolAllocator::Bin::Bin(size_t len,
    size_t chunk_size, size_t alignment,
    std::vector<VirtualAllocBlock*>& vblocks,
    SubAllocator* sub_allocator, TensorPoolAllocator* tp) :
  buffer_(len, chunk_size, alignment, sub_allocator),
  virtual_buffer_(vblocks, tp), sub_allocator_(sub_allocator) {
}

void* TensorPoolAllocator::Bin::Allocate(size_t total, size_t header_size) {
  auto ptr = buffer_.Allocate();
  if (ptr != nullptr) {
    return SetHeader(ptr, total, header_size, (void*)this, nullptr);
  } 
  auto info = virtual_buffer_.Allocate();
  if (info.first != nullptr) {
    return SetHeader(info.first, total, header_size, (void*)this,
        info.second);
  }
  return nullptr;
}

void TensorPoolAllocator::Bin::Deallocate(Header* header) {
  if (header->internal_bin == nullptr) {
    buffer_.Deallocate(header->raw_ptr);
  } else {
    virtual_buffer_.Deallocate(header->raw_ptr,
        (TensorPoolAllocator::Bin*)(header->internal_bin));
  }
}

void* TensorPoolAllocator::Bin::AllocateRaw() {
  return buffer_.Allocate();
}

void TensorPoolAllocator::Bin::DeallocateRaw(void* p) {
  buffer_.Deallocate(p);
}

TensorPoolAllocator::Buffer::Buffer(size_t len, size_t chunk_size,
    size_t alignment, SubAllocator* sub_allocator) {
  auto rounded_bytes = RoundedBytes(chunk_size, alignment);
  auto buffer_size = rounded_bytes * len;
  size_t bytes_recived;
  auto p = static_cast<char*>(sub_allocator->Alloc(alignment, buffer_size, &bytes_recived));
  begin_ = p;
  end_ = p + buffer_size;

  for (auto i = 0; i < len; ++i) {
    buffer_.emplace(p + rounded_bytes *i);
  }
}

void* TensorPoolAllocator::Buffer::Allocate() {
  std::lock_guard<spin_lock> l(lock_);
  if (unlikely(buffer_.empty())) {
    return nullptr;
  }
  auto ptr = buffer_.top();
  buffer_.pop();
  return ptr;
}

void TensorPoolAllocator::Buffer::Deallocate(void* p) {
  if (unlikely(p < begin_ || p > end_)) {
    LOG(WARNING) << "probabaly memory corruption!!";
  }
  std::lock_guard<spin_lock> l(lock_);
  buffer_.emplace(p);
}

TensorPoolAllocator::VirtualBuffer::VirtualBuffer(
    std::vector<VirtualAllocBlock*>& vblocks,
    TensorPoolAllocator* tp) {
  for (auto vblock : vblocks) {
    auto bin_index = vblock->BinIndex();
    auto internal_bin = tp->GetBin(bin_index);
    if (internal_bin == nullptr) {
      LOG(WARNING) << "logic error or not allocate correctly";
    }
    internal_bins_.emplace(internal_bin);
  }
}

std::pair<void*, TensorPoolAllocator::Bin*>
TensorPoolAllocator::VirtualBuffer::Allocate() {
  std::lock_guard<spin_lock> l(lock_);
  if (unlikely(internal_bins_.empty())) {
    return std::make_pair(nullptr, nullptr);
  }
  auto internal_bin = internal_bins_.top();
  auto ptr = internal_bin->AllocateRaw();
  if (ptr == nullptr) {
    // todo: need to loop as much as internal_bins
    // not only just one
    return std::make_pair(nullptr, nullptr);
  }
  internal_bins_.pop();
  return std::make_pair(ptr, internal_bin);
}

void TensorPoolAllocator::VirtualBuffer::Deallocate(
    void* p, TensorPoolAllocator::Bin* bin) {
  std::lock_guard<spin_lock> l(lock_);
  bin->DeallocateRaw(p);
  internal_bins_.emplace(bin);
}

void TensorPoolAllocator::DumpStats() {
  if (stats_) {
    double hit_rate = (double)hit_counter_ /
      (hit_counter_ + missed_counter_ + null_bin_counter_);
    LOG(INFO) << "If you're TensorFlow user, "
      << "please ignore following debugging statistic."
      << "TensorPoolAllocator Statistic:"
      << " hit_counter[" << hit_counter_
      << "], missed_counter[" << missed_counter_
      << "], null_bin_counter[" << null_bin_counter_
      << "], hit_rate[" << hit_rate
      << "]";

    stats_ = false;
    hit_counter_ = 0;
    missed_counter_ = 0;
    null_bin_counter_ = 0;
  } else {
    stats_ = true;
    LOG(INFO) << "Start counting TensorPoolAllocator";
  }
}

void* TensorPoolAllocator::BigAllocate(size_t alignment,
    size_t num_bytes) {
  auto header_size = std::max(sizeof(Header), alignment);
  auto total = num_bytes + header_size; 
  size_t bytes_recived;
  if (!inited_.load()) {
    mem_planner_->TrackAllocate(alignment, total);
    auto ptr = sub_allocator_->Alloc(alignment, total, &bytes_recived);
    return SetDefaultHeader(!inited_.load(), ptr, total, header_size);
  }

  auto id = Index(total, alignment_, alignment_offset_);
  if (unlikely(id < 0)) {
    auto ptr = sub_allocator_->Alloc(alignment, total, &bytes_recived);
    return SetDefaultHeader(!inited_.load(), ptr, total, header_size);
  }

  auto b = GetBin(id);
  if (unlikely(b == nullptr)) {
    auto ptr = sub_allocator_->Alloc(alignment, total, &bytes_recived);
    return SetDefaultHeader(!inited_.load(), ptr, total, header_size);
  }

  auto ptr = b->Allocate(total, header_size);
  if (likely(ptr != nullptr)) {
    return ptr;
  }
  ptr = sub_allocator_->Alloc(alignment, total, &bytes_recived);
  return SetDefaultHeader(!inited_.load(), ptr, total, header_size);
}

// unlikely execute this path which do some atomic operations
void* TensorPoolAllocator::BigAllocateStatistic(size_t alignment,
    size_t num_bytes) {
  auto header_size = std::max(sizeof(Header), alignment);
  auto total = num_bytes + header_size; 
  size_t bytes_recived;
  if (!inited_.load()) {
    mem_planner_->TrackAllocate(alignment, total);
    auto ptr = sub_allocator_->Alloc(alignment, total, &bytes_recived);
    return SetDefaultHeader(!inited_.load(), ptr, total, header_size);
  }

  auto id = Index(total, alignment_, alignment_offset_);
  if (unlikely(id < 0)) {
    auto ptr = sub_allocator_->Alloc(alignment, total, &bytes_recived);
    return SetDefaultHeader(!inited_.load(), ptr, total, header_size);
  }

  auto b = GetBin(id);
  if (unlikely(b == nullptr)) {
    ++null_bin_counter_;
    auto ptr = sub_allocator_->Alloc(alignment, total, &bytes_recived);
    return SetDefaultHeader(!inited_.load(), ptr, total, header_size);
  }

  auto ptr = b->Allocate(total, header_size);
  if (likely(ptr != nullptr)) {
    ++hit_counter_;
    return ptr;
  }
  ++missed_counter_;
  ptr = sub_allocator_->Alloc(alignment, total, &bytes_recived);
  return SetDefaultHeader(!inited_.load(), ptr, total, header_size);
}

void TensorPoolAllocator::BigDeallocate(Header* header) {
  auto ptr = header->raw_ptr;
  auto num_bytes = header->total_size;
  if (!inited_.load()) {
    mem_planner_->TrackDeallocate(header);
    sub_allocator_->Free(ptr, num_bytes);
    return;
  }
  
  if (header->bin == nullptr) {
    sub_allocator_->Free(ptr, num_bytes);
    return;
  }

  auto bin = (Bin*) (header->bin);
  bin->Deallocate(header);
}

class TensorPoolAllocatorFactory : public AllocatorFactory {
 public:
  Allocator* CreateAllocator() override { return new TensorPoolAllocator(); }
  
  SubAllocator* CreateSubAllocator(int numa_node) override {
    return new TensorPoolSubAllocator(new TensorPoolAllocator());
  }

 private:
  class TensorPoolSubAllocator : public SubAllocator {
   public:
    explicit TensorPoolSubAllocator(TensorPoolAllocator* allocator)
      : SubAllocator({}, {}), allocator_(allocator) {}

    void* Alloc(size_t alignment, size_t num_bytes, size_t* bytes_received) override {
      return allocator_->AllocateRaw(alignment, num_bytes);
    }
    
    void Free(void* ptr, size_t num_bytes) override {
      allocator_->DeallocateRaw(ptr);
    }

    virtual bool SupportsCoalescing() const override { 
      return false; 
    }

   private:
    TensorPoolAllocator* allocator_;
  };
};

REGISTER_MEM_ALLOCATOR("TensorPoolAllocator", 300, TensorPoolAllocatorFactory);
} // tensorflow
