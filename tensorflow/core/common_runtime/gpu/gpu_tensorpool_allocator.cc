#include "tensorflow/core/common_runtime/gpu/gpu_tensorpool_allocator.h"
#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/platform/mem.h"
#include <sys/time.h>
#include <iostream>
#include <iomanip>

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

namespace tensorflow {

namespace {
constexpr int64 DEFAULT_START_STATISTIC_STEP = 10;
constexpr int64 DEFAULT_STOP_STATISTIC_STEP = 110;

bool AllocTimeCompare(GPUAllocStats* s1, GPUAllocStats*s2) {
  return s1->begin < s2->begin;
}

int GlobalStepFromEnv() {
  const char* global_step = getenv("GLOBAL_STEP");
  if (global_step == nullptr) {
    return -1;
  }
  string integer_str(global_step, strlen(global_step));
  std::istringstream ss(integer_str);
  int step = -1;
  ss >> step;
  return step;
}

std::string doubleToStringWithoutScientificNotation(double value, int precision=5) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}
}

bool GPUAllocStats::IsOverlap(const GPUAllocStats* other) {
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

std::string GPUAllocStats::DebugString() {
  return "(" + std::to_string(begin) + ", " + std::to_string(end) + "; " + std::to_string(size) + ")";
}

GPUMemoryPlanner::GPUMemoryPlanner() :
    is_stats_(false),
    inited_(false),
    allocator_(nullptr),
    thread_pool_(nullptr),
    counter_(0),
    logic_step_(0),
    start_step_(DEFAULT_START_STATISTIC_STEP),
    stop_step_(DEFAULT_STOP_STATISTIC_STEP) {
  InitStepInfo();
  InitPolicy();
}

GPUMemoryPlanner::~GPUMemoryPlanner() {
}

void GPUMemoryPlanner::InitPolicy() {
  lifetime_stats_polices_.emplace_back(
      new GPULifetimePolicy(_4KB, _4KB_OFFSET, _32KB));
  lifetime_stats_polices_.emplace_back(
      new GPULifetimePolicy(_8KB, _8KB_OFFSET, _32KB));
  lifetime_stats_polices_.emplace_back(
      new GPULifetimePolicy(_16KB, _16KB_OFFSET, _32KB));
  small_bins_.resize(kClassNum);
  for (int i = 0; i < kClassNum; ++i) {
    small_bins_[i] = new GPULifetimeBin(i, kSizeClass[i]);
  }
}

void GPUMemoryPlanner::InitStepInfo() {
  Status s = ReadInt64FromEnvVar("START_STATISTIC_STEP",
      DEFAULT_START_STATISTIC_STEP,
      &start_step_);
  s = ReadInt64FromEnvVar("STOP_STATISTIC_STEP",
      DEFAULT_STOP_STATISTIC_STEP,
      &stop_step_);
}

// lifetime policy
GPULifetimePolicy* GPUMemoryPlanner::BestLifetimePolicy() {
  GPULifetimePolicy* best_policy = nullptr;
  auto total_mem = std::numeric_limits<size_t>::max();
  for (auto policy : lifetime_stats_polices_) {
    auto policy_mem = policy->TotalMem();
    if (policy_mem < total_mem) {
      best_policy = policy;
      total_mem = policy_mem;
    }
  }
  return best_policy;
}

std::vector<GPULifetimeBin*>& GPUMemoryPlanner::GetSmallBins() {
  return small_bins_;
}

GPULifetimeBin* GPUMemoryPlanner::GetSmallBin(size_t size) {
  return small_bins_[kSmallSizeMap.GetClass(size)];
}

void GPUMemoryPlanner::Reset() {
  counter_.store(0);
  Cleanup();
}

void GPUMemoryPlanner::StartCollect() {
  // std::lock_guard<spin_lock> l(allocate_lock_);
  if (is_stats_.load()) {
    VLOG(2) << allocator_->Name() << " at " << allocator_ << " call BestFit() of planner " << this;
    BestFit();
    ResetStats();
  }

  auto current = counter_.fetch_add(1);
  VLOG(1) << "GPUMemoryPlanner::StartCollect with current step is " << current 
          << " start_step is " << start_step_ << " stop_step is " << stop_step_;
  if (current == start_step_) {
    is_stats_.store(true);
  } else if (current == stop_step_) {
    VLOG(1) << "call GPUMemoryPlanner::CollectDone";
    is_stats_.store(false);
    CollectDone();
  }
  if (allocator_ != nullptr) {
    allocator_->BeginStep();
  }
  logic_step_.store(0);
}

void GPUMemoryPlanner::BestFit() {
  for (auto policy : lifetime_stats_polices_) {
    VLOG(2) << "GPUMemoryPlanner " << this << " call BestFit with policy " << policy;
    policy->BestFit();
  }
  for (auto bin : small_bins_) {
    VLOG(2) << "GPUMemoryPlanner " << this << " call SmallFit with bin " << bin;
    bin->SmallFit();
  }
}

void GPUMemoryPlanner::ResetStats() {
  for (auto policy : lifetime_stats_polices_) {
    policy->ResetStats();
  }
  for (auto bin : small_bins_) {
    bin->ResetStats();
  }
  std::lock_guard<spin_lock> l(stats_lock_);
  for (auto s : alloc_stats_) {
    delete s;
  }
  alloc_stats_.clear();
}

void GPUMemoryPlanner::StopCollect() {
  // Make sure counter_ load is atomic.
}

void GPUMemoryPlanner::CollectDone() {
  Schedule([this]() {
    if (allocator_ != nullptr) {
      VLOG(1) << "CollectDone and call allocator_->Init()";
      allocator_->Init();
    }
    Cleanup();
    inited_.store(true);
    VLOG(1) << "CollectDone and set inited_ to " << inited_.load();
  });
}

void GPUMemoryPlanner::Cleanup() {
  for (auto policy : lifetime_stats_polices_) {
    policy->Cleanup();
  }
  for (auto bin : small_bins_) {
    bin->Cleanup();
  }
  std::lock_guard<spin_lock> l(stats_lock_);
  for (auto it : ptr_stats_) {
    delete it.second;
  }
  ptr_stats_.clear();
  for (auto s : alloc_stats_) {
    delete s;
  }
  alloc_stats_.clear();
}

void GPUMemoryPlanner::SetAllocator(GPUTensorPoolAllocator* allocator) {
  allocator_ = allocator;
}

void GPUMemoryPlanner::SetThreadPool(thread::ThreadPool* thread_pool) {
  if (thread_pool_ == nullptr) {
    thread_pool_ = thread_pool;
  }
}
void GPUMemoryPlanner::Schedule(std::function<void()> f) {
  if (thread_pool_ == nullptr) {
    VLOG(2) << "GPUMemoryPlanner::Schedule directly because thread pool is nullptr";
    f();
  } else {
    thread_pool_->Schedule(std::move(f));
  }
}

void GPUMemoryPlanner::TrackAllocate(size_t alignment, size_t num_bytes, void* ptr) {
  if (!is_stats_.load()) {
    return;
  }

  timeval tmp;
  gettimeofday(&tmp, nullptr);

  {
    std::lock_guard<spin_lock> l(stats_lock_);
    auto alloc_stats = new GPUAllocStats;
    // alloc_stats->begin = Timeval2Double(tmp);
    alloc_stats->begin = logic_step_;
    logic_step_.fetch_add(1);
    alloc_stats->size = num_bytes;
    ptr_stats_[ptr] = alloc_stats;
  }
  VLOG(2) << "TrackAllocate with ptr " << ptr << " size " << num_bytes << " allocated at " << doubleToStringWithoutScientificNotation(Timeval2Double(tmp));

  if (SmallAlloc(num_bytes)) {
    GetSmallBin(num_bytes)->TrackAllocate(alignment);
    return;
  }

  for (auto lifetime_policy : lifetime_stats_polices_) {
    lifetime_policy->TrackAllocate(alignment, num_bytes);
  }
}

void GPUMemoryPlanner::TrackDeallocate(void* ptr) {
  if (!is_stats_.load()) {
    return;
  }
  timeval tmp;
  gettimeofday(&tmp, nullptr);

  GPUAllocStats* alloc_stats;
  {
    std::lock_guard<spin_lock> l(stats_lock_);
    auto iter = ptr_stats_.find(ptr);
    if (iter == ptr_stats_.end()) {
      return;
    }
    alloc_stats = iter->second;
    ptr_stats_.erase(iter);
    alloc_stats_.emplace_back(alloc_stats);
    // alloc_stats->end = Timeval2Double(tmp);
    alloc_stats->end = logic_step_;
    logic_step_.fetch_add(1);
  }

  VLOG(2) << "TrackDeallocate with ptr " << ptr << " deallocated at " << doubleToStringWithoutScientificNotation(Timeval2Double(tmp));
  if (SmallAlloc(alloc_stats->size)) {
    GetSmallBin(alloc_stats->size)->TrackDeallocate(alloc_stats);
    return;
  }

  for (auto lifetime_policy : lifetime_stats_polices_) {
    lifetime_policy->TrackDeallocate(alloc_stats);
  }
}

GPULifetimePolicy::GPULifetimePolicy(size_t interval,
    size_t interval_offset, size_t start) :
    interval_(interval), interval_offset_(interval_offset), start_(start),
    large_bin_index_(Index(_32MB, interval_, interval_offset_) + 1) {
  auto cur = start_ + interval_;
  bins_.resize(large_bin_index_);
  for (auto i = 0; i < large_bin_index_; ++i) {
    bins_[i] = new GPULifetimeBin(i, cur);
    cur += interval_;
  }
}

void GPULifetimePolicy::TrackAllocate(size_t alignment, size_t num_bytes) {
  auto index = Index(num_bytes, interval_, interval_offset_);
  if (index < 0) {
    LOG(ERROR) << "GPUTensorPoolAllocator Invalid Index:" << index
               << ", size:" << num_bytes;
    return;
  }
  GetBin(index)->TrackAllocate(alignment);
}

GPULifetimeBin* GPULifetimePolicy::GetBin(size_t index) {
  if (index >= large_bin_index_) {
    std::lock_guard<spin_lock> l(large_bin_lock_);
    auto bin = large_bins_.find(index);
    if (bin == large_bins_.end()) {
      auto chunk_size = start_ + interval_ * (index + 1);
      bin = large_bins_.emplace(index, new GPULifetimeBin(index, chunk_size)).first;
    }
    return bin->second;
  } else {
    return bins_[index];
  }
}

void GPULifetimePolicy::TrackDeallocate(GPUAllocStats* alloc_stats) {
  auto index = Index(alloc_stats->size, interval_, interval_offset_);
  if (index < 0) {
    LOG(ERROR) << "GPUTensorPoolAllocator Invalid Index:" << index
               << ", size:" << alloc_stats->size;
    return;
  }
  GetBin(index)->TrackDeallocate(alloc_stats);
}

size_t GPULifetimePolicy::TotalMem() const {
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

void GPULifetimePolicy::Dump() const {
  // LOG(INFO_DEV) << "GPULifetimePolicy, start:" << start_
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

void GPULifetimePolicy::Cleanup() {
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

GPULifetimeBin::GPULifetimeBin(size_t bin_index, size_t chunk_size)
    : bin_index_(bin_index),
      chunk_size_(chunk_size),
      max_alignment_(Allocator::kAllocatorAlignment) {
}

GPULifetimeBin::~GPULifetimeBin() {
}

void GPULifetimeBin::TrackAllocate(size_t alignment) {
  std::lock_guard<spin_lock> l(stats_lock_);
  max_alignment_ = std::max<int64_t>(max_alignment_, alignment);
}

void GPULifetimeBin::TrackDeallocate(GPUAllocStats* stats) {
  // multiple thread enter
  std::lock_guard<spin_lock> l(stats_lock_);
  stats_.emplace_back(stats);
}

void GPULifetimePolicy::BestFit() {
  std::lock_guard<spin_lock> l(large_bin_lock_);
  for (auto it = large_bins_.rbegin(); it != large_bins_.rend(); ++it) {
    auto bin_info = it->second;
    VLOG(2) << "policy " << this << " BestFit of policy large bin:" << bin_info->BinIndex();
    bin_info->BestFit(this);
  }
  for (auto it = bins_.rbegin(); it != bins_.rend(); ++it) {
    VLOG(2) << "policy " << this << " BestFit of policy small bin:" << (*it)->BinIndex();
    (*it)->BestFit(this);
  }
}

std::vector<GPULifetimeBin*>& GPULifetimePolicy::GetBins() {
  return bins_;
}

std::map<size_t, GPULifetimeBin*>& GPULifetimePolicy::GetLargeBins() {
  return large_bins_;
}

size_t GPULifetimePolicy::Alignment() const {
  return interval_;
}

size_t GPULifetimePolicy::AlignmentOffset() const {
  return interval_offset_;
}

size_t GPULifetimePolicy::Interval() {
  return interval_;
}

void GPULifetimePolicy::ResetStats() {
  {
    std::lock_guard<spin_lock> l(large_bin_lock_);
    for (auto it : large_bins_) {
      auto bin_info = it.second;
      bin_info->ResetStats();
    }
  }
  for (auto bin : bins_) {
    bin->ResetStats();
  }
}

void GPULifetimeBin::Cleanup() {
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

void GPULifetimeBin::BestFit(GPULifetimePolicy* policy) {
  std::lock_guard<spin_lock> l(stats_lock_);
  for (auto vb : virtual_blocks_) {
    delete vb;
  }
  virtual_blocks_.clear();
  if (stats_.empty()) {
    return;
  }
  // sort by alloc time
  std::sort(stats_.begin(), stats_.end(), AllocTimeCompare);
  for (auto s : stats_) {
    VLOG(2) << "GPULifetimeBin " << this << " BestFit of State " << s << " " << s->DebugString();
    auto block = FindBlock(s);
    if (block != nullptr) {
      block->Insert(s);
      continue;
    }
    block = policy->FindBlock(s, bin_index_+1);
    if (block != nullptr) {
      block->Insert(s);
      auto vblock = new VirtualGPUAllocBlock(block, chunk_size_);
      virtual_blocks_.emplace_back(vblock);
      continue;
    }
    block = new GPUAllocBlock(chunk_size_, bin_index_);
    block->Insert(s);
    blocks_.emplace_back(block);
  }
}

void GPULifetimeBin::SmallFit() {
  std::lock_guard<spin_lock> l(stats_lock_);
  if (stats_.empty()) {
    return;
  }
  for (auto s : stats_) {
    VLOG(2) << "GPULifetimeBin " << this << " SmallFit of State " << s << " " << s->DebugString();
    auto block = FindBlock(s);
    if (block != nullptr) {
      block->Insert(s);
      continue;
    }
    block = new GPUAllocBlock(chunk_size_, bin_index_);
    block->Insert(s);
    blocks_.emplace_back(block);
  }
}

void GPULifetimeBin::ResetStats() {
  std::lock_guard<spin_lock> l(stats_lock_);
  for (auto b : blocks_) {
    b->ResetStats();
  }
  stats_.clear();
}

GPUAllocBlock* GPULifetimeBin::FindBlock(GPUAllocStats* stats) {
  for (auto block : blocks_) {
    if (block->CanInsert(stats)) {
      return block;
    }
  }
  return nullptr;
}

size_t GPULifetimeBin::BlockSize() const {
  return blocks_.size();
}

size_t GPULifetimeBin::ChunkSize() const {
  return chunk_size_;
}

size_t GPULifetimeBin::Alignment() const {
  return max_alignment_;
}

GPUAllocBlock* GPULifetimePolicy::FindBlock(GPUAllocStats* stats, size_t bindex) {
  for ( ; bindex < large_bin_index_; ++bindex) {
    auto block = bins_[bindex]->FindBlock(stats);
    if (block != nullptr) {
      return block;
    }
  }
  // no need to lock, BestFit already hold large_bin_lock_ firstly
  for (auto it = large_bins_.lower_bound(bindex); it != large_bins_.end(); ++it) {
    auto block = (it->second)->FindBlock(stats);
    if (block != nullptr) {
      return block;
    }
  }
  return nullptr;
}

size_t GPULifetimeBin::TotalMem() const {
  return blocks_.size() * RoundedBytes(chunk_size_, max_alignment_);
}

void GPULifetimeBin::Dump() const {
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

GPUAllocBlock::GPUAllocBlock(size_t size, size_t bin_index)
    : size_(size), bin_index_(bin_index) {
}

bool GPUAllocBlock::CanInsert(GPUAllocStats* alloc_stats) {
  for (auto s : stats_) {
    if (s->IsOverlap(alloc_stats)) {
      return false;
    }
  }
  return true;
}

void GPUAllocBlock::Insert(GPUAllocStats* alloc_stats) {
  // single thread enter
  stats_.emplace_back(alloc_stats);
}

void GPUAllocBlock::ResetStats() {
  stats_.clear();
}

GPUMemoryPlannerFactory::GPUMemoryPlannerFactory() {
  // Enable Memory Optimization by default
  Status s = ReadBoolFromEnvVar("ENABLE_MEMORY_OPTIMIZATION",
      true,
      &enable_memory_opt_);
  if (enable_memory_opt_) {
    //LOG(INFO_DEV) << "Enable Memory Optimization!";
    memory_planner_ = new GPUMemoryPlanner();
  } else {
    memory_planner_ = new NullableGPUMemoryPlanner();
  }
}

GPUTwoLevelTreeMemoryManager::Node::~Node() {
  left.reset();
  right.reset();
  if (nullptr == parent.get()) {
    outside_allocator->Free(pointer, size);
    outside_allocator.reset();
  }
}

std::string GPUTwoLevelTreeMemoryManager::Node::DebugString() {
  std::stringstream ss;
  ss << "0x" << std::hex << reinterpret_cast<uintptr_t>(pointer);
  return "Node(" + ss.str() + ", " + std::to_string(size) + ")";
}

GPUTwoLevelTreeMemoryManager::GPUTwoLevelTreeMemoryManager(SubAllocator* allocator_) 
    : allocator_ptr_(allocator_) {}

GPUTwoLevelTreeMemoryManager::~GPUTwoLevelTreeMemoryManager() {
  for(auto iter : used_list_) {
    auto node = iter.second;
    (node->left).reset();
    (node->right).reset();
  }
  used_list_.clear();
  for(auto iter: free_list_) {
    auto node = iter.second;
    (node->left).reset();
    (node->right).reset();
  }
  free_list_.clear();
  total_size_ = 0;
}

bool GPUTwoLevelTreeMemoryManager::IsAllocatedBuffer(void* ptr) {
  if (used_list_.find(ptr) != used_list_.end()) {
    return true;
  }
  return false;
}

void* GPUTwoLevelTreeMemoryManager::AllocateBuffer(size_t N) {
  void* ptr = GetFromFreeList(N);
  if (nullptr != ptr) {
    VLOG(2) << "GPUTwoLevelTreeMemoryManager::AllocateBuffer and get " << ptr << " from free list";
    return ptr;
  }
  // alloc new buffer
  size_t bytes_received;
  ptr = allocator_ptr_->Alloc(1, N, &bytes_received);
  if (nullptr == ptr) {
    VLOG(2) << "AllocateBuffer but get nullptr, may because of OOM!";
    return ptr;
  }
  total_size_ += N;
  std::shared_ptr<Node> root_node(new Node);
  root_node->size = N;
  root_node->pointer = ptr;
  root_node->outside_allocator = allocator_ptr_;

  std::shared_ptr<Node> leaf_node(new Node);
  leaf_node->size = N;
  leaf_node->pointer = ptr;
  leaf_node->parent = root_node;

  used_list_.insert(std::make_pair(ptr, leaf_node));
  VLOG(2) << "AllocateBuffer and get " << ptr << " from OS";
  return ptr;
}

void GPUTwoLevelTreeMemoryManager::ReleaseBuffer(void* ptr) {
  VLOG(2) << "call ReleaseBuffer: " << ptr;
  auto iter = used_list_.find(ptr);
  if (iter == used_list_.end()) {
    LOG(FATAL) << "GPUBinaryTreeMemoryManager Invalid Release Buffer: " << ptr;
    return;
  }
  auto node = iter->second;
  used_list_.erase(iter);
  returnMemory(node);
}

void GPUTwoLevelTreeMemoryManager::returnMemory(std::shared_ptr<Node> node) {
  free_list_.insert(std::make_pair(node->size, node));
  if(node->parent == nullptr) {
    LOG(FATAL) << "GPUBinaryTreeMemoryManager Invalid Return root node";
    return;  // this is the root node, not need merge
  }
  // find left nodes that can be merged
  if(node->left != nullptr) {
    VLOG(2) << node->DebugString() << " has left " << node->left->DebugString();
    bool can_merge = false;
    for(auto iter: free_list_) {
      if(iter.second == node->left && (iter.second)->parent == node->parent) {
        can_merge = true;
        break;
      }
    }
    if (can_merge) {
      VLOG(2) << node->DebugString() << " can merge with left " << node->left->DebugString();
      auto original_right = node;
      node = node->left;
      VLOG(2) << "update node to " << node->DebugString() << ", and has right " << node->right;
      if (node->right != original_right) {
        LOG(FATAL) << "Error bidirection link list";
        return;
      }
      // merge the node->right into node
      node->size += original_right->size;
      VLOG(2) << "update node->size to " << node->size;
      node->right = original_right->right;
      VLOG(2) << "update node->right to " << node->right->DebugString();
      if (node->right != nullptr) {
        VLOG(2) << "update node->right->left to node";
        node->right->left = node;
      }
      VLOG(2) << node->DebugString() << " finish merge";
      // delete the node->right from free list
      for(auto iter = free_list_.begin(); iter != free_list_.end(); iter++) {
        if(iter->second == original_right) {
          VLOG(2) << "erase " << original_right->DebugString() << " from free list";
          free_list_.erase(iter);
          break;
        }
      }
    }
  }
  // find right nodes that can be merged
  if (node->right != nullptr) {
    VLOG(2) << node->DebugString() << " has right " << node->right->DebugString();
    bool can_merge = false;
    for(auto iter: free_list_) {
      if(iter.second == node->right && (iter.second)->parent == node->parent) {
        can_merge = true;
        break;
      }
    }
    if (can_merge) {
      auto original_right = node->right;
      VLOG(2) << node->DebugString() << " can merge with right";
      // merge the node->right into node
      node->size += node->right->size;
      VLOG(2) << "update " << node->DebugString() << " size to " << node->size;
      node->right = node->right->right;
      VLOG(2) << "update " << node->DebugString() << " right to " << node->right;
      if (node->right != nullptr) {
        VLOG(2) << "update node->right->left to node";
        node->right->left = node;
      }
      VLOG(2) << node->DebugString() << " finish merge";
      // delete the node->right from free list
      for(auto iter = free_list_.begin(); iter != free_list_.end(); iter++) {
        if(iter->second == original_right) {
          VLOG(2) << "erase " << original_right->DebugString() << " from free list";
          free_list_.erase(iter);
          break;
        }
      }
    }
  }
}

void* GPUTwoLevelTreeMemoryManager::GetFromFreeList(size_t N) {
  auto iter = free_list_.lower_bound(N);  //  iter point to pair(size_t, std::shared_ptr<Node>)
  if (iter == free_list_.end()) {
    return nullptr;
  }
  auto node = iter->second;
  if (iter->first == N) { // lower_bound  ensure iter->first>=N
    // uses up all aligned space
    used_list_.insert(std::make_pair(node->pointer, node));
    free_list_.erase(iter);
    return node->pointer;
  }
  // split the node into two nodes
  std::shared_ptr<Node> left_node(new Node);
  std::shared_ptr<Node> right_node(new Node);

  left_node->size = N;
  left_node->pointer = node->pointer;
  left_node->parent = node->parent;
  left_node->left = node->left;
  left_node->right = right_node;
  if(left_node->left != nullptr) {
    left_node->left->right = left_node;
  }

  right_node->size = node->size - N;
  right_node->pointer = (void*)((uint8_t*)node->pointer + N);
  right_node->parent = node->parent;
  right_node->left = left_node;
  right_node->right = node->right;
  if(right_node->right != nullptr) {
    right_node->right->left = right_node;
  }

  // update the used_list_ and free_list_
  used_list_.insert(std::make_pair(left_node->pointer, left_node));
  free_list_.erase(iter);
  free_list_.insert(std::make_pair(right_node->size, right_node));
  return left_node->pointer;
}

GPUBinaryTreeMemoryManager::Node::~Node() {
  if (nullptr == parent.get()) {
    outside_allocator->Free(pointer, size);
    outside_allocator.reset();
  }
}

GPUBinaryTreeMemoryManager::GPUBinaryTreeMemoryManager(SubAllocator* allocator_) 
    : allocator_ptr_(allocator_) {}

GPUBinaryTreeMemoryManager::~GPUBinaryTreeMemoryManager() {
  used_list_.clear();
  free_list_.clear();
  total_size_ = 0;
}

bool GPUBinaryTreeMemoryManager::IsAllocatedBuffer(void* ptr) {
  if (used_list_.find(ptr) != used_list_.end()) {
    return true;
  }
  return false;
}

void* GPUBinaryTreeMemoryManager::AllocateBuffer(size_t N) {
  void* ptr = GetFromFreeList(N);
  if (nullptr != ptr) {
    VLOG(2) << "AllocateBuffer and get " << ptr << " from free list";
    return ptr;
  }
  // alloc otherwise
  size_t bytes_received;
  ptr = allocator_ptr_->Alloc(1, N, &bytes_received);
  if (nullptr == ptr) {
    VLOG(2) << "AllocateBuffer but get nullptr, may because of OOM!";
    return ptr;
  }
  total_size_ += N;
  std::shared_ptr<Node> node(new Node);
  node->size = N;
  node->pointer = ptr;
  node->outside_allocator = allocator_ptr_;
  used_list_[ptr] = node;
  VLOG(2) << "AllocateBuffer and get " << ptr << " from OS";
  return ptr;
}

void GPUBinaryTreeMemoryManager::ReleaseBuffer(void* ptr) {
  VLOG(2) << "call ReleaseBuffer: " << ptr;
  auto iter = used_list_.find(ptr);
  if (iter == used_list_.end()) {
    LOG(FATAL) << "GPUBinaryTreeMemoryManager Invalid Release Buffer: " << ptr;
    return;
  }
  // mark as reusable
  auto node = iter->second;
  used_list_.erase(iter);
  returnMemory(node);
}

void GPUBinaryTreeMemoryManager::returnMemory(std::shared_ptr<Node> node) {
  free_list_.insert(std::make_pair(node->size, node));
  if (nullptr != node->parent.get()) {
    auto parent = node->parent;
    parent->use_count -= 1;
    // merge if all subnodes were freed
    auto needMerge = parent->use_count == 0;
    while (needMerge) {
      // collect all subnodes
      for (auto iter = free_list_.begin(); iter != free_list_.end();) {
        if (iter->second->parent.get() == parent.get()) {
          iter = free_list_.erase(iter);
        } else {
          iter++;
        }
      }

      // do merge downside up
      free_list_.insert(std::make_pair(parent->size, parent));
      needMerge = false;
      if (parent->parent.get() != nullptr) {
          parent = parent->parent;
          parent->use_count -= 1;
          needMerge = parent->use_count == 0;
      }
    }
  }
}

void* GPUBinaryTreeMemoryManager::GetFromFreeList(size_t N) {
  // get node larger than size
  auto iter = free_list_.lower_bound(N);  //  iter point to pair(size_t, std::shared_ptr<Node>)
  if (iter == free_list_.end()) {
    return nullptr;
  }
  // update parent use count
  auto node = iter->second;
  if (node->parent.get() != nullptr) {
    node->parent->use_count += 1;
  }

  if (iter->first == N) { // lower_bound  ensure iter->first>=N
    // uses up all aligned space
    used_list_.insert(std::make_pair(node->pointer, node));
    free_list_.erase(iter);
    return node->pointer;
  }

  // split otherwise, the non-root Node does not hold the outside_allocator_
  std::shared_ptr<Node> first(new Node);
  first->parent = node;
  first->size = N;
  first->pointer = node->pointer;
  used_list_.insert(std::make_pair(first->pointer, first));
  node->use_count += 1;

  std::shared_ptr<Node> second(new Node);
  second->parent = node;
  second->size = node->size - N;
  second->pointer = (void*)((uint8_t*)node->pointer + N);
  free_list_.erase(iter);
  free_list_.insert(std::make_pair(second->size, second));
  return node->pointer;
}


GPUTensorPoolAllocator::GPUTensorPoolAllocator(
      SubAllocator* sub_allocator, std::string name, size_t total_memory) :
    name_(name),
    stats_(false),
    inited_(false),
    initing_(false),
    step_id_(-1),
    sub_allocator_(sub_allocator),
    mem_planner_(nullptr),
    large_bin_index_(0),
    null_bin_counter_(0),
    hit_counter_(0),
    missed_counter_(0),
    big_mem_begin_(nullptr),
    big_mem_end_(nullptr),
    small_mem_begin_(nullptr),
    small_mem_end_(nullptr) {
  bool enable_memory_opt_ = true;
  ReadBoolFromEnvVar("ENABLE_MEMORY_OPTIMIZATION", true, &enable_memory_opt_);
  if (enable_memory_opt_) {
    //LOG(INFO_DEV) << "Enable Memory Optimization!";
    mem_planner_ = new GPUMemoryPlanner();
  } else {
    mem_planner_ = new NullableGPUMemoryPlanner();
  }
  mem_planner_->SetAllocator(this);
  alloc_stats_.bytes_limit = static_cast<int64>(total_memory);
  fallback_memory_manager_ = std::make_shared<GPUBinaryTreeMemoryManager>(sub_allocator);
}

GPUTensorPoolAllocator::~GPUTensorPoolAllocator() {
  if (big_mem_begin_ != nullptr) {
    sub_allocator_->Free(big_mem_begin_, big_bytes_);
    directly_allocated_pointers_.erase(big_mem_begin_);
  }
  if (small_mem_begin_ != nullptr) {
    sub_allocator_->Free(small_mem_begin_, small_bytes_);
    directly_allocated_pointers_.erase(small_mem_begin_);
  }
  for (auto bin : lifetime_bins_) {
    if (bin != nullptr) {
      delete bin;
    }
  }
  for (auto it : large_lifetime_bins_) {
    delete it.second;
  }
  for (auto bin : small_bins_) {
    if (bin != nullptr) {
      delete bin;
    }
  }
  fallback_memory_manager_.reset();
}

void GPUTensorPoolAllocator::Init() {
  bool tmp = false;
  // this function is executed only once ensured by ``compare_exchange_strong''
  if (initing_.compare_exchange_strong(tmp, true)) {
    auto lifetime_policy = mem_planner_->BestLifetimePolicy();

    alignment_ = lifetime_policy->Alignment();
    alignment_offset_ = lifetime_policy->AlignmentOffset();

    big_bytes_ = 0;
    std::map<size_t, size_t> bin_to_offset;  // bin_index to offset

    auto policy_bins = lifetime_policy->GetBins(); // small bins
    large_bin_index_ = policy_bins.size();
    lifetime_bins_.resize(large_bin_index_);

    size_t max_alignment = 0;

    for (auto it = policy_bins.begin(); it != policy_bins.end(); ++it) {
      VLOG(2) << "bin index " << (*it)->BinIndex() << " block size " << (*it)->BlockSize() << " chunksize " << (*it)->ChunkSize()
              << " alignment " << (*it)->Alignment();
      if ((*it)->BlockSize() > 0) {
        VLOG(2) << "\tbin index " << (*it)->BinIndex() << " block size " << (*it)->BlockSize() << " chunksize " << (*it)->ChunkSize() 
                  << " alignment " << (*it)->Alignment() << " totalMemory " << (*it)->TotalMem();
        // add padding between two bins
        big_bytes_ = RoundedBytes(big_bytes_, (*it)->Alignment());
        bin_to_offset[(*it)->BinIndex()] = big_bytes_;
        big_bytes_ += (*it)->TotalMem();
        max_alignment = std::max<size_t>(max_alignment, (*it)->Alignment());
      }
    }

    auto policy_large_bins = lifetime_policy->GetLargeBins();
    for (auto it = policy_large_bins.begin(); it != policy_large_bins.end(); ++it) {
      auto bin_info = it->second;
      VLOG(2) << "bin index " << bin_info->BinIndex() << " block size " << bin_info->BlockSize() << " chunksize " << bin_info->ChunkSize()
              << " alignment " << bin_info->Alignment();
      if (bin_info->BlockSize() > 0) {
        VLOG(2) << "\tbin index " << bin_info->BinIndex() << " block size " << bin_info->BlockSize() << " chunksize " << bin_info->ChunkSize()
                << " alignment " << bin_info->Alignment() << " totalMemory " << bin_info->TotalMem();
        // add padding between two bins
        big_bytes_ = RoundedBytes(big_bytes_, bin_info->Alignment());
        bin_to_offset[bin_info->BinIndex()] = big_bytes_;
        big_bytes_ += bin_info->TotalMem();
        max_alignment = std::max<size_t>(max_alignment, bin_info->Alignment());
      }
    }

    VLOG(2) << "big_bytes_ " << big_bytes_ << " max_alignment " << max_alignment;
    size_t bytes_received;
    big_mem_begin_ = sub_allocator_->Alloc(max_alignment, big_bytes_, &bytes_received);
    directly_allocated_pointers_.insert(big_mem_begin_);
    if (big_bytes_ > 0 && big_mem_begin_ == nullptr) {
      LOG(FATAL) << "OOM!!! Try to alloc(" << max_alignment << ", " << big_bytes_ << ")";
    }
    if (big_bytes_ > 0) {
      big_mem_end_ = static_cast<char*>(big_mem_begin_) + big_bytes_;
    } else {
      big_mem_end_ = nullptr;
    }

    // create bigger bin first
    for (auto rit = policy_large_bins.rbegin(); rit != policy_large_bins.rend(); ++rit) {
      auto bin_info = rit->second;
      Bin* bin = nullptr;
      if (bin_info->BlockSize() > 0) {
        auto offset = bin_to_offset[rit->first];
        bin = new Bin(bin_info->BlockSize(), bin_info->ChunkSize(),
            bin_info->Alignment(), bin_info->VBlocks(),
            this, static_cast<char*>(big_mem_begin_) + offset);
        offset_to_bin_[offset] = bin;
      } else if (bin_info->VBlocks().size() > 0) {
        bin = new Bin(bin_info->BlockSize(), bin_info->ChunkSize(),
            bin_info->Alignment(), bin_info->VBlocks(),
            this, nullptr);
      }
      if (bin != nullptr) {
        large_lifetime_bins_.emplace(rit->first, bin);
      }
    }

    for (auto it = policy_bins.rbegin(); it != policy_bins.rend(); ++it) {
      Bin* bin = nullptr;
      if ((*it)->BlockSize() > 0) {
        auto offset = bin_to_offset[(*it)->BinIndex()];
        bin = new Bin((*it)->BlockSize(), (*it)->ChunkSize(),
            (*it)->Alignment(), (*it)->VBlocks(),
            this, static_cast<char*>(big_mem_begin_) + offset);
        offset_to_bin_[offset] = bin;
      } else if ((*it)->VBlocks().size() > 0) {
        bin = new Bin((*it)->BlockSize(), (*it)->ChunkSize(),
            (*it)->Alignment(), (*it)->VBlocks(),
            this, nullptr);
      }
      lifetime_bins_[(*it)->BinIndex()] = bin;
    }

    auto small_bins = mem_planner_->GetSmallBins();
    small_bins_.resize(small_bins.size());
    small_bytes_ = 0;
    max_alignment = 0;
    bin_to_offset.clear();
    for (auto b : small_bins) {
      if (b->BlockSize() > 0) {
        small_bytes_ = RoundedBytes(small_bytes_, b->Alignment());
        bin_to_offset[b->BinIndex()] = small_bytes_;
        small_bytes_ +=  b->TotalMem();
        max_alignment = std::max<size_t>(max_alignment, b->Alignment());
      }
    }

    small_mem_begin_ = sub_allocator_->Alloc(max_alignment, small_bytes_, &bytes_received);
    directly_allocated_pointers_.insert(small_mem_begin_);
    if (small_bytes_ > 0 && small_mem_begin_ == nullptr) {
      LOG(FATAL) << "OOM!!! Try to alloc(" << max_alignment << ", " << small_bytes_ << ")";
    }
    if (small_bytes_ > 0) {
      small_mem_end_ = static_cast<char*>(small_mem_begin_) + small_bytes_;
    } else {
      small_mem_end_ = nullptr;
    }

    for (auto b : small_bins) {
      SmallBin* bin = nullptr;
      if (b->BlockSize() > 0) {
        auto offset = bin_to_offset[b->BinIndex()];
        bin = new SmallBin(b->BlockSize(), b->ChunkSize(),
                           b->Alignment(), static_cast<char*>(small_mem_begin_) + offset);
        offset_to_small_bin_[offset] = bin;
      }
      small_bins_[b->BinIndex()] = bin;
    }

    inited_.store(true);
    VLOG(1) << name_ <<" GPUTensorPoolAllocator init done and set inited_ to " << inited_.load();
  }
}

void GPUTensorPoolAllocator::BeginStep() {
  if (inited_.load()) {
    for (auto b : lifetime_bins_) {
      if (b != nullptr) {
        b->BeginStep();
      }
    }
    for (auto it : large_lifetime_bins_) {
      it.second->BeginStep();
    }
  }
  std::lock_guard<spin_lock> l(free_lock_);
  for (auto ptr : async_free_list_) {
    sub_allocator_->Free(ptr, 0);
    directly_allocated_pointers_.erase(ptr);
  }
  async_free_list_.clear();
}

void* GPUTensorPoolAllocator::AllocateRaw(size_t alignment,
    size_t num_bytes, const AllocationAttributes& allocation_attr) {
  return AllocateRaw(alignment, num_bytes);
}

void* GPUTensorPoolAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  // VLOG(1) << name_ << " Calling AllocateRaw, env PRMALLOC_STAGE is " << std::getenv("PRMALLOC_STAGE");
  std::lock_guard<spin_lock> l(allocate_lock_);
  auto real_num_bytes = RoundedBytes(num_bytes, alignment);
  VLOG(2) << name_ << " Calling AllocateRaw: using optimized allocation planner, requiring " << real_num_bytes << " bytes";
  auto ptr = fallback_memory_manager_->AllocateBuffer(real_num_bytes);
  return ptr;

  auto current_step = GlobalStepFromEnv();
  if (step_id_.load() != current_step) {
    // VLOG(1) << name_ << " Calling AllocateRaw, step_id is " << step_id_.load() << ", current_step is " << current_step << std::endl;
    mem_planner_->StartCollect();
    step_id_.store(current_step);
    VLOG(2) << name_ << " Calling AllocateRaw, after planner.collect, step_id is " << step_id_.load() << ", current_step is " << current_step << " inited_ is " << inited_.load();
  }
  VLOG(2) << name_ << " Calling AllocateRaw with inited_ " << inited_.load();
  if (!inited_.load()) {
    VLOG(2) << name_ << " Calling AllocateRaw: Allocate from OS directly, requiring " << num_bytes << " bytes";
    size_t bytes_received;
    auto ptr = sub_allocator_->Alloc(alignment, num_bytes, &bytes_received);
    directly_allocated_pointers_.insert(ptr);
    mem_planner_->TrackAllocate(alignment, num_bytes, ptr);
    return ptr;
  }
  

  if (SmallAlloc(num_bytes)) {
    return SmallAllocate(alignment, num_bytes);
  } else if (unlikely(stats_)) {
    return BigAllocateStatistic(alignment, num_bytes);
  } else {
    return BigAllocate(alignment, num_bytes);
  }
}

void GPUTensorPoolAllocator::DeallocateRaw(void* ptr) {
  std::lock_guard<spin_lock> l(allocate_lock_);
  fallback_memory_manager_->ReleaseBuffer(ptr);
  return;

  if (!inited_.load()) {
    mem_planner_->TrackDeallocate(ptr);
    sub_allocator_->Free(ptr, 0);
    directly_allocated_pointers_.erase(ptr);
  } else if (IsBigOwned(ptr) && false) {
    BigDeallocate(ptr);
  } else if (IsSmallOwned(ptr) && false) {
    SmallDeallocate(ptr);
  } else if (fallback_memory_manager_->IsAllocatedBuffer(ptr)) {
    fallback_memory_manager_->ReleaseBuffer(ptr);
  } else {
    if (directly_allocated_pointers_.find(ptr) == directly_allocated_pointers_.end()) {
      LOG(FATAL) << "DeallocateRaw Error: Unknown pointer " << ptr;
    }
    sub_allocator_->Free(ptr, 0);
    directly_allocated_pointers_.erase(ptr);
  }
}

absl::optional<AllocatorStats> GPUTensorPoolAllocator::GetStats() {
  return alloc_stats_;
}

GPUTensorPoolAllocator::Bin* GPUTensorPoolAllocator::GetBin(size_t bin_index) {
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

GPUTensorPoolAllocator::SmallBin* GPUTensorPoolAllocator::GetSmallBin(
    size_t size) {
  auto id = kSmallSizeMap.GetClass(size);
  if (unlikely(id >= small_bins_.size())) {
    LOG(FATAL) << "logic error";
    return nullptr;
  }
  return small_bins_[id];
}

GPUTensorPoolAllocator::SmallBin::SmallBin(size_t len,
    size_t chunk_size, size_t alignment, void* begin) {
  auto rounded_bytes = RoundedBytes(chunk_size, alignment);
  auto buffer_size = rounded_bytes * len;
  begin_ = begin;
  if (begin != nullptr) {
    end_ = static_cast<char*>(begin) + buffer_size;
  } else {
    end_ = nullptr;
  }

  for (auto i = 0; i < len; ++i) {
    buffer_.emplace(static_cast<char*>(begin) + rounded_bytes *i);
  }
}

void* GPUTensorPoolAllocator::SmallBin::AllocateRaw() {
  std::lock_guard<spin_lock> l(lock_);
  if (unlikely(buffer_.empty())) {
    return nullptr;
  }
  auto ptr = buffer_.top();
  buffer_.pop();
  return ptr;
}

void GPUTensorPoolAllocator::SmallBin::DeallocateRaw(void* p) {
  if (unlikely(begin_ == nullptr || p < begin_ || p > end_)) {
    LOG(WARNING) << "probabaly memory corruption!! begin_: " << begin_
                 << " end_: " << end_ << " p: " << p;
  }
  std::lock_guard<spin_lock> l(lock_);
  buffer_.emplace(p);
}

GPUTensorPoolAllocator::Bin::Bin(size_t len,
    size_t chunk_size, size_t alignment,
    std::vector<VirtualGPUAllocBlock*>& vblocks,
    GPUTensorPoolAllocator* tp, void* begin) :
  buffer_(len, chunk_size, alignment, begin),
  virtual_buffer_(vblocks, tp) {
}

void* GPUTensorPoolAllocator::Bin::Allocate() {
  auto ptr = buffer_.Allocate();
  if (ptr != nullptr) {
    return ptr;
  }
  return virtual_buffer_.Allocate();
}

void* GPUTensorPoolAllocator::Bin::AllocateRaw() {
  return buffer_.Allocate();
}

void GPUTensorPoolAllocator::Bin::DeallocateRaw(void* p) {
  buffer_.Deallocate(p);
}

void GPUTensorPoolAllocator::Bin::BeginStep() {
  return virtual_buffer_.BeginStep();
}

GPUTensorPoolAllocator::Buffer::Buffer(size_t len, size_t chunk_size,
    size_t alignment, void* begin) {
  auto rounded_bytes = RoundedBytes(chunk_size, alignment);
  auto buffer_size = rounded_bytes * len;
  begin_ = begin;
  if (begin != nullptr) {
    end_ = static_cast<char*>(begin) + buffer_size;
  } else {
    end_ = nullptr;
  }

  for (auto i = 0; i < len; ++i) {
    buffer_.emplace(static_cast<char*>(begin) + rounded_bytes *i);
  }
}

void* GPUTensorPoolAllocator::Buffer::Allocate() {
  std::lock_guard<spin_lock> l(lock_);
  if (unlikely(buffer_.empty())) {
    return nullptr;
  }
  auto ptr = buffer_.top();
  buffer_.pop();
  return ptr;
}

void GPUTensorPoolAllocator::Buffer::Deallocate(void* p) {
  if (unlikely(begin_ == nullptr || p < begin_ || p > end_)) {
    LOG(WARNING) << "probabaly memory corruption!! begin_: " << begin_
                 << " end_: " << end_ << " p: " << p;
  }
  std::lock_guard<spin_lock> l(lock_);
  buffer_.emplace(p);
}

GPUTensorPoolAllocator::VirtualBuffer::VirtualBuffer(
    std::vector<VirtualGPUAllocBlock*>& vblocks,
    GPUTensorPoolAllocator* tp) {
  for (auto vblock : vblocks) {
    auto bin_index = vblock->BinIndex();
    auto internal_bin = tp->GetBin(bin_index);
    if (internal_bin == nullptr) {
      LOG(WARNING) << "logic error or not allocate correctly";
    }
    internal_bins_.emplace_back(internal_bin);
  }
  curr_index_ = 0;
}

void* GPUTensorPoolAllocator::VirtualBuffer::Allocate() {
  if (unlikely(internal_bins_.empty())) {
    return nullptr;
  }
  auto index = curr_index_.fetch_add(1) % internal_bins_.size();
  auto bin = internal_bins_[index];
  return bin->AllocateRaw();
}

void GPUTensorPoolAllocator::VirtualBuffer::BeginStep() {
  curr_index_ = 0;
}

void GPUTensorPoolAllocator::DumpStats() {
  if (stats_) {
    double hit_rate = (double)hit_counter_ / (hit_counter_ + missed_counter_ + null_bin_counter_);
    LOG(INFO) << "If you're TensorFlow user, "
      << "please ignore following debugging statistic."
      << "GPUTensorPoolAllocator Statistic:"
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
    LOG(INFO) << "Start counting GPUTensorPoolAllocator";
  }
}

bool GPUTensorPoolAllocator::IsBigOwned(void *ptr) {
  return (ptr >= big_mem_begin_ && ptr <= big_mem_end_);
}

bool GPUTensorPoolAllocator::IsSmallOwned(void *ptr) {
  return (ptr >= small_mem_begin_ && ptr <= small_mem_end_);
}

void* GPUTensorPoolAllocator::SmallAllocate(size_t alignment, size_t num_bytes) {
  auto bin = GetSmallBin(num_bytes);
  size_t bytes_received;
  auto real_num_bytes = RoundedBytes(num_bytes, alignment);
  void* ptr;
  if (unlikely(bin == nullptr)) {
    // return sub_allocator_->Alloc(alignment, num_bytes, &bytes_received);
    ptr = fallback_memory_manager_->AllocateBuffer(real_num_bytes);
    VLOG(2) << "call SmallAllocate but still require from OS for " << real_num_bytes << " bytes because bin is nullptr and get " << ptr;
    return ptr;
  }
  ptr = bin->AllocateRaw();
  if (likely(ptr != nullptr)) {
    return ptr;
  }
  ptr = fallback_memory_manager_->AllocateBuffer(real_num_bytes);
  VLOG(2) << "call SmallAllocate but still require from OS for " << real_num_bytes << " bytes and get " << ptr;
  // return sub_allocator_->Alloc(alignment, num_bytes, &bytes_received);
  return ptr;
}

void* GPUTensorPoolAllocator::BigAllocate(size_t alignment, size_t num_bytes) {
  size_t bytes_received;
  auto id = Index(num_bytes, alignment_, alignment_offset_);
  auto real_num_bytes = RoundedBytes(num_bytes, alignment);
  void* ptr;
  if (unlikely(id < 0)) {
    // return sub_allocator_->Alloc(alignment, num_bytes, &bytes_received);
    ptr = fallback_memory_manager_->AllocateBuffer(real_num_bytes);
    VLOG(2) << "call BigAllocate but still require from OS for " << real_num_bytes << " bytes beacuse id < 0 and get " << ptr;
    return ptr;
  }

  auto b = GetBin(id);
  if (unlikely(b == nullptr)) {
    // return sub_allocator_->Alloc(alignment, num_bytes, &bytes_received);
    ptr = fallback_memory_manager_->AllocateBuffer(real_num_bytes);
    VLOG(2) << "call BigAllocate but still require from OS for " << real_num_bytes << " bytes beacuse bin is nullptr and get " << ptr;
    return ptr;
  }

  ptr = b->Allocate();
  if (likely(ptr != nullptr)) {
    return ptr;
  }

  // return sub_allocator_->Alloc(alignment, num_bytes, &bytes_received);
  ptr = fallback_memory_manager_->AllocateBuffer(real_num_bytes);
  VLOG(2) << "call BigAllocate but still require from OS for " << real_num_bytes << " bytes and get " << ptr;
  return ptr;
}

// unlikely execute this path which do some atomic operations
void* GPUTensorPoolAllocator::BigAllocateStatistic(size_t alignment, size_t num_bytes) {
  size_t bytes_received;
  auto id = Index(num_bytes, alignment_, alignment_offset_);
  auto real_num_bytes = RoundedBytes(num_bytes, alignment);
  void* ptr;
  if (unlikely(id < 0)) {
    // return sub_allocator_->Alloc(alignment, num_bytes, &bytes_received);
    ptr = fallback_memory_manager_->AllocateBuffer(real_num_bytes);
    VLOG(2) << "call BigAllocateStatistic but still require from OS for " << real_num_bytes << " bytes because id < 0 and get " << ptr;
    return ptr;
  }

  auto b = GetBin(id);
  if (unlikely(b == nullptr)) {
    ++null_bin_counter_;
    // return sub_allocator_->Alloc(alignment, num_bytes, &bytes_received);
    ptr = fallback_memory_manager_->AllocateBuffer(real_num_bytes);
    VLOG(2) << "call BigAllocateStatistic but still require from OS for " << real_num_bytes << " bytes because bin is nullptr and get " << ptr;
    return ptr;
  }

  ptr = b->Allocate();
  if (likely(ptr != nullptr)) {
    ++hit_counter_;
    return ptr;
  }

  ++missed_counter_;
  // return sub_allocator_->Alloc(alignment, num_bytes, &bytes_received);
  ptr = fallback_memory_manager_->AllocateBuffer(real_num_bytes);
  VLOG(2) << "call BigAllocateStatistic but still require from OS for " << real_num_bytes << " bytes and get " << ptr;
  return ptr;
}

void GPUTensorPoolAllocator::SmallDeallocate(void* ptr) {
  size_t offset = reinterpret_cast<uint8_t *>(ptr) -
                  reinterpret_cast<uint8_t *>(small_mem_begin_);
  auto it = offset_to_small_bin_.upper_bound(offset);
  it = std::prev(it);
  it->second->DeallocateRaw(ptr);
}

void GPUTensorPoolAllocator::BigDeallocate(void* ptr) {
  size_t offset = reinterpret_cast<uint8_t *>(ptr) -
                  reinterpret_cast<uint8_t *>(big_mem_begin_);
  auto it = offset_to_bin_.upper_bound(offset);
  it = std::prev(it);
  it->second->DeallocateRaw(ptr);
}

} // tensorflow
