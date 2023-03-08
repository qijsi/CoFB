#ifndef __LOCKFREEQ_H_
#define __LOCKFREEQ_H_
#include <atomic>
#include <memory>
#include <vector>

template <typename T>
class LockFreeQueue {
 private:
  struct Node {
    T data_;
    Node *next_;
    Node() : data_(), next_(nullptr) {}
    Node(T &data) : data_(data), next_(nullptr) {}
    Node(const T &data) : data_(data), next_(nullptr) {}
  };

  Node *head_, *tail_;

 public:
  LockFreeQueue() : head_(new Node()), tail_(head_) {}
  LockFreeQueue(LockFreeQueue&& lfq) {std::swap(head_, lfq.head_); std::swap(tail_,lfq.tail_);}
  ~LockFreeQueue() {
    Node *tmp;
    while (head_ != nullptr) {
      tmp = head_;
      head_ = head_->next_;
      delete tmp;
    }
   
    tmp = nullptr;
    tail_ = nullptr;
  }

  bool Try_Dequeue(T &data); 
  void Enqueue(const T &data); 
};

template<typename T> bool LockFreeQueue<T>::Try_Dequeue(T &data) {
    Node *old_head, *old_tail, *first_node;
    while (true) {
      old_head = head_;
      old_tail = tail_;
      first_node = old_head->next_;
      if (old_head != head_) continue;
      if (old_head == old_tail) {
        if (first_node == nullptr) return false;
        ::__sync_bool_compare_and_swap(&tail_, old_tail, first_node);
        continue;
      } else {
        data = first_node->data_;
        if (::__sync_bool_compare_and_swap(&head_, old_head, first_node)) break;
      }
    }
    if (old_head)
    delete old_head;

    return true;
  }

template<typename T> void LockFreeQueue<T>::Enqueue(const T &data) {
    Node *enqueue_node = new Node(data);
    Node *old_tail, *old_tail_next;
    while (true) {
      old_tail = tail_;
      old_tail_next = old_tail->next_;

      if (old_tail != tail_) continue;

      if (old_tail_next == nullptr) {
        if (::__sync_bool_compare_and_swap(&(old_tail->next_), old_tail_next,
                                           enqueue_node)) {
          break;
        }
      } else {
        ::__sync_bool_compare_and_swap(&tail_, old_tail, old_tail_next);
        continue;
      }
    }
    ::__sync_bool_compare_and_swap(&tail_, old_tail, enqueue_node);
  }
#endif
