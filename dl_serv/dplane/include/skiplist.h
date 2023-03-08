#pragma once
#include <assert.h>
#include <stdio.h>
#include <time.h>

#include <iostream>
#include <set>
#include <unordered_map>
#include <vector>

// static const double level_rangdom_p = 1 / 4.0;
//  static const int stand_max_level = 32;
static const double level_rangdom_p = 1 / 8.0;
static const int stand_max_level = 8;

template <class K, class V>
class skiplist {
 public:
  skiplist() : current_level(0), max_level(0), node_count(0), header(NULL) {}
  ~skiplist() { release(); }

  bool init(int max_level) {
    if (header != NULL) {
      return false;
    }
    this->max_level = max_level;
    srand((unsigned int)time(NULL));
    header = init_node(max_level, {}, {});
    return header != NULL;
  }

  uint32_t get_count() { return node_count; }

  bool find(const K& k, V& v) {
    node_ptr prev[stand_max_level] = {0};
    node_ptr p = find_node(k, prev);
    if (p != NULL) {
      v = p->value;
      return true;
    }
    return false;
  }

  void insert(const K& k, const V& v) {
    node_ptr prev[stand_max_level] = {0};
    node_ptr p = find_node(k, prev);
    if (p != NULL) {
    //  std::cout << "p != NULL" << p->value << std::endl;
      p->value = v;
      return;
    }

    int new_level = random_level();
    node_ptr new_node = init_node(new_level, k, v);
    node_count++;
#if 0
    std::cout << " current level: " << current_level
              << " new level: " << new_level << std::endl;
#endif

    for (int i = 0; i < current_level && i < new_level; i++) {
#if 0
				std::cout<<"i: "<<i<<" current level: "<<current_level<<" new level: "<<new_level<<std::endl;
				if (prev[i]==NULL)
				std::cout<<"prev[i]->level_arr = NULL"<<std::endl;
				std::cout<<"pre->level_arr.size: "<<prev[i]->level_arr.size()<<std::endl;
				std::cout<<"new_node->level_arr.size:  "<<new_node->level_arr.size()<<std::endl;
				if (prev[i]->level_arr.at(i)==NULL)
				std::cout<<"level_arr[i] is NULL"<<std::endl;
				if (new_node->level_arr.at(i)==NULL)
				std::cout<<"new_node[i] is NULL"<<std::endl;
#endif

      new_node->level_arr[i] = prev[i]->level_arr[i];
      prev[i]->level_arr[i] = new_node;
    }

    if (new_level > current_level) {
      for (int i = current_level; i < new_level; i++)
        header->level_arr[i] = new_node;
      current_level = new_level;
      header->level = current_level;
    }
  }

  bool get_range(int num, std::vector<V>& out) {
    int i, actual_num;
    node_ptr test;
    //std::unordered_map<V, int> key_idx;
    std::unordered_map<K, int> key_idx;
    node_ptr p, max_level_p;
    int tmp_max_level;

    actual_num = std::min(node_count, num);

    if (actual_num <= 0) return false;

    max_level_p = header->level_arr[0];
    p = header->level_arr[0];

    if (actual_num == node_count) {
      for (i = 0; i < actual_num; i++) {
        if (p==nullptr) fprintf(stderr, "actual_num:%d p is null\n", actual_num);
        out.emplace_back(p->value);
        p = p->level_arr[0];
      }

      for (i = current_level-1; i >= 0; i--) header->level_arr[i] = NULL;
      header->level_arr.resize(0);
      current_level = 0;
    } else {
      tmp_max_level = max_level_p->level_arr.size();

      for (i = 0; i < actual_num; i++) {
       // V v = p->key;
       K k = p->key;
        key_idx.emplace(k, i);
        // ret.emplace_back(p->value);
        out.emplace_back(p->value);

        if (max_level_p->level_arr.size() < p->level_arr.size()) {
          max_level_p = p;
          tmp_max_level = p->level_arr.size();
        }

        p = p->level_arr[0];
      }

      for (i = tmp_max_level - 1; i >= 0; i--) {
        if (max_level_p->level_arr[i] == NULL) {
          header->level_arr[i] = NULL;
          assert(current_level > 0);
          current_level--;
          header->level_arr.resize(current_level);
        } else {
          test = max_level_p->level_arr[i];
          // header->leval_arr[i] points to the first element beyond parameter
          // "num". Currently, we havn't consider elements with the same key.
          while (key_idx.find(test->key) != key_idx.end()) {
            if (test->level_arr[i] != NULL)
              test = test->level_arr[i];
            else
              break;
          }

          header->level_arr[i] = test;
        }
      }
    }
    node_count -= actual_num;

    return true;
  }

  K get_firstkey() {
    assert(header->level_arr[0] != NULL);
    return header->level_arr[0]->key;
  }

  K get_Nthkey(int n) {
    int i=0;
    node_ptr p = header;
    while(p->level_arr[0] != NULL && i<n) {
      p = p->level_arr[0];
      i++;
    }

    return p->key;
  }

  V get_firstvalue() {
    assert(header->level_arr[0] != NULL);
    return header->level_arr[0]->value;
  }

  V get_Nthvalue(int n) {
    int i=0;
    node_ptr p = header;
    while(p->level_arr[0] != NULL && i<n) {
      p = p->level_arr[0];
      i++;
    }

    return p->value;
  }

  bool remove(const K& k) {
    assert(header != NULL);
    node_ptr p = header;
    for (int i = current_level - 1; i >= 0; --i) {
      while (p->level_arr[i] != NULL) {
        if (p->level_arr[i]->key == k) {
          node_ptr rm = p->level_arr[i];
          p->level_arr[i] = rm->level_arr[i];
          if (i == 0) {
            delete rm;
            node_count--;
            return true;
          }

          if (p == header && p->level_arr[i] == NULL) {
            current_level--;
          }

          break;
        }

        if (p->level_arr[i]->key > k) break;
        p = p->level_arr[i];
      }
    }
    return false;
  }


 protected:
  typedef struct node {
    K key;
    V value;
    int level;
    std::vector<node*> level_arr;
  } * node_ptr;

  node_ptr init_node(int level, const K& k, const V& v) {
    node_ptr p = new node;
    if (p != NULL) {
      p->key = k;
      p->value = v;
      p->level_arr.resize(level, NULL);
      p->level = level;
    }
    return p;
  }

  node_ptr first() {
    if (header == NULL) return NULL;
    return header->level_arr[0];
  }

  node_ptr find_node(const K& k, node_ptr* prev) {
    assert(header != NULL);
    if (node_count < 1) {
      //    std::cout << "node_count<1" << std::endl;
      prev[0] = header;
      return NULL;
    }
    // assert(node_count >= 1 && "node_count < 1");

    node_ptr p = header;
    int step = 0;
    for (int i = current_level - 1; i >= 0; --i) {
      step++;
      while (p->level_arr[i] != NULL) {
#if 0
        std::cout << "i: " << i
                  << " p->level_arr[i]->key: " << p->level_arr[i]->key
                  << " insert k: " << k << std::endl;
#endif
        if (p->level_arr[i]->key == k) {
          return p->level_arr[i];
        }

        if (p->level_arr[i]->key > k) break;
        step++;
        p = p->level_arr[i];
      }
      prev[i] = p;
    }
    return NULL;
  }

  int random_level() {
    int level = 1;
    float rand_ = rand() % 10 / 10.0;
    //  std::cout << "\r\nrandom number: " << rand_ << std::endl;
    while (rand_ < level_rangdom_p && level < max_level) level++;
    return level;
  }

  void release() {
    if (header == NULL) return;
    node_ptr p = header->level_arr[0];
    while (p != NULL) {
      node_ptr next = p->level_arr[0];
      delete p;
      p = next;
    }
    delete header;
    header = NULL;
  }

 private:
  int current_level;
  int max_level;
  int node_count;
  node_ptr header;
};
