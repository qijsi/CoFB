#ifndef _GDP_H_
#define _GDP_H_

#include <stdbool.h>
#include <stdint.h>
#include <sys/queue.h>
#include <rte_mbuf.h>

#define NUM_BINS_FLOWS (8000) /* 8000 entries*/
#define PRIORITY_NUM 4
#define QUEUE_CAPACITY 200

typedef enum {
  NEW,
  RECV,
  FIN,
  IDLE,
} flow_state;


typedef struct flow_info {
  flow_state state;
  uint32_t fdir_id;
  uint32_t pkt_num; // num of queued pkts
  uint64_t start_time;
  uint64_t last_active;
//  uint32_t weight; // processed_pkt / flow_size;
  uint32_t processed_mbuf;
  uint32_t pending_mbuf;
  uint32_t flow_size;
  uint32_t weight;
} flow_info;

typedef struct flow_table {
  flow_info info;
  struct rte_mbuf *head;
  struct rte_mbuf *tail;
} flow_table;

typedef struct flow_elem {
  uint32_t fdir_id;
  TAILQ_ENTRY(flow_elem) next;
} flow_elem;

typedef struct priority_queue {
  TAILQ_HEAD(, flow_elem) head[PRIORITY_NUM];
  uint32_t capacity;
  uint32_t size;
} priority_queue;

typedef struct gdp_flowtable {
  uint32_t bins;
  int (*eqfn)(uint32_t, uint32_t);
  flow_table *table;
} gdp_flowtable;


gdp_flowtable *CreateGDPHashtable(int (*eqfn)(uint32_t, uint32_t), int bins);

int GDPInsertPkt(struct gdp_flowtable *ht, uint32_t fdir_id,
                  struct rte_mbuf *mbuf);

struct rte_mbuf *GDPIssuePkt(struct gdp_flowtable *ht, uint32_t fdir_id);

void DestroyGDPHashtable(struct gdp_flowtable *ht);

void InitPriorityQueue(priority_queue queue); 
uint32_t DoubleCapacity(priority_queue queue);
void QueueInsert(priority_queue queue, uint32_t fdir_id);
#endif /* FHASH_H */
