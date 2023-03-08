#include "gdphash.h"

#include <arpa/inet.h>
#include <assert.h>
#include <math.h>
#include <netinet/in.h>
#include <rte_cycles.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/queue.h>
//#include "tailq.h"

#define IS_FLOW_TABLE(x) (x == FDirHashFlow)

/*----------------------------------------------------------------------------*/
gdp_flowtable *CreateGDPHashtable(int (*eqfn)(uint32_t, uint32_t), int bins) {
  int i;
  gdp_flowtable *ht = calloc(1, sizeof(gdp_flowtable));
  if (!ht) {
    fprintf(stderr, "calloc: CreateHashtable");
    return 0;
  }
  ht->eqfn = eqfn;
  ht->bins = bins;
  ht->table = calloc(bins, sizeof(flow_table));
  if (!ht->table) {
    fprintf(stderr, "calloc: Createflowtable bins!\n");
    free(ht);
    return 0;
  }
  /* init the tables */
  for (i = 0; i < bins; i++) {
    ht->table[i].head = (struct rte_mbuf *)malloc(sizeof(struct rte_mbuf));
    ht->table[i].tail = ht->table[i].head;
    ht->table[i].info.fdir_id = i;
    ht->table[i].info.flow_size = 0;
    ht->table[i].info.pkt_num = 0;
    ht->table[i].info.processed_pkt = 0;
    ht->table[i].info.weight = 0;
    ht->table[i].info.state = IDLE;
    ht->table[i].info.start_time = 0;
    ht->table[i].info.last_active = 0;
  }

  return ht;
}
/*----------------------------------------------------------------------------*/
void DestroyGDPHashtable(struct gdp_flowtable *ht) {
  free(ht->table);
  free(ht);
}
/*----------------------------------------------------------------------------*/
int GDPInsertPkt(struct gdp_flowtable *ht, uint32_t fdir_id,
                  struct rte_mbuf *mbuf) {
  /* create an entry*/
  ht->table[fdir_id].info.pending_mbuf += 1;
  ht->table[fdir_id].tail->next = mbuf;
  ht->table[fdir_id].tail = mbuf;
  while (ht->table[fdir_id].tail->next) {
    ht->table[fdir_id].tail = ht->table[fdir_id].tail->next;
  }
  if (ht->table[fdir_id].info.state == IDLE)
    ht->table[fdir_id].info.state = RECV;

  return ht->table[fdir_id].info.state;
}
/*----------------------------------------------------------------------------*/
struct rte_mbuf *GDPIssuePkt(struct gdp_flowtable *ht, uint32_t fdir_id) {
  struct rte_mbuf *pkt;
  if (ht->table[fdir_id].head != ht->table[fdir_id].tail) {
    pkt = ht->table[fdir_id].head->next;
    if (pkt == ht->table[fdir_id].tail)
      ht->table[fdir_id].tail = ht->table[fdir_id].head;
    else {
      ht->table[fdir_id].head->next = pkt->next;
    }
    ht->table[fdir_id].info.pending_mbuf--;
    return pkt;
  } else
    return NULL;
}
/*----------------------------------------------------------------------------*/
int FDirEqualFlow(uint32_t fdir1, uint32_t fdir2) { return fdir1 == fdir2; }

void InitPriorityQueue(priority_queue queue) {
  int i, num;
  flow_elem *item;
  for (i=0; i<PRIORITY_NUM; i++)
  TAILQ_INIT(&queue.head[i]);

  for (i=0, num=0; i<QUEUE_CAPACITY; i++) {
    item = (flow_elem *)malloc(sizeof(flow_elem));
    if (item) {
      num++;
      TAILQ_INSERT_TAIL(&queue.head[IDLE], item, next);
    }
  }
  queue.capacity = num;
  queue.size = 0;
  
}

uint32_t DoubleCapacity(priority_queue queue) {
    int i, num;
    flow_elem *item;
    for (i=0, num=0; i<queue.capacity; i++) {
    item = (flow_elem *)malloc(sizeof(flow_elem));
    if (item) {
      num++;
      TAILQ_INSERT_TAIL(&queue.head[IDLE], item, next);
    }
  }
  queue.capacity += num;
  return queue.capacity;
}

void QueueInsert(priority_queue queue, uint32_t fdir_id) {
  if (queue.size == queue.capacity)
  DoubleCapacity(queue);
  flow_elem *item = TAILQ_FIRST(&queue.head[IDLE]);
  item->fdir_id = fdir_id;
  TAILQ_INSERT_TAIL(&queue.head[RECV], item, next);
  TAILQ_REMOVE(&queue.head[IDLE], item, next);
  queue.size++;
}
/*----------------------------------------------------------------------------*/
#if 0
unsigned int
HashFlow(const void *f)
{
	tcp_stream *flow = (tcp_stream *)f;
	unsigned int hash, i;
	char *key = (char *)&flow->saddr;

	for (hash = i = 0; i < 12; ++i) {
		hash += key[i];
		hash += (hash << 10);
		hash ^= (hash >> 6);
	}
	hash += (hash << 3);
	hash ^= (hash >> 11);
	hash += (hash << 15);

	return hash & (NUM_BINS_FLOWS - 1);
}


unsigned int FDirHashFlow(uint32_t fdir_hash) {
  return fdir_hash & (NUM_BINS_FLOWS - 1);
}
/*---------------------------------------------------------------------------*/

#endif