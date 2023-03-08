#include "fdirhash.h"

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
struct fdir_hashtable *CreateFDirHashtable(
    unsigned int (*hashfn)(uint32_t),  // key function
    int (*eqfn)(uint32_t, uint32_t),   // equality
    int bins)                          // no of bins
{
  int i;
  struct fdir_hashtable *ht = calloc(1, sizeof(struct fdir_hashtable));
  if (!ht) {
    fprintf(stderr, "calloc: CreateHashtable");
    return 0;
  }

  ht->hashfn = hashfn;
  ht->eqfn = eqfn;
  ht->bins = bins;

  /* creating bins */
  if (IS_FLOW_TABLE(hashfn)) {
    ht->ht_table = calloc(bins, sizeof(fdir_hash_bucket_head));
    if (!ht->ht_table) {
      fprintf(stderr, "calloc: CreateHashtable bins!\n");
      free(ht);
      return 0;
    }
    /* init the tables */
    for (i = 0; i < bins; i++) TAILQ_INIT(&ht->ht_table[i]);
  }

  return ht;
}
/*----------------------------------------------------------------------------*/
void DestroyFDirHashtable(struct fdir_hashtable *ht) {
  if (IS_FLOW_TABLE(ht->hashfn)) free(ht->ht_table);
  free(ht);
}
/*----------------------------------------------------------------------------*/
fdir_flow *FDirStreamHTInsert(struct fdir_hashtable *ht, uint32_t fdir_id,
                              uint32_t pkt_size, uint32_t start_idx) {
  /* create an entry*/
  int idx;
  fdir_flow *item = (fdir_flow *)malloc(sizeof(*item));
  assert(ht);
  if (NULL == item) return NULL;
  item->fdir_id = fdir_id;
  item->size = pkt_size;
  //printf("insert fdirstream, flow id:%u start_idx:%u\n", fdir_id, start_idx);
  item->start_idx = start_idx;
  item->tmp_end_idx = start_idx;
  item->start_time = rte_rdtsc();
  item->state = RECV;
  item->num = 1;
  //item->tmp_end_idx = INVALID_ARRAY_INDEX;
  idx = ht->hashfn(fdir_id);
  assert(idx >= 0 && idx < NUM_BINS_FLOWS);
  TAILQ_INSERT_TAIL(&ht->ht_table[idx], item, fd_link);
  ht->flow_num++;
  // item->ht_idx = TCP_AR_CNT;
  return item;
}
/*----------------------------------------------------------------------------*/
void FDirStreamHTRemove(struct fdir_hashtable *ht, fdir_flow *item) {
  fdir_hash_bucket_head *head;

  int idx = ht->hashfn(item->fdir_id);
  head = &ht->ht_table[idx];
  TAILQ_REMOVE(head, item, fd_link);
}
/*----------------------------------------------------------------------------*/

fdir_flow *FDirStreamHTSearch(struct fdir_hashtable *ht, uint32_t fdir_id) {
  int idx;
  fdir_flow *walk;
  fdir_hash_bucket_head *head;
  idx = ht->hashfn(fdir_id);

  head = &ht->ht_table[idx];
  TAILQ_FOREACH(walk, head, fd_link) {
    if (ht->eqfn(walk->fdir_id, fdir_id)) {
      return walk;
    }
  }

  //	UNUSED(idx);
  return NULL;
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
#endif

unsigned int FDirHashFlow(uint32_t fdir_hash) {
  return fdir_hash & (NUM_BINS_FLOWS - 1);
}
/*---------------------------------------------------------------------------*/
int FDirEqualFlow(uint32_t fdir1, uint32_t fdir2) { return fdir1 == fdir2; }

#if 0
int
EqualFlow(const void *f1, const void *f2)
{
	tcp_stream *flow1 = (tcp_stream *)f1;
	tcp_stream *flow2 = (tcp_stream *)f2;

	return (flow1->saddr == flow2->saddr && 
			flow1->sport == flow2->sport &&
			flow1->daddr == flow2->daddr &&
			flow1->dport == flow2->dport);
}
#endif

#if 0
unsigned int HashSID(const void *f) {
	tcp_stream *flow = (tcp_stream *)f;
	return (flow->id % (NUM_BINS_FLOWS -1));
}

int
EqualSID(const void *f1, const void *f2) {
	return (((tcp_stream *)f1)->id == ((tcp_stream *)f2)->id);
}
#endif
