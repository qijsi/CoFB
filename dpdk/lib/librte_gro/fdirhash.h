#ifndef FDIR_FHASH_H
#define FDIR_FHASH_H

#include <stdbool.h>
#include <stdint.h>
#include <sys/queue.h>

#define INVALID_ARRAY_INDEX 0xffffffffUL

#define NUM_BINS_FLOWS (131072) /* 132 K entries per thread*/
#define TCP_AR_CNT (3)

typedef enum {
  RECV,
  FLUSHABLE,
} flow_state;

typedef struct fdir_flow {
  flow_state state;
  uint32_t size;
  uint32_t start_idx;
  uint32_t tmp_end_idx;
  uint32_t fdir_id;
  uint32_t num;
  uint64_t start_time;
  TAILQ_ENTRY(fdir_flow) fd_link;
} fdir_flow;

typedef struct fdir_hash_bucket_head {
  fdir_flow *tqh_first;
  fdir_flow **tqh_last;
} fdir_hash_bucket_head;



/* hashtable structure */
struct fdir_hashtable {
  uint32_t bins;
  uint32_t flow_num;
  fdir_hash_bucket_head *ht_table;
  // functions
  unsigned int (*hashfn)(uint32_t);
  int (*eqfn)(uint32_t, uint32_t);
};

/*functions for hashtable*/
struct fdir_hashtable *CreateFDirHashtable(unsigned int (*hashfn)(uint32_t),
                                           int (*eqfn)(uint32_t, uint32_t),
                                           int bins);
void DestroyFDirHashtable(struct fdir_hashtable *ht);
fdir_flow *FDirStreamHTInsert(struct fdir_hashtable *ht, uint32_t fdir_id,
                              uint32_t pkt_size, uint32_t start_idx);
void FDirStreamHTRemove(struct fdir_hashtable *ht, fdir_flow *fl);
fdir_flow *FDirStreamHTSearch(struct fdir_hashtable *ht, uint32_t fdir_id);
unsigned int FDirHashFlow(uint32_t fdir_id);
int FDirEqualFlow(uint32_t fdir1, uint32_t fdir2);
#endif /* FHASH_H */
