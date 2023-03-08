/*
 * defs.h - local definitions for networking
 */

#pragma once
#ifdef __cplusplus
extern "C" {
#endif

// struct rte_mempool *mempool;
//

#define QUEUE_SIZE 256
//#define QUEUE_MASK (QUEUE_SIZE -1)
#define CYC_NUM 8

#define TARGET_TL 200000
#define INTERVAL_MIN 0

#define IDLE_THR 0
#define LIGHT_THR 0.25
#define MID_THR 0.5
#define HEAVY_THR 0.75


//#define MAX_RX_BURST 32 // for new connection

#define BATCH_SIZE 64
#define MAXTER_BATCH_SIZE 4*BATCH_SIZE
#define BATCH_MAX 2*BATCH_SIZE
#define BATCH_MIN 0

#define BTH 0.5*BATCH_SIZE

#define LOOP_MIN 0
#define LOOP_MAX TARGET_TL

#define LOAD_LEVEL 5

#if 0
enum {
  IDLE = 0,
  LIGHT, 
  MIDL,
  MIDH,
  HEAVY,
};
#endif

struct avg_poll_cycle {
  int core_id;
  int cyc_index;
  int target_level;
  int last_level;
  uint64_t interval[CYC_NUM];
  uint64_t interval_sum;
  uint64_t bt[CYC_NUM];
  uint64_t bt_sum;
//  TAIL_HEAD(, avg_pool_cycle) *curlist;
  //struct timespec last_poll_time;
  uint64_t last_poll_time;
  struct load_level *ll;

  float score;

  uint32_t conn[MAX_CPUS];
  uint32_t last_num_conn;
  uint32_t last_num_data;
  
//  TAILQ_ENTRY(avg_poll_cycle) lnode;
};

#if 0
struct load_level {
	int num;
	TAILQ_HEAD(, avg_poll_cycle) llist;
};

struct load_status {
  volatile  char status;
  struct load_level ll[LOAD_LEVEL];
};

extern struct load_status ls;
#endif 
extern struct rte_ring *conn_ring[MAX_CPUS];
extern struct avg_poll_cycle apc[MAX_CPUS];

extern volatile uint64_t coremask[LOAD_LEVEL];
extern volatile int worker_ready;

#ifdef __cplusplus
}
#endif
