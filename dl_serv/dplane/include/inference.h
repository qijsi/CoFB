#ifndef __INFERENCE_H_
#define __INFERENCE_H_

#include <NvInfer.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <list>
#include <memory>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <map>

#include "modelinfo.h"
#include "utili.h"

#define DEVICE 0
#define CYCLE 4000

#define MAX_BATCH 16
#define CMP_STREAM_SIZE 4
#define BUCKET_STEP 8
#define BUCKET_NUM 4

struct req_message {
  uint32_t workerid;
  uint32_t sockid;
  char *req_addr;
  uint32_t size;
 // uint64_t query_recv_latency;
  uint64_t recv_start_time, recv_fin_time, queue_time;
  uint64_t recv_latency, queue_latency, reqtrans_latency, rsptrans_latency, inference_latency;
  uint64_t missed_enqueue_time;
  bool operator>(const req_message &req2) const {
    return recv_start_time > req2.recv_start_time;
  }
};

struct batch_message {
  uint32_t batch_idx;
  uint32_t batch_size;
  uint32_t batch_limit;

  bool operator>(const batch_message &req) const {
    return batch_limit > req.batch_limit;
  }

  bool operator==(const batch_message &req) const {
    return batch_idx == req.batch_idx;
  }
};

struct rsp_message {
  uint32_t workerid;
  uint32_t sockid;
  uint32_t batchid;
  uint32_t batchsize;
  uint32_t offset;
  uint32_t size;
  
  char *rsp_addr;

  uint32_t recv_latency, queue_latency, reqtrans_latency, rsptrans_latency, inference_latency;
  uint64_t fin_time;
};

struct Query {
  bool head_processed;
  std::string id;
  uint32_t read_offset;
  uint32_t req_offset;
  uint32_t req_len;
  uint32_t cont_len;
  uint32_t json_start;
  uint32_t json_len;
  uint32_t act_len;
  uint32_t recv_num;

  struct timeval recv_start_io;
  uint64_t recv_start, last, recv_fin;
  std::chrono::time_point<std::chrono::system_clock> mem_fin, queue_fin,
      infer_fin, ifer_rsp, rsp_fin;

  //char data[655360];
   char data[2800000];
 // char data[100000];
};

typedef struct {
  nvinfer1::IRuntime *runtime;
  nvinfer1::ICudaEngine *engine;
  nvinfer1::IExecutionContext *context;
} trt_t;

typedef struct {
  nvinfer1::IExecutionContext *context;
  cudaStream_t stream;
  cudaEvent_t event;
} trt_ctx;

class Inference {
 public:
  void Init(const std::string &model);
  // std::shared_ptr<cppflow::model> LoadModel(const std::string& model_pb);
  int LoadTFModel(const std::string &model_pb, const std::string &i_operator,
                  const std::string &o_operator);
  void SetTrtEngine(std::shared_ptr<trt_t> trtVar,
                    const std::string &model_file);
  void PrepareInfer(struct Query *query, int sockid);
  void CopyreqAsync(uint32_t buf_idx, uint32_t batch_size);

  void DoInference(int device, uint32_t buf_idx, uint32_t batch_size);

  bool AdaptiveBatching(uint32_t pending_requests);
  void CopyrspAsync(uint32_t buf_idx, uint32_t batch_size);
  void InitInputOutput(struct model_info minfo);
  uint32_t InitBuffer();
  nvinfer1::IExecutionContext *getCudaContext(int i) { return ctx[i].context; }
  const trt_ctx& getTrTContext(int i) { return ctx[i]; }
  void setCudaContext(nvinfer1::IExecutionContext *context, int i) {
    ctx[i].context = context;
  }
  void CreateSmPool();
  // uint32_t get_size(std::string itype);
  std::vector<std::string> get_input();
  std::vector<std::string> get_output();
  std::vector<uint32_t> get_ibinding_size();
  std::vector<uint32_t> get_obinding_size();
  void **get_hbuf(int i) { return hbuf[i]; }
  void **get_dbuf(int i) { return dbuf[i]; }
  uint32_t get_inputdims() { return num_inputbinding; }
  uint32_t *get_ptail() { return ptail; }
  void increase_access_count(int i, int num) { access_count[i].fetch_add(num); }
  void decrease_access_count(int i, int num) { access_count[i].fetch_sub(num); }
  uint32_t get_access_count(int i) { return access_count[i].load(); }
  uint32_t get_freebuf_size() { return free_buf.size(); }
  uint32_t get_frontbuf() { return free_buf.front(); }
  uint32_t pop_freebuf() {
    uint32_t idx = free_buf.front();
    free_buf.pop();
    return idx;
  }
  void push_freebuf(uint32_t idx) { free_buf.push(idx); }
  std::vector<nvinfer1::Dims> get_idims_vector();
  void set_idims_vector(std::vector<nvinfer1::Dims> idims);
  std::shared_ptr<trt_t> get_trtengine();
  std::shared_ptr<trt_t> crerate_trtmodel();
  void SetBinding(std::shared_ptr<trt_t> trtVar, int idx, int total,
                  int buf_size);
  ~Inference();
  trt_ctx ctx[CMP_STREAM_SIZE];
 private:
  uint32_t device_;

  std::shared_ptr<trt_t> trtEngine_;
  
  void **hbuf[CMP_STREAM_SIZE];
  void **dbuf[CMP_STREAM_SIZE];
  uint32_t *ptail;
  std::atomic<uint32_t> *access_count;
  std::queue<uint32_t> free_buf;
  // nvinfer1::IExecutionContext *context[CMP_STREAM_SIZE];
  int total_req_size;
  std::atomic<int> iIdx;
  std::vector<uint32_t> ibinding_size;
  std::vector<uint32_t> obinding_size;
  std::vector<nvinfer1::Dims> idims_vector;
  struct model_info mio;
  uint32_t num_inputbinding;
};

typedef struct {
  uint32_t time;
  uint32_t socketid;
} rinfo;

typedef struct batch_info {
  uint32_t batch_idx;
  uint32_t buf_idx;
  uint32_t batch_size;
  cudaEvent_t event;
  /// std::vector<uint32_t> socket_ids;
  std::vector<struct req_message> einfo;
  //std::chrono::time_point<std::chrono::system_clock> queue_time, reqtrans_time, infer_time, rsptrans_time;
  uint64_t queue_time, reqtrans_time, infer_time, rsptrans_time;
} batch_info;

typedef enum { IDLE, TRANS_INP, INFER, TRANS_OUP } TSTATE;

typedef struct {
  bool do_infer;
  bool waiting;
  uint32_t schedule_size;
  uint32_t do_infer_batch;
  uint32_t batch_size;
  uint32_t workerid;
  std::list<uint32_t> concurrent_batch;

  uint32_t buf_access[CMP_STREAM_SIZE];

  TSTATE state;

  std::queue<uint32_t> free_shared;

  std::list<batch_info> req_list;
  std::list<batch_info> inf_list;
  std::list<batch_info> rsp_list;
  std::chrono::time_point<std::chrono::system_clock> wait_time;

  // std::vector<int> b_ts;
  uint32_t bsize;
///  std::chrono::time_point<std::chrono::system_clock> stime;
  uint64_t stime;
  std::chrono::time_point<std::chrono::system_clock> wtime;

} context_buf;

#define CHECK(status)                                       \
  do {                                                      \
    if (status != 0) {                                      \
      std::cerr << "Cuda failure: " << status << std::endl; \
      abort();                                              \
    }                                                       \
  } while (0)

#endif
