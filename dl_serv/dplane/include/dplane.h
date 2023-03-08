//#define _LARGEFILE64_SOURCE
#ifndef __DPLANE_H_
#define __DPLANE_H_

#include <rte_lcore.h>

#include <atomic>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <regex>
#include <thread>

#include "inference.h"
//#include "lockfreeq.h"
#include "modelinfo.h"
#include "sm_pool.h"
//#include "skiplist.h"
#include "concurrentqueue.h"

extern "C" {
#include "mtcp_api.h"
#include "mtcp_epoll.h"
}

#define PROFILING 0
#define MAX_FLOW_NUM (10000)

#define RCVBUF_SIZE (2 * 1024)
#define SNDBUF_SIZE (8 * 1024)

#define MAX_EVENTS (MAX_FLOW_NUM * 2)

#define MAX_FILES 30

#ifndef MAX_CPUS
#define MAX_CPUS 48
#endif

#define BATCH_LIMIT 32
#define PIXEL 224 * 224 * 3

#define BATCH_PER_RSP 1
#define MAXBATCHID 0xfffffff

#define MAX_JSON_CONTENT_SIZE 65535
#define HTTP_SEPARATOR "\r\n\r\n"
/*----------------------------------------------------------------------------*/
typedef enum { FORMULA, LOOKUP } BTMODE;

typedef std::pair<int, int> eb_type;

template <typename T, class Container = std::vector<T>,
          class Compare = std::greater<typename Container::value_type>>
class custom_priority_queue
    : public std::priority_queue<T, Container, Compare> {
 public:
  bool remove(T &value) {
    auto it = std::find(this->c.begin(), this->c.end(), value);
    if (it == this->c.end()) {
      return false;
    }
    if (it == this->c.begin()) {
//      this->top();
      this->pop();
    } else {
      this->c.erase(it);
      std::make_heap(this->c.begin(), this->c.end(), this->comp);
    }
    return true;
  }
};

class Server {
 public:
  int AcceptConnection(uint32_t listener);
  void CloseConnection(uint32_t sockid);
  int InitServer(uint32_t numworker, uint32_t thoffset, uint32_t backlog,
                 uint32_t port, uint32_t max_batch,
                 const std::string &mtcp_config_file,
                 const std::string &model_config_file,
                 const std::string &batch_config_file, uint32_t sla);
  void Warmup();
  void SetTrtEngine(std::shared_ptr<trt_t> trtVar,
                    const std::string &model_file);
  static void AppLogic(void *arg);
  static void DispLogic(void *arg);
  void AppLogicLocal(void *arg);
  void DispLogicLocal(void *arg);
  static int ThreadRun(void *core);
  int ThreadRunLocal(void *core);
  static void SignalHandler(int signum);
  void StartWorker();
  int ReqProcess(uint32_t socketid, bool waiting);
  int RspProcess(struct rsp_message rsp);
  int Process_config(int sockid);
  int Process_metadata(int sockid);
  int Process_servermeta(int sockid);
  uint32_t evaluate_sla();

  int send_response(struct Response rsp, uint32_t buf_idx, uint32_t offset,
                    uint32_t sockid, std::string id, uint32_t status_code);
  int get_numctx();
  int get_thoffset();
  int get_numworker();
  int get_bsize(int headromm);
  std::string get_rspjson();
  std::string cache_rsp_json(int batchsize, std::string id);
  //  void PrepareInfer(struct Query* query, int sockid);
  struct model_info get_modelinfo();
  static Server &getInstance() { return instance; }
  inline std::shared_ptr<struct Inference> getinfer() { return infer; }
  void parsejsonfile(const std::string filename, struct model_info &minfo);
  void parsebatchfile(const std::string filename);
  inline uint64_t get_sla() { return sla_; }
  void check_notify();
  uint32_t sloslack_batch();
  uint32_t adaptive_adjust(uint32_t ss_batch);
  bool dequeue_req(uint32_t num, std::vector<struct req_message> &reqs);
  uint32_t gen_batchidx() { return batch_idx++ & MAXBATCHID; }
  uint32_t get_batchidx() { return batch_idx; }
  std::unordered_map<uint64_t, uint32_t> btable;
  uint32_t minitime;
//  std::vector<uint32_t> btable;
  std::regex sregex_;
  std::regex pregex_;
  BTMODE mode;
  int pre_batch;
  uint32_t pending_requests;
  uint32_t *gslack;
  bool allow;
  uint32_t setworker;
  uint32_t batch_idx;
  uint32_t buf_size;
  uint32_t mini_batchlimit;
  uint32_t mini_batchidx;
  uint32_t ongoing_batch;
  uint32_t actual_instance;

  context_buf cbuf;
  std::shared_ptr<moodycamel::ConcurrentQueue<struct req_message>> dis_lfq;
  std::vector<std::shared_ptr<moodycamel::ConcurrentQueue<struct rsp_message>>>
      worker_lfq;
  thread_local static std::unordered_map<int, std::unique_ptr<struct Query>>
      queryv;
  std::unordered_map<uint32_t, struct batch_message> batchmap;
  custom_priority_queue<struct batch_message, std::vector<struct batch_message>,
                        std::greater<struct batch_message>>
      batch_queue;

  std::priority_queue<struct req_message, std::vector<struct req_message>,
                      std::greater<struct req_message>>
      queue;
  std::priority_queue<struct req_message, std::vector<struct req_message>,
                      std::greater<struct req_message>>
      missed_reqs;

 private:
  Server(){};
  bool done[MAX_CPUS];
  std::random_device rd;
  std::mt19937 eng;
  std::uniform_int_distribution<uint32_t> distr;
  std::shared_ptr<struct Inference> infer;
  struct model_info minfo;
  std::vector<uint32_t> ibinding_size;
  std::vector<uint32_t> obinding_size;
  std::vector<nvinfer1::Dims> idims_vector;
  std::list<trt_ctx> ctx_pool;
  pthread_spinlock_t qlock[32];
  pthread_spinlock_t batchlock;

  uint32_t listener;
  static uint32_t backlog_;
  uint32_t port_;
  uint32_t nworker_;
  uint32_t offset_;
  uint32_t sla_;
  static struct sockaddr_in saddr;
  std::string rsp_json;
  struct mtcp_conf mcfg;
  pthread_t app_thread[MAX_CPUS];
  struct thread_context *ctxs[MAX_CPUS];
  static Server instance;
  thread_local static uint32_t workerid;
};

#endif
