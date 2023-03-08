//#define _LARGEFILE64_SOURCE

#include "dplane.h"

#include <absl/strings/numbers.h>
#include <absl/strings/str_split.h>
#include <absl/strings/string_view.h>
#include <arpa/inet.h>
#include <cuda.h>
#include <math.h>
#include <netinet/in.h>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/stringbuffer.h>
#include <sched.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <opencv2/core/version.hpp>

#include "cpu.h"
#include "debug.h"
#include "http.h"
#include "logging.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "skiplist.h"
/*----------------------------------------------------------------------------*/

static pthread_mutex_t init_lock;
static int init_count = 0;
static pthread_cond_t init_cond;

static void wait_for_thread_registration(int nthreads) {
  while (init_count < nthreads) {
    pthread_cond_wait(&init_cond, &init_lock);
  }
}

static void register_thread_initialized(void) {
  pthread_mutex_lock(&init_lock);
  init_count++;
  pthread_cond_signal(&init_cond);
  pthread_mutex_unlock(&init_lock);
}

uint32_t Server::backlog_;
struct sockaddr_in Server::saddr;
Server Server::instance;
thread_local uint32_t Server::workerid;
thread_local std::unordered_map<int, std::unique_ptr<struct Query>>
    Server::queryv;

int Server::RspProcess(struct rsp_message rsp) {
  int sent_byte, total_byte = 0;

  struct Response reply;

  if (rsp.batchid != 0xffffffff)
    reply.status = 200;
  else {
    std::cout << "RspProcess, sockid: " << rsp.sockid << std::endl;
    reply.status =
        200;  // for evaluation with perf_client, we set the http code to 200,
              // but delay send response to time of 2*sla.
    //    reply.status = 400;
  }

  reply.version = "1.1";

  if (queryv.at(rsp.sockid) != nullptr) {
    queryv.at(rsp.sockid)->infer_fin = std::chrono::system_clock::now();
  }

  if (queryv.at(rsp.sockid) == NULL) {
    std::cout << "worker: " << workerid << " batch id: " << rsp.batchid
              << " batch_size: " << rsp.batchsize << " rsp sockid: is null"
              << rsp.sockid << std::endl;
    return 0;
  }

  assert(queryv.at(rsp.sockid) != NULL);

  sent_byte = send_response(reply, rsp.batchid, rsp.offset, rsp.sockid,
                            queryv.at(rsp.sockid)->id, reply.status);
  total_byte += sent_byte;

  queryv.erase(rsp.sockid);

  return total_byte;
}

void Server::CloseConnection(uint32_t sockid) {
  auto ctx = ctxs[workerid];
  mtcp_epoll_ctl(ctx->mctx, ctx->ep, MTCP_EPOLL_CTL_DEL, sockid, NULL);
  mtcp_close(ctx->mctx, sockid);
}

int Server::ReqProcess(uint32_t sockid, bool waiting) {
  static std::chrono::time_point<std::chrono::system_clock> start, end;
  int rd;
  auto ctx = ctxs[workerid];
  int req_length = 0;
  int json_len = 0;
  absl::string_view header;
  absl::string_view content;
  absl::string_view payload;
  std::vector<absl::string_view> sub_header;
  struct Request req;
  std::unique_ptr<struct Query> query;
  int idx;
  bool newq;
  if (queryv.find(sockid) == queryv.end()) {
    query = std::make_unique<struct Query>();
    newq = true;
    query->read_offset = 0;
    query->req_offset = 0;
    query->cont_len = 0;
    query->act_len = 0;
    query->head_processed = 0;
    query->recv_num = 0;
  } else {
    query = std::move(queryv[sockid]);
    newq = false;
  }

  if (query == nullptr) {
    std::cout << "sockid: " << sockid << " query is null" << std::endl;
    exit(1);
  }

  rd =
      mtcp_recv(ctx->mctx, sockid, query->data + query->read_offset, 100000, 0);

  if (rd <= 0) {
    if (queryv.find(sockid) != queryv.end())
      queryv[sockid] = std::move(query);
    else {
      queryv.emplace(std::piecewise_construct, std::forward_as_tuple(sockid),
                     std::forward_as_tuple(std::move(query)));
    }
    std::cout << "sockid: " << sockid << "rd <=0 " << std::endl;
    return rd;
  } else {
    query->read_offset += rd;
    if (query->head_processed) {
      query->act_len += rd;
      query->recv_num++;
      query->last = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();

      if (query->cont_len &&
          query->cont_len == query->act_len - query->json_len) {
        query->recv_fin =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count();
        rapidjson::Document doc;
        doc.Parse(std::string(query->data + query->json_start, query->json_len)
                      .data());
        if (doc.HasMember("id")) {
          query->id = doc["id"].GetString();
        }

        assert(query != nullptr && "query is null");
        struct req_message req;
        req.workerid = workerid;
        req.sockid = sockid;
        req.req_addr = query->data + query->req_offset;

        req.size = query->cont_len;
        req.recv_start_time = query->recv_start;
        req.recv_fin_time = query->recv_fin;
        req.recv_latency = req.recv_fin_time - req.recv_start_time;
        query->read_offset = 0;
        query->cont_len = 0;
        query->act_len = 0;
        query->json_len = 0;
        query->head_processed = false;
        query->mem_fin = std::chrono::system_clock::now();

        dis_lfq->enqueue(req);
      }

      if (queryv.find(sockid) == queryv.end())
        std::cout << "queryv[" << sockid << "] is null" << std::endl;
      queryv.at(sockid) = std::move(query);
    } else {
      absl::string_view data(query->data, rd);
      auto hc_separator = data.find(HTTP_SEPARATOR);
      if (hc_separator != absl::string_view::npos) {
        start = std::chrono::system_clock::now();
        query->head_processed = true;
        header = data.substr(0, hc_separator);
        content = data.substr(hc_separator + sizeof(HTTP_SEPARATOR) - 1);
        sub_header = absl::StrSplit(header, "\r\n");
        std::vector<absl::string_view> m_h_v =
            absl::StrSplit(sub_header[0], " ");
        if (m_h_v[0] == "GET" && m_h_v[1] == "/v2")
          return Process_servermeta(sockid);
        else {
          std::vector<absl::string_view> path = absl::StrSplit(m_h_v[1], "/");
          if (m_h_v[0] == "GET") {
            if (path.size() == 4) {
              return Process_metadata(sockid);
            } else if (path.size() == 5 && path[4] == "config") {
              return Process_config(sockid);
            }
            // } else if (m_h_v[0] == "POST" && req.operation == "infer") {
          } else if (m_h_v[0] == "POST" && path.size() == 5 &&
                     path[4] == "infer") {
            for (auto &iter : sub_header) {
              std::vector<absl::string_view> k_v = absl::StrSplit(iter, ":");
              if (k_v[0] == "Content-Length")
                req_length = std::stoi(k_v[1].data());
              else if (k_v[0] == "Inference-Header-Content-Length")
                json_len = std::stoi(k_v[1].data());
              if (req_length && json_len) break;
            }

            assert(req_length && "req_length is 0");
            assert(json_len && "json_length is 0");

            idx = hc_separator + sizeof(HTTP_SEPARATOR) - 1;
            query->json_start = idx;
            query->json_len = json_len;
            query->req_offset = idx + json_len;
            query->req_len = req_length;
            query->cont_len = req_length - json_len;
            query->recv_start =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count();
            query->last = query->recv_start;

            query->act_len = (idx < rd) ? rd - idx : 0;
            query->head_processed = true;
            query->recv_num = 1;
            if (query->cont_len &&
                (query->cont_len == query->act_len ||
                 query->cont_len + query->json_len == query->act_len)) {
              query->recv_fin =
                  std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count();
              query->read_offset = 0;
              query->cont_len = 0;
              query->act_len = 0;
              query->json_len = 0;
              query->head_processed = false;
              query->mem_fin = std::chrono::system_clock::now();

              struct req_message req;
              req.workerid = workerid;
              req.sockid = sockid;
              req.req_addr = query->data + query->req_offset;

              req.recv_fin_time =
                  std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count();
              dis_lfq->enqueue(req);
            }

            if (queryv.find(sockid) != queryv.end())
              queryv[sockid] = std::move(query);
            else {
              queryv.emplace(std::piecewise_construct,
                             std::forward_as_tuple(sockid),
                             std::forward_as_tuple(std::move(query)));
            }
          } else {
            std::cout << m_h_v[0] << " " << path.size() << " " << path[4]
                      << std::endl;
          }
        }

      } else {
        absl::string_view data(query->data, 10);
        std::cout << "cpu: " << workerid << " sockid: " << sockid
                  << "Cannot find http header\n"
                  << " cont_len: " << query->cont_len
                  << " act_len: " << query->act_len << " read len: " << rd
                  << data << " newq: " << newq << std::endl;
      }
    }
  }
  return rd;
}

int Server::Process_config(int sockid) {
  uint32_t sent;
  struct Response rsp;
  BufferStream bstrm;
  rapidjson::StringBuffer strBuf;
  rapidjson::Writer<rapidjson::StringBuffer> writer(strBuf);
  writer.StartObject();
  writer.Key("name");
  writer.String(minfo.name.data());
  writer.Key("platform");
  writer.String("tensorrt_plan");
  writer.Key("backend");
  writer.String("");
  writer.Key("version_policy");
  writer.StartObject();
  writer.Key("latest");
  writer.StartObject();
  writer.Key("num_versions");
  writer.Int(1);
  writer.EndObject();
  writer.EndObject();
  if (minfo.max_batch_size) {
    writer.Key("max_batch_size");
    writer.Int(minfo.max_batch_size);
  }
  writer.Key("input");
  writer.StartArray();
  for (uint32_t i = 0; i < minfo.minput.size(); i++) {
    writer.StartObject();
    writer.Key("name");
    writer.String(minfo.minput[i].name.data());
    writer.Key("data_type");
    writer.String(minfo.minput[i].data_type.data());
    writer.Key("format");
    writer.String("FORMAT_NONE");
    writer.Key("dims");
    writer.StartArray();
    for (uint32_t j = 0; j < minfo.minput[i].dims.size(); j++) {
      writer.Int(minfo.minput[i].dims[j]);
    }
    writer.EndArray();
    writer.Key("is_shape_tensor");
    writer.Bool(false);
    writer.Key("allow_ragged_batch");
    writer.Bool(false);
    writer.EndObject();
  }
  writer.EndArray();
  writer.Key("output");
  writer.StartArray();
  for (uint32_t i = 0; i < minfo.moutput.size(); i++) {
    writer.StartObject();
    writer.Key("name");
    writer.String(minfo.moutput[i].name.data());
    writer.Key("data_type");
    writer.String(minfo.moutput[i].data_type.data());
    writer.Key("dims");
    writer.StartArray();
    for (uint32_t j = 0; j < minfo.moutput[i].dims.size(); j++) {
      writer.Int(minfo.moutput[i].dims[j]);
    }
    writer.EndArray();
    writer.Key("label_filename");
    writer.String("");
    writer.Key("is_shape_tensor");
    writer.Bool(false);
    writer.EndObject();
  }
  writer.EndArray();
  writer.Key("batch_input");
  writer.StartArray();
  writer.EndArray();
  writer.Key("batch_output");
  writer.StartArray();
  writer.EndArray();
  // writer.EndObject();
  writer.Key("optimization");
  writer.StartObject();
  writer.Key("priority");
  writer.String("PRIORITY_DEFAULT");
  writer.Key("cuda");
  writer.StartObject();
  writer.Key("graphs");
  writer.Bool(false);
  writer.Key("busy_wait_events");
  writer.Bool(false);
  writer.Key("graph_spec");
  writer.StartArray();
  writer.EndArray();
  writer.Key("output_copy_stream");
  writer.Bool(false);
  writer.EndObject();
  writer.Key("input_pinned_memory");
  writer.StartObject();
  writer.Key("enable");
  writer.Bool(true);
  writer.EndObject();
  writer.Key("output_pinned_memory");
  writer.StartObject();
  writer.Key("enable");
  writer.Bool(true);
  writer.EndObject();
  writer.Key("gather_kernel_buffer_threshold");
  writer.Int(0);
  writer.Key("eager_batching");
  writer.Bool(false);
  writer.EndObject();
  writer.Key("instance_group");
  writer.StartArray();
  writer.StartObject();
  writer.Key("kind");
  writer.String("KIND_GPU");
  writer.Key("count");
  writer.Int(1);
  writer.Key("gpus");
  writer.StartArray();
  writer.Int(0);
  writer.EndArray();
  writer.Key("profile");
  writer.StartArray();
  writer.EndArray();
  writer.Key("passive");
  writer.Bool(false);
  writer.Key("host_policy");
  writer.String("");
  writer.EndObject();
  writer.EndArray();
  writer.Key("default_model_filename");
  writer.String("model.plan");
  writer.Key("cc_model_filenames");
  writer.StartObject();
  writer.EndObject();
  writer.Key("metric_tags");
  writer.StartObject();
  writer.EndObject();
  writer.Key("parameters");
  writer.StartObject();
  writer.EndObject();
  writer.Key("model_warmup");
  writer.StartArray();
  writer.EndArray();
  writer.EndObject();

  std::string data = strBuf.GetString();
  char json_len[10];
  sprintf(json_len, "%lu", data.size());
  rsp.set_header("Content-Type", "application/json");
  rsp.body = std::move(data);

  rsp.status = 200;
  write_response(bstrm, rsp);
  auto rsp_data = bstrm.get_buffer();
  auto ctx = ctxs[workerid];
  sent = mtcp_write(ctx->mctx, sockid, rsp_data.data(), rsp_data.size());
  if (sent != rsp_data.size()) {
    std::cout << "error in mtcp_write: sent size(" << sent << ")"
              << " rsp size(" << rsp_data.size() << ")" << std::endl;
  }
  queryv.erase(sockid);
  return sent;
}

int Server::Process_servermeta(int sockid) {
  struct Response rsp;
  uint32_t sent;
  BufferStream bstrm;
  rapidjson::StringBuffer strBuf;
  rapidjson::Writer<rapidjson::StringBuffer> writer(strBuf);
  writer.StartObject();
  writer.Key("name");
  writer.String("triton");
  writer.Key("version");
  writer.String("2.12.0");
  writer.Key("extensions");
  writer.StartArray();
  writer.String("classification");
  writer.String("sequence");
  writer.String("model_repository");
  writer.String("schedule_policy");
  writer.String("model_configuration");
  writer.String("binary_tensor_data");
  writer.EndArray();
  writer.EndObject();

  std::string data = strBuf.GetString();
  char json_len[10];
  sprintf(json_len, "%lu", data.size());
  rsp.set_header("Content-Type", "application/json");
  rsp.body = std::move(data);
  rsp.status = 200;
  write_response(bstrm, rsp);
  auto rsp_data = bstrm.get_buffer();
  auto ctx = ctxs[workerid];
  sent = mtcp_write(ctx->mctx, sockid, rsp_data.data(), rsp_data.size());
  if (sent != rsp_data.size()) {
    std::cout << "error in mtcp_write: sent size(" << sent << ")"
              << " rsp size(" << rsp_data.size() << ")" << std::endl;
  }
  queryv.erase(sockid);
  return sent;
}

int Server::Process_metadata(int sockid) {
  struct Response rsp;
  uint32_t sent;
  BufferStream bstrm;
  rapidjson::StringBuffer strBuf;
  rapidjson::Writer<rapidjson::StringBuffer> writer(strBuf);
  writer.StartObject();
  writer.Key("name");
  writer.String(minfo.name.c_str());
  writer.Key("versions");
  writer.StartArray();
  std::string sversion = std::move(std::to_string(minfo.version));
  writer.String(sversion.c_str());  // different from triton, ["1"]
  writer.EndArray();
  writer.Key("platform");
  writer.String("tensorrt_plan");
  writer.Key("inputs");
  writer.StartArray();
  for (uint32_t i = 0; i < minfo.minput.size(); i++) {
    writer.StartObject();
    writer.Key("name");
    writer.String(minfo.minput[i].name.data());
    writer.Key("datatype");
    std::string dt =
        std::move(minfo.minput[i].data_type.substr(sizeof("TYPE_") - 1));
    writer.String(dt.data());
    writer.Key("shape");
    writer.StartArray();
    if (minfo.batch_dim == true) {
      // std::cout << "batch dim:" << minfo.batch_dim << std::endl;
    } else
      writer.Int(-1);
    for (uint32_t j = 0; j < minfo.minput[i].dims.size(); j++) {
      writer.Int(minfo.minput[i].dims[j]);
    }
    writer.EndArray();
    writer.EndObject();
  }
  writer.EndArray();
  writer.Key("outputs");
  writer.StartArray();
  for (uint32_t i = 0; i < minfo.moutput.size(); i++) {
    writer.StartObject();
    writer.Key("name");
    writer.String(minfo.moutput[i].name.c_str());
    writer.Key("datatype");
    std::string dt =
        std::move(minfo.moutput[i].data_type.substr(sizeof("TYPE_") - 1));
    writer.String(dt.data());
    writer.Key("shape");
    writer.StartArray();
    if (minfo.batch_dim != true) writer.Int(-1);
    for (uint32_t j = 0; j < minfo.moutput[i].dims.size(); j++) {
      writer.Int(minfo.moutput[i].dims[j]);
    }
    writer.EndArray();
    writer.EndObject();
  }
  writer.EndArray();
  writer.EndObject();

  std::string data = strBuf.GetString();
  std::cout << "meta data: " << data << std::endl;
  char json_len[10];
  sprintf(json_len, "%lu", data.size());
  rsp.set_header("Content-Type", "application/json");
  rsp.body = std::move(data);

  rsp.status = 200;
  write_response(bstrm, rsp);
  auto rsp_data = bstrm.get_buffer();
  auto ctx = ctxs[workerid];
  sent = mtcp_write(ctx->mctx, sockid, rsp_data.data(), rsp_data.size());
  queryv.erase(sockid);
  return sent;
}

std::string Server::cache_rsp_json(int batchsize, std::string id) {
  rapidjson::StringBuffer strBuf;
  rapidjson::Writer<rapidjson::StringBuffer> writer(strBuf);
  int tot_dim = 1;
  unsigned long elesize = 0;
  writer.StartObject();
  writer.Key("id");
  writer.String(id.data());
  writer.Key("model_name");
  writer.String(minfo.name.data());
  writer.Key("model_version");
  std::string sversion = std::move(std::to_string(minfo.version));
  writer.String(sversion.data());
  writer.Key("outputs");
  writer.StartArray();
  for (uint32_t i = 0; i < minfo.moutput.size(); i++) {
    writer.StartObject();
    writer.Key("name");
    writer.String(minfo.moutput[i].name.data());
    writer.Key("datatype");
    std::string dt =
        std::move(minfo.moutput[i].data_type.substr(sizeof("TYPE_") - 1));
    writer.String(dt.data());
    elesize = get_size(minfo.moutput[i].data_type);
    writer.Key("shape");
    writer.StartArray();
    writer.Int(batchsize);
    for (uint32_t j = 1; j < minfo.moutput[i].dims.size(); j++) {
      writer.Int(minfo.moutput[i].dims[j]);
      tot_dim *= minfo.moutput[i].dims[j];
    }
    writer.EndArray();
    writer.Key("parameters");
    writer.StartObject();
    writer.Key("binary_data_size");
    writer.Int(tot_dim * elesize);
    writer.EndObject();
    writer.EndObject();
    tot_dim = 1;
  }
  writer.EndArray();
  writer.EndObject();
  return strBuf.GetString();
}

std::string Server::get_rspjson() { return rsp_json; }

int Server::send_response(struct Response rsp, uint32_t buf_idx,
                          uint32_t offset, uint32_t sockid, std::string id,
                          uint32_t status_code) {
  int sent_num = 0;
  uint32_t i;
  uint32_t sent;
  std::string result_data;
  BufferStream bstrm;

  auto obinding_size = infer->get_obinding_size();
  uint32_t num_inputdim = infer->get_inputdims();

  if (buf_idx != 0xffffffff) {
    void **buf = infer->get_hbuf(buf_idx);
    for (i = 0; i < obinding_size.size(); i++) {
      char const *p = reinterpret_cast<char const *>(buf[i + num_inputdim]) +
                      offset * obinding_size[i];
      result_data.append(p, obinding_size[i]);
    }

  } else {
    uint32_t len = 0;
    for (i = 0; i < obinding_size.size(); i++) {
      len += obinding_size[i];
    }
    result_data.resize(len, '0');
    std::cout << "result for drop req, len: " << len << std::endl;
  }

  auto data = cache_rsp_json(1, id);
  char json_len[10];
  sprintf(json_len, "%lu", data.size());
  rsp.set_header("Inference-Header-Content-Length", json_len);
  data += result_data;
  rsp.body = std::move(data);
  write_response(bstrm, rsp);
  auto rsp_data = bstrm.get_buffer();
  auto ctx = ctxs[workerid];

  sent = mtcp_write(ctx->mctx, sockid, rsp_data.data(), rsp_data.size());

  if (sent != rsp_data.size()) {
    std::cout << "error in mtcp_write: sent size(" << sent << ")"
              << " rsp size(" << rsp_data.size() << ")" << std::endl;
  }
  sent_num++;
  return sent;
}

int Server::AcceptConnection(uint32_t listener) {
  auto ctx = ctxs[workerid];
  mctx_t mctx = ctx->mctx;
  struct mtcp_epoll_event ev;
  int c;

  c = mtcp_accept(mctx, listener, NULL, NULL);

  if (c >= 0) {
    if (c >= MAX_FLOW_NUM) {
      TRACE_ERROR("Invalid socket id %d.\n", c);
      return -1;
    }

    TRACE_APP("New connection %d accepted.\n", c);
    ev.events = MTCP_EPOLLIN;
    ev.data.sockid = c;
    mtcp_setsock_nonblock(ctx->mctx, c);
    mtcp_epoll_ctl(mctx, ctx->ep, MTCP_EPOLL_CTL_ADD, c, &ev);
    TRACE_APP("Socket %d registered.\n", c);
  } else {
    if (errno != EAGAIN) {
      TRACE_ERROR("mtcp_accept() error %s\n", strerror(errno));
    }
  }

  return c;
}
/*----------------------------------------------------------------------------*/
void Server::SignalHandler(int signum) {
  uint32_t i;
  std::cout << "nworker: " << instance.nworker_ << std::endl;
  for (i = 0; i < instance.nworker_; i++) {
    if (instance.app_thread[i] == pthread_self()) {
      instance.done[i] = true;
      std::cout << "thread: " << i << "set done" << std::endl;
    } else {
      std::cout << "thread: " << i << "exit" << std::endl;
      if (!instance.done[i]) pthread_kill(instance.app_thread[i], signum);
    }
  }
}

void Server::parsebatchfile(const std::string filename) {
  std::ifstream infile;
  uint32_t batchsize, last_latency, last_batchsize = 1;
  int latency, count = 0;
  if (filename.empty()) {
    std::cerr << "No batch_latency configuration file" << std::endl;
    return;
  }
  infile.open(filename.c_str());

  while (infile >> batchsize >> latency) {
    while (count < latency) {
      if (count == latency - 1) {
        btable[latency] = batchsize;
        count++;
      } else {
        btable[++count] =
            RTE_MAX(floor((last_batchsize + batchsize) / 2), btable[count]);
        last_batchsize = btable[count - 1];
        last_latency = latency;
      }
    }

    btable[count] = batchsize;
    last_batchsize = batchsize;
    last_latency = latency;
  }
}

void Server::parsejsonfile(const std::string filename,
                           struct model_info &minfo) {
  if (filename.empty()) {
    std::cerr << "No json configuration file" << std::endl;
    return;
  }

  FILE *fp = fopen(filename.c_str(), "rb");
  if (fp == NULL) {
    std::cerr << "Can't open json configuration file" << std::endl;
    return;
  }

  char tmpBuf[MAX_JSON_CONTENT_SIZE];
  rapidjson::FileReadStream rfstream(fp, tmpBuf, sizeof(tmpBuf));
  rapidjson::Document doc;
  doc.Parse(tmpBuf);
  fclose(fp);

  if (doc.HasParseError()) {
    rapidjson::ParseErrorCode code = doc.GetParseError();
    std::cerr << "Parse json configuration error: " << GetParseError_En(code)
              << " offset: " << doc.GetErrorOffset() << std::endl;
    return;
  }

  if (doc.HasMember("batch_dim"))
    minfo.batch_dim = doc["batch_dim"].GetBool();
  else
    minfo.batch_dim = false;
  if (doc.HasMember("name")) minfo.name = doc["name"].GetString();
  if (doc.HasMember("path")) minfo.path = doc["path"].GetString();
  if (doc.HasMember("version")) minfo.version = doc["version"].GetInt();

  if (doc.HasMember("max_batch_size"))
    minfo.max_batch_size = doc["max_batch_size"].GetInt();
  else
    minfo.max_batch_size = 0;

  rapidjson::Value &inputs = doc["inputs"];
  if (inputs.IsArray()) {
    struct model_io tmp_input;
    for (uint32_t i = 0; i < inputs.Capacity(); i++) {
      rapidjson::Value &val = inputs[i];
      tmp_input.name = val["name"].GetString();
      tmp_input.data_type = val["data_type"].GetString();
      for (uint32_t j = 0; j < val["dims"].Capacity(); j++)
        tmp_input.dims.emplace_back(val["dims"][j].GetInt());
      minfo.minput.emplace_back(std::move(tmp_input));
    }
  }

  rapidjson::Value &outputs = doc["outputs"];
  if (outputs.IsArray()) {
    struct model_io tmp_output;
    for (uint32_t i = 0; i < outputs.Capacity(); i++) {
      rapidjson::Value &val = outputs[i];
      tmp_output.name = val["name"].GetString();
      tmp_output.data_type = val["data_type"].GetString();
      for (uint32_t j = 0; j < val["dims"].Capacity(); j++)
        tmp_output.dims.emplace_back(val["dims"][j].GetInt());
      minfo.moutput.emplace_back(std::move(tmp_output));
    }
  }
}

uint32_t Server::sloslack_batch() {
  struct req_message req = queue.top();
  uint64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now().time_since_epoch())
                     .count();
  int slack = get_sla() - (now - req.recv_start_time) / 1000000;
  if (slack > 0)
    return RTE_MAX(btable.at(slack), 1);
  else
    return 0;
}

uint32_t Server::adaptive_adjust(uint32_t batch) {
  uint32_t actual_batch = 0;
  if (ongoing_batch) {
    if (mini_batchlimit > batch + ongoing_batch) {
      actual_batch = batch;
    } else if (mini_batchlimit > ongoing_batch) {
      actual_batch = RTE_MAX(mini_batchlimit - ongoing_batch, (uint32_t)0);
    } else
      actual_batch = 0;
  } else
    actual_batch = batch;
  return actual_batch;
}

void Server::AppLogic(void *arg) { instance.AppLogicLocal(arg); }

void Server::AppLogicLocal(void *arg) {
  struct thread_context *ctx = reinterpret_cast<struct thread_context *>(arg);
  struct mtcp_epoll_event *events;
  int nevents;
  int do_accept;
  int ret;
  events = ctx->events;
  int i;
  struct rsp_message rsp;

  nevents = mtcp_epoll_wait(ctx->mctx, ctx->ep, events, MAX_EVENTS, -1);

  if (nevents < 0) {
    if (errno != EINTR) {
      return;
    }
  }

  do_accept = false;
  for (i = 0; i < nevents; i++) {
    if (events[i].data.sockid == ctx->listener) {
      /* if the event is for the listener, accept connection */
      do_accept = true;
    } else if (events[i].events & MTCP_EPOLLERR) {
      int err;
      socklen_t len = sizeof(err);

      if (mtcp_getsockopt(ctx->mctx, events[i].data.sockid, SOL_SOCKET,
                          SO_ERROR, (void *)&err, &len) == 0) {
        if (err != ETIMEDOUT) {
          fprintf(stderr, "Error on socket %d: %s\n", events[i].data.sockid,
                  strerror(err));
        }
      } else {
        perror("mtcp_getsockopt");
      }

      CloseConnection(events[i].data.sockid);

    } else if (events[i].events & MTCP_EPOLLIN) {
      ret = ReqProcess(events[i].data.sockid, 1);
      if (ret == 0) {
        CloseConnection(events[i].data.sockid);
      } else if (ret < 0) {
        /* if not EAGAIN, it's an error */
        if (errno != EAGAIN) {
          CloseConnection(events[i].data.sockid);
        }
      }
    } else if (events[i].events & MTCP_EPOLLOUT) {
      printf("MTCP_EPOLLOUT\n");
    } else {
      assert(0 && "unknown event");
    }
  }

  while (worker_lfq[workerid]->try_dequeue(rsp)) {
#if 0
    auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
                   .count();
    std::cout << "worker: " << rsp.workerid
              << " recv one rsp, batch size: " << rsp.batchsize
              << " recv: " << rsp.recv_latency/1000000.0
              << " queue: " << rsp.queue_latency/1000000.0
              << " reqtrans: " << rsp.reqtrans_latency/1000000.0
              << " infer: " << rsp.inference_latency/1000000.0
              << " rsptrans: " << rsp.rsptrans_latency/1000000.0
              << " rsprecv: " << now - rsp.fin_time/1000000.0 << std::endl;
#endif
    RspProcess(rsp);
  }

  /* if do_accept flag is set, accept connections */
  if (do_accept) {
    while (1) {
      ret = AcceptConnection(ctx->listener);
      if (ret < 0) {
        break;
      }
    }
  }
}

void Server::DispLogic(void *arg) { instance.DispLogicLocal(arg); }

bool Server::dequeue_req(uint32_t num, std::vector<struct req_message> &reqs) {
  for (uint32_t i = 0; i < num; i++) {
    reqs.emplace_back(std::move(queue.top()));
    queue.pop();
  }
  return true;
}

void Server::DispLogicLocal(void *arg) {
  uint32_t bsize = 0;
  uint32_t i;
  bool tf;
  struct req_message req;

  int count = 0;
  while (dis_lfq->try_dequeue(req)) {
    struct req_message task = req;
    count++;
    queue.push(task);
  }

#ifdef SLOSLACK
  if (infer->get_freebuf_size() && queue.size()) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    uint32_t qss_batch = 0;
    uint32_t ss_batch = sloslack_batch();

    if (ss_batch > 0) {
      qss_batch = RTE_MIN(ss_batch, queue.size());
      bsize = RTE_MIN(adaptive_adjust(qss_batch), (uint32_t)MAX_BATCH);
    }

    if (bsize && ss_batch) {
      auto ibinding_size = infer->get_ibinding_size();

      batch_info binfo;
      batch_message bmesg;
      binfo.batch_idx = gen_batchidx();
      bmesg.batch_idx = binfo.batch_idx;
      bmesg.batch_size = bsize;
      bmesg.batch_limit = ss_batch;

      if (ss_batch < mini_batchlimit) {
        if (ss_batch == 1) {
          uint64_t now =
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now().time_since_epoch())
                  .count();
          struct req_message req = queue.top();

          bmesg.batch_limit = 9999;  // for evaluate tai latency with
                                     // perf_client
        } else {
          mini_batchlimit = ss_batch;
          mini_batchidx = bmesg.batch_idx;
        }
      }

      ongoing_batch += bsize;
      batch_queue.push(bmesg);
      batchmap.insert(std::make_pair(bmesg.batch_idx, bmesg));

      tf = dequeue_req(bsize, binfo.einfo);
      if (tf) {
        uint32_t buf_idx = infer->pop_freebuf();
        binfo.batch_size = bsize;
        binfo.buf_idx = buf_idx;
        void **hbuf = infer->get_hbuf(buf_idx);
        uint32_t *ptail = infer->get_ptail();
        uint32_t offset = 0;
        auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now().time_since_epoch())
                       .count();
        binfo.queue_time = now;

        for (auto &&iter : binfo.einfo) {
          offset = 0;
          iter.queue_latency = now - iter.recv_fin_time;
          iter.queue_time = now;
          for (i = 0; i < ibinding_size.size(); i++) {
            memcpy(reinterpret_cast<char *>(hbuf[i]) + ptail[i],
                   iter.req_addr + offset, ibinding_size[i]);
            offset += ibinding_size[i];
            ptail[i] += ibinding_size[i];
          }
        }

        for (i = 0; i < ibinding_size.size(); i++) ptail[i] = 0;
        infer->CopyreqAsync(buf_idx, bsize);
        binfo.event = infer->ctx[buf_idx].event;
        cbuf.req_list.push_back(binfo);
      } else
        std::cout << "failed to get a batch of socketid" << std::endl;
    } else if (!ss_batch) {
      struct req_message req = queue.top();
      uint64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count();
      int slack = get_sla() - (now - req.recv_start_time) / 1000000;
      req.missed_enqueue_time = now;
      req.queue_latency = now - req.recv_fin_time;
      missed_reqs.push(req);
      queue.pop();
    }
  }

#endif

  if (!cbuf.req_list.empty()) {
    for (auto iter = cbuf.req_list.begin(); iter != cbuf.req_list.end();) {
      if (cudaEventQuery(iter->event) == cudaSuccess) {
        iter->reqtrans_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count();

        infer->DoInference(DEVICE, iter->buf_idx, iter->batch_size);
        auto elem = *iter;
        cbuf.inf_list.push_back(elem);
        iter = cbuf.req_list.erase(iter);
      } else
        iter++;
    }
  }

  if (!cbuf.inf_list.empty()) {
    for (auto iter = cbuf.inf_list.begin(); iter != cbuf.inf_list.end();) {
      if (cudaEventQuery(iter->event) == cudaSuccess) {
        iter->infer_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count();
        infer->CopyrspAsync(iter->buf_idx, iter->batch_size);
        batch_info elem(*iter);
        cbuf.rsp_list.push_back(elem);
        iter = cbuf.inf_list.erase(iter);
      } else
        iter++;
    }
  }

  if (!cbuf.rsp_list.empty()) {
    for (auto iter = cbuf.rsp_list.begin(); iter != cbuf.rsp_list.end();) {
      if (cudaEventQuery(iter->event) == cudaSuccess) {
        iter->rsptrans_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count();
        void **hbuf = infer->get_hbuf(iter->buf_idx);
        for (uint32_t i = 0; i < iter->batch_size; i++) {
          struct rsp_message rsp;
          rsp.workerid = (iter->einfo[i]).workerid;
          rsp.sockid = (iter->einfo[i]).sockid;
          rsp.batchid = iter->buf_idx;
          rsp.batchsize = iter->batch_size;
          rsp.offset = i;
          rsp.rsp_addr = (char *)hbuf[0];
          rsp.recv_latency = (iter->einfo[i]).recv_latency;
          rsp.reqtrans_latency = iter->reqtrans_time - iter->queue_time;
          rsp.inference_latency = iter->infer_time - iter->reqtrans_time;
          rsp.rsptrans_latency = iter->rsptrans_time - iter->infer_time;
          rsp.queue_latency = iter->einfo[i].queue_latency;
          rsp.fin_time = iter->rsptrans_time;

          worker_lfq[(iter->einfo[i]).workerid]->enqueue(rsp);
        }

#ifdef SLOSLACK

        ongoing_batch -= iter->batch_size;
        auto message = batchmap[iter->batch_idx];

        batchmap.erase(iter->batch_idx);
        if (!batch_queue.remove(message))
          std::cout << "remove message from queue failed" << std::endl;

        if (mini_batchidx == iter->batch_idx && ongoing_batch) {
          mini_batchidx = batch_queue.top().batch_idx;
          mini_batchlimit = batch_queue.top().batch_limit;
        } else if (ongoing_batch == 0) {
          mini_batchlimit = 0xffffffff;
        }

#endif

        infer->push_freebuf(iter->buf_idx);
        iter = cbuf.rsp_list.erase(iter);

      } else
        iter++;
    }
  }

  if (missed_reqs.size()) {
    struct req_message req = missed_reqs.top();
    if (1) {
      struct rsp_message rsp;
      missed_reqs.pop();
      rsp.workerid = req.workerid;
      rsp.sockid = req.sockid;
      rsp.batchid = 0xffffffff;
      rsp.batchsize = 0;
      rsp.offset = 0;
      // rsp.rsp_addr = (char *)hbuf[0];
      rsp.recv_latency = req.recv_latency;
      rsp.reqtrans_latency = 0;
      rsp.inference_latency = 0;
      rsp.rsptrans_latency = 0;
      rsp.queue_latency = req.queue_latency;
      rsp.fin_time = 0;
      worker_lfq[req.workerid]->enqueue(rsp);
    }
  }
}

int Server::InitServer(uint32_t numworker, uint32_t thoffset, uint32_t backlog,
                       uint32_t port, uint32_t max_batch,
                       const std::string &mtcp_config_file,
                       const std::string &model_config_file,
                       const std::string &batch_config_file, uint32_t sla) {
  cudaSetDevice(0);
  struct cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, 0);

  int ret;
  uint32_t i;
  nworker_ = numworker;
  offset_ = thoffset;
  backlog_ = backlog;
  port_ = port;
  sla_ = sla;
  batch_idx = 0;
  ongoing_batch = 0;
  mini_batchlimit = 0xffffffff;
  mini_batchidx = 0;

  saddr.sin_family = AF_INET;
  saddr.sin_addr.s_addr = INADDR_ANY;
  saddr.sin_port = htons(port);

  for (i = 0; i < numworker; i++) {
    done[i] = false;
  }

  if (mtcp_config_file.empty()) {
    std::cerr << "No mTCP config_file!" << std::endl;
    exit(EXIT_FAILURE);
  }

  ret = mtcp_init(mtcp_config_file.c_str());
  if (ret) {
    std::cerr << "Failed to initialize mtcp!\n" << std::endl;
    exit(EXIT_FAILURE);
  }

  mtcp_getconf(&mcfg);

  if (numworker != (uint32_t)mcfg.num_cores) {
    std::cerr << "mcfg.num_cores: " << mcfg.num_cores << " != nworker "
              << numworker << std::endl;
    exit(EXIT_FAILURE);
  }

  if (backlog_ > (uint32_t)mcfg.max_concurrency) {
    std::cerr << "backlog " << backlog_
              << "can not be set larger than CONFIG.max_concurrency "
              << mcfg.max_concurrency << "!" << std::endl;
    backlog_ = mcfg.max_concurrency;
    std::cerr << "Set backlog to " << mcfg.max_concurrency << std::endl;
  }

  eng = std::mt19937(rd());
  distr = std::uniform_int_distribution<uint32_t>(0, 31);

  parsejsonfile(model_config_file, minfo);
  parsebatchfile(batch_config_file);

  infer = std::make_shared<Inference>();
  infer->Init(minfo.path);

  struct model_info tinfo = get_modelinfo();
  infer->InitInputOutput(tinfo);
  actual_instance = infer->InitBuffer();
  cbuf.stime = std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
                   .count();
  ibinding_size = infer->get_ibinding_size();
  obinding_size = infer->get_obinding_size();

  dis_lfq = std::make_shared<moodycamel::ConcurrentQueue<struct req_message>>();
  for (uint32_t i = 0; i < nworker_; i++) {
    std::shared_ptr<moodycamel::ConcurrentQueue<struct rsp_message>> lfq =
        std::make_shared<moodycamel::ConcurrentQueue<struct rsp_message>>();
    worker_lfq.emplace_back(lfq);
  }

  Warmup();

  pthread_spin_init(&batchlock, PTHREAD_PROCESS_PRIVATE);
  register_appfunc(DispLogic, true);
  register_appfunc(AppLogic, false);
  return 0;
}

void Server::Warmup() {
  uint32_t i;
  uint32_t finish = 0;
  for (i = 0; i < actual_instance; i++) {
    uint32_t buf_idx = infer->pop_freebuf();
    batch_info binfo;
    binfo.batch_idx = gen_batchidx();
    binfo.buf_idx = buf_idx;
    binfo.batch_size = 1;
    infer->CopyreqAsync(i, 1);
    binfo.event = infer->ctx[buf_idx].event;
    cbuf.req_list.push_back(binfo);
  }

  while (finish < actual_instance) {
    if (!cbuf.req_list.empty()) {
      for (auto iter = cbuf.req_list.begin(); iter != cbuf.req_list.end();) {
        if (cudaEventQuery(iter->event) == cudaSuccess) {
          infer->DoInference(DEVICE, iter->buf_idx, iter->batch_size);
          auto elem = *iter;
          cbuf.inf_list.push_back(elem);
          iter = cbuf.req_list.erase(iter);
          finish++;
        } else
          iter++;
      }
    }
  }

  finish = 0;
  while (finish < actual_instance) {
    if (!cbuf.inf_list.empty()) {
      for (auto iter = cbuf.inf_list.begin(); iter != cbuf.inf_list.end();) {
        if (cudaEventQuery(iter->event) == cudaSuccess) {
          infer->CopyrspAsync(iter->buf_idx, iter->batch_size);
          batch_info elem(*iter);
          cbuf.rsp_list.push_back(elem);
          iter = cbuf.inf_list.erase(iter);
          finish++;
        } else
          iter++;
      }
    }
  }

  finish = 0;
  while (finish < actual_instance) {
    for (auto iter = cbuf.rsp_list.begin(); iter != cbuf.rsp_list.end();) {
      if (cudaEventQuery(iter->event) == cudaSuccess) {
        infer->push_freebuf(iter->buf_idx);
        iter = cbuf.rsp_list.erase(iter);
        finish++;
      }
    }
  }
  std::cout << "Complete Warmup" << std::endl;
}

int Server::get_thoffset() { return offset_; }

struct model_info Server::get_modelinfo() {
  return minfo;
}

int Server::get_numworker() { return nworker_; }

int Server::ThreadRun(void *core) {
  instance.ThreadRunLocal(core);
  return 0;
}

int Server::ThreadRunLocal(void *core) {
  struct mtcp_epoll_event ev;
  int ret;
  uint32_t cpu;
  cpu_set_t mask;
  CPU_ZERO(&mask);

  workerid = *(reinterpret_cast<int *>(core));
  app_thread[workerid] = pthread_self();

  cpu = workerid + get_thoffset();

  CPU_SET(cpu, &mask);
  if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
    std::cerr << "Cannot set cpu affinity for thread: " << workerid
              << std::endl;
  }

  auto ctx = static_cast<struct thread_context *>(
      calloc(1, sizeof(struct thread_context)));

  ctx->mctx = mtcp_create_context(workerid);
  assert(ctx->mctx && "ctx->mctx is null");

  ctx->ep = mtcp_epoll_create(ctx->mctx, MAX_EVENTS);
  ctx->listener = mtcp_socket(ctx->mctx, AF_INET, SOCK_STREAM, 0);
  assert(ctx->listener > 0 && "invalid listener");
  ret = mtcp_setsock_nonblock(ctx->mctx, ctx->listener);
  assert(!ret && "mtcp_setsock_nonblock error");
  ret = mtcp_bind(ctx->mctx, ctx->listener, (struct sockaddr *)&saddr,
                  sizeof(struct sockaddr_in));
  assert(!ret && "mtcp_bind err");

  ret = mtcp_listen(ctx->mctx, ctx->listener, backlog_);
  assert(!ret && "mtcp_listen err");

  ev.events = MTCP_EPOLLIN;
  ev.data.sockid = ctx->listener;
  mtcp_epoll_ctl(ctx->mctx, ctx->ep, MTCP_EPOLL_CTL_ADD, ctx->listener, &ev);
  ctxs[workerid] = ctx;
  ctx->events = (struct mtcp_epoll_event *)calloc(
      MAX_EVENTS, sizeof(struct mtcp_epoll_event));
  if (workerid) {
    register_thread_initialized();
    RunWorkerLoop(reinterpret_cast<void *>(ctx));
  } else {
    pthread_mutex_lock(&init_lock);
    wait_for_thread_registration(get_numworker() - 1);
    pthread_mutex_unlock(&init_lock);
    fprintf(stdout, "all %d workerthreads are ready\n", (get_numworker()));
    fflush(stdout);
    RunMasterLoop(reinterpret_cast<void *>(ctx));
  }

  return 0;
}

void Server::StartWorker() {
  int worker_id = 1;
  int lcore_id;
  int master = 0;

  RTE_LCORE_FOREACH_SLAVE(lcore_id) {
    int tmp = worker_id;
    rte_eal_remote_launch(ThreadRun, (void *)&tmp, lcore_id);
    done[worker_id] = false;
    worker_id++;
    sleep(0.1);
  }

  done[master] = false;
  ThreadRun((void *)&master);
}
