#include <unistd.h>
#include <cstring>
#include <iostream>
#include "dplane.h"

extern "C" {
#include "cpu.h"
#include "netlib.h"
}

#define HTTP_HEADER_LEN 1024
#define URL_LEN 128

int main(int argc, char** argv) {
  int worker_id = 1;
  int ret;
  struct mtcp_conf mcfg;
  int process_cpu;
  int i, o;
  int master = 0;
  int lcore_id;

  int num_cores = 11;
  int core_limit = num_cores;
  process_cpu = -1;
  int max_batch;
  std::string io_config_file;
  std::string model_config_file;
  std::string batch_config_file;
  int backlog;
  int offset;
  uint32_t sla;
  uint32_t max_batch_size;

  while (-1 != (o = getopt(argc, argv, "m:N:f:g:B:e:c:b:l:h"))) {
    switch (o) {
      case 'm':
        model_config_file = std::string(optarg);
        break;
      case 'N':
        core_limit = mystrtol(optarg, 10);
        if (core_limit > num_cores) {
          std::cerr << "CPU limit should be smaller than the number of CPUs: "
                    << num_cores << std::endl;
          return FALSE;
        }
        mtcp_getconf(&mcfg);
        mcfg.num_cores = core_limit;
        mtcp_setconf(&mcfg);
        break;
      case 'B':
        max_batch = mystrtol(optarg, 10);
        if (max_batch < 0) {
          std::cerr << "Batch size cannot be negative" << std::endl;
          ;
          return FALSE;
        } else if (max_batch > BATCH_LIMIT) {
          std::cerr << "Batch size should smaller than " << BATCH_LIMIT
                    << std::endl;
        }
        break;
      case 'f':
        io_config_file = std::string(optarg);
        break;
      case 'g':
        batch_config_file = std::string(optarg);
        break;
      case 'e':
        offset = mystrtol(optarg, 10);
        break;
      case 'c':
        process_cpu = mystrtol(optarg, 10);
        if (process_cpu > core_limit) {
          std::cerr << "Starting CPU is way off limits!" << std::endl;
          return FALSE;
        }
        break;
      case 'b':
        backlog = mystrtol(optarg, 10);
        break;

      case 'l':
        sla = mystrtol(optarg, 10);  // milliseond

      case 'h':
        std::cout << argv[0]
                  << " -p <path_to_model> -f <mtcp_conf_file> "
                     "[-N num_cores] [-e coreoffset] [-B max_batch_size] [-l "
                     "latency_threshold] [-c <per-process core_id>] [-h]"
                  << std::endl;
        break;
    }
  }

  Server& iServer = Server::getInstance();

  iServer.InitServer(core_limit, offset, backlog, 8080, max_batch, 
                     io_config_file, model_config_file, batch_config_file, sla);
  iServer.StartWorker();
  return 0;
}