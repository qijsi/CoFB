#ifndef _MODEL_H_
#define _MODEL_H_

#include <vector>
#include <string>

struct model_io {
    std::string name;
    std::string data_type;
    std::vector<int> dims;
};

struct model_info {
  bool batch_dim;
  std::string name;
  std::string path;
  int version;
  int max_batch_size;
  std::vector<struct model_io> minput;
  std::vector<struct model_io> moutput;
};
#endif
