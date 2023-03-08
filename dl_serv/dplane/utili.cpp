#include "utili.h"

unsigned long get_size(std::string itype) {
  if (itype == "TYPE_BOOL" || itype == "BOOL")
    return sizeof(bool);
  else if (itype == "TYPE_UINT8" || itype == "UINT8")
    return sizeof(uint8_t);
  else if (itype == "TYPE_UINT16" || itype == "UINT16")
    return sizeof(uint16_t);
  else if (itype == "TYPE_UINT32" || itype == "UINT32")
    return sizeof(uint32_t);
  else if (itype == "TYPE_UINT64" || itype == "UINT64")
    return sizeof(uint64_t);
  else if (itype == "TYPE_INT8" || itype == "INT8")
    return sizeof(int8_t);
  else if (itype == "TYPE_INT16" || itype == "INT16")
    return sizeof(int16_t);
  else if (itype == "TYPE_INT32" || itype == "INT32")
    return sizeof(int32_t);
  else if (itype == "TYPE_INT64" || itype == "INT64")
    return sizeof(int64_t);
  else if (itype == "TYPE_FP16" || itype == "FP16")
    return 2;
  else if (itype == "TYPE_FP32" || itype == "FP32")
    return 4;
  else if (itype == "TYPE_FP64" || itype == "FP64")
    return 8;

  return 0;
}