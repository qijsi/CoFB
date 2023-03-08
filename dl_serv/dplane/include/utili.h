#ifndef __UTILI_H_
#define __UTILI_H_
#include <string>

typedef enum {
  TYPE_INVALID,
  TYPE_BOOL,
  TYPE_UINT8,
  TYPE_UINT16,
  TYPE_UINT32,
  TYPE_UINT64,
  TYPE_INT8,
  TYPE_INT16,
  TYPE_INT32,
  TYPE_INT64,
  TYPE_FP16,
  TYPE_FP32,
  TYPE_FP64
} DATA_TYPE;

unsigned long get_size(std::string itype); 
#endif