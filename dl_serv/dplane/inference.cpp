#include "inference.h"

#include <assert.h>
#include <unistd.h>

#include <fstream>

#include "logging.h"

static Logger gLogger;

thread_local context_buf cbuf;

std::vector<uint32_t> Inference::get_ibinding_size() { return ibinding_size; }
std::vector<uint32_t> Inference::get_obinding_size() { return obinding_size; }

std::shared_ptr<trt_t> Inference::get_trtengine() {
  if (trtEngine_ == nullptr) trtEngine_ = std::make_shared<trt_t>();
  return trtEngine_;
}

std::vector<nvinfer1::Dims> Inference::get_idims_vector() {
  return idims_vector;
}

void Inference::set_idims_vector(std::vector<nvinfer1::Dims> idims) {
  idims_vector = idims;
}

void Inference::SetTrtEngine(std::shared_ptr<trt_t> trtVar,
                             const std::string &model_file) {
  char *trtModelStream{nullptr};
  std::ifstream file(model_file, std::ios::binary);
  if (file.good()) {
    file.seekg(0, file.end);
    int size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    file.read(trtModelStream, size);
    file.close();
    trtVar->runtime = nvinfer1::createInferRuntime(gLogger);
    assert(trtVar->runtime != nullptr && "trt runtime is null");
    trtVar->engine =
        trtVar->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(trtVar->engine != nullptr && " trt engine is null");
    delete[] trtModelStream;
  } else {
    std::cerr << "Cannot read model file" << std::endl;
    exit(1);
  }
}

void Inference::CopyreqAsync(uint32_t buf_idx, uint32_t batch_size) {
  uint32_t i;
  void **hbuf = get_hbuf(buf_idx);
  void **dbuf = get_dbuf(buf_idx);

  auto ibinding_size = get_ibinding_size();

  for (i = 0; i < ibinding_size.size(); ++i) {
    CHECK(cudaMemcpyAsync(dbuf[i], hbuf[i], batch_size * ibinding_size[i],
                          cudaMemcpyHostToDevice, ctx[buf_idx].stream));
  }
  cudaEventRecord(ctx[buf_idx].event, ctx[buf_idx].stream);
}

void Inference::CopyrspAsync(uint32_t buf_idx, uint32_t batch_size) {
  uint32_t i;
  auto obinding_size = get_obinding_size();
  void **hbuf = get_hbuf(buf_idx);
  void **dbuf = get_dbuf(buf_idx);
  for (i = 0; i < obinding_size.size(); ++i) {
    CHECK(cudaMemcpyAsync(hbuf[i + num_inputbinding],
                          dbuf[i + num_inputbinding],
                          batch_size * obinding_size[i], cudaMemcpyDeviceToHost,
                          ctx[buf_idx].stream));
  }
  cudaEventRecord(ctx[buf_idx].event, ctx[buf_idx].stream);
}

uint32_t Inference::InitBuffer() {
  uint32_t i, j, k;
  num_inputbinding = 0;
  auto trtVar = get_trtengine();
  uint64_t mem_size = 0;
  uint32_t actual_instance;

  size_t avail, total, used;

  for (i = 0; i < CMP_STREAM_SIZE; i++) {
    ctx[i].context = trtVar->engine->createExecutionContext();
    mem_size += trtVar->engine->getDeviceMemorySize();
    CHECK(cudaStreamCreateWithFlags(&ctx[i].stream, cudaStreamNonBlocking));
    cudaEventCreateWithFlags(&ctx[i].event, cudaEventDisableTiming);

    hbuf[i] = (void **)malloc(sizeof(void *) *
                              (ibinding_size.size() + obinding_size.size()));
    dbuf[i] = (void **)malloc(sizeof(void *) *
                              (ibinding_size.size() + obinding_size.size()));

    mem_size += 2 * sizeof(hbuf[i]);

    for (j = 0; j < ibinding_size.size(); j++) {
      void *tmp;
      void *dtmp;
      cudaMallocHost(&tmp, MAX_BATCH * ibinding_size[j]);
      hbuf[i][j] = std::move(tmp);
      CHECK(cudaMalloc(&dtmp, MAX_BATCH * ibinding_size[j]));
      dbuf[i][j] = std::move(dtmp);
      if (i == 0) num_inputbinding++;

      mem_size += 2 * MAX_BATCH * sizeof(ibinding_size[j]);
    }

    for (j = 0; j < obinding_size.size(); j++) {
      void *tmp;
      void *dtmp;
      cudaMallocHost(&tmp, MAX_BATCH * obinding_size[j]);
      hbuf[i][j + num_inputbinding] = std::move(tmp);
      CHECK(cudaMalloc(&dtmp, MAX_BATCH * obinding_size[j]));
      dbuf[i][j + num_inputbinding] = std::move(dtmp);

      mem_size += 2 * MAX_BATCH * sizeof(obinding_size[j]);
    }

    free_buf.push(i);
    cudaMemGetInfo(&avail, &total);
    used = total - avail;
    if (used > 0.8 * total) {
      std::cout << "Generate " << i + 1 << " contexts, due to GPU memory limit"
                << std::endl;
      break;
    }
  }

  actual_instance = i;

  nvinfer1::Dims idims, idims2;

  for (i = 0; i < mio.minput.size(); i++) {
    idims.nbDims = mio.batch_dim ? mio.minput[i].dims.size()
                                 : mio.minput[i].dims.size() + 1;

    for (j = 0; j < mio.minput.size(); j++) {
      idims.d[0] = 1;
      k = mio.batch_dim ? 1 : 0;
      for (; k < mio.minput[j].dims.size(); k++) {
        idims.d[k] = mio.minput[j].dims[k];
      }

      idims2 = ctx[i].context->getBindingDimensions(j);
      ctx[i].context->setBindingDimensions(j, idims);
      if (i == 0) idims_vector.emplace_back(idims);
    }
  }

  ptail = (uint32_t *)calloc(ibinding_size.size(), sizeof(uint32_t));
  return actual_instance;
}

void Inference::DoInference(int device, uint32_t buf_idx, uint32_t batch_size) {
  uint32_t i;
  std::vector<nvinfer1::Dims> dims = get_idims_vector();

  for (i = 0; i < dims.size(); i++) {
    dims[i].d[0] = batch_size;
    ctx[buf_idx].context->setBindingDimensions(i, dims[i]);
  }

  if (!ctx[buf_idx].context->allInputDimensionsSpecified()) {
    std::cerr << "Somme input dimension is not specified" << std::endl;
  }

  void **hbuf = get_hbuf(buf_idx);
  ctx[buf_idx].context->enqueueV2(hbuf, ctx[buf_idx].stream, nullptr);
  cudaEventRecord(ctx[buf_idx].event, ctx[buf_idx].stream);
}

Inference::~Inference() {}

void Inference::Init(const std::string &model) {
  auto trtEngine_ = get_trtengine();
  SetTrtEngine(trtEngine_, model);
}

void Inference::InitInputOutput(struct model_info minfo) {
  uint32_t tot_dim = 1;
  uint32_t elesize = 0;
  uint32_t i;
  for (auto &min : minfo.minput) {
    elesize = get_size(min.data_type);
    for (i = 0; i < min.dims.size(); i++) {
      if (min.dims[i] == -1) min.dims[i] = 1;
      tot_dim *= min.dims[i];
    }
    ibinding_size.emplace_back(std::move(tot_dim * elesize));
    tot_dim = 1;
  }

  for (auto &mout : minfo.moutput) {
    elesize = get_size(mout.data_type);
    for (i = 0; i < mout.dims.size(); i++) {
      if (mout.dims[i] == -1) mout.dims[i] = 1;
      tot_dim *= mout.dims[i];
    }
    obinding_size.emplace_back(std::move(tot_dim * elesize));
    tot_dim = 1;
  }
  mio = minfo;
}