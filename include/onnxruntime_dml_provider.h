typedef struct OrtDmlDeviceOptions OrtDmlDeviceOptions;

struct OrtDmlApi {
  OrtStatus*(* SessionOptionsAppendExecutionProvider_DML)(OrtSessionOptions* options, int device_id);
  OrtStatus*(* SessionOptionsAppendExecutionProvider_DML2)(OrtSessionOptions* options, OrtDmlDeviceOptions* device_opts);
};

typedef struct OrtDmlApi OrtDmlApi;
