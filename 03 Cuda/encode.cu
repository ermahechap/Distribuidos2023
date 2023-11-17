#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <sndfile.h>
#include <complex.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define MAX_BLOCKS 2147483647
#define MAX_THREADS 1024

#define CALCULATE_BLOCKS(nThreads, N) (N + nThreads - 1) / nThreads

#ifdef _CUFFT_H_
// cuFFT API errors
static const char *_cudaGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
    }

    return "<unknown>";
}
#endif

#define CHECK_CUFFT_ERRORS(call) { \
    cufftResult_t err; \
    if ((err = (call)) != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error %d:%s at %s:%d\n", err, _cudaGetErrorEnum(err), \
                __FILE__, __LINE__); \
        exit(1); \
    } \
}

char *stringToBinary(char* s) {
  if(s == NULL) return (char *)"/n"; /* no input string */
  int len = strlen(s);
  char *binary = (char *)malloc((len*8 + 1)); // each char is one byte (8 bits) and + 1 at the end for null terminator
  
  int binaryIndex = 0;
  for(int i = 0; i < len; ++i) {
    char currentChar = s[i];
    for(int j = 7; j >= 0; j--){
      binary[binaryIndex++] = ((currentChar >> j) & 1) ? '1' : '0';
    }
  }
  binary[binaryIndex] = '\0';
  return binary;
}

char* binaryToString(char *binary) {
  int binaryLength = strlen(binary);
  int stringLength = binaryLength / 8;
  char* originalString = (char *)malloc(stringLength + 1);

  int binaryIndex = 0;
  for (int i = 0; i < stringLength; i++) {
    char currentChar = 0;
    for (int j = 0; j < 8; j++) {
      currentChar = (currentChar << 1) | (binary[binaryIndex++] - '0');
    }
    originalString[i] = currentChar;
  }
  originalString[stringLength] = '\0';
  return originalString;
}

void readWav(char *path, float **readDataPtr, SF_INFO* sfInfoPtr, SNDFILE *sfFile, int verbosity) {
  if (verbosity) printf("Reading File: %s\n", path);

  sfFile = sf_open(path, SFM_READ, sfInfoPtr);
  float *buffer = (float *)malloc(sfInfoPtr->channels * sfInfoPtr->frames * sizeof(float));
  *readDataPtr = (float *) malloc(sfInfoPtr->frames * sizeof(float));

  sf_readf_float(sfFile, buffer, sfInfoPtr->channels * sfInfoPtr->frames);

  if (sfInfoPtr->channels > 1) {
    if (verbosity) printf("WARNING - More than one channel, taking first channel");
    // #pragma omp for
    for (int i = 0; i < sfInfoPtr->frames; i++) {
      (*readDataPtr)[i] = buffer[i * sfInfoPtr->channels];
    }
  } else {
    memcpy(*readDataPtr, buffer, sfInfoPtr->frames * sizeof(float));
  }
  free(buffer);
  
  sf_close(sfFile);
  if (verbosity) {
    printf("File Read... DONE!\n");
    printf("File Info:\n");
    printf("Sample Rate: %d Hz\n", sfInfoPtr->samplerate);
    printf("Frames: %d\n", sfInfoPtr->frames);
    printf("Channels: %d\n", sfInfoPtr->channels);
  }
}

void writeWav(char *path, float *data, int samplerate, int N, int verbosity) {
  if (verbosity) printf("Writing File: %s\n", path);
  SNDFILE *outFile; SF_INFO outInfo;
  outInfo.samplerate = samplerate;
  outInfo.channels = 1;
  outInfo.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;

  outFile = sf_open(path, SFM_WRITE, &outInfo);
  sf_writef_float(outFile, data, N);
  sf_close(outFile);
  if (verbosity) printf("File Writing... DONE!\n");
}

int parseLenFromFilename (char *str) {
  char *lastSlash = strrchr(str, '/');
  char *lastDot = strrchr(str, '.');
  if (lastSlash == NULL) lastSlash = str;
  if (lastDot == NULL) lastDot = str + strlen(str);
  lastSlash++;

  int span = lastDot - lastSlash;
  char filename[span]; memcpy(filename, lastSlash, span);
  return atoi(filename);
}

unsigned int nextPower(unsigned int v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

__global__ void getMagnitudeAngle(cufftComplex *data, float *magnitude, float *angle, int n_samples) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < n_samples / 2 + 1) { // Optimization... only need half of it
    magnitude[id] = sqrt(data[id].x * data[id].x) + (data[id].y * data[id].y);
    angle[id] = atan2(data[id].y, data[id].x);
  };
}

__global__ void encodeMsg(float *X_embed, char *binary_msg, float a, int frame, int start_embed, int embed_sample_sz, int verbosity) {
  int char_id = blockIdx.x * blockDim.x + threadIdx.x;
  int start = char_id * 8;
  int end = start + 8;
  if (start < frame) {
    if(verbosity) printf("Embedding character: %d -> %d - %d\n",char_id, start, end);
    for(int k = start; k < end; k++) {
      float avg = 0;
      for (int l = 0; l < embed_sample_sz; l++) { //col
        avg += X_embed[start_embed + (k * embed_sample_sz + l)];
      }
      avg /= embed_sample_sz;
      //printf("<%d> (%c): [%d, %d) ->> avg: %.3f\n", k, binary_msg[k], start_embed + (k*embed_sample_sz), start_embed + (k*embed_sample_sz + embed_sample_sz - 1, avg));

      if (binary_msg[k] == '0') {
        for (int l = 0; l < embed_sample_sz; l++) {
          X_embed[start_embed + (k * embed_sample_sz + l)] = avg;
        }
      } else {
        for (int l = 0; l < embed_sample_sz / 2; l++) {
          X_embed[start_embed + (k * embed_sample_sz + l)] = a * avg;
        }
        for (int l = embed_sample_sz / 2; l < embed_sample_sz; l++){
          X_embed[start_embed + (k * embed_sample_sz + l)] = (2 - a) * avg;
        }
      }
    }
  }
}

__global__ void restore(cufftComplex *Y1, float *X_abs, float *X_angle, int n_samples) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n_samples / 2 + 1) { // Optimization... only need half of it
      Y1[id].x = X_abs[id] * cos(X_angle[id]);
      Y1[id].y = X_abs[id] * sin(X_angle[id]);
    } else if (id < n_samples) {
      Y1[id].x = 0; Y1[id].y = 0;
    }
}

__global__ void normalize(float *embedded_signal, int n_samples) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < n_samples) embedded_signal[id] = embedded_signal[id] / n_samples;
}


__global__ void show(char *t, int n) {
  int i = blockIdx.x;
  if (i < n) printf("%d - %c\n", i, t[i]);
}

void printCudaInfo() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  for (int device = 0; device < deviceCount; ++device) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    printf("Device %d: %s\n", device, deviceProp.name);
    printf("  Total Global Memory: %lu bytes\n", deviceProp.totalGlobalMem);
    printf("  Max Grid Dimensions: (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("  Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
  }
}


int main(int argc, char **argv) {
  // ------------------------ Setup -----------------------
  int verbosity = 0;
  int timing = 0;
  int nThreads = 32;
  char *in_filename;
  char *out_filename;
  char *message_path;

  int opt;
  while((opt = getopt(argc, argv, "i:m:n:o:vt")) != -1) {
    switch (opt) {
      case 'i': // in audio filepath
        in_filename = optarg;
        break;
      case 'o': // out audio filepath
        out_filename = optarg;
        break;
      case 'm': // message path
        message_path = optarg;
        break;
      case 'n': // numBlocks
        nThreads = atoi(optarg);
        break;
      case 'v': // verbosity (optional, no argument)
        verbosity = 1;
        break;
      case 't': // print execution time (optional, no argument)
        timing = 1;
        break;
      default:
        break;
    }
  }

  // Read msg file
  int msgLen = parseLenFromFilename(message_path);
  FILE *msgFptr;
  msgFptr = fopen(message_path, "r");
  char message[msgLen];
  fgets(message, msgLen + 1, msgFptr);
  
  // convert message to binary
  char *binary_msg = stringToBinary(message);
  char *GPU_binary_msg;
  cudaMalloc((void**)&GPU_binary_msg, sizeof(char) * strlen(binary_msg));
  cudaMemcpy(GPU_binary_msg, binary_msg, sizeof(char) * strlen(binary_msg), cudaMemcpyHostToDevice);

  if (verbosity){
    printCudaInfo();
    printf("Setup Info:\n");
    printf("nThreads: %d\n", nThreads);
    printf("Message N bits: %d\n", strlen(binary_msg));
    if (verbosity >= 2) {
      printf("Message: %s\n", message);
      printf("Message encoded: ");
      for (int i = 0; i < strlen(binary_msg); i++){
        if ((i % 8 == 0)) printf(" [%c]", message[i/8]);
        printf("%c", binary_msg[i]);
      }
      puts("");
    }
    printf("------------------------\n");
  }

  // ---------------------- Load WAV ---------------------
  SNDFILE *inFile; SF_INFO inInfo;
  float *data; // Not allocated yet
  readWav(in_filename, &data, &inInfo, inFile, verbosity);
  if (verbosity) printf("-------------------------\n");

  int Fs = inInfo.samplerate;
  int N = inInfo.frames;

  if (verbosity) {
    printf("nThreads: %d\n", nThreads);
    printf("Blocks???: %d - %d\n", CALCULATE_BLOCKS(nThreads, N), CALCULATE_BLOCKS(nThreads, N / 2 +1));
    printf("-------------------------\n");
  }

  // ------------------------ FFT ------------------------
  struct timeval  tv1, tv2;
  gettimeofday(&tv1, NULL); // Timing start (for benchmarking)
  // FFT
  if (verbosity) printf ("FFT over read data\n");

  cufftHandle plan; // Create plan first to be able to allocate memory successfully
  cufftReal *GPU_data;
  cufftComplex *GPU_data_ft;
  
  cudaMalloc((void**)&GPU_data, sizeof(cufftReal) * N);
  cudaMalloc((void**)&GPU_data_ft, sizeof(cufftComplex) * (N / 2 + 1));
  cudaMemcpy(GPU_data, (cufftReal*)data, sizeof(cufftReal) * N, cudaMemcpyHostToDevice); // send audio to GPU
  
  if (verbosity) printf("Creating plan ...\n");
  CHECK_CUFFT_ERRORS(cufftPlan1d(&plan, N, CUFFT_R2C, 1));
  
  if (verbosity) printf("Executing plan ...\n");
  CHECK_CUFFT_ERRORS(cufftExecR2C(plan, GPU_data, GPU_data_ft));
  
  free(data);
  cudaFree(GPU_data);

  if (verbosity) {
    printf("FFT... DONE!\n");
    printf("--------------------------\n");
  }
  // ------------------- Embed meesage -------------------
  if (verbosity) printf ("Embedding message into signal:\n");
  int frame = strlen(binary_msg);
  int embed_sample_sz = 10;
  int p = frame * embed_sample_sz; // total number of samples used for embedding data
  int embedding_freq = 5000; // in hz
  float a = 0.1;
  
  int start_embed = embedding_freq + 1;
  int end_embed = embedding_freq + p + 1;
  if(end_embed > N / 2 + 1){
    if(timing) printf("elapsed: too_large\n");
    return 1;
  }

  if (verbosity) {
    printf("Settings:\n");
    printf("frame: %d\n", frame);
    printf("embed_sample_sz: %d\n", embed_sample_sz);
    printf("embedding_freq: %d\n", embedding_freq);
  }

  float *GPU_X_abs;
  float *GPU_X_angle;
  cudaMalloc((void**)&GPU_X_abs, sizeof(float) * (N / 2 + 1));
  cudaMalloc((void**)&GPU_X_angle, sizeof(float) * (N / 2 + 1));

  // Get MagnitudeAngle (only N / 2)
  int nBlocks = CALCULATE_BLOCKS(nThreads, N / 2 + 1 );
  if(verbosity) printf("getMagnitudeAngle<<<%d, %d>>>\n", nBlocks, nThreads);
  getMagnitudeAngle<<<nBlocks, nThreads>>>(GPU_data_ft, GPU_X_abs, GPU_X_angle, N);

  if (verbosity) printf("Embedding range: [%d, %d)\n", start_embed, end_embed);

  nBlocks = CALCULATE_BLOCKS(nThreads, msgLen);
  if (nBlocks == 0) nBlocks = 1;
  if(verbosity) printf("encodeMsg<<<%d, %d>>>\n", nBlocks, nThreads);
  encodeMsg<<<nBlocks, nThreads>>>(GPU_X_abs, GPU_binary_msg, a, frame, start_embed, embed_sample_sz, 0);
  
  // Get complex back
  cufftComplex *GPU_Y1;
  cudaMalloc((void **)&GPU_Y1, sizeof(cufftComplex) * (N / 2 + 1));
  nBlocks = CALCULATE_BLOCKS(nThreads, N / 2 + 1);
  if(verbosity) printf("restore<<<%d, %d>>>\n", nBlocks, nThreads);
  restore<<<nBlocks, nThreads>>>(GPU_Y1, GPU_X_abs, GPU_X_angle, N);

  cudaFree(GPU_X_abs);
  cudaFree(GPU_X_angle);
  cudaFree(GPU_data_ft);

  if (verbosity){
    printf("Embedding in signal... DONE!\n");
    printf("--------------------------\n");
  }
  // ----------------------- IFFT --------------------------
  cufftHandle inverse_plan;
  cufftReal *GPU_embedded_signal;
  cudaMalloc((void **)&GPU_embedded_signal, sizeof(cufftReal) * N);
  
  if (verbosity) printf("Creating inverse plan ...\n");
  CHECK_CUFFT_ERRORS(cufftPlan1d(&inverse_plan, N, CUFFT_C2R, 1));
  
  if (verbosity) printf("Executing IFFT ...\n");

  CHECK_CUFFT_ERRORS(cufftExecC2R(inverse_plan, GPU_Y1, GPU_embedded_signal));
  
  nBlocks = CALCULATE_BLOCKS(nThreads, N);
  normalize<<<nBlocks, nThreads>>>(GPU_embedded_signal, N);
  if(verbosity) printf("normalize<<<%d, %d>>>\n", nBlocks, nThreads);

  // Send data back to device
  float *embedded_signal = (float *)malloc(N * sizeof(float));
  cudaMemcpy(embedded_signal, (float*)GPU_embedded_signal, N * sizeof(cufftReal), cudaMemcpyDeviceToHost); // send audio to GPU
  
  cudaFree(GPU_Y1);
  cudaFree(GPU_embedded_signal);
  
  if (timing) {
    gettimeofday(&tv2, NULL); // Timing end (for benchmarking)
    printf ("elapsed: %f\n",
      (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
      (double) (tv2.tv_sec - tv1.tv_sec));
  }
  // -------------------- Write Embedded WAV -----------------

  // for(int i = 4251890; i < 4251900; i++){
  //   printf("%d, %f\n", i, embedded_signal[i]);
  // }

  writeWav(out_filename, embedded_signal, Fs, N, verbosity);
  free(embedded_signal);
}