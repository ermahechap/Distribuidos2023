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

char *stringToBinary(char* s) {
  if(s == NULL) return "/n"; /* no input string */
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

void readWav(char *path, double **readDataPtr, SF_INFO* sfInfoPtr, SNDFILE *sfFile, int verbosity) {
  if (verbosity) printf("Reading File: %s\n", path);

  sfFile = sf_open(path, SFM_READ, sfInfoPtr);
  double *buffer = (double *)malloc(sfInfoPtr->channels * sfInfoPtr->frames * sizeof(double));
  *readDataPtr = (double *) malloc(sfInfoPtr->frames * sizeof(double));

  sf_read_double(sfFile, buffer, sfInfoPtr->channels * sfInfoPtr->frames);

  if (sfInfoPtr->channels > 1) {
    if (verbosity) printf("WARNING - More than one channel, taking first channel");
    // #pragma omp for
    for (int i = 0; i < sfInfoPtr->frames; i++) {
      (*readDataPtr)[i] = buffer[i * sfInfoPtr->channels];
    }
  } else {
    memcpy(*readDataPtr, buffer, sfInfoPtr->frames * sizeof(double));
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

void writeWav(char *path, double *data, int samplerate, int N, int verbosity) {
  if (verbosity) printf("Writing File: %s\n", path);
  SNDFILE *outFile; SF_INFO outInfo;
  outInfo.samplerate = samplerate;
  outInfo.channels = 1;
  outInfo.format = SF_FORMAT_WAV | SF_FORMAT_DOUBLE;

  outFile = sf_open(path, SFM_WRITE, &outInfo);
  sf_write_double(outFile, data, N);
  sf_close(outFile);
  if (verbosity) printf("File Writing... DONE!\n", path);
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

__global__ void getMagnitudeAngle(cufftDoubleComplex *data, double *magnitude, double *angle, int n_samples) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < (n_samples + 1) / 2) { // Optimization... only need half of it
    magnitude[id] = sqrt(data[id].x * data[id].x) + (data[id].y * data[id].y);
    angle[id] = atan2(data[id].y, data[id].x);
    if (id < 10){
      printf("%d: %f, %f\n", id, magnitude[id], angle[id]);
    }
  };
}


__global__ void encodeMsg(double *X_embed, char *binary_msg, double a, int frame, int start_embed, int embed_sample_sz, int verbosity) {
  int start = threadIdx.x * 8;
  int end = (threadIdx.x + blockIdx.x * blockDim.x) * 8;
  
  for(int k = start; k < end; k++) {
    double avg = 0;
    for (int l = 0; l < embed_sample_sz; l++) { //col
      avg += X_embed[start_embed + (k * embed_sample_sz + l)];
    }
    avg /= embed_sample_sz;

    // embed_sample_sz is small enough - Not paralellized
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

__global__ void restore(cufftDoubleComplex *Y1, double *X_abs, double *X_angle, int n_samples) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < (n_samples + 1) / 2) { // Optimization... only need half of it
      Y1[id].x = X_abs[id] * cos(X_angle[id]);
      Y1[id].y = X_abs[id] * sin(X_angle[id]);
    } else if (id < n_samples) {
      Y1[id].x = 0; Y1[id].y = 0;
    }
}

__global__ void normalize(double *embedded_signal, int n_samples) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < n_samples) embedded_signal[id] / n_samples;
}


int main(int argc, char **argv) {
  // ------------------------ Setup --------------------
  int verbosity = 0;
  int timing = 0;
  int nBlocks = 256;
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
        nBlocks = atoi(optarg);
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
  cudaMalloc((void**)&GPU_binary_msg, sizeof(char)*msgLen);
  cudaMemcpy(GPU_binary_msg, binary_msg, sizeof(char)*msgLen, cudaMemcpyHostToDevice);

  if (verbosity){
    
    printf("nBlocks: %d\n", nBlocks);
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
  double *data; // Not allocated yet
  readWav(in_filename, &data, &inInfo, inFile, verbosity);
  if (verbosity) printf("-------------------------\n");

  int Fs = inInfo.samplerate;
  int N = inInfo.frames;

  // ------------------------ FFT ------------------------
  // FFT
  if (verbosity) printf ("FFT over read data\n");

  cufftHandle plan; // Create plan first to be able to allocate memory successfully
  cufftPlan1d(&plan, N, CUFFT_R2C, 1);
  double * GPU_data;
  cufftDoubleComplex *GPU_data_ft;
  cudaMalloc((void**)&GPU_data, sizeof(double) * N);
  cudaMalloc((void**)&GPU_data_ft, sizeof(cufftDoubleComplex) * N);
  cudaMemcpy(GPU_data, data, sizeof(double) * N, cudaMemcpyHostToDevice); // send audio to GPU

  cufftExecD2Z(plan, GPU_data, GPU_data_ft); // Excute plan

  //cudaFree(GPU_data);
  free(data);

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
  double a = 0.1;

  if (verbosity) {
    printf("Settings:\n");
    printf("frame: %d\n", frame);
    printf("embed_sample_sz: %d\n", embed_sample_sz);
    printf("embedding_freq: %d\n", embedding_freq);
  }

  double *GPU_X_abs;
  double *GPU_X_angle;
  cudaMalloc((void**)&GPU_X_abs, sizeof(double) * N);
  cudaMalloc((void**)&GPU_X_angle, sizeof(double) * N);

  
  // Get MagnitudeAngle (only N / 2)
  int blockSize = 256;
  int numThreads = (N / 2 + blockSize - 1) / blockSize; // N threads for magnitudeAngle

  if(verbosity) printf("getMagnitudeAngle<<<%d, %d>>>\n", blockSize, numThreads);
  getMagnitudeAngle<<<blockSize, numThreads>>>(GPU_data_ft, GPU_X_abs, GPU_X_angle, N);

  int start_embed = embedding_freq + 1;
  int end_embed = embedding_freq + p + 1;

  numThreads = (msgLen + blockSize - 1) / blockSize; // N threads for message encoding/decoding
  if (numThreads == 0) numThreads = 1;
  if(verbosity) printf("encodeMsg<<<%d, %d>>>\n", blockSize, numThreads);
  encodeMsg<<<blockSize, numThreads>>>(GPU_X_abs, GPU_binary_msg, a, frame, start_embed, embed_sample_sz, verbosity);


  // Restore
  cufftDoubleComplex *GPU_Y1;
  cudaMalloc((void **)&GPU_Y1, sizeof(cufftDoubleComplex) * N);
  numThreads = (N / 2 + blockSize - 1) / blockSize; // N threads for magnitudeAngle
  if(verbosity) printf("restore<<<%d, %d>>>\n", blockSize, numThreads);
  restore<<<blockSize, numThreads>>>(GPU_Y1, GPU_X_abs, GPU_X_angle, N);

  if (verbosity){
    printf("Embedding in signal... DONE!\n");
    printf("--------------------------\n");
  }
  // ----------------------- IFFT --------------------------
  cufftHandle ifft_plan;
  cufftPlan1d(&ifft_plan, N, CUFFT_C2R, 1);
  double *GPU_embedded_signal;
  cudaMalloc((void **)&GPU_embedded_signal, sizeof(double) * N);
  cufftExecZ2D(ifft_plan, GPU_data_ft, GPU_embedded_signal);
  
  numThreads = (N + blockSize - 1) / blockSize; // N threads for magnitudeAngle
  if(verbosity) printf("normalize<<<%d, %d>>>\n", blockSize, numThreads);
  normalize<<<blockSize, numThreads>>>(GPU_embedded_signal, N);

  // Restore data back to device
  double *embedded_signal = (double *)malloc(N * sizeof(double));
  cudaMemcpy(embedded_signal, GPU_embedded_signal, N * sizeof(double), cudaMemcpyDeviceToHost); // send audio to GPU
  // -------------------- Write Embedded WAV -----------------
  writeWav(out_filename, embedded_signal, Fs, N, verbosity);
  free(embedded_signal);

  // msgLen, blockSize
  // Rules:
  // blockSize * nThreads >= msgLen and 
  // nThreads = 2**i
  // nThreads <= 1024
  

}