#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <sndfile.h>
#include <fftw3.h>
#include <complex.h>

extern char *optarg;
extern int optind, opterr, optopt;

char *stringToBinary(char* s) {
  if(s == NULL) return ""; /* no input string */
  int len = strlen(s);
  char *binary = malloc((len*8 + 1)); // each char is one byte (8 bits) and + 1 at the end for null terminator
  
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
  char* originalString = malloc(stringLength + 1);

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
    for (int i = 0; i < sfInfoPtr->frames; i++) {
      (*readDataPtr)[i] = buffer[i * sfInfoPtr->channels];
    }
  } else {
    memcpy(*readDataPtr, buffer, sfInfoPtr->frames * sizeof(double));
  }
  
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

double *magnitude(fftw_complex *data, int n) {
  double *m = malloc(sizeof(double) * n);
  for (int i = 0; i < n; i++) {
    m[i] = sqrt(data[i][0]*data[i][0] + data[i][1]*data[i][1]);
  }
  return m;
}

double *angle(fftw_complex *data, int n) {
  double *a = malloc(sizeof(double) * n);
  for (int i = 0; i < n; i++) {
    a[i] = atan2(data[i][1], data[i][0]);
  }
  return a;
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

// It should return the message already encoded
double *encode_msg(double *data, int N, char *binary_msg, int n_ranks, int rank, int verbosity) {
  // ------------------ FFT --------------------
  fftw_complex *data_ft = fftw_malloc(sizeof(fftw_complex) * N);
  fftw_plan plan = fftw_plan_dft_r2c_1d(N, data, data_ft, FFTW_ESTIMATE);
  fftw_execute(plan);\

  // ---------------- Embed msg ----------------
  int frame = strlen(binary_msg);
  int embed_sample_sz = 10;
  int p = frame * embed_sample_sz;
  int embedding_freq = 5000 / n_ranks;
  double a = 0.1;

  int start_embed = embedding_freq + 1;
  int end_embed = embedding_freq + p + 1;

  double *X_abs = magnitude(data_ft, N);
  double *X_angle = angle(data_ft, N);
  fftw_free(data_ft);

  // Encode
  for (int k = 0; k < frame; k++) { // row
    double avg = 0;
    for (int l = 0; l < embed_sample_sz; l++) { //col
      avg += X_abs[start_embed + (k * embed_sample_sz + l)];
    }
    avg /= (double)embed_sample_sz;

    if (binary_msg[k] == '0') {
      for (int l = 0; l < embed_sample_sz; l++) {
        X_abs[start_embed + (k * embed_sample_sz + l)] = avg;
      }
    } else {
      for (int l = 0; l < embed_sample_sz / 2; l++) {
        X_abs[start_embed + (k * embed_sample_sz + l)] = a * avg;
      }
      for (int l = embed_sample_sz / 2; l < embed_sample_sz; l++){
        X_abs[start_embed + (k * embed_sample_sz + l)] = (2 - a) * avg;
      }
    }
  }

  // Multiply
  fftw_complex *Y1 = fftw_malloc(sizeof(fftw_complex)* N);
  for (int i = 0; i < N; i++) {
    Y1[i][0] = X_abs[i] * cos(X_angle[i]); 
    Y1[i][1] = X_abs[i] * sin(X_angle[i]);
  }
  free(X_abs);
  free(X_angle);

  double *embedded_signal = malloc(N * sizeof(double));
  fftw_plan inverse_plan = fftw_plan_dft_c2r_1d(N, Y1, embedded_signal, FFTW_ESTIMATE);
  fftw_execute(inverse_plan);
  fftw_free(Y1);

  for (int i = 0; i < N; i++) embedded_signal[i] = embedded_signal[i] / N;

  return embedded_signal;
}


int main(int argc, char** argv) {
  int verbosity = 1;
  int timing = 0;
  // char *in_filename;
  // char *out_filename;
  // char *message_path;
  char in_filename[] = "../Samples/custom_testcases/05.wav"; // Input filename
  char out_filename[] = "../Outputs/c_out.wav"; // Output filename
  char message_path[] = "../MessageSamples/100.txt"; // Message
  
  int rank, n_ranks;
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //TODO - read params

  // RANK 0 params
  double *data; // Not allocated yet
  int N;
  int Fs;

  //TAGS
  int chunk_binary_msg_tag = 1;
  int chunk_data_tag = 2;
  int data_start_tag = 3;
  int data_end_tag = 4;
  int msg_start_tag = 5;
  int msg_end_tag = 6;

  char *chunk_binary_msg; 
  double *chunk_data;
  int data_start, data_end, data_span;
  int msg_start, msg_end, msg_span;

  if (rank == 0) {
    int msgLen = parseLenFromFilename(message_path);
    FILE *msgFptr;
    msgFptr = fopen(message_path, "r");
    char message[msgLen];
    fgets(message, msgLen + 1, msgFptr);
    // convert message to binary
    char *binary_msg = stringToBinary(message);

    if (verbosity){
      printf("n_ranks: %d\n", n_ranks);
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
    
    //Read data on root node
    SNDFILE *inFile; SF_INFO inInfo;
    readWav(in_filename, &data, &inInfo, inFile, verbosity);
    if (verbosity) printf("-------------------------\n");

    Fs = inInfo.samplerate;
    N = inInfo.frames;

    for (int i = n_ranks - 1; i >= 0; i--) {
      // Split message in n ranks
      int base_split = (msgLen / n_ranks);
      int redistribution = (i < msgLen % n_ranks) ? 1: 0;
      msg_start = ((base_split * i) + fmin(i, msgLen % n_ranks)) * 8;
      msg_span = (base_split + redistribution) * 8;
      msg_end = msg_start + msg_span;
      chunk_binary_msg = malloc(sizeof(char) * msg_span);
      memcpy(chunk_binary_msg, binary_msg + msg_start, msg_span);

      // split data in n ranks
      base_split = (N / n_ranks);
      redistribution = (i < N % n_ranks) ? 1: 0;
      data_start = (base_split * i) + fmin(i, N % n_ranks);
      data_span = base_split + redistribution;
      data_end = data_start + data_span;
      chunk_data = malloc(sizeof(double) * data_span);
      memcpy(chunk_data, data + data_start, sizeof(double) * data_span);

      if(i > 0) {
        // printf("SENDING %d to %d\n", msg_start, i);
        MPI_Send(&msg_start, 1, MPI_INT, i, msg_start_tag, MPI_COMM_WORLD);
        MPI_Send(&msg_end, 1, MPI_INT, i, msg_end_tag, MPI_COMM_WORLD);
        MPI_Send(&data_start, 1, MPI_INT, i, data_start_tag, MPI_COMM_WORLD);
        MPI_Send(&data_end, 1, MPI_INT, i, data_end_tag, MPI_COMM_WORLD);
        
        MPI_Send(chunk_binary_msg, msg_span, MPI_CHAR, i, chunk_binary_msg_tag, MPI_COMM_WORLD);
        MPI_Send(chunk_data, data_span, MPI_DOUBLE, i, chunk_data_tag, MPI_COMM_WORLD);
        free(chunk_binary_msg);
        free(chunk_data);
      }
    }
    free(data);
  } else {
    MPI_Recv(&msg_start, 1, MPI_INT, 0, msg_start_tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&msg_end, 1, MPI_INT, 0, msg_end_tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&data_start, 1, MPI_INT, 0, data_start_tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&data_end, 1, MPI_INT, 0, data_end_tag, MPI_COMM_WORLD, &status);
    msg_span = msg_end - msg_start;
    data_span = data_end - data_start;

    chunk_binary_msg = malloc(msg_span * sizeof(char));
    chunk_data = malloc(data_span * sizeof(double));

    MPI_Recv(chunk_binary_msg, msg_span, MPI_CHAR, 0, chunk_binary_msg_tag, MPI_COMM_WORLD, &status);
    MPI_Recv(chunk_data, data_span, MPI_DOUBLE, 0, chunk_data_tag, MPI_COMM_WORLD, &status);
  }

  if (verbosity) {
    printf("RANK %d - msg_start: %d, msg_end: %d, data_start: %d, data_end: %d\n", rank, msg_start, msg_end, data_start, data_end);
    printf(">>> Start encoding for chunk in RANK %d\n", rank);
  }
  double *embeded_chunk_data = encode_msg(chunk_data, data_span, chunk_binary_msg, n_ranks, rank, verbosity);
  if (verbosity) {
    printf("<<< End encoding for chunk in RANK %d\n", rank);
  }

  double *embedded_signal = NULL;
  if (rank == 0){
    embedded_signal = malloc(N * sizeof(double));
  }
  MPI_Gather(embeded_chunk_data, data_span, MPI_DOUBLE,
             embedded_signal, data_span, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    // -------------------- Write Embedded WAV -----------------
    writeWav(out_filename, embedded_signal, Fs, N, verbosity);
    free(embedded_signal);
  }

  MPI_Finalize();
  return 0;
}