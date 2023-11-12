#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
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

void fftshift(fftw_complex **dataPtr, int N) {
  int c = (int) floor((float) N / 2);
  if (N % 2 == 0) { //even
    for (int k = 0; k < c; k++){
      fftw_complex tmp = {(*dataPtr)[k][0], (*dataPtr)[k][1]};
      (*dataPtr)[k][0] = (*dataPtr)[k + c][0]; (*dataPtr)[k][1] = (*dataPtr)[k + c][1];
      (*dataPtr)[k + c][0] = tmp[0]; (*dataPtr)[k + c][1] = tmp[1];
    }
  } else { // odd
    fftw_complex tmp = {(*dataPtr)[0][0], (*dataPtr)[0][1]};
    for (int k = 0; k < c; k++){
      (*dataPtr)[k][0] = (*dataPtr)[c + k + 1][0]; (*dataPtr)[k][1] = (*dataPtr)[c + k + 1][1];
      (*dataPtr)[c + k + 1][0] = (*dataPtr)[k + 1][0]; (*dataPtr)[c + k + 1][1] = (*dataPtr)[k + 1][1];
    }
    (*dataPtr)[c][0] = tmp[0]; (*dataPtr)[c][1] = tmp[1];
  }
}

void ifftshift(fftw_complex **dataPtr, int N){
  int c = (int) floor((float) N / 2);
  if (N % 2 == 0) { // even
    for (int k = 0; k < c; k++){
      fftw_complex tmp = {(*dataPtr)[k][0], (*dataPtr)[k][1]};
      (*dataPtr)[k][0] = (*dataPtr)[k + c][0]; (*dataPtr)[k][1] = (*dataPtr)[k + c][1];
      (*dataPtr)[k + c][0] = tmp[0]; (*dataPtr)[k + c][1] = tmp[1];
    }
  } else { // odd
    fftw_complex tmp = {(*dataPtr)[N - 1][0], (*dataPtr)[N - 1][1]};
    for (int k = c - 1; k >= 0; k--){
      (*dataPtr)[c + k + 1][0] = (*dataPtr)[k][0]; (*dataPtr)[c + k + 1][1] = (*dataPtr)[k][1];
      (*dataPtr)[k][0] = (*dataPtr)[c + k][0]; (*dataPtr)[k][1] = (*dataPtr)[c + k][1];
    }
    (*dataPtr)[c][0] = tmp[0]; (*dataPtr)[c][1] = tmp[1];
  }
}

int main(int argc, char **argv) {
  // ------------------------ Setup --------------------
  int verbosity = 0;
  int timing = 0;
  char *in_filename;
  char *out_decoded_filename;
  int n_chars;

  // int verbosity = 1;
  // char in_filename[] = "../../Outputs/c_out.wav"; // Output filename
  // int n_chars = 21; // msg n characters

  int opt;
  while((opt = getopt(argc, argv, "i:l:o:vt")) != -1) {
    switch (opt) {
      case 'i': // in audio filepath
        in_filename = optarg;
        break;
      case 'o': // out txt file
        out_decoded_filename = optarg;
        break;
      case 'l': // 
        n_chars = atoi(optarg);
        break;
      case 'v': // verbosity (optional, no argument)
        verbosity = 1;
        break;
      case 't': // print execution time
        timing = 1;
        break;
      default:
        break;
    }
  }
  // ---------------------- Load WAV ---------------------
  SNDFILE *inFile; SF_INFO inInfo;
  double *data; // Not allocated yet
  readWav(in_filename, &data, &inInfo, inFile, verbosity);
  if (verbosity) printf("-------------------------\n");

  int Fs = inInfo.samplerate;
  int N = inInfo.frames;

  // --------------------- FFT -----------------------------
  struct timeval  tv1, tv2;
  gettimeofday(&tv1, NULL); // Timing start (for benchmarking)
  // FFT
  if (verbosity) printf ("FFT over read data\n");
  fftw_complex *data_ft = fftw_malloc(sizeof(fftw_complex) * N);
  fftw_plan plan = fftw_plan_dft_r2c_1d(N, data, data_ft, FFTW_ESTIMATE); // Create fftw execution plan
  fftw_execute(plan);

  if (verbosity) {
    printf("FFT... DONE!\n");
    printf("--------------------------\n");
  }

  // ----------------- Decode Embedded msg ----------------
  if (verbosity) printf ("Decoding message:\n");
  int frame = n_chars * 8;
  int embed_sample_sz = 10;
  int p = frame * embed_sample_sz;
  int embedding_freq = 5000;
  double a = 0.1;

  double *Y2_abs = magnitude(data_ft, N);
  double *Y2_angle = angle(data_ft, N);

  int start_embed = embedding_freq + 1;
  int end_embed = embedding_freq + p + 1;
  if (verbosity) printf("Embedding range: [%d, %d)\n", start_embed, end_embed);

  char *recovered_binary = malloc(frame * sizeof(char)); // recovered msg in binary string
  // Decode loop
  for (int k = 0; k < frame; k++){
    double avg = 0; int b = 0, c = 0;
    for (int l = 0; l < embed_sample_sz; l++) {
      avg = avg + Y2_abs[start_embed + (k * embed_sample_sz + l)];
    }
    avg = avg / embed_sample_sz;
        
    for (int l = 0; l < embed_sample_sz / 2; l++){
        if (Y2_abs[start_embed + (k * embed_sample_sz + l)] >= (1 + a) * avg / 2) c++;
        else b++;
      }

    for (int l = embed_sample_sz / 2; l < embed_sample_sz - 1; l++){
      if (Y2_abs[start_embed + (k * embed_sample_sz + l)] < (3 - a) * avg / 2)c++;
      else b++;
    }
    recovered_binary[k] = (b > c) ? '1': '0';
  }

  char *recoveredMsg = binaryToString(recovered_binary);
  if (verbosity) {
    printf("Recovered Encoded Message: %s\n", recovered_binary);
    printf("%s\n", recoveredMsg);
  }

  if (timing) {
    gettimeofday(&tv2, NULL); // Timing end (for benchmarking)
    printf ("elapsed: %f\n",
      (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
      (double) (tv2.tv_sec - tv1.tv_sec));
  }
  // Read outmsg file
  FILE *msgFptr;
  msgFptr = fopen(out_decoded_filename, "w");
  fprintf(msgFptr, recoveredMsg);

}