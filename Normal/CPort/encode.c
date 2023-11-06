#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sndfile.h>
#include <fftw3.h>
#include <complex.h>

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

int main(void) {
  // ------------------------ Setup --------------------
  int verbosity = 1;
  char in_filename[] = "../../Samples/ImperialMarch60.wav"; // Input filename
  char out_filename[] = "../../Outputs/c_out.wav"; // Output filename
  char message[] = "my name is slim shady"; // Message

  // convert message to binary
  char *binary_msg = stringToBinary(message);
  if (verbosity){
    printf("Message N bits: %d\n", strlen(binary_msg));
    if (verbosity > 2) {
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

  // --------------------- FFT -----------------------------
  // FFT
  if (verbosity) printf ("FFT over read data\n");
  fftw_complex *data_ft = fftw_malloc(sizeof(fftw_complex) * N);
  fftw_plan plan = fftw_plan_dft_r2c_1d(N, data, data_ft, FFTW_ESTIMATE); // Create fftw execution plan
  // fftw_plan plan = fftw_plan_dft_1d(N, data, data_ft, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(plan);
  fftshift(&data_ft, N); // Shift the data

  if (verbosity) {
    printf("FFT... DONE!\n");
    printf("--------------------------\n");
  }
  // --------------------- Embed Message ----------------------
  if (verbosity) printf ("Embedding message into signal:\n");
  int frame = strlen(binary_msg);
  int embed_sample_sz = 10;
  int p = frame * embed_sample_sz;
  int centre = N / 2 + 1;
  int embedding_freq = 5000;
  double a = 0.1;
  if (verbosity) {
    printf("Settings:\n");
    printf("frame: %d\n", frame);
    printf("embed_sample_sz: %d\n", embed_sample_sz);
    printf("embedding_freq: %d\n", embedding_freq);
  }

  double *X_abs = magnitude(data_ft, N);
  double *X_angle = angle(data_ft, N);
  fftw_free(data_ft);

  int start_embed = centre + embedding_freq + 1;
  int end_embed = centre + embedding_freq + p + 1;
  if (verbosity) printf("Embedding range: [%d, %d)\n", start_embed, end_embed);
  double *X_embed = malloc(sizeof(double) * p);
  memcpy(X_embed, &X_abs[start_embed], sizeof(double) * p);

  // Loop
  for (int k = 0; k < frame; k++) { // row
    double avg = 0;
    for (int l = 0; l < embed_sample_sz; l++) { //col
      avg += X_embed[k * embed_sample_sz + l];
    }
    avg /= (double)embed_sample_sz;

    if (binary_msg[k] == '0') {
      for (int l = 0; l < embed_sample_sz; l++) {
        X_embed[k * embed_sample_sz + l] = avg;
      }
    } else {
      for (int l = 0; l < embed_sample_sz / 2; l++) {
        X_embed[k * embed_sample_sz + l] = a * avg;
      }
      for (int l = embed_sample_sz / 2; l < embed_sample_sz; l++){
        X_embed[k * embed_sample_sz + l] = (2 - a) * avg;
      }
    }
  }
  // Y[range_2] = X_embed
  int range_2[] = {centre + embedding_freq + 1, centre + embedding_freq + p + 1}; // [centre + freq, centre + freq + p]
  for (int i = range_2[0]; i < range_2[1]; i++){
    X_abs[i] = X_embed[i - range_2[0]];
  }

  // Multiply
  fftw_complex *Y1 = fftw_malloc(sizeof(fftw_complex)* N);
  for (int i = 0; i < N; i++) {
    Y1[i][0] = X_abs[i] * cos(X_angle[i]); 
    Y1[i][1] = X_abs[i] * sin(X_angle[i]);
  }

  free(X_abs);
  free(X_angle);
  
  if (verbosity){
    printf("Embedding in signal... DONE!\n");
    printf("--------------------------\n");
  }
  // --------------------- Unshift FFT -----------------------
  if (verbosity) printf("IFFT over embedded data!\n");
  ifftshift(&Y1, N); // unshift
  double *embedded_signal = malloc(N * sizeof(double));
  fftw_plan inverse_plan = fftw_plan_dft_c2r_1d(N, Y1, embedded_signal, FFTW_ESTIMATE);
  fftw_execute(inverse_plan);
  fftw_free(Y1);

  // Normalization
  for (int i = 0; i < N; i++) embedded_signal[i] = embedded_signal[i] / N;
  
  if (verbosity){
    printf("IFFT... DONE!\n");
    printf("--------------------------\n");
  }
  // -------------------- Write Embedded WAV -----------------
  writeWav(out_filename, embedded_signal, Fs, N, verbosity);
  free(embedded_signal);
}