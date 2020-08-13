/* Program :  Speedup calculation of matrix multiplication with
 *            multi-processing and multi-threading.
 * Author  :  changeme  changeme
 * Roll#   :  changeme
 */

#include <assert.h> /* for assert */
#include <errno.h>  /* for error code eg. E2BIG */
#include <getopt.h> /* for getopt */
#include <pthread.h>
#include <stdio.h>  /* for fprintf */
#include <stdlib.h> /* for exit, atoi */
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#define T 3
pthread_t thread_id[T];
void usage(int argc, char *argv[]);
void input_matrix(int *mat, int nrows, int ncols);
void output_matrix(int *mat, int nrows, int ncols);
void init_matrix(int *mat, int nrows, int ncols);
int *A, *B, *C;
int crows, ccols;
int arows, acols, brows, bcols;
char interactive = 0;

unsigned long long int single_thread_mm() {
  A = (int *)malloc(arows * acols * sizeof(int));
  B = (int *)malloc(brows * bcols * sizeof(int));
  if (interactive == 1) {
    printf("Enter A:\n");
    fflush(stdout);
    input_matrix(A, arows, acols);
    printf("Enter B:\n");
    fflush(stdout);
    input_matrix(B, brows, bcols);
  } else {
    init_matrix(A, arows, acols);
    init_matrix(B, brows, bcols);
  }
  C = (int *)malloc(arows * bcols * sizeof(int));
  struct timeval start, end;
  gettimeofday(&start, NULL);
  for (int i = 0; i < arows; i++) {
    for (int j = 0; j < bcols; j++) {
      *(C + (i * acols + j)) = 0;
      for (int p = 0; p < acols; p++) {
        *(C + (i * acols + j)) +=
            (*(A + (i * acols + p))) * (*(B + (p * bcols + j)));
      }
    }
  }
  gettimeofday(&end, NULL);

  output_matrix(C, arows, bcols);
  return ((end.tv_usec - start.tv_usec) +
          (end.tv_sec - start.tv_sec) * 1000000);
}

unsigned long long int multi_process_mm() {

  int SIZE1, SIZE2, SIZE3;
  SIZE1 = arows * acols * sizeof(int);
  SIZE2 = brows * bcols * sizeof(int);
  SIZE3 = arows * bcols * sizeof(int);
  int segment_id1 = shmget(IPC_PRIVATE, SIZE1, IPC_CREAT | 0666);
  int segment_id2 = shmget(IPC_PRIVATE, SIZE2, IPC_CREAT | 0666);
  int segment_id3 = shmget(IPC_PRIVATE, SIZE3, IPC_CREAT | 0666);

  A = (int *)malloc(arows * acols * sizeof(int));
  B = (int *)malloc(brows * bcols * sizeof(int));
  C = (int *)malloc(arows * bcols * sizeof(int));
  A = (int *)shmat(segment_id1, NULL, 0);
  B = (int *)shmat(segment_id2, NULL, 0);
  C = (int *)shmat(segment_id3, NULL, 0);
  if (interactive == 1) {
    printf("Enter A:\n");
    fflush(stdout);
    input_matrix(A, arows, acols);
    printf("Enter B:\n");
    fflush(stdout);
    input_matrix(B, brows, bcols);
  } else {
    init_matrix(A, arows, acols);
    init_matrix(B, brows, bcols);
  }
  struct timeval start, end;
  gettimeofday(&start, NULL);
  int pid = fork();
  if (pid == 0) {

    for (int i = 0; i < arows / 2; i++) {
      for (int j = 0; j < bcols; j++) {
        *(C + (i * acols + j)) = 0;
        for (int p = 0; p < brows; p++) {
          *(C + (i * acols + j)) +=
              (*(A + (i * acols + p))) * (*(B + (p * bcols + j)));
        }
      }
    }
    exit(0);
  } else {

    for (int i = arows / 2; i < arows; i++) {
      for (int j = 0; j < bcols; j++) {
        *(C + (i * arows + j)) = 0;
        for (int p = 0; p < brows; p++) {
          *(C + (i * acols + j)) +=
              (*(A + (i * acols + p))) * (*(B + (p * bcols + j)));
        }
      }
    }
  }
  wait(NULL);
  gettimeofday(&end, NULL);
  output_matrix(C, arows, bcols);
  return ((end.tv_usec - start.tv_usec) +
          (end.tv_sec - start.tv_sec) * 1000000);
}

void *multi_mul(void *arg) {

  int x = *(int *)arg;

  x = (x * arows) / T;
  int y = (x + (arows / T)) + 1;
  for (int i = x; i < arows && i < y; i++) {

    for (int j = 0; j < bcols; j++) {
      *(C + (i * acols + j)) = 0;

      for (int k = 0; k < acols; k++) {

        *(C + (i * acols + j)) +=
            (*(A + (i * acols + k))) * (*(B + (k * bcols + j)));
      }
    }
  }
}

unsigned long long int multi_thread_mm() {
  A = (int *)malloc(arows * acols * sizeof(int));
  B = (int *)malloc(brows * bcols * sizeof(int));

  if (interactive == 1) {
    printf("Enter A:\n");
    fflush(stdout);
    input_matrix(A, arows, acols);
    printf("Enter B:\n");
    fflush(stdout);
    input_matrix(B, brows, bcols);
  } else {
    init_matrix(A, arows, acols);
    init_matrix(B, brows, bcols);
  }
  C = (int *)malloc(arows * bcols * sizeof(int));
  int row_numberOfTHread[T];

  struct timeval start, end;
  gettimeofday(&start, NULL);
  for (int i = 0; i < T; i++) {
    row_numberOfTHread[i] = i;

    pthread_create(&thread_id[i], NULL, multi_mul, &row_numberOfTHread[i]);
  }
  for (int i = 0; i < T; i++) {
    pthread_join(thread_id[i], NULL);
  }
  gettimeofday(&end, NULL);
  output_matrix(C, arows, bcols);
  return ((end.tv_usec - start.tv_usec) +
          (end.tv_sec - start.tv_sec) * 1000000);
}

int main(int argc, char *argv[]) {
  int c;

  /* Loop through each option (and its's arguments) and populate variables */
  while (1) {
    int this_option_optind = optind ? optind : 1;
    int option_index = 0;
    static struct option long_options[] = {{"help", no_argument, 0, 'h'},
                                           {"ar", required_argument, 0, '1'},
                                           {"ac", required_argument, 0, '2'},
                                           {"br", required_argument, 0, '3'},
                                           {"bc", required_argument, 0, '4'},
                                           {"interactive", no_argument, 0, '5'},
                                           {0, 0, 0, 0}};

    c = getopt_long(argc, argv, "h1:2:3:4:5", long_options, &option_index);
    if (c == -1)
      break;

    switch (c) {
    case 0:
      fprintf(stdout, "option %s", long_options[option_index].name);
      if (optarg)
        fprintf(stdout, " with arg %s", optarg);
      fprintf(stdout, "\n");
      break;

    case '1':
      arows = atoi(optarg);
      break;

    case '2':
      acols = atoi(optarg);
      break;

    case '3':
      brows = atoi(optarg);
      break;

    case '4':
      bcols = atoi(optarg);
      break;

    case '5':
      interactive = 1;
      break;

    case 'h':
    case '?':
      usage(argc, argv);

    default:
      fprintf(stdout, "?? getopt returned character code 0%o ??\n", c);
      usage(argc, argv);
    }
  }

  if (optind != argc) {
    fprintf(stderr, "Unexpected arguments\n");
    usage(argc, argv);
  }

  unsigned long long time_single, time_multi_process, time_multi_thread;
  if (acols != brows)
    exit(EXIT_FAILURE);

  time_single = single_thread_mm();
  time_multi_process = multi_process_mm();
  time_multi_thread = multi_thread_mm();

  fprintf(stdout, "Time taken for single threaded: %llu us\n", time_single);
  fflush(stdout);
  fprintf(stdout, "Time taken for multi process: %llu us\n",
          time_multi_process);
  fflush(stdout);
  fprintf(stdout, "Time taken for multi threaded: %llu us\n",
          time_multi_thread);
  fflush(stdout);
  fprintf(stdout, "Speedup for multi process : %4.2f x\n",
          (double)time_single / time_multi_process);
  fflush(stdout);
  fprintf(stdout, "Speedup for multi threaded : %4.2f x\n",
          (double)time_single / time_multi_thread);
  fflush(stdout);

  exit(EXIT_SUCCESS);
}

/*
 * Show usage of the program
 */
void usage(int argc, char *argv[]) {
  fprintf(stderr, "Usage:\n");
  fprintf(stderr,
          "%s --ar <rows_in_A>  --ac <cols_in_A>"
          " --br <rows_in_B>  --bc <cols_in_B>"
          " [--interactive]",
          argv[0]);
  exit(EXIT_FAILURE);
}

/*
 * Input a given 2D matrix
 */
void input_matrix(int *mat, int rows, int cols) {

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      fscanf(stdin, "%d", mat + (i * cols + j));
    }
  }
}

void init_matrix(int *mat, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      *(mat + (i * cols + j)) = rand() % 1000;
    }
  }
}
/*
 * Output a given 2D matrix
 */
void output_matrix(int *mat, int rows, int cols) {
  if (interactive == 1) {

    printf("Result:\n");
    fflush(stdout);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        fprintf(stdout, "%d ", *(mat + (i * cols + j)));
      }
      fprintf(stdout, "\n");
    }
  }
}
