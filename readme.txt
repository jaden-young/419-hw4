Jaden Young

ASSIGNMENT 4: CLUSTERING COEFFICIENT ON GPU

To build:
   make

To run clustering coefficients on every argument that is a file:
   ./run file1 file2 file3

   Results in:
   file1: 12345
   file2: 34123
   file3: 25234

To run all graphs in local 'data' directory:
   ./run $(ls data/*)

Threads are run in blocks of 128.

