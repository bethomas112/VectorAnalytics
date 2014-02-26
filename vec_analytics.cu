#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#define FLOAT_SIZE 4


int main(int argc, char **argv) {
   int fp;
   int numElements;

   if (argc < 2) {
      fprintf(stderr, "Usage: %s <infile>\n", *argv);
      exit(1);
   }

   if ((fp = open(argv[1], O_RDONLY)) == -1) {
      perror("open");
      exit(1);
   }
   
   read(fp, &numElements, FLOAT_SIZE);

   

   return 0; 
   
}
