#include <stdio.h>
#include <stdlib.h>
#include "UTILS.h"

/*
    This removes all solutions that exceed the bounds
    and outputs them into a separate file

    Mostly used for bookkeeping purposes since the
    other files all utilise the same function to
    extract charges
*/

int main(int argc, char *argv[]) {
    FILE *sol = fopen(argv[1], "r");
    int num_sol = atoi(argv[2]);
    int num_valid = 0;
    int *charges = extract_charges(sol, num_sol, &num_valid);

    char *output_filename = argv[3];
    FILE *rmed = fopen(output_filename, "w");
    
    for (int i = 0; i < num_valid; i++) {
        for (int j = 0; j < 18; j++) {
            fprintf(rmed, "  % d", *(charges + 18*i+j));
        }
        fprintf(rmed, "\n");
    }
    printf("There are %d valid solutions\n", num_valid);

    fclose(sol);
    fclose(rmed);

    free(charges);
}