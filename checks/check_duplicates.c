#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "UTILS.h"

/*
    This program finds duplicates solutions between
    two files.
    
    It can also be used to check if a solution
    is missing from the complete set.

    Args are:
    1. Path to file containing solutions to search from

    2. Number of solutions in the above file

    2. Path to file containing solutions to search for

    3. Number of solutions in the above file
*/

int main(int argc, char *argv[]) {
    FILE *sol_1 = fopen(argv[1], "r");
    int num_sol_1 = atoi(argv[2]);
    int num_valid_1 = 0;
    int *charges_1 = extract_charges(sol_1, num_sol_1, &num_valid_1);

    FILE *sol_2 = fopen(argv[3], "r");
    int num_sol_2 = atoi(argv[4]);
    int num_valid_2 = 0;
    int *charges_2 = extract_charges(sol_2, num_sol_2, &num_valid_2);

    FILE *dupes = fopen(argv[5], "w");
    FILE *uniques = fopen(argv[6], "w");

    for (int i = 0; i < num_valid_1; i++) {
        int found_at = is_multiple(charges_2, charges_1 + 18*i, num_valid_2, 0);

        if (found_at == -1) {
            for (int j = 0; j < 18; j++) {
                fprintf(uniques, "  % d", *(charges_1 + 18*i+j));
            }
            fprintf(uniques, "\n");
            
            continue;
        }

        if (found_at != -1) {
            for (int j = 0; j < 18; j++) {
                fprintf(dupes, "  % d", *(charges_1 + 18*i+j));
            }
            fprintf(dupes, "\n");
        }
    }

    fclose(sol_1);
    fclose(sol_2);
    fclose(dupes);
    fclose(uniques);

    free(charges_1);
    free(charges_2);
}