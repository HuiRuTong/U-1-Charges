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
    
    4. Whether to print duplicates or non duplicates
       0 for duplicates, 1 for non-duplicates
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

    int dupe_or_unique = atoi(argv[5]);
    char *file_name;
    
    if (dupe_or_unique) {
        file_name = "./output/uniques.txt";
    } else {
        file_name = "./output/duplicates.txt";
    }
    FILE *logged_sol = fopen(file_name, "w");

    for (int i = 0; i < num_valid_2; i++) {
        int found_at = is_multiple(charges_1, charges_2 + 18*i, num_valid_1);

        if (dupe_or_unique && found_at == -1) {
            for (int j = 0; j < 18; j++) {
                fprintf(logged_sol, "  % d", *(charges_2 + 18*i+j));
            }
            fprintf(logged_sol, "\n");
            
            printf("Solution %d in file 2 is missing from file 1\n", i+1);
            continue;
        }

        if (!dupe_or_unique && found_at != -1) {
            for (int j = 0; j < 18; j++) {
                fprintf(logged_sol, "  % d", *(charges_2 + 18*i+j));
            }
            fprintf(logged_sol, "\n");
            
            printf("Solution %d in file 2 is a duplicate of solution %d in file 1\n", i+1, found_at+1);
        }
    }

    fclose(sol_1);
    fclose(sol_2);
    fclose(logged_sol);

    free(charges_1);
    free(charges_2);
}