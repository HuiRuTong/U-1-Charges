#include <stdio.h>
#include <stdlib.h>
#include "UTILS.h"

/*
    This sorts the complete set of solutions
    according to increasing magitude
*/

int main(int argc, char *argv[]) {
    FILE *complete_sol = fopen(argv[1], "r");
    int num_complete_sol = atoi(argv[2]);
    int num_valid = 0;
    int *complete_charges = extract_charges(complete_sol, num_complete_sol, &num_valid);

    char *output_filename = argv[3];
    FILE *sorted_sol = fopen(output_filename, "w");

    sort_abs(complete_charges, num_valid);
    for (int i = 0; i < num_valid; i++) {
        for (int j = 0; j < 18; j++) {
            fprintf(sorted_sol, "  % d", *(complete_charges + 18*i+j));
        }
        fprintf(sorted_sol, "\n");
    }

    fclose(complete_sol);
    free(complete_charges);
}