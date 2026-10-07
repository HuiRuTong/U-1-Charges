#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "UTILS.h"

int main(int argc, char *argv[]) {
    FILE *sol = fopen(argv[1], "r");
    int num_sol = atoi(argv[2]);
    int num_bounded = 0;
    int *charges = extract_charges(sol, num_sol, &num_bounded);

    bool is_dupe[num_bounded];
    int num_dupes = 0;

    for (int i = 0; i < num_bounded; i++) {
        is_dupe[i] = false;
    }

    char *output_name = argv[3];
    FILE *cleaned = fopen(output_name, "w");

    for (int i = 0; i < num_bounded; i++) {
        int found_at = is_multiple(charges, charges + 18*i, num_bounded, i+1);

        if (found_at != -1) {
            is_dupe[found_at] = true;
            num_dupes++;
        }
    }

    for (int i = 0; i < num_bounded; i++) {
        if (!is_dupe[i]) {
            for (int j = 0; j < 18; j++) {
                fprintf(cleaned, "  % d", *(charges + 18*i+j));
            }
            fprintf(cleaned, "\n");
        }
    }
    printf("There are a total of %d bounded solutions. Of which, %d are unique\n",
           num_bounded, num_bounded-num_dupes);

    fclose(sol);
    fclose(cleaned);

    free(charges);
}