#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "UTILS.h"

int main(int argc, char *argv[]) {
    FILE *sol = fopen(argv[1], "r");
    int num_sol = atoi(argv[2]);
    int num_bounded = 0;
    int *charges = extract_charges(sol, num_sol, &num_bounded);

    int *dupes = malloc(sizeof(int) * BUFFER_SIZE);
    int dupes_len = 0;

    char *output_name = argv[3];
    FILE *cleaned = fopen(output_name, "w");

    for (int i = 0, j = 0; i < num_bounded; i++) {
        if (dupes_len > 0 && i == dupes[j]) {
            j++;
            continue;
        }
        int found_at = is_multiple(charges, charges + 18*i, num_bounded, i+1);

        if (found_at != -1) {
            if (dupes_len && !(dupes_len % BUFFER_SIZE)) {
                dupes = realloc(dupes, sizeof(int) * (dupes_len / BUFFER_SIZE));
            }
            dupes[dupes_len] = found_at;
            dupes_len++;
        }

        for (int j = 0; j < 18; j++) {
            fprintf(cleaned, "  % d", *(charges + 18*i+j));
        }
        fprintf(cleaned, "\n");
    }
    printf("There are a total of %d bounded solutions. Of which, %d are unique\n",
           num_bounded, num_bounded-dupes_len);

    fclose(sol);
    fclose(cleaned);

    free(charges);
}