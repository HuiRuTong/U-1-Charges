#include <stdio.h>
#include <stdlib.h>
#include "UTILS.h"

/*
    This checks every solution in a file to see if it satisfies
    all the conditions.

    Because once isn't enough, I suppose.
    
    The actual motivation is that many solutions appear to
    be missing from the "complete" set which likely indicates
    something is wrong with the way I'm checking for dupes
*/

int main(int argc, char *argv[]) {
    FILE *sol = fopen(argv[1], "r");
    int num_sol = atoi(argv[2]);
    int num_bounded = 0;
    int *charges = extract_charges(sol, num_sol, &num_bounded);

    int check_lin = atoi(argv[3]);

    char *output_filename = argv[4];
    FILE *invalid = fopen(output_filename, "w");
    int num_invalid = 0;

    for (int i = 0; i < num_bounded; i++) {
        int quad = 0;
        int cube = 0;
        int ykwa = 0;
        int lin1 = 0;
        int lin2 = 0;
        int lin3 = 0;
        int lin4 = 0;
        
        for (int j = 0; j < 3; j++) {
            quad += charges[18*i + j]*charges[18*i + j] + charges[18*i + j+6]*charges[18*i + j+6]
                    - 2*charges[18*i + j+9]*charges[18*i + j+9] - charges[18*i + j+12]*charges[18*i + j+12]
                    + charges[18*i + j+15]*charges[18*i + j+15];

            cube += 6*charges[18*i + j]*charges[18*i + j]*charges[18*i + j]
                    - charges[18*i + j+3]*charges[18*i + j+3]*charges[18*i + j+3]
                    - charges[18*i + j+6]*charges[18*i + j+6]*charges[18*i + j+6]
                    - 3*charges[18*i + j+9]*charges[18*i + j+9]*charges[18*i + j+9]
                    + 2*charges[18*i + j+12]*charges[18*i + j+12]*charges[18*i + j+12]
                    - 3*charges[18*i + j+15]*charges[18*i + j+15]*charges[18*i + j+15];

            ykwa += - charges[18*i + j] - charges[18*i + j+3] - charges[18*i + j+6]
                    - charges[18*i + j+9] + charges[18*i + j+12] - charges[18*i + j+15];

            if (check_lin) {
                lin1 += 2*charges[18*i + j] - charges[18*i + j+9] - charges[18*i + j+15];
                lin2 += 3*charges[18*i + j] + charges[18*i + j+12];
                lin3 += charges[18*i + j] - 6*charges[18*i + j+6]- 8*charges[18*i + j+9]
                         + 3*charges[18*i + j+12] - 2*charges[18*i + j+15];
                lin4 += 6*charges[18*i + j] - charges[18*i + j+3] - charges[18*i + j+6]
                         - 3*charges[18*i + j+9] + 2*charges[18*i + j+12] - 3*charges[18*i + j+15];
            }
        }

        if (quad || cube || ykwa || lin1 || lin2 || lin3 || lin4) {
            num_invalid++;

            for (int j = 0; j < 18; j++) {
                fprintf(invalid, "  % d", *(charges + j));
            }
            fprintf(invalid, "\n");
        }
    }

    printf("There are %d invalid sol\n", num_invalid);

    fclose(sol);
    fclose(invalid);
    free(charges);
}