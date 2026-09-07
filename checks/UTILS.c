#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "UTILS.h"

static void _swap(int *a, int *b) {
    /*
        Swaps the values stored at
        a and b
    */
   
    int temp = *a;
    *a = *b;
    *b = temp;
}

void add_charges(int **a, int *b, int pos, int len) {
    /*
        Appends all elements of b onto a

        To be more specific: a is a
        flattened N x 18 array while
        b is an array with 18 elements

        pos :
            Index of the first unfilled row
        len :
            The number of rows a can actually hold
    */

    // Reserve more space if needed
    if (pos == len - 1) {
        *a = realloc(*a, sizeof(int) * (len + BUFFER_SIZE) * 18);
    }

    for (int i = 0; i < 18; i++) {
        (*a)[18*pos + i] = b[i];
    }
}

int *extract_charges(FILE *sol, int num_sol, int *num_valid) {
    int *all_charges = malloc(sizeof(int) * BUFFER_SIZE * 18);
    int curr_charges[18];
    *num_valid = 0;

    for (int i = 0; i < num_sol; i++) {
        int add = 1;
        for (int j = 0; j < 18; j++) {
            fscanf(sol, " %d", curr_charges + j);
            // Remove any terms that were larger than
            // the specified bounds
            if (abs(curr_charges[j]) > MAX_CHARGE) {
                add = 0;
            }
        }
        if (add) {
            add_charges(&all_charges, curr_charges,
                        *num_valid, BUFFER_SIZE*(1 + *num_valid / BUFFER_SIZE));
            (*num_valid)++;
        }
    }
    return all_charges;
}

void sort(int *charges, int num_sol) {
    /*
        Sorts charges in increasing order
        to match the paper's charge
        arrangements

        Might go unused
    */

    for (int i = 0; i < num_sol*18; i+=3) {
        if (*(charges + i) > *(charges + i+1)) {
            _swap(charges + i, charges + i+1);
        }
        if (*(charges + i+1) > *(charges + i+2)) {
            _swap(charges+ i+1, charges + i+2);
            if (*(charges + i) > *(charges + i+1)) {
                _swap(charges + i, charges+ i+1);
            }
        }
    }
}

void sort_abs(int *charges, int num_sol) {
    /*
        Sorts a 3 element array in
        order of increasing magnitude
    */

    for (int i = 0; i < num_sol*18; i+=3) {
        if (abs(*(charges + i)) > abs(*(charges + i+1))) {
            _swap(charges+i, charges+i+1);
        }
        if (abs(*(charges + i+1)) > abs(*(charges + i+2))) {
            _swap(charges+i+1, charges+i+2);
            if (abs(*(charges + i)) > abs(*(charges + i+1))) {
                _swap(charges+i, charges+i+1);
            }
        }
    }
}

int srch(int *a, int *b, int num_sol_a) {
    /*
        Searches for any rows in a that
        match b exactly

        To be more specific: a is a
        flattened N x 18 array while
        b is an array with 18 elements

        (actually, this is probably not needed)
    */

    for (int i = 0; i < num_sol_a; i++) {
        if (!memcmp(a + 18*i, b, 18*sizeof(int))) {
            return i;
        }
    }
    return -1;
}

int is_multiple(int *a, int *b, int num_sol_a) {
    /*
        Checks to see if b is a multiple of a row
        in a
    */
    
    for (int i = 0; i < num_sol_a; i++) {
        int dot = 0;
        int a_sqr = 0;
        int b_sqr = 0;

        for (int j = 0; j < 18; j++) {
            dot += *(a + 18*i+j) * (*(b + j));
            a_sqr += *(a + 18*i+j) * (*(a + 18*i+j));
            b_sqr += *(b + j) * (*(b + j));
        }
        
        if (dot*dot == a_sqr * b_sqr) {
            return i;
        }
    }
    return -1;
}