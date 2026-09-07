#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

static PyObject *anomaly_quadratic(PyObject *self, PyObject *args) {
    PyArrayObject *charges;
    int coeff = 0;
    
    if (!PyArg_ParseTuple(args, "O", &charges)) {
        return NULL;
    }

    for (int i = 0; i < 3; i++) {
        int q = *((int *) PyArray_DATA(charges) + i);
        int e = *((int *) PyArray_DATA(charges) + 6+i);
        int u = *((int *) PyArray_DATA(charges) + 9+i);
        int l = *((int *) PyArray_DATA(charges) + 12+i);
        int d = *((int *) PyArray_DATA(charges) + 15+i);

        coeff += q*q + e*e - 2*u*u - l*l + d*d;
    }

    return PyLong_FromLong(coeff);
}

static PyObject *anomaly_cubic(PyObject *self, PyObject *args) {
    PyArrayObject *charges;
    int coeff = 0;
    
    if (!PyArg_ParseTuple(args, "O", &charges)) {
        return NULL;
    }

    for (int i = 0; i < 3; i++) {
        int q = *((int *) PyArray_DATA(charges) + i);
        int v = *((int *) PyArray_DATA(charges) + 3+i);
        int e = *((int *) PyArray_DATA(charges) + 6+i);
        int u = *((int *) PyArray_DATA(charges) + 9+i);
        int l = *((int *) PyArray_DATA(charges) + 12+i);
        int d = *((int *) PyArray_DATA(charges) + 15+i);

        coeff += 6*q*q*q - v*v*v - e*e*e - 3*u*u*u + 2*l*l*l - 3*d*d*d;
    }

    return PyLong_FromLong(coeff);
}

static PyObject *yukawa(PyObject *self, PyObject *args) {
    PyArrayObject *charges_sum;
    int q_sum, v_sum, e_sum, u_sum, l_sum, d_sum;
    
    if (!PyArg_ParseTuple(args, "O", &charges_sum)) {
        return NULL;
    }

    q_sum = ((int *) PyArray_DATA(charges_sum))[0];
    v_sum = ((int *) PyArray_DATA(charges_sum))[1];
    e_sum = ((int *) PyArray_DATA(charges_sum))[2];
    u_sum = ((int *) PyArray_DATA(charges_sum))[3];
    l_sum = ((int *) PyArray_DATA(charges_sum))[4];
    d_sum = ((int *) PyArray_DATA(charges_sum))[5];

    return PyLong_FromLong(-q_sum - v_sum - e_sum - u_sum + l_sum - d_sum);
}

static PyObject *multiple_check(PyObject *self, PyObject *args) {
    // Assumes charges have already been sorted w/ _sort_abs
    PyArrayObject *found_charges, *curr_charges;
    int *found_arr; 
    int *curr_arr;

    if (!PyArg_ParseTuple(args, "OO", &found_charges, &curr_charges)) {
        return NULL;
    }

    found_arr = (int *) PyArray_DATA(found_charges);
    curr_arr = (int *) PyArray_DATA(curr_charges);

    int dot = 0;
    int found_sqr = 0;
    int curr_sqr = 0;
    for (int i = 0; i < 18; i++) {
        // After many stack overflow answers, I've concluded that I'm
        // dumb and the best method really was just the dot product
        dot += found_arr[i] * curr_arr[i];
        found_sqr += found_arr[i]*found_arr[i];
        curr_sqr += curr_arr[i]*curr_arr[i];
    }
    
    if (dot*dot == found_sqr * curr_sqr) {
        return Py_True;
    }
    return Py_False;
    
}

// List of callable functions
static PyMethodDef callables[] = {
    
    {"anomaly_quadratic", anomaly_quadratic, METH_VARARGS,
     "Checks if the individual charges fulfill the quadratic condition"},
        
    {"anomaly_cubic", anomaly_cubic, METH_VARARGS,
     "Checks if the individual charges fulfill the cubic conditions"},
            
    {"yukawa", yukawa, METH_VARARGS,
     "Checks yukawa stuff"},
                
    {"multiple_check", multiple_check, METH_VARARGS,
     "Checks if a set of charges is equivalent to a known solution"},

    {NULL, NULL, 0, NULL}
};

// Required API thingy
static struct PyModuleDef conditions = {
    PyModuleDef_HEAD_INIT,
    "conditions",
    "C implementation of various functions. MIGHT provide minimal speedups idk",
    -1,
    callables
};

// CHANGE THIS FUNCTIONS NAME!!! if renaming the module
PyMODINIT_FUNC PyInit_conditions(void) {
    import_array(); // For np
    return PyModule_Create(&conditions);
}