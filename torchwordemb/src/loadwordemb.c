#include <stdio.h>

#include "Python.h"
#include <TH/TH.h>

static int get_line_numbers(FILE *fp){
    int cnt = 0;
    int ch = 0;
    do{
        ch = fgetc(fp);
        if( ch == '\n' ) cnt ++;
    }while( ch != EOF );

    rewind(fp);
    return cnt;
}
static int get_tokens_in_first_line(FILE*fp){
    int cnt = 0;
    int ch = 0, last = 0;

    while(1){
        ch = fgetc(fp);
        if( ch != ' ' ) break;
    }
    last = ' ';
    while( ch != '\n' ){

        if( ch != ' ' && last == ' ' ){
            cnt++;
        }
       
        last = ch;
        ch = fgetc(fp);
    }
    rewind(fp);
    return cnt;
}

int load_glove(THFloatTensor *dest, const char * filename, size_t o){
    PyObject *obj = (PyObject*) (void*)(uintptr_t) o;

    FILE *fp = fopen(filename, "rb");


    if( fp == NULL ){ 
        return -1;
    }

    size_t dim, n_word;
    dim    = get_tokens_in_first_line(fp) - 1;
    n_word = get_line_numbers(fp);
    //printf("%zu %zu\n", n_word, dim );

    THFloatTensor_resize2d(dest, n_word, dim);

    char buf[512];
    float * data = THFloatTensor_data(dest);
    for( int i = 0; i < n_word; i++ ){

        float *vector = data + dim * i;

        fscanf(fp, "%s", buf );
        PyDict_SetItem(obj, PyUnicode_FromString(buf), PyLong_FromLong((long)i) );

        for( int j=0; j<dim; j++){
            fscanf(fp, "%f", vector+j);
        }

    }
    fclose(fp);
    return n_word;
}

int load_word2vec(THFloatTensor *dest, const char * filename, size_t o){
    PyObject *obj = (PyObject*) (void*)(uintptr_t) o;

    FILE *fp = fopen(filename, "rb");


    if( fp == NULL ){ 
        return -1;
    }

    size_t dim, n_word;
    fscanf(fp, "%zu %zu", &n_word, &dim);
    //printf("%zu %zu\n", n_word, dim );

    THFloatTensor_resize2d(dest, n_word, dim);

    char buf[512];
    float * data = THFloatTensor_data(dest);
    for( int i = 0; i < n_word; i++ ){

        float *vector = data + dim * i;

        fscanf(fp, "%s", buf );
        PyDict_SetItem(obj, PyUnicode_FromString(buf), PyLong_FromLong((long)i) );

        for( int j=0; j<dim; j++){
            fscanf(fp, "%f", vector+j);
        }

    }
    fclose(fp);
    return n_word;
}
int load_word2vec_bin(THFloatTensor *dest, const char * filename, size_t o){
    PyObject *obj = (PyObject*) (void*)(uintptr_t) o;

    FILE *fp = fopen(filename, "rb");


    if( fp == NULL ){ 
        return -1;
    }

    size_t dim, n_word;
    fscanf(fp, "%zu %zu\n", &n_word, &dim);
    //printf("%zu %zu\n", n_word, dim );

    THFloatTensor_resize2d(dest, n_word, dim);

    char buf[512];
    float * data = THFloatTensor_data(dest);
    for( int i = 0; i < n_word; i++ ){

        float *vector = data + dim * i;

        fscanf(fp, "%s", buf );
        PyDict_SetItem(obj, PyUnicode_FromString(buf), PyLong_FromLong((long)i) );

        fgetc(fp); // delete ' '

        fread( vector, sizeof(float), dim, fp );
    }
    fclose(fp);
    return n_word;
}

