#include <tuple>
#include <map>

#include <torch/extension.h>
#include <pybind11/pybind11.h>


namespace py = pybind11;

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
static int get_tokens_in_first_line(FILE *fp){
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


using VocabAndTensor = std::tuple<py::dict, torch::Tensor> ;

VocabAndTensor load_glove(const char * filename){

    py::dict vocab;

    auto options = torch::TensorOptions()
            .dtype(torch::kFloat32);

    torch::Tensor dest = torch::empty(torch::IntArrayRef{0}, options);


    FILE *fp = fopen(filename, "rb");

    if( fp != NULL ){ 

        size_t dim, n_word;
        dim    = get_tokens_in_first_line(fp) - 1;
        n_word = get_line_numbers(fp);

        dest.resize_(torch::IntArrayRef{n_word, dim});

        char buf[512];
        float * data = static_cast<float*>(dest.storage().data());

        for( size_t i = 0; i < n_word; i++ ){

            float *vector = data + dim * i;

            fscanf(fp, "%s", buf );
            vocab[ py::str(buf) ] = py::int_(i);

            for( size_t j=0; j<dim; j++){
                fscanf(fp, "%f", vector+j);
            }

        }
        fclose(fp);
    }
    return {vocab, dest};
}


VocabAndTensor load_word2vec(const char * filename){
    py::dict vocab;

    auto options = torch::TensorOptions()
            .dtype(torch::kFloat32);

    torch::Tensor dest = torch::empty(torch::IntArrayRef{0}, options);

    FILE *fp = fopen(filename, "rb");
    if( fp != NULL ){ 

        size_t dim, n_word;
        fscanf(fp, "%zu %zu", &n_word, &dim);

        dest.resize_(torch::IntArrayRef{n_word, dim});

        char buf[512];
        float * data = static_cast<float*>(dest.storage().data());

        for( size_t i = 0; i < n_word; i++ ){

            float *vector = data + dim * i;

            fscanf(fp, "%s", buf );
            vocab[ py::str(buf) ] = py::int_(i);

            for( size_t j=0; j<dim; j++){
                fscanf(fp, "%f", vector+j);
            }

        }
        fclose(fp);
    }
    return {vocab, dest};
}

VocabAndTensor load_word2vec_bin(const char * filename){
    py::dict vocab;

    auto options = torch::TensorOptions()
            .dtype(torch::kFloat32);

    torch::Tensor dest = torch::empty(torch::IntArrayRef{0}, options);
    FILE *fp = fopen(filename, "rb");


    if( fp != NULL ){ 

        size_t dim, n_word;
        fscanf(fp, "%zu %zu\n", &n_word, &dim);
        //printf("%zu %zu\n", n_word, dim );

        dest.resize_(torch::IntArrayRef{n_word, dim});

        char buf[512];
        float * data = static_cast<float*>(dest.storage().data());
        for( size_t i = 0; i < n_word; i++ ){

            float *vector = data + dim * i;

            fscanf(fp, "%s", buf );
            vocab[ py::str(buf) ] = py::int_(i);

            fgetc(fp); // delete ' '

            fread( vector, sizeof(float), dim, fp );
        }
        
        fclose(fp);
    }
    return {vocab, dest};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("load_glove_text", &load_glove, "Load GloVE vectors (text format)");
  m.def("load_word2vec_text", &load_word2vec, "Load Word2Vec vectors (text format)");
  m.def("load_word2vec_bin", &load_word2vec_bin, "Load Word2Vec vectors (binary format)");
}