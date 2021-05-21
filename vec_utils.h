#include <iostream>
#include <stdlib.h> 


namespace vecutils {

typedef struct {
    int blocks;
    int threads;
} Grid;

typedef struct {
    dim3 blocks;
    dim3 threads;
} HyperGrid;

template <class T>
void printVector(T* vec,int N,char delim=' ') {
    #ifndef _VUTILS_SUPPRESS_
    for(int i =0;i < N;++i)
        std::cout << vec[i] << delim;
    std::cout << std::endl;
    #endif
}

template <class T>
void printMatrix(T* vec,int R,int C,char delim=' ') {
    #ifndef _VUTILS_SUPPRESS_
    for(int i =0;i < R;++i) {
        for(int j = 0; j < C; ++j) 
            std::cout << vec[i*C+j] << delim;
        std::cout << std::endl;
    }
    std::cout << std::endl;
    #endif
}



template <class T>
void initVect(T* vec,int N,int min=0,int max=RAND_MAX) {
    for(int i =0;i < N;++i)
        vec[i] = (T)(rand() % max + min);
}

template <class T>
void initMatrix(T* vec,int R,int C,int min=0,int max=RAND_MAX) {
    for(int i =0;i < R*C;++i)
            vec[i] = (T)(rand() % max + min);
}


Grid evalLinearGrid(int N,int threads) {
    Grid g{(N+threads - 1)/threads,threads};
    return g;
}

HyperGrid evalHyperGrid(int R,int C,int threadsPerSide) {
    dim3 threadsPerBlock(threadsPerSide,threadsPerSide);
    unsigned int rowBlocks = std::max(1u,(unsigned int)ceil((float)R / (float)threadsPerBlock.x)),
                 colBlocks = std::max(1u,(unsigned int)ceil((float)C / (float)threadsPerBlock.y));

    dim3 numBlocks(rowBlocks, colBlocks);
    #ifndef _VUTILS_SUPPRESS_
    std::cout << "Requesting grid for " << R << "x" << C << std::endl;
    std::cout << "Block size: " << threadsPerSide << "x" << threadsPerSide << std::endl;
    std::cout << "Block count" << rowBlocks << "x" << colBlocks << std::endl;
    #endif 
    HyperGrid g{numBlocks,threadsPerBlock};
    return g;
}

}