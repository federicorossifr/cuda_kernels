#include "vec_utils.h"
#include "chrono_utils.h"
#include <chrono>


__global__ void convKernel(float* a,float* f,float* o,int R,int C,int K,int stride=1) {
    // tId is the index of the destination element in c to be computed
    int tIdX = blockIdx.x * blockDim.x + threadIdx.x,
        tIdY = blockIdx.y * blockDim.y + threadIdx.y;

    int outRows = (R-K+1)/stride, outCols = (C-K+1)/stride,
        rowPin  = tIdX*stride, colPin = tIdY*stride;
    // Dot product: tIdx-th row of a times tIdy-th column of b
    // Common lenght equal to C
    if(tIdX >= outRows || tIdY >= outCols) return;

    float pinValue = 0;
    for(int i = 0; i < K; ++i)
        for(int j = 0; j < K; ++j) {
            pinValue += a[(rowPin+i)*C+(colPin+j)]*f[i*K+j];
        }
    
    o[tIdX*outCols+tIdY] = pinValue;
}

int main(int argc,char* argv[]) {
    int R = atoi(argv[1]),C = atoi(argv[2]), K = atoi(argv[3]), stride = 1;
    int outRows = (R-K+1)/stride, outCols = (C-K+1)/stride;        
    int sizeA = R*C*sizeof(float), 
        sizeF = K*K*sizeof(float),
        sizeO = outRows*outCols*sizeof(float);
    float* a = new float[R*C], *aD;
    float* f = new float[K*K], *fD;
    float* o = new float[outRows*outCols], *oD;
    vecutils::initMatrix(a,R,C,1,1);
    vecutils::printMatrix(a,R,C,'\t');

    vecutils::initMatrix(f,K,K,1,1);
    vecutils::printMatrix(f,K,K,'\t');

    cudaMalloc(&aD,sizeA);
    cudaMalloc(&fD,sizeF);
    cudaMalloc(&oD,sizeO);

    cudaMemcpy(aD,a,sizeA,cudaMemcpyHostToDevice);
    cudaMemcpy(fD,f,sizeF,cudaMemcpyHostToDevice);


    vecutils::HyperGrid hg = vecutils::evalHyperGrid(outRows,outCols,16);
    chronoIt({convKernel<<<hg.blocks,hg.threads>>>(aD,fD,oD,R,C,K);})

    cudaMemcpy(o,oD,sizeO,cudaMemcpyDeviceToHost);
    vecutils::printMatrix(o,outRows,outCols,'\t');

    return 0;
}