#include "vec_utils.h"
#include "chrono_utils.h"
#include <chrono>


__global__ void matMulKernel(float* a,float* b,float* c,int R,int C,int K) {
    // tId is the index of the destination element in c to be computed
    int tIdX = blockIdx.x * blockDim.x + threadIdx.x,
        tIdY = blockIdx.y * blockDim.y + threadIdx.y;

    // Dot product: tIdx-th row of a times tIdy-th column of b
    // Common lenght equal to C
    if(tIdX >= R || tIdY >= K) return;
    c[tIdX*K+tIdY] = 0;
    for(int i = 0; i < C; ++i)
        c[tIdX*K+tIdY] += a[tIdX*C+i]*b[i*K+tIdY];
}

int main(int argc,char* argv[]) {
    int R = atoi(argv[1]),C = atoi(argv[2]), K = atoi(argv[3]);
    int sizeA = R*C*sizeof(float), 
        sizeB = C*K*sizeof(float),
        sizeC = R*K*sizeof(float);
    
    float* a = new float[R*C], *aD;
    float* b = new float[C*K], *bD;
    float* c = new float[R*K], *cD;
    vecutils::initMatrix(a,R,C,1,2);
    vecutils::printMatrix(a,R,C,'\t');

    vecutils::initMatrix(b,C,K,1,2);
    vecutils::printMatrix(b,C,K,'\t');

    cudaMalloc(&aD,sizeA);
    cudaMalloc(&bD,sizeB);
    cudaMalloc(&cD,sizeC);

    cudaMemcpy(aD,a,sizeA,cudaMemcpyHostToDevice);
    cudaMemcpy(bD,b,sizeB,cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    unsigned int rowBlocks = std::max(1u,R / threadsPerBlock.x),
                 colBlocks = std::max(1u,K / threadsPerBlock.y);

    dim3 numBlocks(rowBlocks, colBlocks);
    vecutils::HyperGrid hg = vecutils::evalHyperGrid(R,K,16);
    chronoIt({matMulKernel<<<hg.blocks,hg.threads>>>(aD,bD,cD,R,C,K);})

    cudaMemcpy(c,cD,sizeC,cudaMemcpyDeviceToHost);
    vecutils::printMatrix(c,R,K,'\t');

    return 0;
}