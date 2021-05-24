#include "vec_utils.h"



__global__ void vecAdd(float* a,float* b,float* dst,int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < N)
        dst[tid] = a[tid]+b[tid];
}

int main(int argc,char* argv[]) {
    int N = atoi(argv[1]);
    int size = N*sizeof(float);
    float* a = new float[N], *aD;
    float* b = new float[N], *bD;
    float* c = new float[N], *cD;
    vecutils::initVect(a,N,0,100);
    vecutils::printVector(a,N,'\t');
    vecutils::initVect(b,N,0,100);
    vecutils::printVector(b,N,'\t');


    cudaMalloc(&aD,size);
    cudaMalloc(&bD,size);
    cudaMalloc(&cD,size);

    cudaMemcpy(aD,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(bD,b,size,cudaMemcpyHostToDevice);

    vecutils::Grid dGrid = vecutils::evalLinearGrid(N,256);

    vecAdd<<<dGrid.blocks,dGrid.threads>>>(aD,bD,cD,N);

    cudaMemcpy(c,cD,size,cudaMemcpyDeviceToHost);
    vecutils::printVector(c,N,'\t');
    return 0;
}