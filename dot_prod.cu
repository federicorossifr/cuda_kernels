#include "vec_utils.h"

#define LOG_1(n) (((n) >= 2) ? 1 : 0)
#define LOG_2(n) (((n) >= 1<<2) ? (2 + LOG_1((n)>>2)) : LOG_1(n))
#define BLOCK_SIZE 256
#define REDO_TREE_HEIGHT LOG_2(BLOCK_SIZE)


__global__ void dot_prod(float* a,float* b,float* dst,int N) {
    int idx = threadIdx.x;
    int workers = blockDim.x, workPoint;
    // Shared memory area for partial products
    // This area is only shared inside the block
    __shared__ float partials[BLOCK_SIZE];

    // Shared memory area for partial sums
    __shared__ float scratchpad[BLOCK_SIZE/2];
    
    // Ptr collection for buffers at different reduction stages
    // Given 
    float* reductionPtrs[REDO_TREE_HEIGHT];

    // Init ptrs (ToDo: is there a better way to initialize them?)
    for(int i = 0; i < REDO_TREE_HEIGHT; i+=2) {
        reductionPtrs[i] = scratchpad;
        reductionPtrs[i+1] = partials;
    }

    // Evaluate the partial product at each thread
    partials[idx] = a[idx]*b[idx];

    // Start tree-depth reduction of partials
    // At most we need to go deep for 8 tree levels
    int l = 0;
    for(; l < REDO_TREE_HEIGHT; ++l) {
        
        // Only first half of current workers will execute
        // each stage
        workPoint = workers >> 1;

        // Take pin for the current buffer
        float* dstPin = reductionPtrs[l], *srcPin = (l > 0)?reductionPtrs[l-1]: partials;
        if(idx < workPoint) {
            // Each thread in a block computes sum between two subsequent elements
            dstPin[idx] = srcPin[2*idx]+srcPin[2*idx + 1];
        }
        __syncthreads();
        workers = workers >> 1;
    }

    // Each threadBlock has produced a single value from reduction
    // We can add them using atomicAdd
    // Performance wise it is not the better way to accomplish a reduction
    // However there are not too much additions, so serializing them can be ok
    // Only the first thread in each block will add the reduction
    if(threadIdx.x == 0)
        atomicAdd(dst, reductionPtrs[l-1][0]);
    __syncthreads();
    

}

int main(int argc,char* argv[]) {
    int N = atoi(argv[1]);
    int size = N*sizeof(float);
    float* a = new float[N], *aD;
    float* b = new float[N], *bD;
    float* c = new float(0), *cD;
    vecutils::initVect(a,N,0,100);
    vecutils::printVector(a,N,'\t');
    vecutils::initVect(b,N,0,100);
    vecutils::printVector(b,N,'\t');


    cudaMalloc(&aD,size);
    cudaMalloc(&bD,size);
    cudaMalloc(&cD,sizeof(float));

    cudaMemcpy(aD,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(bD,b,size,cudaMemcpyHostToDevice);

    vecutils::Grid dGrid = vecutils::evalLinearGrid(N,256);

    dot_prod<<<dGrid.blocks,dGrid.threads>>>(aD,bD,cD,N);

    cudaMemcpy(c,cD,sizeof(float),cudaMemcpyDeviceToHost);
    vecutils::printVector(c,1,'\t');
    return 0;
}