#include <iostream>
#include <complex>
#include <iostream>
#include <fstream>
#include <thrust/complex.h>

__device__
void mandelbrot(double x, double y, double bound, int maxIters, int *outArr, int idx) {
    using namespace std::complex_literals;
    using namespace thrust;

    // Check for value inside bulbs
    double p = (x - 0.25) * (x - 0.25) + y * y;
    if (p * (p + (x - 0.25)) <= 0.25 * y * y) {
        outArr[idx] = 0;
        return;
    }

    complex<double> c = complex<double>(x, y);
    complex<double> z;

    int i = 0;
    while (i < maxIters & norm(z) < bound * bound) {
        z = z * z + c;
        i++;
    }

    if (i == maxIters) {
        outArr[idx] = 0;
    } else {
        outArr[idx] = i;
    }
}

__global__
void mandelbrotRow(double y, double xmin, double stepSize, int numSteps, double bound, int maxIters, int *outArr,
                   int idxStart) {
    for (int j = 0; j < numSteps; ++j) {
        double x = xmin + stepSize * j;
        mandelbrot(x, y, bound, maxIters, outArr, idxStart + j);
    }
}

__global__
void
mandelbrotSingle(double xmin, double ymin, double stepSize, int numSteps, double bound, int maxIters, int *outArr) {
    int idxX = blockIdx.x * blockDim.x + threadIdx.x;
    int idxY = blockIdx.y * blockDim.y + threadIdx.y;
    if (idxX >= numSteps || idxY >= numSteps)
        return;

    double x = xmin + stepSize * idxX;
    double y = ymin + stepSize * idxY;
    int idx = numSteps * idxY + idxX;
    mandelbrot(x, y, bound, maxIters, outArr, idx);
}


void mandelbrotSquareArr(double centreX, double centreY, double size, int numSteps, double bound, int maxIters,
                         int *outArr) {
    double stepSize = size / numSteps;
    double xmin = centreX - size / 2;
    double ymin = centreY - size / 2;
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(std::ceil((double) numSteps / threadsPerBlock.x),
                   std::ceil((double) numSteps / threadsPerBlock.y));


    mandelbrotSingle << < numBlocks, threadsPerBlock >> > (xmin, ymin, stepSize, numSteps, bound, maxIters, outArr);
    cudaDeviceSynchronize();

}

void writeArrToFile(int width, int height, int *outArr, const std::string &filename) {
    std::ofstream file;
    file.open(filename);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            file << outArr[i * height + j];
            if (j < width - 1)
                file << ",";
        }
        file << std::endl;
    }

    file.close();
}

void writeArrToCout(int width, int height, int *outArr) {
    using namespace std;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            cout << outArr[i * height + j];
            if (j < width - 1)
                cout << ",";
        }
        cout << endl;
    }

}


int main() {
    // Generate mandelbrot results
    int n = 100;
//    int *mandels = new int[n * n];
    int *mandels;
    cudaMallocManaged(&mandels, n * n * sizeof(int));
//    mandelbrotSquareArr(0.02445758009307159, 0.6296933276014198, 0.000001, n, 2, 10000, mandels);
    mandelbrotSquareArr(-1.5636314569242658, 0.000016435977612209248, 0.00000000001, n, 2, 10000, mandels);
//    mandelbrotSquareArr(-0.5, 0, 2, n, 2, 250, mandels);

//    writeArrToFile(n, n, mandels, "mandelbrot.csv");
    writeArrToCout(n, n, mandels);


    cudaFree(mandels);

    return 0;
}