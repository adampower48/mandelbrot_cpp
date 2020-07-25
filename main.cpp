#include <iostream>
#include <complex>
#include <fstream>
#include <opencv2/opencv.hpp>

void displayImage(int height, int width, const int *data) {
    using namespace cv;
    int *cycledData = new int[height * width];
    for (int i = 0; i < height * width; ++i) {
        cycledData[i] = data[i] % 256;
    }


    Mat img(height, width, CV_32S, cycledData), mask;

    img.convertTo(mask, CV_8U);
    normalize(img, img, 0, 255, NORM_MINMAX, CV_32S, mask);

    img.convertTo(img, CV_8U);
    applyColorMap(img, img, COLORMAP_TWILIGHT_SHIFTED);

    Mat img2(1080, 1080, CV_8U);
    resize(img, img2, img2.size());

    namedWindow("Display window", WINDOW_AUTOSIZE);
    imshow("Display window", img2);
    waitKey(0);
}

int mandelbrot(double x, double y, double bound, int maxIters) {
    using namespace std::complex_literals;

    // Check for value inside bulbs
    double p = (x - 0.25) * (x - 0.25) + y * y;
    if (p * (p + (x - 0.25)) <= 0.25 * y * y)
        return 0;


    // Check for value inside 2nd bulb
    if ((x + 1) * (x + 1) + y * y < 1. / 16)
        return 0;


    std::complex<double> c = x + y * 1i;
    std::complex<double> z;

    int i = 0;
    while (i < maxIters & std::norm(z) < bound * bound) {
        z = z * z + c;
        i++;
    }

    if (i == maxIters) {
        return 0;
    } else {
        return i;
    }
}

void mandelbrotSquareArr(double centreX, double centreY, double size, int numSteps, double bound, int maxIters,
                         int *outArr) {
    double stepSize = size / numSteps;
    double xmin = centreX - size / 2;
    double ymin = centreY - size / 2;

    double x, y;
    for (int i = 0; i < numSteps; ++i) {
        for (int j = 0; j < numSteps; ++j) {
            x = xmin + stepSize * j;
            y = ymin + stepSize * i;
            outArr[i * numSteps + j] = mandelbrot(x, y, bound, maxIters);
        }
    }
}

void mandelbrotInplace(int i, double bound, std::complex<double> *c, std::complex<double> *curVals, int *iters,
                       bool *active) {
    curVals[i] = curVals[i] * curVals[i] + c[i];
    iters[i]++;

    if (std::norm(curVals[i]) >= bound * bound) {
        active[i] = false;
    }
}

void multibrotInplace(int i, double power, double bound, std::complex<double> *c, std::complex<double> *curVals,
                      int *iters, bool *active) {
    curVals[i] = pow(curVals[i], power) + c[i];
    iters[i]++;

    if (std::norm(curVals[i]) >= bound * bound) {
        active[i] = false;
    }
}

void juliaInplace(int i, double bound, std::complex<double> c, std::complex<double> *curVals,
                  int *iters, bool *active) {
    curVals[i] = curVals[i] * curVals[i] + c;
    iters[i]++;

    if (std::norm(curVals[i]) >= bound * bound) {
        active[i] = false;
    }
}

void juliasetSquareArr(double centreX, double centreY, double radius, int numSteps, double bound, double c_real,
                       double c_im, int maxIters, int *outArr) {
    double stepSize = 2 * radius / numSteps;
    double xmin = centreX - radius;
    double ymin = centreY - radius;


    // Set up data arrays
    auto c = std::complex(c_real, c_im);
    auto *curVals = new std::complex<double>[numSteps * numSteps];
    bool *active = new bool[numSteps * numSteps];
    std::vector<int> activeIndices1;
    std::vector<int> activeIndices2;

    // Init data arrays
    double x, y;
    for (int i = 0; i < numSteps; ++i) {
        for (int j = 0; j < numSteps; ++j) {
            x = xmin + stepSize * j;
            y = ymin + stepSize * i;

            curVals[i * numSteps + j] = std::complex(x, y);
            active[i * numSteps + j] = true;
            activeIndices1.push_back(i * numSteps + j);
        }
    }

    // Main iteration loop
    for (int iter = 0; iter < maxIters; ++iter) {
        for (int i : activeIndices1) {
            juliaInplace(i, bound, c, curVals, outArr, active);

            if (active[i]) {
                activeIndices2.push_back(i);
            }
        }
        std::swap(activeIndices1, activeIndices2);
        activeIndices2.clear();
    }

    // Set still unknown values to 0
    for (int i : activeIndices1) {
        outArr[i] = 0;
    }

}

void mandelbrotSquareArr2(double centreX, double centreY, double radius, int numSteps, double bound, int maxIters,
                          int *outArr) {
    double stepSize = 2 * radius / numSteps;
    double xmin = centreX - radius;
    double ymin = centreY - radius;

    // Set up data arrays
    auto *c = new std::complex<double>[numSteps * numSteps];
    auto *curVals = new std::complex<double>[numSteps * numSteps];
    bool *active = new bool[numSteps * numSteps];
    std::vector<int> activeIndices1;
    std::vector<int> activeIndices2;

    // Init data arrays
    double x, y, p;
    for (int i = 0; i < numSteps; ++i) {
        for (int j = 0; j < numSteps; ++j) {
            x = xmin + stepSize * j;
            y = ymin + stepSize * i;

            c[i * numSteps + j] = std::complex(x, y);
            curVals[i * numSteps + j] = 0;
            active[i * numSteps + j] = true;
            activeIndices1.push_back(i * numSteps + j);

            // Check for main cardioid bulb
            p = (x - 0.25) * (x - 0.25) + y * y;
            if (p * (p + (x - 0.25)) <= 0.25 * y * y)
                active[i * numSteps + j] = false;

            // Check for 2nd cardioid bulb
            if ((x + 1) * (x + 1) + y * y < 1. / 16)
                active[i * numSteps + j] = false;

        }
    }

    // Main iteration loop
    for (int iter = 0; iter < maxIters; ++iter) {
        for (int i : activeIndices1) {
            mandelbrotInplace(i, bound, c, curVals, outArr, active);

            if (active[i]) {
                activeIndices2.push_back(i);
            }
        }
        std::swap(activeIndices1, activeIndices2);
        activeIndices2.clear();
    }

    // Set still unknown values to 0
    for (int i : activeIndices1) {
        outArr[i] = 0;
    }


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


int main(int argc, char **argv) {
    // Parse args
    std::string setType = argv[1];
    int imgSteps = std::stoi(argv[2]);
    int maxIters = std::stoi(argv[3]);
    double centreX = std::stod(argv[4]);
    double centreY = std::stod(argv[5]);
    double radius = std::stod(argv[6]);
    double bound = std::stod(argv[7]);

    // Calculate
    int *mandels = new int[imgSteps * imgSteps]();
    if (setType == "julia") {
        // Julia
        double juliaCReal = std::stod(argv[8]);
        double juliaRIm = std::stod(argv[9]);

        juliasetSquareArr(centreX, centreY, radius, imgSteps, bound, juliaCReal, juliaRIm, maxIters, mandels);
    } else {
        // Mandelbrot
        mandelbrotSquareArr2(centreX, centreY, radius, imgSteps, bound, maxIters, mandels);
    }

    // Output/show results
    writeArrToFile(imgSteps, imgSteps, mandels, "mandelbrot.csv");
    displayImage(imgSteps, imgSteps, mandels);


    return 0;
}