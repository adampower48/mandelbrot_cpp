#include <iostream>
#include <complex>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

int mandelbrot(double x, double y, double bound, int maxIters) {
    using namespace std::complex_literals;

    // Check for value inside bulbs
    double p = (x - 0.25) * (x - 0.25) + y * y;
    if (p * (p + (x - 0.25)) <= 0.25 * y * y)
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

void displayImage(int height, int width, int *data) {
    using namespace cv;
    Mat img(height, width, CV_32S, data), mask;
    img.convertTo(mask, CV_8U);
    normalize(img, img, 0, 255, NORM_MINMAX, CV_32S, mask);
    img.convertTo(img, CV_8U);
    applyColorMap(img, img, COLORMAP_VIRIDIS);

    namedWindow("Display window", WINDOW_AUTOSIZE);
    imshow("Display window", img);
    waitKey(0);
}

int main() {
    // Generate mandelbrot results
    int n = 1000;
    int *mandels = new int[n * n];
    mandelbrotSquareArr(0.02445758009307159, 0.6296933276014198, 0.00001, n, 2, 10000, mandels);

//    writeArrToFile(n, n, mandels, "mandelbrot.csv");
//    writeArrToCout(n, n, mandels);

    displayImage(n, n, mandels);



    return 0;
}