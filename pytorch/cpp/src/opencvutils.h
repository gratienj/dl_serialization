#pragma once
#include <iostream>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

cv::Mat preprocess(cv::Mat, int, int,
  std::vector<double>,
  std::vector<double>);