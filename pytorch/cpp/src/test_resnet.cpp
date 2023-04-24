
#include <iostream>
#include <vector>
#include <tuple>
#include <chrono>
#include <fstream>
#include <random>
#include <string>
#include <memory>

#include <torch/script.h>
#include <torch/serialize/tensor.h>
#include <torch/serialize.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>

#include "torchutils.h"
#include "opencvutils.h"

std::tuple<std::string, std::string>
infer(
  cv::Mat image,
  int image_height,
  int image_width,
  std::vector<double> mean,
  std::vector<double> std,
  std::vector<std::string> labels,
  torch::jit::script::Module model,
  bool usegpu)
{
  if (image.empty()) {
    std::cout << "WARNING: Cannot read image!" << std::endl;
  }

  std::string pred = "";
  std::string prob = "0.0";

  // Predict if image is not empty
  if (!image.empty()) {

    // Preprocess image
    image = preprocess(image, image_height, image_width,
      mean, std);

    // Forward
    std::vector<float> probs = forward({image, }, model, usegpu);

    // Postprocess
    tie(pred, prob) = postprocess(probs, labels);
  }

  return std::make_tuple(pred, prob);
}

int main(int argc, char **argv)
{

  namespace po = boost::program_options;

  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("use-gpu",     po::value<int>()->default_value(0), "use gpu option")
    ("image-file",  po::value<std::string>(),           "image file path")
    ("model-file",  po::value<std::string>(),           "model file path")
    ("labels-file", po::value<std::string>(),           "labels file path") ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
  }

  std::string image_path  = vm["image-file"].as<std::string>();
  std::string model_path  = vm["model-file"].as<std::string>();
  std::string labels_path = vm["labels-file"].as<std::string>();
  bool usegpu             = vm["use-gpu"].as<int>() == 1 ;

  int image_height = 224;
  int image_width = 224;

// Read labels
  std::vector<std::string> labels;
  std::string label;
  std::ifstream labelsfile (labels_path);
  if (labelsfile.is_open())
  {
    while (getline(labelsfile, label))
    {
      labels.push_back(label);
    }
    labelsfile.close();
  }

  std::vector<double> mean = {0.485, 0.456, 0.406};
  std::vector<double> std = {0.229, 0.224, 0.225};

  cv::Mat image = cv::imread(image_path);
  torch::jit::script::Module model = read_model(model_path, usegpu);

  std::string pred, prob;
  tie(pred, prob) = infer(image, image_height, image_width, mean, std, labels, model, usegpu);

  std::cout << "PREDICTION  : " << pred << std::endl;
  std::cout << "CONFIDENCE  : " << prob << std::endl;

  return 0;
}