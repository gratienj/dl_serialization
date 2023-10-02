

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>



#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#include <filesystem>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>

#include <exception>

#include "engine.h"


template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}


std::vector<std::string> readLabels(std::string& labelFilepath)
{
    std::vector<std::string> labels;
    std::string line;
    std::ifstream fp(labelFilepath);
    while (std::getline(fp, line))
    {
        labels.push_back(line);
    }
    return labels;
}





int main(int argc, char* argv[])
{
  namespace fs = std::filesystem;
  namespace po = boost::program_options;

  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
         ("help", "produce help message")
         ("use-gpu",    po::value<int>()->default_value(0), "use gpu option")
         ("data-dir",  po::value<std::string>(), "data dir path")
         ("model-file", po::value<std::string>(), "model file path")
         ("data-file",  po::value<std::string>(), "data file path")
         ("test-id",    po::value<int>()->default_value(0), "test id");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
  }
  bool useCUDA = vm["use-gpu"].as<int>()==1;
  const char* useCUDAFlag = "--use_cuda";
  const char* useCPUFlag = "--use_cpu";

  if (useCUDA)
  {
      std::cout << "Inference Execution Provider: CUDA" << std::endl;
  }
  else
  {
      std::cout << "Inference Execution Provider: CPU" << std::endl;
  }
  auto data_dir      = fs::path(vm["data-dir"].as<std::string>().c_str()) ;
  auto instanceName  = (data_dir / "image-classification-inference").string();
  auto modelFilepath = (data_dir / "models/squeezenet1.1-7.onnx").string();
  auto imageFilepath = (data_dir / "images/european-bee-eater-2115564_1920.jpg").string();
  auto labelFilepath = (data_dir / "labels/synset.txt").string();

  std::vector<std::string> labels{readLabels(labelFilepath)};


  // Specify our GPU inference configuration options
  Options options;
  // Specify what precision to use for inference
  // FP16 is approximately twice as fast as FP32.
  options.precision = Precision::FP16;
  // If using INT8 precision, must specify path to directory containing calibration data.
  options.calibrationDataDirectoryPath = "";
  // If the model does not support dynamic batch size, then the below two parameters must be set to 1.
  // Specify the batch size to optimize for.
  options.optBatchSize = 1;
  // Specify the maximum batch size we plan on running.
  options.maxBatchSize = 1;

  Engine engine(options);

  // Define our preprocessing code
  // The default Engine::build method will normalize values between [0.f, 1.f]
  // Setting the normalize flag to false will leave values between [0.f, 255.f] (some converted models may require this).

  // For our YoloV8 model, we need the values to be normalized between [0.f, 1.f] so we use the following params
  std::array<float, 3> subVals {0.f, 0.f, 0.f};
  std::array<float, 3> divVals {1.f, 1.f, 1.f};
  bool normalize = true;
  // Note, we could have also used the default values.

  // If the model requires values to be normalized between [-1.f, 1.f], use the following params:
  //    subVals = {0.5f, 0.5f, 0.5f};
  //    divVals = {0.5f, 0.5f, 0.5f};
  //    normalize = true;

  // Build the onnx model into a TensorRT engine file.
  bool succ = engine.build(modelFilepath, subVals, divVals, normalize);
  if (!succ) {
      throw std::runtime_error("Unable to build TRT engine.");
  }

  // Load the TensorRT engine file from disk
  succ = engine.loadNetwork();
  if (!succ) {
      throw std::runtime_error("Unable to load TRT engine.");
  }

  // Read the input image
  // TODO: You will need to read the input image required for your model
  const std::string inputImage = "../inputs/team.jpg";
  auto cpuImg = cv::imread(imageFilepath);
  if (cpuImg.empty()) {
      throw std::runtime_error("Unable to read image at path: " + inputImage);
  }

  // Upload the image GPU memory
  cv::cuda::GpuMat img;
  img.upload(cpuImg);

  // The model expects RGB input
  cv::cuda::cvtColor(img, img, cv::COLOR_BGR2RGB);

  // In the following section we populate the input vectors to later pass for inference
  const auto& inputDims = engine.getInputDims();
  std::vector<std::vector<cv::cuda::GpuMat>> inputs;

  // Let's use a batch size which matches that which we set the Options.optBatchSize option
  size_t batchSize = options.optBatchSize;

  // TODO:
  // For the sake of the demo, we will be feeding the same image to all the inputs
  // You should populate your inputs appropriately.
  for (const auto & inputDim : inputDims) { // For each of the model inputs...
      std::vector<cv::cuda::GpuMat> input;
      for (size_t j = 0; j < batchSize; ++j) { // For each element we want to add to the batch...
          // TODO:
          // You can choose to resize by scaling, adding padding, or a combination of the two in order to maintain the aspect ratio
          // You can use the Engine::resizeKeepAspectRatioPadRightBottom to resize to a square while maintain the aspect ratio (adds padding where necessary to achieve this).
          auto resized = Engine::resizeKeepAspectRatioPadRightBottom(img, inputDim.d[1], inputDim.d[2]);
          // You could also perform a resize operation without maintaining aspect ratio with the use of padding by using the following instead:
//            cv::cuda::resize(img, resized, cv::Size(inputDim.d[2], inputDim.d[1])); // TRT dims are (height, width) whereas OpenCV is (width, height)
          input.emplace_back(std::move(resized));
      }
      inputs.emplace_back(std::move(input));
  }

  // Warm up the network before we begin the benchmark
  std::cout << "\nWarming up the network..." << std::endl;
  std::vector<std::vector<std::vector<float>>> featureVectors;
  for (int i = 0; i < 100; ++i) {
      succ = engine.runInference(inputs, featureVectors);
      if (!succ) {
          throw std::runtime_error("Unable to run inference.");
      }
  }

  // Benchmark the inference time
  size_t numIterations = 1000;
  std::cout << "Warmup done. Running benchmarks (" << numIterations << " iterations)...\n" << std::endl;
  preciseStopwatch stopwatch;
  for (size_t i = 0; i < numIterations; ++i) {
      featureVectors.clear();
      engine.runInference(inputs, featureVectors);
  }
  auto totalElapsedTimeMs = stopwatch.elapsedTime<float, std::chrono::milliseconds>();
  auto avgElapsedTimeMs = totalElapsedTimeMs / numIterations / static_cast<float>(inputs[0].size());

  std::cout << "Benchmarking complete!" << std::endl;
  std::cout << "======================" << std::endl;
  std::cout << "Avg time per sample: " << std::endl;
  std::cout << avgElapsedTimeMs << " ms" << std::endl;
  std::cout << "Batch size: " << std::endl;
  std::cout << inputs[0].size() << std::endl;
  std::cout << "Avg FPS: " << std::endl;
  std::cout << static_cast<int>(1000 / avgElapsedTimeMs) << " fps" << std::endl;
  std::cout << "======================\n" << std::endl;

  // Print the feature vectors
  for (size_t batch = 0; batch < featureVectors.size(); ++batch) {
      for (size_t outputNum = 0; outputNum < featureVectors[batch].size(); ++outputNum) {
          std::cout << "Batch " << batch << ", " << "output " << outputNum << std::endl;
          int i = 0;
          for (const auto &e:  featureVectors[batch][outputNum]) {
              std::cout << e << " ";
              if (++i == 10) {
                  std::cout << "...";
                  break;
              }
          }
          std::cout << "\n" << std::endl;
      }
  }

  return 0 ;

  /*
  cv::Mat imageBGR = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);
  cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
  cv::resize(imageBGR, resizedImageBGR,
             cv::Size(inputDims.at(2), inputDims.at(3)),
             cv::InterpolationFlags::INTER_CUBIC);
  cv::cvtColor(resizedImageBGR, resizedImageRGB,
               cv::ColorConversionCodes::COLOR_BGR2RGB);
  resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);

  cv::Mat channels[3];
  cv::split(resizedImage, channels);
  // Normalization per channel
  // Normalization parameters obtained from
  // https://github.com/onnx/models/tree/master/vision/classification/squeezenet
  channels[0] = (channels[0] - 0.485) / 0.229;
  channels[1] = (channels[1] - 0.456) / 0.224;
  channels[2] = (channels[2] - 0.406) / 0.225;
  cv::merge(channels, 3, resizedImage);
  // HWC to CHW
  cv::dnn::blobFromImage(resizedImage, preprocessedImage);

  size_t inputTensorSize = vectorProduct(inputDims);
  std::vector<float> inputTensorValues(inputTensorSize);
  inputTensorValues.assign(preprocessedImage.begin<float>(),
                           preprocessedImage.end<float>());

  size_t outputTensorSize = vectorProduct(outputDims);
  assert(("Output tensor size should equal to the label set size.",
          labels.size() == outputTensorSize));
  std::vector<float> outputTensorValues(outputTensorSize);

  //std::vector<char const *> inputNames{*inputName};
  //std::vector<char const *> outputNames{*outputName};
  const char* input_names[] = {inputName.get()};
  char* output_names[] = {outputName.get()};
  */
}
