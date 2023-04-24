#include <iostream>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "cppflow/cppflow.h"


int main(int argc, char **argv) {

    std::string image_path = argv[1];
    std::string model_path = argv[2];
    std::string label_path = argv[3];

    auto input = cppflow::decode_jpeg(cppflow::read_file(image_path));
    input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
    input = cppflow::expand_dims(input, 0);
    cppflow::model model(model_path);
    auto output = model(input);

    int label_id = cppflow::arg_max(output, 1).get_data<int64_t>()[0] ;
    std::cout << "ImageNet class  id : " << label_id << std::endl;
    {
       // Short alias for this namespace
       namespace pt = boost::property_tree;

      // Create a root
      pt::ptree root;

      // Load the json file in this ptree
      pt::read_json(label_path, root);
      std::cout << "ImageNet class : ";
      for (pt::ptree::value_type &label : root.get_child(std::to_string(label_id)))
      {
         std::cout<<label.second.data()<<" " ;
      }
      std::cout<<std::endl ;
    }


    return 0;
}
