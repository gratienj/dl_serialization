#include "inference.h"

Inferator::Inferator(std::string model_path)
{
    std::string json_data = read_kernel_bias_and_jsondata(model_path);
    read_activations(json_data);
    // for debug purpose
    // for (uint i = 0; i < topology.size(); i++) {
    //     cout << "topology[" << i << "] = " << topology[i] << endl;
    //     //
    //     if (i > 0) {
    //         cout << "bias[" << i - 1 << "] = " << *bias[i-1] << endl;
    //         cout << "weights[" << i - 1 << "] = " << (*weights[i-1]).transpose() << endl;
    //     }
    //     cout << endl << endl;
    // }
}

Inferator::~Inferator() {
    if (norm_param_X != NULL) delete[] norm_param_X;
    if (norm_param_Y != NULL) delete[] norm_param_Y;
}

std::string Inferator::read_kernel_bias_and_jsondata(std::string path) {
    hid_t h5Model;
    //
    // Open .h5 file
    //
	try {
        h5Model = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    }
    catch (...) {
        std::string msg("Could not open HDF5 file containing NN\n");
        throw msg;
    }
    //
    // Read root attribute (name = model_config) to get activation functions
    //
    std::string attribute_name("model_config");
    hid_t attr, grp, dset, filetype, memtype;
    size_t sdim;
    try {
        attr = H5Aopen(h5Model,attribute_name.c_str(),H5P_DEFAULT);
        filetype = H5Aget_type(attr);
        sdim = H5Tget_size(filetype);
        sdim++;
        memtype = H5Tcopy(filetype);
        H5Tset_size(memtype,sdim);
    } catch (...) {
        std::string msg("Could not find \"model_config\" attribute in NN .h5 file\n");
        throw msg;
    }
    char** rdata = new char*[1];
    hid_t status = H5Aread(attr, filetype, (void*)rdata);
    std::string json_data(rdata[0]);
    delete[] rdata;
    //
    std::string grp_path = "/model_weights/dense/dense";
    grp = H5Gopen(h5Model,grp_path.c_str(),H5P_DEFAULT);

    dset = H5Dopen(grp,"kernel:0",H5P_DEFAULT);
    add_weight_from_dataset(dset);
    H5Dclose(dset);

    dset = H5Dopen(grp,"bias:0",H5P_DEFAULT);
    add_bias_from_dataset(dset);
    H5Dclose(dset);
    H5Gclose(grp);
    //
    H5Eset_auto(NULL, NULL, NULL);
    size_t nlayer = 1;
    grp_path = "/model_weights/dense_"+std::to_string(nlayer)+"/dense_"+std::to_string(nlayer);
    while (H5Lexists(h5Model,grp_path.c_str(),H5P_DEFAULT) > 0) {
        grp = H5Gopen(h5Model,grp_path.c_str(),H5P_DEFAULT);

        dset = H5Dopen(grp,"kernel:0",H5P_DEFAULT);
        add_weight_from_dataset(dset);
        H5Dclose(dset);

        dset = H5Dopen(grp,"bias:0",H5P_DEFAULT);
        add_bias_from_dataset(dset);
        H5Dclose(dset);
        H5Gclose(grp);

        nlayer+=1;
        grp_path = "/model_weights/dense_"+std::to_string(nlayer)+"/dense_"+std::to_string(nlayer);
    }
    //
    H5Aclose(attr);
    H5Tclose(filetype);
    H5Tclose(memtype);
    //
    return json_data;
}

void Inferator::read_activations(std::string data) {
    rapidjson::Document doc;
    std::stringstream content;
    doc.Parse(data.c_str());
    //
    try {
        bool check = doc["config"].HasMember("layers");
    }
    catch (...) {
        std::string msg("Could not find \"layers\" object in NN json attributes \n");
        throw msg;
    }
    rapidjson::Value& layers = doc["config"]["layers"];
    assert(layers.IsArray());
    for(size_t i = 0; i < layers.Size(); i++) {
        std::string class_name(layers[i]["class_name"].GetString());
        // cout << layers[i]["class_name"].GetString() << endl; // for debug purpose
        //
        if (class_name == "Dense") { // to be checked (might depend on NN generation method)
            std::string act_func(layers[i]["config"]["activation"].GetString());
            // cout << act_func << endl; // for debug purpose
            //
            if (act_func == "tanh") {
                activationFunctions.push_back(tanh);
            } else if (act_func == "relu") {
                activationFunctions.push_back(reLU);
            } else {
                activationFunctions.push_back(id);
            }
        }
    }
    // //
    // // For standalone testing purpose (activation function cost eval)
    // //
    // // activationFunctions.push_back(tanh);
    // activationFunctions.push_back(reLU);
    // // activationFunctions.push_back(id);
    // for (uint i = activationFunctions.size()-1; i < topology.size()-2; i++) {
    //         // activationFunctions.push_back(tanh);
    //         activationFunctions.push_back(reLU);
    //         // activationFunctions.push_back(id);
    // }
    // activationFunctions.push_back(id);

}

void Inferator::add_bias_from_dataset(hid_t h5_dset) {
    size_t i = topology.size();
    hid_t dspace = H5Dget_space(h5_dset);
    double* tmp = new double[topology[i-1]];
    H5Dread(h5_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,tmp);
    H5Sclose(dspace);
    
    bias.push_back(RowVector(topology[i-1]));
    bias[i-2] = Eigen::Map<RowVector>(tmp,topology[i-1]);
    
    delete[] tmp;
}

void Inferator::add_weight_from_dataset(hid_t h5_dset) {
    hid_t dspace = H5Dget_space(h5_dset);
    hsize_t ndim = H5Sget_simple_extent_ndims(dspace);
    hsize_t* dims = new hsize_t[ndim];
    H5Sget_simple_extent_dims(dspace,dims,NULL);

    size_t i = topology.size();
    if (i == 0) {
        topology.push_back(dims[0]);
        neuronLayers.push_back(RowVector(topology[i]));
        i++;
    }
    topology.push_back(dims[1]);
    neuronLayers.push_back(RowVector(topology[i]));

    double* tmp = new double[dims[0]*dims[1]];
    H5Dread(h5_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,tmp);
    H5Sclose(dspace);
    weights.push_back(Matrix(topology[i - 1], topology[i]));
    Matrix mat = Eigen::Map<Matrix>(tmp, topology[i], topology[i - 1]);
    weights[i-1] = mat.transpose();
    // weights[i-1] = Eigen::Map<Matrix>(tmp, topology[i - 1], topology[i]); // issue with Stride...
    //
    delete[] tmp;
    delete[] dims;
}

void Inferator::propagateForward(RowVector& input) {
    neuronLayers[0] = input; 
    // propagate the data forward and then
    // apply the activation function to your network
    for (uint i = 1; i < topology.size(); i++) {
        // already explained above
        neuronLayers[i] = neuronLayers[i - 1] * weights[i - 1] + bias[i-1];
        activationFunctions[i-1](neuronLayers[i]);
    }
}

double* Inferator::run_ai(double* input_ai)
{
    RowVector input_vector = Eigen::Map<RowVector>(input_ai,topology[0]); // DAK : check dans TotalView
    propagateForward(input_vector);
    return neuronLayers.back().data();
}

// Function to normalize input vector
double* Inferator::normalize_input(double* state_X)
{
    // Declare normalized state
    double *state_X_norm = new double[n_input_ai];

    for(int i = 0; i < n_input_ai; i++) state_X_norm[i] = (state_X[i]-norm_param_X[i][0])/(sqrt(norm_param_X[i][1]));

    return state_X_norm;
}

// Function to denormalize output vector
double* Inferator::denormalize_output(double* state_Y_norm)
{
    // Declare denormalized state
    double *state_Y = new double[n_output_ai];

    for(int i = 0; i < n_output_ai; i++) state_Y[i] = state_Y_norm[i]*sqrt(norm_param_Y[i][1]) + norm_param_Y[i][0];

    return state_Y;
}