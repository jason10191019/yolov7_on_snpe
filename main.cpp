#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <typeinfo>
#include <numeric>
#include <math.h>
#include <cmath>
#include <chrono>
#include <unordered_map>

#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DlSystem/DlEnums.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlContainer/IDlContainer.hpp"

std::vector<std::vector<float>> sigmoid(std::vector<std::vector<float>> in){
    for(size_t i=0;i<in.size();i++){
        for(size_t j=0;j<in[i].size();j++){
            float x = in[i][j];
            in[i][j]=1.0/(1.0+exp(-static_cast<double>(x)));
        }
    }
    float strides[3] = {32, 8, 16};
    float anchorGrid[][6] = {
        {142,110, 192,243, 459,401},  // 32*32
        {12,16, 19,36, 40,28},       // 8*8
        {36,75, 76,55, 72,146},      // 16*16
    };

    // copy all outputs to one array.
    // [80 * 80 * 3 * 85]----\
    // [40 * 40 * 3 * 85]--------> [25200 * 85]
    // [20 * 20 * 3 * 85]----/
    float size[3]={20,80,40};
    for (size_t i = 0; i < 3; i++) {
        int height = size[i];
        int width = size[i];
        for (int j = 0; j < height; j++) {      // 80/40/20
            for (int k = 0; k < width; k++) {   // 80/40/20
                int anchorIdx = 0;
                for (int l = 0; l < 3; l++) {   // 3
                    int index = 3*(size[i]*j+k)+l;
                    if(i==1){
                        index+=1200;
                    }else if(i==2){
                        index+=20400;
                    }

                    for (int m = 0; m < 4; m++) {     // 85
                        if (m < 2) {
                            float value = in[index][m];
                            float gridValue = m == 0 ? k : j;
                            in[index][m] = (value * 2 - 0.5 + gridValue) * strides[i];
                        } else if (m < 4) {
                            float value = in[index][m];
                            in[index][m] = value * value * 4 * anchorGrid[i][anchorIdx++];
                        } 
                    }
                }
            }
        }
    }
    return in;
}

float calculateIoU(std::vector<float> input1, std::vector<float> input2) {
    // 计算边界框的坐标
    if(input1[5]!=input2[5]){
        return 0;
    }
    float x1 = std::max(input1[0] - input1[2]/2 , input2[0] - input2[2]/2);
    float y1 = std::max(input1[1] - input1[3]/2 , input2[1] - input2[3]/2);
    float x2 = std::min(input1[0] + input1[2]/2 , input2[0] + input2[2]/2);
    float y2 = std::min(input1[1] + input1[3]/2 , input2[1] + input2[3]/2);

    // 计算交集区域的面积
    float intersectionArea = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);

    // 计算并集区域的面积
    float unionArea = input1[2] * input1[3] + input2[2] * input2[3] - intersectionArea;

    // 计算 IoU
    return intersectionArea / unionArea;
}
bool compareByFifthElement(const std::vector<float>& a, const std::vector<float>& b) {
    return a[4] > b[4];
}
std::vector<std::vector<float>> nms(std::vector<std::vector<float>> input,float thres){
    std::sort(input.begin(), input.end(),compareByFifthElement);
    std::vector<bool> flag(input.size(), false);
    for(size_t i=0;i<input.size();i++) {
        if (flag[i]) continue;
        for (unsigned int j = i + 1; j < input.size(); j++) {
            if (calculateIoU(input[i], input[j]) > thres) flag[j] = true;
        }
    }
    std::vector<std::vector<float>> output;
    for (size_t i = 0; i < input.size(); i++) {
        if (!flag[i]) 
            output.push_back(input[i]);
    }
    return output;

}

zdl::DlSystem::Runtime_t checkRuntime()
{
    static zdl::DlSystem::Version_t Version = zdl::SNPE::SNPEFactory::getLibraryVersion();
    static zdl::DlSystem::Runtime_t Runtime;
    if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::GPU)) {
        Runtime = zdl::DlSystem::Runtime_t::GPU;
        std::cout<<"using GPU"<<std::endl;
    } else {
        Runtime = zdl::DlSystem::Runtime_t::CPU;
        std::cout<<"using CPU"<<std::endl;
    }
    return Runtime;
}

std::unique_ptr<zdl::SNPE::SNPE> initializeSNPE(zdl::DlSystem::Runtime_t runtime) {
    zdl::DlSystem::StringList outputLayers = {};
    outputLayers.append("Conv_134");
    outputLayers.append("Conv_148");
    outputLayers.append("Conv_162");
    /*outputLayers.append("Conv_296");
    outputLayers.append("Conv_310");
    outputLayers.append("Conv_324");*/
    std::unique_ptr<zdl::DlContainer::IDlContainer> container;
    container = zdl::DlContainer::IDlContainer::open("/root/models/yolo_new_640x640.dlc");   
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
    std::unique_ptr<zdl::SNPE::SNPE> snpe = snpeBuilder.setOutputLayers(outputLayers)
                      .setRuntimeProcessor(runtime)
                      .setCPUFallbackMode(true)
                      .setUseUserSuppliedBuffers(false)
                      .setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE)
                      .build();                  
  return snpe;
}

std::unique_ptr<zdl::DlSystem::ITensor> loadInputTensor(std::unique_ptr<zdl::SNPE::SNPE> &snpe, std::vector<float> inputVec) {
  std::unique_ptr<zdl::DlSystem::ITensor> input;
  const auto &strList_opt = snpe->getInputTensorNames();
  if (!strList_opt) throw std::runtime_error("Error obtaining Input tensor names");
  const auto &strList = *strList_opt;

  const auto &inputDims_opt = snpe->getInputDimensions(strList.at(0));
  const auto &inputShape = *inputDims_opt;
  input = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);

  std::copy(inputVec.begin(), inputVec.end(), input->begin());
  return input;
}

std::vector<float> executeNetwork(std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                    zdl::DlSystem::TensorMap inputTensorMap) {
  static zdl::DlSystem::TensorMap outputTensorMap;
  snpe->execute(inputTensorMap, outputTensorMap);
  std::vector<float> output;
  zdl::DlSystem::StringList tensorNames = outputTensorMap.getTensorNames();
  std::for_each( tensorNames.begin(), tensorNames.end(), [&](const char* name)
    {
        auto tensorPtr = outputTensorMap.getTensor(name);
        for ( auto it = tensorPtr->cbegin(); it != tensorPtr->cend(); ++it )
        {
            float f = *it;
            output.push_back(f);
        }
    });

    return output;
}

int main() {
    //class 
    std::vector<std::string> class_names = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
         "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
         "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
         "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
         "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
         "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
         "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
         "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
         "hair drier", "toothbrush" };
    std::unordered_map<int, std::string> ClassMap;
    for (int i = 5; i <= 84; ++i) {
        ClassMap[i] = class_names[i - 5];
    }

    //load_pic
    std::string image_path = "/root/models/picture4.jpg";
    cv::Mat image = cv::imread(image_path);

    //resize
    float width = image.cols;
    float height = image.rows;
    float down_width = 640;
    float down_height = 640;
    float scale = std::min(down_width/width,down_height/height);
    width*=scale;
    height*=scale;
    cv::Mat image_new;
    resize(image, image_new, cv::Size(width, height), cv::INTER_LINEAR);

    //push_input_vector
    std::vector<float> pixel;
    for (float y = 0; y < down_width; y++) {
        for (float x = 0; x < down_height; x++) {
            cv::Vec3b pixel_value = image_new.at<cv::Vec3b>(y, x);
            float red   = static_cast<float>(pixel_value[2]) / 255.0f;
            float green = static_cast<float>(pixel_value[1]) / 255.0f;
            float blue  = static_cast<float>(pixel_value[0]) / 255.0f;
            pixel.push_back(red);
            pixel.push_back(green);
            pixel.push_back(blue);
        }
    }

    //建立snpe model
    zdl::DlSystem::Runtime_t runt=checkRuntime();
    std::unique_ptr<zdl::SNPE::SNPE> snpe = initializeSNPE(runt);

    //建立input_tensor
    std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = loadInputTensor(snpe, pixel);
    static zdl::DlSystem::TensorMap inputTensorMap;
    const auto strList = snpe->getInputTensorNames();
    inputTensorMap.add((*strList).at(0), inputTensor.get());

    //execute
    std::vector<float> output = executeNetwork(snpe, inputTensorMap);

    //整理出box
    int groupsize = 85;
    int group = output.size()/groupsize;
    std::vector<std::vector<float>> box(group, std::vector<float>(groupsize));
    for(size_t i=0;i<output.size();i++){
        int x = i/groupsize;
        int y = i%groupsize;
        box[x][y] = output[i];
    }

    //處理model預測出的data
    box = sigmoid(box);
    
    //進行篩選
    std::vector<std::vector<float>> cond;
    for(size_t i=0;i<box.size();i++){
        if(box[i][4]>0.3){
            float large=0;
            int large_id=5;
            for(int j=5;j<85;j++){
                if(box[i][j]>large){
                    large=box[i][j];
                    large_id=j;
                }
            }
            if(box[i][4]*box[i][large_id] > 0.3){
                std::vector<float> temp{
                    box[i][0],
                    box[i][1],
                    box[i][2],
                    box[i][3],
                    box[i][4],
                    static_cast<float>(large_id)
                };
                cond.push_back(temp);
            }
        }
    }
    std::vector<std::vector<float>> result = nms(cond,0.5);

    //畫圖
    cv::Scalar color(0, 255, 0);
    for(size_t i=0;i<result.size();i++){
        cv::Rect bbox;
        bbox.width=result[i][2];
        bbox.height=result[i][3];
        bbox.x=result[i][0]-bbox.width/2;
        bbox.y=result[i][1]-bbox.height/2;
        cv::rectangle(image_new, bbox, color, 2);  // 2 是边框的宽度
        stream << std::fixed << std::setprecision(3) << result[i][4];
        std::string roundedString = stream.str();
        std::string className = ClassMap[result[i][5]]+roundedString;
        cv::putText(image_new, className, cv::Point(bbox.x, bbox.y -5),
                    cv::FONT_HERSHEY_SIMPLEX, 1 , cv::Scalar(0, 255, 0), 2 , cv::LINE_AA );
    }

    //resize回原本大小
    width/=scale;
    height/=scale;
    resize(image_new, image_new, cv::Size(width, height), cv::INTER_LINEAR);
    cv::imwrite("/root/models/picture_detection4.jpg", image_new);

    return 0;
}
