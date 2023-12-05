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
    float strides[3] = {32, 8, 16};
    float anchorGrid[][6] = {
        {116,90, 156,198, 373,326},  // 32*32
        {10,13, 16,30, 33,23},       // 8*8
        {30,61, 62,45, 59,119},      // 16*16
    };

    // copy all outputs to one array.
    // [80 * 80 * 3 * 85]----\
    // [40 * 40 * 3 * 85]--------> [25200 * 85]
    // [20 * 20 * 3 * 85]----/
    float size[3]={13,52,26};
    for (size_t i = 0; i < 3; i++) {
        int height = size[i];
        int width = size[i];
        for (int j = 0; j < height; j++) {      // 80/40/20
            for (int k = 0; k < width; k++) {   // 80/40/20
                int anchorIdx = 0;
                for (int l = 0; l < 3; l++) {   // 3
                    int index = 3*(size[i]*j+k)+l;
                    if(i==1){
                        index+=507;  //507 1200
                    }else if(i==2){
                        index+=8619;  //8619 20400
                    }

                    for (int m = 0; m < 14; m++) {     // 28 85
                        if (m < 2) {
                            float value = in[index][m];
                            value = 1.0/(1.0+exp(static_cast<double>(-value)));
                            float gridValue = m == 0 ? k : j;
                            in[index][m] = (value * 2 - 0.5 + gridValue) * strides[i];
                        } else if (m < 4) {
                            float value = in[index][m];
                            value = 1.0/(1.0+exp(static_cast<double>(-value)));
                            in[index][m] = value * value * 4 * anchorGrid[i][anchorIdx++];
                        } else {
                            in[index][m]=1.0/(1.0+exp(static_cast<double>(-in[index][m])));
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
    
    /*outputLayers.append("Conv_198");
    outputLayers.append("Conv_232");
    outputLayers.append("Conv_266");*/

    outputLayers.append("Conv_134");
    outputLayers.append("Conv_148");
    outputLayers.append("Conv_162");
    
    /*outputLayers.append("Conv_296");
    outputLayers.append("Conv_310");
    outputLayers.append("Conv_324");*/
    std::unique_ptr<zdl::DlContainer::IDlContainer> container;
    container = zdl::DlContainer::IDlContainer::open("/root/models/9classes.dlc");   
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
        std::cout<<name<<std::endl;
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
    std::vector<std::string> class_names = {"person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "traffic light",
         "stop sign" };
    std::unordered_map<int, std::string> ClassMap;
    for (int i = 5; i <= 13; ++i) {
        ClassMap[i] = class_names[i - 5];
    }

    //load_pic
    std::string image_path = "/root/models/1212.jpg";
    cv::Mat image = cv::imread(image_path);

    //resize
    float width = image.cols;
    float height = image.rows;
    float down_width = 416;
    float down_height = 416;
    float scale = std::min(down_width/width,down_height/height);
    width*=scale;
    height*=scale;
    cv::Mat image_new;
    resize(image, image_new, cv::Size(width, height), cv::INTER_LINEAR);

    //push_input_vector
    /*cv::dnn::blobFromImage(
        input_image,           // Input image
        blob,                  // Output blob
        1./255.,               // Scale factor (divides pixel values by this value)
        cv::Size(INPUT_WIDTH, INPUT_HEIGHT),  // Size of the output blob
        cv::Scalar(),          // Mean subtraction (subtract this mean value from each channel)
        true,                  // Swap RB channels (OpenCV uses BGR order, but networks often expect RGB)
        false                  // Indicate if the input image is in BGR order (true) or RGB order (false)
    );*/
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
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<float> output = executeNetwork(snpe, inputTensorMap);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "img size:416x416 經過的時間：" << elapsed_time.count() << " 毫秒" << std::endl;

    //整理出box
    int groupsize = 14;
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
            int large_id=0;
            for(int j=5;j<14;j++){
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
                    static_cast<float>(large_id),
                    box[i][large_id],
                    static_cast<float>(i)
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
        int cf = result[i][4]*100;
        std::string className = ClassMap[result[i][5]]+std::to_string(cf)+"%";
        cv::putText(image_new, className, cv::Point(bbox.x, bbox.y -5),
                    cv::FONT_HERSHEY_SIMPLEX, 1 , cv::Scalar(0, 255, 0), 2 , cv::LINE_AA );
    }
        cv::putText(image_new, std::to_string(result.size()),cv::Point (10,30),
                    cv::FONT_HERSHEY_SIMPLEX, 1 , cv::Scalar(0, 255, 0), 2 , cv::LINE_AA );
    //resize回原本大小
    width/=scale;
    height/=scale;
    resize(image_new, image_new, cv::Size(width, height), cv::INTER_LINEAR);
    cv::imwrite("/root/models/1234.jpg", image_new);

    return 0;
}
