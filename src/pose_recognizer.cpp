include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ros_caffe/caffe.h>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <message_filters/time_synchronizer.h>
#include <string>
#include <utility>
#include <vector>
#include <stdlib.h>
#include <cudnn.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <rgbd_process/gpu_process.hu>

#include <rgbd_process/pose_recognizer.hpp>


const float FACTOR = 8.f;
const unsigned int MAX_PEAKS = 96u;
const float NMS_THRESHOLD = 0.f;
const std::vector<unsigned int> BODY_PART_PAIRS = {1, 2,
                                                   1, 5,
                                                   2, 3,
                                                   3, 4,
                                                   5, 6,
                                                   6, 7,
                                                   1, 8,
                                                   8, 9,
                                                   9, 10,
                                                   1, 11,
                                                   11, 12,
                                                   12, 13,
                                                   1, 0,
                                                   0, 14,
                                                   14, 16,
                                                   0, 15,
                                                   15, 17,
                                                   2, 16,
                                                   5, 17};
const std::vector<unsigned int> MAP_IDX = {31, 32,
                                           39, 40,
                                           33, 34,
                                           35, 36,
                                           41, 42,
                                           43, 44,
                                           19, 20,
                                           21, 22,
                                           23, 24,
                                           25, 26,
                                           27, 28,
                                           29, 30,
                                           47, 48,
                                           49, 50,
                                           53, 54,
                                           51, 52,
                                           55, 56,
                                           37, 38,
                                           45, 46};
const unsigned int NUMBER_BODY_PARTS = 18u;
  

PoseRecognizer::PoseRecognizer(const std::string& model_file, const std::string& trained_file,
                               const std::string& img_topic)
  : it_(nh_)
    //color_image_sub_(nh_, img_topic, 1)
    //sync_(color_image_sub_, color_info_suredo vimb_, depth_image_sub_, 10)
{
  //Gets the pose from an image

  //sync_.registerCallback(boost::bind(&PoseRecognizer::matchColorDepth, this, _1, _2, _3));
  //color_image_pub_ = it_.advertise("matched_camera/color/image_raw", 1);
  //color_info_pub_ = it_.advertise("matched_camera/color/camera_info", 1);
  #ifdef CPU_ONLY
    ROS_ERROR("CPUs are tooooo slow... so GPUs only please :)");
  #else
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
  #endif

  num_channels_ = 3;
  net_sp_.reset(new caffe::Net<float>(model_file, caffe::TEST));
  net_sp_->CopyTrainedLayersFrom(trained_file);

  image_sub_ = nh_.subscribe(img_topic, 10, &PoseRecognizer::recognize, this);
}

void PoseRecognizer::recognize(const sensor_msgs::ImageConstPtr&
                                     color_image_data)
{
  const long MAX_AREA = 150000;
  cv::Mat data_img = cv_bridge::toCvCopy(color_image_data,
                                         sensor_msgs::image_encodings::RGB8)->image;
  cv::Size data_size = data_img.size();
  unsigned long data_area = data_size.area();
  if(data_area > MAX_AREA)
  {
    float shrink_factor = std::sqrt(float(MAX_AREA)/data_area);
    cv::resize(data_img, data_img, cv::Size(), shrink_factor, shrink_factor, CV_INTER_AREA);
  }
  PoseRecognizer::poseExtract(data_img);
}

void PoseRecognizer::poseExtract(const cv::Mat& img)
{
  ROS_INFO("Processing...");
  
  input_geometry_ = img.size();
  
/*---------------------------------------*/
//Seting up input and output layer

  boost::shared_ptr<caffe::Blob<float> > input_bsp = net_sp_->blobs()[0];
  input_bsp->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);

  net_sp_->Reshape();
  
  boost::shared_ptr<caffe::Blob<float> > output_bsp = net_sp_->blob_by_name("net_output");
  std::vector<int> output_shape = output_bsp->shape();

/*---------------------------------------*/
//Setting up heat maps layer

  std::shared_ptr<caffe::Blob<float> > heat_maps_sp = {std::make_shared<caffe::Blob<float> >
                                                       (1, 1, 1, 1)};
  std::vector<int> heat_maps_shape{output_shape};
  heat_maps_shape[0] = 1;
  heat_maps_shape[2] = heat_maps_shape[2]*FACTOR;
  heat_maps_shape[3] = heat_maps_shape[3]*FACTOR;
  heat_maps_sp->Reshape(heat_maps_shape);
  ROS_INFO("%d %d %d", output_shape[1], output_shape[2], output_shape[3]);

/*---------------------------------------*/
//Setting up peak map layer

  std::shared_ptr<caffe::Blob<int> > nms_sp = {std::make_shared<caffe::Blob<int> >                     //non-maximum suppression
                                                 (1, 1, 1, 1)};
  nms_sp->Reshape(heat_maps_shape);

/*---------------------------------------*/
//Setting up peak info layer

  std::shared_ptr<caffe::Blob<float> > peaks_sp = {std::make_shared<caffe::Blob<float> >
                                                   (1, 1, 1, 1)};
  std::vector<int> peaks_shape{heat_maps_shape};
  peaks_shape[1] = peaks_shape[1] - 1;
  peaks_shape[2] = MAX_PEAKS + 1;
  peaks_shape[3] = 3;
  peaks_sp->Reshape(peaks_shape);

/*---------------------------------------*/
//Setting up pose info layer
 
  std::shared_ptr<caffe::Blob<float> > pose_sp = {std::make_shared<caffe::Blob<float> >
                                                  (1, 1, 1, 1)};
  const int body_part_num = peaks_shape[1];
  pose_sp->Reshape({1, MAX_PEAKS, body_part_num, 3});

  float* input_gpu_data = input_bsp->mutable_gpu_data();

  unsigned long input_vol = input_bsp->count();
  float image_data[input_vol];
  PoseRecognizer::uCharCvMatToFloatPtr(image_data, img, true);

  cudaMemcpy(input_gpu_data, image_data,
             input_vol*sizeof(float),
             cudaMemcpyHostToDevice);

/*---------------------------------------*/
//Running neural net to get heat maps and vector fields

  net_sp_->ForwardFrom(0);

/*---------------------------------------*/
//GPUPU resizing 

  gproc::resizeAndMerge(heat_maps_sp->mutable_gpu_data(), output_bsp->gpu_data(),
                        heat_maps_shape[3], heat_maps_shape[2],
                        output_shape[3], output_shape[2], output_shape[1]);
  
/*---------------------------------------*/

  gproc::findPeaks(peaks_sp->mutable_gpu_data(), nms_sp->mutable_gpu_data(),
                   heat_maps_sp->gpu_data(), heat_maps_shape[3],
                   heat_maps_shape[2], peaks_shape[1], MAX_PEAKS, NMS_THRESHOLD);

/*---------------------------------------*/

//  const unsigned int num_body_part_pairs = BODY_PART_PAIRS.size()/2;

//  std::vector<std::pair<std::vector

}

unsigned long PoseRecognizer::shapeSize(const std::vector<int>& shape, const int& unit_size)
{
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>())*unit_size;
}

void PoseRecognizer::uCharCvMatToFloatPtr(float* float_image, const cv::Mat& cv_image,
                                          const bool normalize)
{ 
  //float* (deep net format): C x H x W
  //cv::Mat (OpenCV format): H x W x C
  const int width = cv_image.cols;
  const int height = cv_image.rows;
  const int channels = cv_image.channels();

  const uchar* origin_frame_ptr = cv_image.data;
  for(int c = 0; c < channels; c++)
  {
    const int float_image_offset_c = c*height;
    for(int y = 0; y < height; y++)
    {
      const int float_image_offset_y = (float_image_offset_c + y)*width;
      const int origin_frame_ptr_offset_y = y*width;
      for(int x = 0; x < width; x++)
      {
        float_image[float_image_offset_y + x] = float(origin_frame_ptr[
                                                        (origin_frame_ptr_offset_y + x)*
                                                         channels + c]);
      }
    }
  }

  if(normalize)
  {
    cv::Mat float_image_cv_wrapper{height, width, CV_32FC3, float_image};
    float_image_cv_wrapper = float_image_cv_wrapper/256.f - 0.5f;
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "pose_recognizer");
  const std::string ROOT_SAMPLE = ros::package::getPath("rgbd_process");
  std::string model_path = ROOT_SAMPLE + "/models/pose_deploy_linevec.prototxt";
  std::string weights_path = ROOT_SAMPLE + "/models/pose_iter_440000.caffemodel";
  PoseRecognizer pr(model_path, weights_path, "camera/color/image_raw");
  ros::spin();
  return 0;
}

