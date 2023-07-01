// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//   T. Shan and B. Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain
//      IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). October 2018.
#include "utility.h"
#include <std_srvs/Empty.h>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam/nonlinear/ISAM2.h>

// #include <ORB_SLAM2/wy_keyframe.h>
// #include <ORB_SLAM2/wy_keyframedb.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <visualization_msgs/Marker.h>
#include <fstream>

#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Path.h>

#include <Eigen/Dense>
#include <queue>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/ply_io.h>
#include <pcl/registration/ndt.h>

#include "LidarIris.h"
#include "tic_toc.h"

using namespace gtsam;

#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

// ground plane fit 杨航的代码部分
// #define PI 3.1415926535

// 类型定义
typedef pcl::PointXYZI PointT;
typedef pcl::PointCloud<PointT> PointCloud;

// For disable PCL complile lib, to use PointXYZIR    
#define PCL_NO_PRECOMPILE
// #include <pcl_conversions/pcl_conversions.h>

//Customed Point Struct for holding clustered points

/** Euclidean Velodyne coordinate, including intensity and ring number, and label. */
struct PointXYZIRL
{
    PCL_ADD_POINT4D;                    // quad-word XYZ
    float    intensity;                 ///< laser intensity reading
    uint16_t ring;                      ///< laser ring number
    uint16_t label;                     ///< point label
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW     // ensure proper alignment
} EIGEN_ALIGN16;


#define PointL PointXYZIRL
// #define VPoint velodyne_pointcloud::PointXYZIR
#define RUN pcl::PointCloud<PointL>
// Register custom point struct according to PCL
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRL,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, intensity, intensity)
                                  (uint16_t, ring, ring)
                                  (uint16_t, label, label))

// using eigen lib
#include <Eigen/Dense>
using Eigen::MatrixXf;
using Eigen::JacobiSVD;
using Eigen::VectorXf;
using namespace std;
using namespace cv;
class GroundPlaneFit{
public:
    void extract_road_from_submap_(pcl::PointCloud<PointL>::Ptr& originCloud){
        inputCloud = originCloud;
        int centerx = 4, centery = 0;
        double squareCenAxisx[] = {-6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6};
        double squareCenAxisy[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        // store point cloud index
        std::vector<vector<int>> MapGridArray;         
        MapGridArray.resize(9);
        // put point cloud into grid
        for (int i = 0; i < inputCloud->points.size(); i++)
        {
            for (int j = 0; j < 9; j++)
            {
                if (abs(inputCloud->points[i].x - squareCenAxisx[j]) <= 1 &&
                    abs(inputCloud->points[i].z - squareCenAxisy[j]) <= 1.5)
                {
                    MapGridArray[j].push_back(i);
                }
            }
        }
        cout << "size of grid map: ";
        for (int i = 0; i < 9; i++){
            cout << MapGridArray[i].size() << " ";
        }
        cout<< endl; //centery = 0
        extract_initial_seeds_from_dist_(MapGridArray[centerx]);
        groundCloud->clear();
        *groundCloud += *seedsCloud;
        fit_plane_(MapGridArray[centerx],1);
        // pcl::io::savePCDFileBinary("/home/yangh/Data/result/ground" + std::to_string(submapIdx) + ".pcd", *groundCloud);
        // pcl::io::savePCDFileBinary("/home/yangh/Data/result/not_ground" + std::to_string(submapIdx) + ".pcd", *notGroundCloud);
        int turn = 1;
        for (int i = 0; i < 2;i++) //两个方向
        {
            for (int j = 1; j <= 4; j++) //往其中一个方向延伸
            {
                int ind = centerx + j * turn;
                extract_initial_seeds_from_around_(MapGridArray[ind]);
                if(seedsCloud->points.size()<th_seed_num_)
                    break;
                groundCloud = seedsCloud;
                fit_plane_(MapGridArray[ind],j+2);
            }
            turn *= -1;
        }
        static int idx = 0;
        //pcl::io::savePCDFileBinary("/media/yangh/Backup/rosdata/dongshan/result/" + to_string(idx++) + ".pcd", *inputCloud);
    }
    GroundPlaneFit():
        inputCloud(new pcl::PointCloud<PointL>()),
        seedsCloud(new pcl::PointCloud<PointL>()),
        groundCloud(new pcl::PointCloud<PointL>()),
        notGroundCloud(new pcl::PointCloud<PointL>()),
        labeledCloud(new pcl::PointCloud<PointL>()),
        originPointCloud(new pcl::PointCloud<PointL>()),
        labelCloud(new pcl::PointCloud<PointL>())
    {

    }
private:
    int th_seed_num_ = 100;
    int sensor_model_ = 16;
    double sensor_height_ = 1.0;
    int num_seg_ = 1;
    int num_iter_ = 5;
    int num_lpr_ = 20;
    double th_seeds_ = 1.2;
    double th_dist_ = 0.02;
    float d_;
    MatrixXf normal_;
    float th_dist_d_;
    pcl::PointCloud<PointL>::Ptr inputCloud;
    pcl::PointCloud<PointL>::Ptr seedsCloud;
    pcl::PointCloud<PointL>::Ptr groundCloud ;
    pcl::PointCloud<PointL>::Ptr notGroundCloud;
    pcl::PointCloud<PointL>::Ptr labeledCloud;
    pcl::PointCloud<PointL>::Ptr originPointCloud;
    pcl::PointCloud<PointL>::Ptr labelCloud;
    void estimate_plane_()
    {
        // Create covarian matrix in single pass.
        // TODO: compare the efficiency.
        Eigen::Matrix3f cov;
        Eigen::Vector4f pc_mean;
        pcl::computeMeanAndCovarianceMatrix(*groundCloud, cov, pc_mean);
        // Singular Value Decomposition: SVD
        JacobiSVD<MatrixXf> svd(cov,Eigen::DecompositionOptions::ComputeFullU);
        // use the least singular vector as normal
        normal_ = (svd.matrixU().col(2));
        // mean ground seeds value
        Eigen::Vector3f seeds_mean = pc_mean.head<3>();
        // according to normal.T*[x,y,z] = -d
        d_ = -(normal_.transpose()*seeds_mean)(0,0);
        // set distance threshold to `th_dist - d`
        th_dist_d_ = th_dist_ - d_;
        th_dist_ = normal_.norm() * th_dist_;
        // return the equation parameters
    }
    void extract_initial_seeds_from_height_(const pcl::PointCloud<PointL>& p_sorted){
        // LPR is the mean of low point representative
        double sum = 0;
        int cnt = 0;
        // Calculate the mean height value.
        for(int i=0;i<p_sorted.points.size() && cnt<num_lpr_;i++){
            sum += p_sorted.points[i].z;
            cnt++;
        }
        double lpr_height = cnt!=0?sum/cnt:0;// in case divide by 0
        seedsCloud->clear();
        // iterate pointcloud, filter those height is less than lpr.height+th_seeds_
        for(int i=0;i<p_sorted.points.size();i++){
            if(p_sorted.points[i].z < lpr_height + th_seeds_){
                seedsCloud->points.push_back(p_sorted.points[i]);
            }
        }
        // return seeds points
    }
    void extract_initial_seeds_from_dist_(std::vector<int>& mapidx){
        seedsCloud->clear();
        for (int i = 0; i < mapidx.size();i++)
        {
            double dist = inputCloud->points[mapidx[i]].x * inputCloud->points[mapidx[i]].x
                        + inputCloud->points[mapidx[i]].z * inputCloud->points[mapidx[i]].z;
            if (dist<0.25)
            {
                seedsCloud->points.push_back(inputCloud->points[mapidx[i]]);
            }
        }
        cout << "initial seeds num: " << seedsCloud->points.size() << endl;
    }
    void extract_initial_seeds_from_around_(std::vector<int>& mapidx){
        seedsCloud->clear();
        for (int i = 0; i < mapidx.size(); i++)
        {
            if(inputCloud->points[mapidx[i]].label != 0)
            {
                seedsCloud->points.push_back(inputCloud->points[mapidx[i]]);
            }
        }
        cout << "around seeds num: " << seedsCloud->points.size() << endl;
    }
    // 5. Ground plane fitter mainloop
    void fit_plane_(std::vector<int>& mapidx,unsigned int label)
    {
        for(int i=0;i<num_iter_;i++){
            estimate_plane_();
            groundCloud->clear();
            notGroundCloud->clear();

            //pointcloud to matrix
            MatrixXf points(mapidx.size(),3);
            int j = 0;
            for (auto p : mapidx)
            {
                points.row(j++) << inputCloud->points[p].x, inputCloud->points[p].y, inputCloud->points[p].z;
            }
            // ground plane model
            VectorXf result = points * normal_;
            // threshold filter
            // cout << "th_dist_d_: " << th_dist_d_ << endl;
            for (int r = 0; r < result.rows(); r++)
            {
                if(result[r]<th_dist_d_)
                // if (result[r] + d_ < th_dist_ && result[r] + d_ > -th_dist_) //result[r] < th_dist_d_
                {
                    inputCloud->points[mapidx[r]].label = label;// means ground
                    groundCloud->points.push_back(inputCloud->points[mapidx[r]]);
                }else{
                    inputCloud->points[mapidx[r]].label = 0;// means not ground and non clusterred
                    notGroundCloud->points.push_back(inputCloud->points[mapidx[r]]);
                }
            }
            cout << "ground points num: " << groundCloud->points.size() << endl;
            cout << "not ground points num: " << notGroundCloud->points.size() << endl;
        }
    }
};

bool point_cmp(PointT a, PointT b){
    return a.z<b.z;
}




void fillHole(const Mat srcBw, Mat &dstBw, Point p)
{
	Size m_Size = srcBw.size();
	Mat Temp=Mat::zeros(m_Size.height+2,m_Size.width+2,srcBw.type());//延展图像
	srcBw.copyTo(Temp(cv::Range(1, m_Size.height + 1), cv::Range(1, m_Size.width + 1)));
 
	cv::floodFill(Temp, p, Scalar(255));
    // cv::dilate( Temp, Temp, cv::Mat());
 
	Mat cutImg;//裁剪延展的图像
	Temp(cv::Range(1, m_Size.height + 1), cv::Range(1, m_Size.width + 1)).copyTo(cutImg);
 
	dstBw = srcBw | (~cutImg);
}
bool has_suffix(const std::string &str, const std::string &suffix) {
  std::size_t index = str.find(suffix, str.size() - suffix.size());
  return (index != std::string::npos);
}






class mapOptimization{

private:
    int loop_id = 0;
    int testnum = 0;
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;

    noiseModel::Diagonal::shared_ptr priorNoise;
    noiseModel::Diagonal::shared_ptr odometryNoise;
    noiseModel::Diagonal::shared_ptr constraintNoise;
    gtsam::Vector6 LMScore;
    double LMNoise;

    ros::NodeHandle nh;

    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubOdomAftMapped;
    ros::Publisher pubKeyPoses;

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubRegisteredCloud;

    ros::Subscriber subLaserCloudCornerLast;
    ros::Subscriber subLaserCloudSurfLast;
    ros::Subscriber subOutlierCloudLast;
    ros::Subscriber subLaserOdometry;
    ros::Subscriber subImu;

    ros::Subscriber subVelodyne;
    pcl::PointCloud<pcl::PointXYZ>::Ptr velodyne_points;

    nav_msgs::Odometry odomAftMapped;
    tf::StampedTransform aftMappedTrans;
    tf::TransformBroadcaster tfBroadcaster;

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> outlierCloudKeyFrames;

    deque<pcl::PointCloud<PointType>::Ptr> recentCornerCloudKeyFrames;
    deque<pcl::PointCloud<PointType>::Ptr> recentSurfCloudKeyFrames;
    deque<pcl::PointCloud<PointType>::Ptr> recentOutlierCloudKeyFrames;
    int latestFrameID;

    vector<int> surroundingExistingKeyPosesID;
    deque<pcl::PointCloud<PointType>::Ptr> surroundingCornerCloudKeyFrames;
    deque<pcl::PointCloud<PointType>::Ptr> surroundingSurfCloudKeyFrames;
    deque<pcl::PointCloud<PointType>::Ptr> surroundingOutlierCloudKeyFrames;
    
    PointType previousRobotPosPoint;
    PointType currentRobotPosPoint;

    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;

    

    pcl::PointCloud<PointType>::Ptr surroundingKeyPoses;
    pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner featuer set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf featuer set from odoOptimization

    pcl::PointCloud<PointType>::Ptr laserCloudOutlierLast; // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudOutlierLastDS; // corner feature set from odoOptimization

    pcl::PointCloud<PointType>::Ptr laserCloudSurfTotalLast; // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfTotalLastDS; // downsampled corner featuer set from odoOptimization

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    
    pcl::PointCloud<PointType>::Ptr nearHistoryCornerKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr nearHistoryCornerKeyFrameCloudDS;
    pcl::PointCloud<PointType>::Ptr nearHistorySurfKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr nearHistorySurfKeyFrameCloudDS;

    pcl::PointCloud<PointType>::Ptr latestCornerKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr latestSurfKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr latestSurfKeyFrameCloudDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap;
    pcl::PointCloud<PointType>::Ptr globalMapKeyPoses;
    pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS;
    pcl::PointCloud<PointType>::Ptr globalMapKeyFrames;
    pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS;

    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterOutlier;
    pcl::VoxelGrid<PointType> downSizeFilterHistoryKeyFrames; // for histor key frames of loop closure
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for global map visualization
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization

    ros::Subscriber subOriginLidar;
    ros::Publisher pubOriginLidar;
    ros::Publisher pubGlobalPoses;
    pcl::PointCloud<PointType>::Ptr originLidar;
    ros::Time originTime;
    ros::Subscriber subOriginImage;
    ros::Publisher pubOriginImage;
    cv::Mat3b originImage;

    double timeLaserCloudCornerLast;
    double timeLaserCloudSurfLast;
    double timeLaserOdometry;
    double timeLaserCloudOutlierLast;
    double timeLastGloalMapPublish;

    bool newLaserCloudCornerLast;
    bool newLaserCloudSurfLast;
    bool newLaserOdometry;
    bool newLaserCloudOutlierLast;


    float transformLast[6];
    float transformSum[6];
    float transformIncre[6];
    float transformTobeMapped[6];
    float transformBefMapped[6];
    float transformAftMapped[6];


    int imuPointerFront;
    int imuPointerLast;

    double imuTime[imuQueLength];
    float imuRoll[imuQueLength];
    float imuPitch[imuQueLength];

    std::mutex mtx;

    double timeLastProcessing;

    PointType pointOri, pointSel, pointProj, coeff;

    cv::Mat matA0;
    cv::Mat matB0;
    cv::Mat matX0;

    cv::Mat matA1;
    cv::Mat matD1;
    cv::Mat matV1;

    bool isDegenerate;
    cv::Mat matP;

    int laserCloudCornerFromMapDSNum;
    int laserCloudSurfFromMapDSNum;
    int laserCloudCornerLastDSNum;
    int laserCloudSurfLastDSNum;
    int laserCloudOutlierLastDSNum;
    int laserCloudSurfTotalLastDSNum;

    bool potentialLoopFlag;
    double timeSaveFirstCurrentScanForLoopClosure;
    int closestHistoryFrameID;
    int latestFrameIDLoopCloure;

    bool aLoopIsClosed;

    float cRoll, sRoll, cPitch, sPitch, cYaw, sYaw, tX, tY, tZ;
    float ctRoll, stRoll, ctPitch, stPitch, ctYaw, stYaw, tInX, tInY, tInZ;

    //add by wy
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;
    cv::Mat img;
    cv::Mat color_;
    bool newimg = false;
    double timeimg;
    // wy::KeyFrame* pcurrkf;
    // std::vector<wy::KeyFrame*> vpkfs;
    // std::queue<wy::KeyFrame*> qpkfs;
    // wy::KeyFrameDB* pkfdb;
    // wy::KeyFrame* kf;
    ros::Publisher loop_pub_;

    ros::Subscriber gps_sub;
    bool newgps;
    double latitude, longtitude, alt;
    std::ofstream gps_p;

    // ORB_SLAM2::ORBVocabulary* porb_vocabulary;
    // ORB_SLAM2::ORBextractor* porb_extractor;

    ros::Publisher path_pub;
    ros::Publisher poseCov_pub;
    nav_msgs::Path path_msg;
    sensor_msgs::PointCloud2 posewithCovariance;

    ros::Subscriber cloud_to_save_sub;
    pcl::PointCloud<PointType>::Ptr cloud_to_save;
    bool newcloudsave = false;
    double timefullcloud;
    int current_frame_id = 1; // save pose with frame id 
    std::vector<pcl::PointCloud<PointType>::Ptr> keyframes_save;


    std::vector<std::pair<int, int>> loop_buffers;
    std::vector<std::pair<int, int>> loop_check;
    std::vector<float> icp_buf;

    pcl::StatisticalOutlierRemoval<PointType> sor;

    ros::ServiceServer save_map_server;
    bool b_save_map = false;


    std::vector<cv::Mat1b> mats;
    std::vector<LidarIris::FeatureDesc> fds;
    int matchBias;
    LidarIris iris = LidarIris(4, 18, 1.6, 0.75, 50);
    int lastkeyframe = -1;

    int last_knn_search = -1;

public:

    void imageCallback(const sensor_msgs::ImageConstPtr& msg)
    {
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::Mat image;
        color_ = cv_ptr->image.clone();
        cv::cvtColor(cv_ptr->image, image, CV_BGR2GRAY);
        img = image.clone();
        timeimg =  msg->header.stamp.toSec();
        newimg = true;
    }
    
    void GPSCallback(const sensor_msgs::NavSatFixConstPtr& gps_msg)
    {
        latitude = gps_msg->latitude;
        longtitude = gps_msg->longitude;
        alt = gps_msg->altitude;
        newgps=true;
    }

    void fullcloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
    {
        velodyne_points->clear();
        pcl::fromROSMsg(*msg, *velodyne_points);
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*velodyne_points, *velodyne_points, indices);
        timefullcloud = msg->header.stamp.toSec();
        newcloudsave = true;
    }
    bool toSaveMap(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res)
    {
        b_save_map = true;
        return true;
    }

    mapOptimization():
        nh("~")
    {
        image_transport::ImageTransport it_(nh);
        // porb_vocabulary = orb_vocabulary;
        // porb_extractor = orb_extractor;
        image_sub_ = it_.subscribe(imageTopic, 2, &mapOptimization::imageCallback, this);
        image_pub_ = it_.advertise("key_frames_image", 2);
        // pkfdb = new wy::KeyFrameDB(*orb_vocabulary);

        loop_pub_ = nh.advertise<visualization_msgs::Marker>("loop_line", 2);
        path_pub = nh.advertise<nav_msgs::Path>("robot_path", 2);
        poseCov_pub = nh.advertise<sensor_msgs::PointCloud2>("pose_with_cov", 2);

        subVelodyne = nh.subscribe("/velodyne_points_3", 2, &mapOptimization::fullcloudCallback, this);
        cloud_to_save.reset(new pcl::PointCloud<PointType>());
        velodyne_points.reset(new pcl::PointCloud<pcl::PointXYZ>());


        gps_sub = nh.subscribe(gpsTopic, 2, &mapOptimization::GPSCallback, this);
        gps_p.open(fileDirectory + "gps.txt");
        gps_p << "type,latitude,longitude,alt"<<std::endl;

        save_map_server = nh.advertiseService("/save_map", &mapOptimization::toSaveMap, this);


    	ISAM2Params parameters;
		parameters.relinearizeThreshold = 0.01;
		parameters.relinearizeSkip = 1;
    	isam = new ISAM2(parameters);

        pubKeyPoses = nh.advertise<sensor_msgs::PointCloud2>("/key_pose_origin", 2);
        pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 2);
        pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> ("/aft_mapped_to_init", 5);

        subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 2, &mapOptimization::laserCloudCornerLastHandler, this);
        subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2, &mapOptimization::laserCloudSurfLastHandler, this);
        subOutlierCloudLast = nh.subscribe<sensor_msgs::PointCloud2>("/outlier_cloud_last", 2, &mapOptimization::laserCloudOutlierLastHandler, this);
        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 5, &mapOptimization::laserOdometryHandler, this);
        subImu = nh.subscribe<sensor_msgs::Imu> (imuTopic, 50, &mapOptimization::imuHandler, this);

        pubHistoryKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/history_cloud", 2);
        pubIcpKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/corrected_cloud", 2);
        pubRecentKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/recent_cloud", 2);
        pubRegisteredCloud = nh.advertise<sensor_msgs::PointCloud2>("/registered_cloud", 2);

        downSizeFilterCorner.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilterSurf.setLeafSize(0.4, 0.4, 0.4);
        downSizeFilterOutlier.setLeafSize(0.4, 0.4, 0.4);

        downSizeFilterHistoryKeyFrames.setLeafSize(0.4, 0.4, 0.4); // for histor key frames of loop closure
        downSizeFilterSurroundingKeyPoses.setLeafSize(1.0, 1.0, 1.0); // for surrounding key poses of scan-to-map optimization

        downSizeFilterGlobalMapKeyPoses.setLeafSize(1.0, 1.0, 1.0); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(0.4, 0.4, 0.4); // for global map visualization

        odomAftMapped.header.frame_id = "camera_init";
        odomAftMapped.child_frame_id = "aft_mapped";

        aftMappedTrans.frame_id_ = "camera_init";
        aftMappedTrans.child_frame_id_ = "aft_mapped";

        subOriginLidar = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 2, &mapOptimization::originLidarHandler, this);
        pubOriginLidar = nh.advertise<sensor_msgs::PointCloud2>("/points_indexed", 2);
        pubGlobalPoses = nh.advertise<nav_msgs::Path>("/global_poses", 2);
        // subOriginImage = nh.subscribe<sensor_msgs::Image>("/camera/image_left", 4, &mapOptimization::originImageHandler, this);
        subOriginImage = nh.subscribe<sensor_msgs::Image>("/kitti/camera_color_left/image_raw", 4, &mapOptimization::originImageHandler, this);
        pubOriginImage = nh.advertise<sensor_msgs::Image>("/image_indexed", 2);

        allocateMemory();
    }

    void allocateMemory(){
        originLidar.reset(new pcl::PointCloud<PointType>());

        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        surroundingKeyPoses.reset(new pcl::PointCloud<PointType>());
        surroundingKeyPosesDS.reset(new pcl::PointCloud<PointType>());        

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization
        laserCloudOutlierLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudOutlierLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner feature set from odoOptimization
        laserCloudSurfTotalLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudSurfTotalLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        
        nearHistoryCornerKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        nearHistoryCornerKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());
        nearHistorySurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        nearHistorySurfKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());

        latestCornerKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        latestSurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        latestSurfKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());

        kdtreeGlobalMap.reset(new pcl::KdTreeFLANN<PointType>());
        globalMapKeyPoses.reset(new pcl::PointCloud<PointType>());
        globalMapKeyPosesDS.reset(new pcl::PointCloud<PointType>());
        globalMapKeyFrames.reset(new pcl::PointCloud<PointType>());
        globalMapKeyFramesDS.reset(new pcl::PointCloud<PointType>());

        timeLaserCloudCornerLast = 0;
        timeLaserCloudSurfLast = 0;
        timeLaserOdometry = 0;
        timeLaserCloudOutlierLast = 0;
        timeLastGloalMapPublish = 0;

        timeLastProcessing = -1;

        newLaserCloudCornerLast = false;
        newLaserCloudSurfLast = false;

        newLaserOdometry = false;
        newLaserCloudOutlierLast = false;

        for (int i = 0; i < 6; ++i){
            transformLast[i] = 0;
            transformSum[i] = 0;
            transformIncre[i] = 0;
            transformTobeMapped[i] = 0;
            transformBefMapped[i] = 0;
            transformAftMapped[i] = 0;
        }

        imuPointerFront = 0;
        imuPointerLast = -1;

        for (int i = 0; i < imuQueLength; ++i){
            imuTime[i] = 0;
            imuRoll[i] = 0;
            imuPitch[i] = 0;
        }

        gtsam::Vector Vector6(6);
        Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-6;
        priorNoise = noiseModel::Diagonal::Variances(Vector6);
        // odometryNoise = noiseModel::Diagonal::Variances(Vector6);

        matA0 = cv::Mat (5, 3, CV_32F, cv::Scalar::all(0));
        matB0 = cv::Mat (5, 1, CV_32F, cv::Scalar::all(-1));
        matX0 = cv::Mat (3, 1, CV_32F, cv::Scalar::all(0));

        matA1 = cv::Mat (3, 3, CV_32F, cv::Scalar::all(0));
        matD1 = cv::Mat (1, 3, CV_32F, cv::Scalar::all(0));
        matV1 = cv::Mat (3, 3, CV_32F, cv::Scalar::all(0));

        isDegenerate = false;
        matP = cv::Mat (6, 6, CV_32F, cv::Scalar::all(0));

        laserCloudCornerFromMapDSNum = 0;
        laserCloudSurfFromMapDSNum = 0;
        laserCloudCornerLastDSNum = 0;
        laserCloudSurfLastDSNum = 0;
        laserCloudOutlierLastDSNum = 0;
        laserCloudSurfTotalLastDSNum = 0;

        potentialLoopFlag = false;
        aLoopIsClosed = false;

        latestFrameID = 0;
    }

    void transformAssociateToMap()
    {
        float x1 = cos(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) 
                 - sin(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);
        float y1 = transformBefMapped[4] - transformSum[4];
        float z1 = sin(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) 
                 + cos(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);

        float x2 = x1;
        float y2 = cos(transformSum[0]) * y1 + sin(transformSum[0]) * z1;
        float z2 = -sin(transformSum[0]) * y1 + cos(transformSum[0]) * z1;

        transformIncre[3] = cos(transformSum[2]) * x2 + sin(transformSum[2]) * y2;
        transformIncre[4] = -sin(transformSum[2]) * x2 + cos(transformSum[2]) * y2;
        transformIncre[5] = z2;

        float sbcx = sin(transformSum[0]);
        float cbcx = cos(transformSum[0]);
        float sbcy = sin(transformSum[1]);
        float cbcy = cos(transformSum[1]);
        float sbcz = sin(transformSum[2]);
        float cbcz = cos(transformSum[2]);

        float sblx = sin(transformBefMapped[0]);
        float cblx = cos(transformBefMapped[0]);
        float sbly = sin(transformBefMapped[1]);
        float cbly = cos(transformBefMapped[1]);
        float sblz = sin(transformBefMapped[2]);
        float cblz = cos(transformBefMapped[2]);

        float salx = sin(transformAftMapped[0]);
        float calx = cos(transformAftMapped[0]);
        float saly = sin(transformAftMapped[1]);
        float caly = cos(transformAftMapped[1]);
        float salz = sin(transformAftMapped[2]);
        float calz = cos(transformAftMapped[2]);

        float srx = -sbcx*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz)
                  - cbcx*sbcy*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                  - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                  - cbcx*cbcy*(calx*salz*(cblz*sbly - cbly*sblx*sblz) 
                  - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx);
        transformTobeMapped[0] = -asin(srx);

        float srycrx = sbcx*(cblx*cblz*(caly*salz - calz*salx*saly)
                     - cblx*sblz*(caly*calz + salx*saly*salz) + calx*saly*sblx)
                     - cbcx*cbcy*((caly*calz + salx*saly*salz)*(cblz*sbly - cbly*sblx*sblz)
                     + (caly*salz - calz*salx*saly)*(sbly*sblz + cbly*cblz*sblx) - calx*cblx*cbly*saly)
                     + cbcx*sbcy*((caly*calz + salx*saly*salz)*(cbly*cblz + sblx*sbly*sblz)
                     + (caly*salz - calz*salx*saly)*(cbly*sblz - cblz*sblx*sbly) + calx*cblx*saly*sbly);
        float crycrx = sbcx*(cblx*sblz*(calz*saly - caly*salx*salz)
                     - cblx*cblz*(saly*salz + caly*calz*salx) + calx*caly*sblx)
                     + cbcx*cbcy*((saly*salz + caly*calz*salx)*(sbly*sblz + cbly*cblz*sblx)
                     + (calz*saly - caly*salx*salz)*(cblz*sbly - cbly*sblx*sblz) + calx*caly*cblx*cbly)
                     - cbcx*sbcy*((saly*salz + caly*calz*salx)*(cbly*sblz - cblz*sblx*sbly)
                     + (calz*saly - caly*salx*salz)*(cbly*cblz + sblx*sbly*sblz) - calx*caly*cblx*sbly);
        transformTobeMapped[1] = atan2(srycrx / cos(transformTobeMapped[0]), 
                                       crycrx / cos(transformTobeMapped[0]));
        
        float srzcrx = (cbcz*sbcy - cbcy*sbcx*sbcz)*(calx*salz*(cblz*sbly - cbly*sblx*sblz)
                     - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx)
                     - (cbcy*cbcz + sbcx*sbcy*sbcz)*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                     - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                     + cbcx*sbcz*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz);
        float crzcrx = (cbcy*sbcz - cbcz*sbcx*sbcy)*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                     - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                     - (sbcy*sbcz + cbcy*cbcz*sbcx)*(calx*salz*(cblz*sbly - cbly*sblx*sblz)
                     - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx)
                     + cbcx*cbcz*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz);
        transformTobeMapped[2] = atan2(srzcrx / cos(transformTobeMapped[0]), 
                                       crzcrx / cos(transformTobeMapped[0]));

        x1 = cos(transformTobeMapped[2]) * transformIncre[3] - sin(transformTobeMapped[2]) * transformIncre[4];
        y1 = sin(transformTobeMapped[2]) * transformIncre[3] + cos(transformTobeMapped[2]) * transformIncre[4];
        z1 = transformIncre[5];

        x2 = x1;
        y2 = cos(transformTobeMapped[0]) * y1 - sin(transformTobeMapped[0]) * z1;
        z2 = sin(transformTobeMapped[0]) * y1 + cos(transformTobeMapped[0]) * z1;

        transformTobeMapped[3] = transformAftMapped[3] 
                               - (cos(transformTobeMapped[1]) * x2 + sin(transformTobeMapped[1]) * z2);
        transformTobeMapped[4] = transformAftMapped[4] - y2;
        transformTobeMapped[5] = transformAftMapped[5] 
                               - (-sin(transformTobeMapped[1]) * x2 + cos(transformTobeMapped[1]) * z2);
    }

    void transformUpdate()
    {
		if (imuPointerLast >= 0) {
		    float imuRollLast = 0, imuPitchLast = 0;
		    while (imuPointerFront != imuPointerLast) {
		        if (timeLaserOdometry + scanPeriod < imuTime[imuPointerFront]) {
		            break;
		        }
		        imuPointerFront = (imuPointerFront + 1) % imuQueLength;
		    }

		    if (timeLaserOdometry + scanPeriod > imuTime[imuPointerFront]) {
		        imuRollLast = imuRoll[imuPointerFront];
		        imuPitchLast = imuPitch[imuPointerFront];
		    } else {
		        int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
		        float ratioFront = (timeLaserOdometry + scanPeriod - imuTime[imuPointerBack]) 
		                         / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
		        float ratioBack = (imuTime[imuPointerFront] - timeLaserOdometry - scanPeriod) 
		                        / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);

		        imuRollLast = imuRoll[imuPointerFront] * ratioFront + imuRoll[imuPointerBack] * ratioBack;
		        imuPitchLast = imuPitch[imuPointerFront] * ratioFront + imuPitch[imuPointerBack] * ratioBack;
		    }

		    transformTobeMapped[0] = 0.998 * transformTobeMapped[0] + 0.002 * imuPitchLast;
		    transformTobeMapped[2] = 0.998 * transformTobeMapped[2] + 0.002 * imuRollLast;
		  }

		for (int i = 0; i < 6; i++) {
		    transformBefMapped[i] = transformSum[i];
		    transformAftMapped[i] = transformTobeMapped[i];
		}
    }

    void updatePointAssociateToMapSinCos(){
        cRoll = cos(transformTobeMapped[0]);
        sRoll = sin(transformTobeMapped[0]);

        cPitch = cos(transformTobeMapped[1]);
        sPitch = sin(transformTobeMapped[1]);

        cYaw = cos(transformTobeMapped[2]);
        sYaw = sin(transformTobeMapped[2]);

        tX = transformTobeMapped[3];
        tY = transformTobeMapped[4];
        tZ = transformTobeMapped[5];
    }

    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        float x1 = cYaw * pi->x - sYaw * pi->y;
        float y1 = sYaw * pi->x + cYaw * pi->y;
        float z1 = pi->z;

        float x2 = x1;
        float y2 = cRoll * y1 - sRoll * z1;
        float z2 = sRoll * y1 + cRoll * z1;

        po->x = cPitch * x2 + sPitch * z2 + tX;
        po->y = y2 + tY;
        po->z = -sPitch * x2 + cPitch * z2 + tZ;
        po->intensity = pi->intensity;
    }

    void updateTransformPointCloudSinCos(PointTypePose *tIn){

        ctRoll = cos(tIn->roll);
        stRoll = sin(tIn->roll);

        ctPitch = cos(tIn->pitch);
        stPitch = sin(tIn->pitch);

        ctYaw = cos(tIn->yaw);
        stYaw = sin(tIn->yaw);

        tInX = tIn->x;
        tInY = tIn->y;
        tInZ = tIn->z;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn){
	// !!! DO NOT use pcl for point cloud transformation, results are not accurate
        // Reason: unkown
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;
        PointType pointTo;

        int cloudSize = cloudIn->points.size();
        cloudOut->resize(cloudSize);

        for (int i = 0; i < cloudSize; ++i){

            pointFrom = &cloudIn->points[i];
            float x1 = ctYaw * pointFrom->x - stYaw * pointFrom->y;
            float y1 = stYaw * pointFrom->x + ctYaw* pointFrom->y;
            float z1 = pointFrom->z;

            float x2 = x1;
            float y2 = ctRoll * y1 - stRoll * z1;
            float z2 = stRoll * y1 + ctRoll* z1;

            pointTo.x = ctPitch * x2 + stPitch * z2 + tInX;
            pointTo.y = y2 + tInY;
            pointTo.z = -stPitch * x2 + ctPitch * z2 + tInZ;
            pointTo.intensity = pointFrom->intensity;

            cloudOut->points[i] = pointTo;
        }
        return cloudOut;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn){

        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;
        PointType pointTo;

        int cloudSize = cloudIn->points.size();
        cloudOut->resize(cloudSize);
        
        for (int i = 0; i < cloudSize; ++i){

            pointFrom = &cloudIn->points[i];
            float x1 = cos(transformIn->yaw) * pointFrom->x - sin(transformIn->yaw) * pointFrom->y;
            float y1 = sin(transformIn->yaw) * pointFrom->x + cos(transformIn->yaw)* pointFrom->y;
            float z1 = pointFrom->z;

            float x2 = x1;
            float y2 = cos(transformIn->roll) * y1 - sin(transformIn->roll) * z1;
            float z2 = sin(transformIn->roll) * y1 + cos(transformIn->roll)* z1;

            pointTo.x = cos(transformIn->pitch) * x2 + sin(transformIn->pitch) * z2 + transformIn->x;
            pointTo.y = y2 + transformIn->y;
            pointTo.z = -sin(transformIn->pitch) * x2 + cos(transformIn->pitch) * z2 + transformIn->z;
            pointTo.intensity = pointFrom->intensity;

            cloudOut->points[i] = pointTo;
        }
        return cloudOut;
    }

    void laserCloudOutlierLastHandler(const sensor_msgs::PointCloud2ConstPtr& msg){
        timeLaserCloudOutlierLast = msg->header.stamp.toSec();
        laserCloudOutlierLast->clear();
        pcl::fromROSMsg(*msg, *laserCloudOutlierLast);
        newLaserCloudOutlierLast = true;
    }

    void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr& msg){
        timeLaserCloudCornerLast = msg->header.stamp.toSec();
        laserCloudCornerLast->clear();
        pcl::fromROSMsg(*msg, *laserCloudCornerLast);
        newLaserCloudCornerLast = true;
    }

    void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr& msg){
        timeLaserCloudSurfLast = msg->header.stamp.toSec();
        laserCloudSurfLast->clear();
        pcl::fromROSMsg(*msg, *laserCloudSurfLast);
        newLaserCloudSurfLast = true;
    }

    void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr& laserOdometry){
        timeLaserOdometry = laserOdometry->header.stamp.toSec();
        double roll, pitch, yaw;
        geometry_msgs::Quaternion geoQuat = laserOdometry->pose.pose.orientation;
        tf::Matrix3x3(tf::Quaternion(geoQuat.z, -geoQuat.x, -geoQuat.y, geoQuat.w)).getRPY(roll, pitch, yaw);
        transformSum[0] = -pitch;
        transformSum[1] = -yaw;
        transformSum[2] = roll;
        transformSum[3] = laserOdometry->pose.pose.position.x;
        transformSum[4] = laserOdometry->pose.pose.position.y;
        transformSum[5] = laserOdometry->pose.pose.position.z;
        newLaserOdometry = true;
    }

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn){
        double roll, pitch, yaw;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(imuIn->orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        // tf::Matrix3x3(orientation).getRPY(pitch, yaw, roll);
        imuPointerLast = (imuPointerLast + 1) % imuQueLength;
        imuTime[imuPointerLast] = imuIn->header.stamp.toSec();
        imuRoll[imuPointerLast] = roll;
        imuPitch[imuPointerLast] = pitch;

        // imuRoll[imuPointerLast] = pitch;
        // imuPitch[imuPointerLast] = yaw;
    }

    void originLidarHandler(const sensor_msgs::PointCloud2ConstPtr& msg){
        originLidar->clear();
        pcl::fromROSMsg(*msg, *originLidar);
        originTime = msg->header.stamp;
    }

    void originImageHandler(const sensor_msgs::ImageConstPtr &msg) {
        // if((msg->header.stamp - originTime).toSec() > 0.05) return;
        originImage = cv_bridge::toCvCopy(msg)->image;
    }

    void publishTF(){

        geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw
                                  (transformAftMapped[2], -transformAftMapped[0], -transformAftMapped[1]);

        odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        odomAftMapped.pose.pose.orientation.x = -geoQuat.y;
        odomAftMapped.pose.pose.orientation.y = -geoQuat.z;
        odomAftMapped.pose.pose.orientation.z = geoQuat.x;
        odomAftMapped.pose.pose.orientation.w = geoQuat.w;
        odomAftMapped.pose.pose.position.x = transformAftMapped[3];
        odomAftMapped.pose.pose.position.y = transformAftMapped[4];
        odomAftMapped.pose.pose.position.z = transformAftMapped[5];
        odomAftMapped.twist.twist.angular.x = transformBefMapped[0];
        odomAftMapped.twist.twist.angular.y = transformBefMapped[1];
        odomAftMapped.twist.twist.angular.z = transformBefMapped[2];
        odomAftMapped.twist.twist.linear.x = transformBefMapped[3];
        odomAftMapped.twist.twist.linear.y = transformBefMapped[4];
        odomAftMapped.twist.twist.linear.z = transformBefMapped[5];
        pubOdomAftMapped.publish(odomAftMapped);

        aftMappedTrans.stamp_ = ros::Time().fromSec(timeLaserOdometry);
        aftMappedTrans.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
        aftMappedTrans.setOrigin(tf::Vector3(transformAftMapped[3], transformAftMapped[4], transformAftMapped[5]));
        tfBroadcaster.sendTransform(aftMappedTrans);
    }
    
    PointTypePose trans2PointTypePose(float transformIn[]){
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }

    void publishKeyPosesAndFrames(){

        if (pubKeyPoses.getNumSubscribers() != 0){
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*cloudKeyPoses3D, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "camera_init";
            pubKeyPoses.publish(cloudMsgTemp);
        }

        if (pubRecentKeyFrames.getNumSubscribers() != 0){
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*laserCloudSurfFromMapDS, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "camera_init";
            pubRecentKeyFrames.publish(cloudMsgTemp);
        }

        // if(path_pub.getNumSubscribers() != 0)
        // {
            path_msg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            path_msg.header.frame_id = "camera_init";
            path_pub.publish(path_msg);
            posewithCovariance.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            posewithCovariance.header.frame_id = "camera_init";
            poseCov_pub.publish(posewithCovariance);
        // }
        
        if (pubRegisteredCloud.getNumSubscribers() != 0){
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut += *transformPointCloud(laserCloudCornerLastDS,  &thisPose6D);
            *cloudOut += *transformPointCloud(laserCloudSurfTotalLast, &thisPose6D);
            
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*cloudOut, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/camera_init";
            pubRegisteredCloud.publish(cloudMsgTemp);
        } 
    }

    void visualizeGlobalMapThread(){
        ros::Rate rate(1);  // the hertz that flushes the map update
        // ros::Rate rate(0.2);
        while (ros::ok()){
            rate.sleep();
            publishGlobalMap();
        }     
         globalMapKeyFrames->clear();

        pcl::PointCloud<PointType>::Ptr wy_map(new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr wy_mapDS(new pcl::PointCloud<PointType>);
        std::cout << cornerCloudKeyFrames.size() << std::endl;
        std::cout << cloudKeyPoses3D->points.size() << std::endl;
        std::cout << cloudKeyPoses6D->points.size() << std::endl;
        std::cout << keyframes_save.size() << std::endl;

        std::cout << "save start..." << std::endl;

        pcl::io::savePCDFileBinary(fileDirectory + "lego_trajectory.pcd", *cloudKeyPoses3D);
        std::cout << "lego_trajectory.pcd saved!" << std::endl;

        std::ofstream of(fileDirectory + "lego_trajectory.txt");
  for(int i = 0; i < cloudKeyPoses6D->size(); i++)
        {
            of.setf(ios::fixed,ios::floatfield);
            of.precision(0);
            of << cloudKeyPoses6D->points[i].time << " ";
            of.precision(5);




            of << cloudKeyPoses6D->points[i].x << " ";
            of << cloudKeyPoses6D->points[i].y << " ";
            of << cloudKeyPoses6D->points[i].z << " ";
            // of << cloudKeyPoses6D->points[i].intensity << " ";

            Eigen::Matrix3f rotation_matrix = Eigen::AngleAxisf(cloudKeyPoses6D->points[i].pitch, Eigen::Vector3f::UnitY()).matrix()
                                        * Eigen::AngleAxisf(cloudKeyPoses6D->points[i].roll, Eigen::Vector3f::UnitX()).matrix()
                                        * Eigen::AngleAxisf(cloudKeyPoses6D->points[i].yaw, Eigen::Vector3f::UnitZ()).matrix();
            Eigen::Quaternionf q_f=Eigen::Quaternionf(rotation_matrix);
            q_f.normalize();
            // of << cloudKeyPoses6D->points[i].roll << " ";
            // of << cloudKeyPoses6D->points[i].pitch << " ";
            // of << cloudKeyPoses6D->points[i].yaw << " ";

            of << q_f.x()<< " ";
            of << q_f.y() << " ";
            of << q_f.z() << " ";
            of << q_f.w() << std::endl;
            
        }
        std::cout << "lego_trajectory.txt save successful" << std::endl;
        of.close();

        for(int i = 0; i < cloudKeyPoses6D->points.size(); ++i)
        {
            *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[i],   &cloudKeyPoses6D->points[i]);
        }
        pcl::io::savePCDFileBinary(fileDirectory + "lego_corner.pcd", *globalMapKeyFrames);
        *wy_map += *globalMapKeyFrames;
        globalMapKeyFrames->clear();

        for(int i = 0; i < cloudKeyPoses6D->points.size(); i++)
        {
			*globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
            // *globalMapKeyFrames += *transformPointCloud(outlierCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
        }

        pcl::io::savePCDFileBinary(fileDirectory + "lego_surf.pcd", *globalMapKeyFrames);
        *wy_map += *globalMapKeyFrames;

        globalMapKeyFrames->clear();

        for(int i = 0; i < cloudKeyPoses6D->points.size(); i++)
        {
			// *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
            *globalMapKeyFrames += *transformPointCloud(outlierCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
        }

        *wy_map += *globalMapKeyFrames;

        // for(int i = 0; i < keyframes_save.size(); i++)
        // {
        //     pcl::io::savePCDFileBinary(fileDirectory+"keyframes/" + std::to_string(i+1) + ".pcd", *keyframes_save[i]);
        // }

        downSizeFilterGlobalMapKeyFrames.setInputCloud(wy_map);
        downSizeFilterGlobalMapKeyFrames.filter(*wy_mapDS);
        pcl::io::savePCDFileBinary(fileDirectory + "lego_finalCloud.pcd", *wy_mapDS);
        std::cout << "lego_finalCloud save successful" << std::endl;
        
        

        
    }

    void transformPointCloud(pcl::PointCloud<PointL>::Ptr& pointcloud,Eigen::Isometry3d T)
    {
        Eigen::MatrixXd points(4,pointcloud->size());
        int j = 0;
        for (auto p : *pointcloud)
        {
            points.col(j++) << p.x, p.y, p.z, 1;
        }
        Eigen::MatrixXd result = T.matrix() * points;
        for (int i = 0; i < pointcloud->size();i++)
        {
            pointcloud->points[i].x = result(0, i) / result(3, i);
            pointcloud->points[i].y = result(1, i) / result(3, i);
            pointcloud->points[i].z = result(2, i) / result(3, i);
        }
    }
    void ExactLowerScan(pcl::PointCloud<pcl::PointXYZI>::Ptr& inputcloud){
        pcl::PointCloud<pcl::PointXYZI>::Ptr lowerScan(new pcl::PointCloud<pcl::PointXYZI>);
        int maxintensity = 0;
        for (auto pt : *inputcloud)
        {
            // calculate vertical point angle and scan ID
            float angle = std::atan(pt.y / std::sqrt(pt.x * pt.x + pt.z * pt.z));
            int ring = int(((angle * 180 / M_PI) + 15) * 0.5 + 0.5);
            if (ring < 5)
            {
                lowerScan->push_back(pt);
            }
        }
        inputcloud->clear();
        *inputcloud += *lowerScan;
    }

    // 将雷达原始坐标系转换到loam坐标系中, 就是坐标轴的切换, 如果已经是loam坐标系则>不用切换
    void CoordinateTransform(pcl::PointCloud<pcl::PointXYZI>::Ptr& inputcloud){
        pcl::PointXYZI point;
        pcl::PointCloud<pcl::PointXYZI>::Ptr newcloud(new pcl::PointCloud<pcl::PointXYZI>());
        for (auto pt : *inputcloud)
        {
            point.x = pt.y;
            point.y = pt.z;
            point.z = pt.x;
            point.intensity = pt.intensity;
            newcloud->push_back(point);
        }
        inputcloud->clear();
        inputcloud = newcloud;
    }



    void publishGlobalMap(){
        nav_msgs::Path poses;
        poses.header.frame_id = "camera_init";
        poses.header.stamp = ros::Time::now();
        for(size_t index = 0; index < cloudKeyPoses6D->size(); index++)
        {
            geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw
                                (cloudKeyPoses6D->points[index].yaw, -cloudKeyPoses6D->points[index].roll, -cloudKeyPoses6D->points[index].pitch);
            //
            Eigen::Quaterniond q(geoQuat.w, -geoQuat.y, -geoQuat.z, geoQuat.x);
            q = q * Eigen::AngleAxisd(-M_PI_2, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(-M_PI_2, Eigen::Vector3d::UnitX());
            Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
            pose.translate(Eigen::Vector3d(cloudKeyPoses6D->points[index].x, cloudKeyPoses6D->points[index].y, cloudKeyPoses6D->points[index].z));
            pose.rotate(q);
            pose = Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitY()) * pose;
            
            const auto &translation = pose.translation();
            const auto &rotation = Eigen::Quaterniond(pose.rotation());
            geometry_msgs::PoseStamped poseMsg;
            poseMsg.header.frame_id = std::to_string(index); // TODO: maybe need another index rule
            poseMsg.pose.position.x = translation.x();
            poseMsg.pose.position.y = translation.y();
            poseMsg.pose.position.z = translation.z();
            poseMsg.pose.orientation.x = rotation.x();
            poseMsg.pose.orientation.y = rotation.y();
            poseMsg.pose.orientation.z = rotation.z();
            poseMsg.pose.orientation.w = rotation.w();
            poses.poses.push_back(poseMsg);
            //
            tf::StampedTransform transform;
            transform.stamp_ = ros::Time().now();
            transform.frame_id_ = "global_map";
            transform.child_frame_id_ = poseMsg.header.frame_id;
            transform.setRotation(tf::Quaternion(rotation.x(), rotation.y(), rotation.z(), rotation.w()));
            transform.setOrigin(tf::Vector3(translation.x(), translation.y(), translation.z()));
            tfBroadcaster.sendTransform(transform);
        }
        pubGlobalPoses.publish(poses);

        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;

        if (cloudKeyPoses3D->points.empty() == true)
            return;
	    // kd-tree to find near key frames to visualize
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
	    // search near key frames to visualize
        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMap->radiusSearch(currentRobotPosPoint, globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < pointSearchIndGlobalMap.size(); ++i)
          globalMapKeyPoses->points.push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);


	    // downsample near selected key frames
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
	    // extract visualized and downsampled key frames
        for (int i = 0; i < globalMapKeyPosesDS->points.size(); ++i){
			int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
			*globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],   &cloudKeyPoses6D->points[thisKeyInd]);
			*globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
			*globalMapKeyFrames += *transformPointCloud(outlierCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
        }
	    // downsample visualized points
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
 
        sensor_msgs::PointCloud2 cloudMsgTemp;
        pcl::toROSMsg(*globalMapKeyFramesDS, cloudMsgTemp);
        cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        cloudMsgTemp.header.frame_id = "camera_init";
        pubLaserCloudSurround.publish(cloudMsgTemp);  

        globalMapKeyPoses->clear();
        globalMapKeyPosesDS->clear();
        globalMapKeyFrames->clear();
        globalMapKeyFramesDS->clear();     
    }

    void loopClosureThread(){

        if (loopClosureEnableFlag == false)
            return;

        ros::Rate rate(1);
        while (ros::ok()){
            rate.sleep();
            performLoopClosure();
        }
    }

    bool detectLoopClosure(){


        latestSurfKeyFrameCloud->clear();
        nearHistorySurfKeyFrameCloud->clear();
        nearHistorySurfKeyFrameCloudDS->clear();

        //// ROS_INFO("detectLoopClosure()");
        std::lock_guard<std::mutex> lock(mtx);
        // std::cout << testnum++ << std::endl;
        if(fds.size() <= 50) return false;
        // if(qpkfs.empty()) return false;

        closestHistoryFrameID = -1;

        // std::vector<wy::KeyFrame*> kfs = vpkfs;
        // wy::KeyFrame* kf = qpkfs.front();
        LidarIris::FeatureDesc kf = fds.back();


        // qpkfs.pop();
        latestFrameIDLoopCloure = kf.index;
        if(latestFrameIDLoopCloure == lastkeyframe) return false;
        lastkeyframe = latestFrameIDLoopCloure;

        // 根据位置来找回环
        if(latestFrameIDLoopCloure - last_knn_search < 10)
        // if(1)
        {
            // last_knn_search = latestFrameIDLoopCloure;
            std::vector<int> pointSearchIndLoop;
            std::vector<float> pointSearchSqDisLoop;
            kdtreeHistoryKeyPoses->setInputCloud(cloudKeyPoses3D);
            kdtreeHistoryKeyPoses->radiusSearch(currentRobotPosPoint, historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, surroundingKeyframeSearchRadius);
        
            closestHistoryFrameID = -1;
            for (int i = 0; i < pointSearchIndLoop.size(); ++i){
                int id = pointSearchIndLoop[i];
                if (abs(cloudKeyPoses6D->points[id].time - timeLaserOdometry) > 60.0){
                    closestHistoryFrameID = id;
                    break;
                }
            }
            if (closestHistoryFrameID == -1){
                return false;
            }
        }
        //knn + lidariris找到回换
        else
        {
            float distance = 1;
            int index = -1;
            
            TicToc tt;
            // iris.Compare(kf, &index, &distance, &matchBias);
            std::vector<int> pointSearchIndLoop;
            std::vector<float> pointSearchSqDisLoop;
        


            kdtreeHistoryKeyPoses->setInputCloud(cloudKeyPoses3D);
            kdtreeHistoryKeyPoses->radiusSearch(currentRobotPosPoint, historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, surroundingKeyframeSearchRadius);
        
            closestHistoryFrameID = -1;
            for (int i = 0; i < pointSearchIndLoop.size(); ++i){
                int id = pointSearchIndLoop[i];
                if (abs(cloudKeyPoses6D->points[id].time - timeLaserOdometry) > 60.0){
                    int bias=0;
                    float dis = iris.Compare(fds[id], kf, &bias);
                    //// cout << dis << " ";
                    if(dis < distance)
                    {
                        distance = dis;
                        matchBias = bias;
                        index = id;
                    }
                }
            }

            //tricks: 只是搜索开始几个关键帧
            // for (int i = 0; i < 50; ++i){
            //     int id = i;
            //     if (abs(cloudKeyPoses6D->points[id].time - timeLaserOdometry) > 60.0){
            //         int bias=0;
            //         float dis = iris.Compare(fds[id], kf, &bias);
            //         cout << dis << " ";
            //         if(dis < distance)
            //         {
            //         distance = dis;
            //         matchBias = bias;
            //         index = id;
            //         }
            //     }
            // }
            //// cout << endl;


            //// std::cout <<"loop compare: "<< tt.toc() << " ms" << std::endl;

            if(distance < 0.4)
            {
                closestHistoryFrameID = index;
            }

            /*
            std::cout << "loop id " << closestHistoryFrameID << "  current id " << latestFrameIDLoopCloure
                    << "distance"<<  distance
                    <<std::endl;
            std::cout << "loop id " << index << "  current id " << latestFrameIDLoopCloure
                    << "distance"<<  distance
                    <<std::endl;
            */



            if(closestHistoryFrameID == -1) 
            {
                // loop_buffers.clear();
                return false;
            }

            // 回环检测增加时间上的约束
            if (abs(cloudKeyPoses6D->points[closestHistoryFrameID].time - cloudKeyPoses6D->points[latestFrameIDLoopCloure].time) <= 60.0){
                return false;
            }
        }

        // pub loop marker
        visualization_msgs::Marker loop_marker;
        loop_marker.header.frame_id = "camera_init";
        loop_marker.header.stamp = ros::Time::now();

        loop_marker.ns = "loop";
        loop_marker.id = 1;
        loop_marker.type = 4;
        loop_marker.action = 0;
        loop_marker.pose.position.x = 0;
        loop_marker.pose.position.y = 0;
        loop_marker.pose.position.z = 0;
        loop_marker.pose.orientation.x = 0;
        loop_marker.pose.orientation.y = 0;
        loop_marker.pose.orientation.z = 0;
        loop_marker.pose.orientation.w = 1;

        loop_marker.scale.x = 1;
        loop_marker.scale.y = 1;
        loop_marker.scale.z = 1;

        loop_marker.color.r = 0;
        loop_marker.color.g = 0;
        loop_marker.color.b = 1;
        loop_marker.color.a = 1;

        loop_marker.lifetime = ros::Duration(3.0);
        geometry_msgs::Point p1, p2;
        p1.x = cloudKeyPoses6D->points[latestFrameIDLoopCloure].x;
        p1.y = cloudKeyPoses6D->points[latestFrameIDLoopCloure].y;
        p1.z = cloudKeyPoses6D->points[latestFrameIDLoopCloure].z;
        p2.x = cloudKeyPoses6D->points[closestHistoryFrameID].x;
        p2.y = cloudKeyPoses6D->points[closestHistoryFrameID].y;
        p2.z = cloudKeyPoses6D->points[closestHistoryFrameID].z;
        
        double diffx = p1.x - p2.x;
        double diffy = p1.y - p2.y;
        double diffz = p1.z - p2.z;

        // if((diffx * diffx + diffz * diffz) > historyKeyframeSearchRadius * historyKeyframeSearchRadius) return false;

        loop_marker.points.push_back(p1);
        loop_marker.points.push_back(p2);

        loop_pub_.publish(loop_marker);

        //// ROS_INFO("detectLoopClosure() ok");


        // *latestSurfKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames[latestFrameIDLoopCloure], &cloudKeyPoses6D->points[latestFrameIDLoopCloure]);
        // *latestSurfKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[latestFrameIDLoopCloure],   &cloudKeyPoses6D->points[latestFrameIDLoopCloure]);

        *latestSurfKeyFrameCloud += *cornerCloudKeyFrames[latestFrameIDLoopCloure];
        *latestSurfKeyFrameCloud += *surfCloudKeyFrames[latestFrameIDLoopCloure];


        pcl::PointCloud<PointType>::Ptr hahaCloud(new pcl::PointCloud<PointType>());
        int cloudSize = latestSurfKeyFrameCloud->points.size();
        for (int i = 0; i < cloudSize; ++i){
            if ((int)latestSurfKeyFrameCloud->points[i].intensity >= 0){
                hahaCloud->push_back(latestSurfKeyFrameCloud->points[i]);
            }
        }
        latestSurfKeyFrameCloud->clear();
        *latestSurfKeyFrameCloud   = *hahaCloud;
        for(int i = 0; i < latestSurfKeyFrameCloud->size(); i++)
        {
            PointType p;
            p.x = latestSurfKeyFrameCloud->points[i].z; 
            p.y = latestSurfKeyFrameCloud->points[i].x; 
            p.z = latestSurfKeyFrameCloud->points[i].y;

            latestSurfKeyFrameCloud->points[i] = p;
        }
	// save history near key frames
         for (int j = -historyKeyframeSearchNum; j <= historyKeyframeSearchNum; ++j){
            if (closestHistoryFrameID + j < 0 || closestHistoryFrameID + j > latestFrameIDLoopCloure)
                continue;
            *nearHistorySurfKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames[closestHistoryFrameID+j], &cloudKeyPoses6D->points[closestHistoryFrameID+j]);
            *nearHistorySurfKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[closestHistoryFrameID+j],   &cloudKeyPoses6D->points[closestHistoryFrameID+j]);
        }
        downSizeFilterHistoryKeyFrames.setInputCloud(nearHistorySurfKeyFrameCloud);
        downSizeFilterHistoryKeyFrames.filter(*nearHistorySurfKeyFrameCloudDS);

        PointTypePose t_prev = cloudKeyPoses6D->points[closestHistoryFrameID];
    
        Eigen::Quaternionf q((Eigen::AngleAxisf(t_prev.pitch, Eigen::Vector3f::UnitY()) *
                                 Eigen::AngleAxisf(t_prev.roll, Eigen::Vector3f::UnitX()) * 
                                 Eigen::AngleAxisf(t_prev.yaw, Eigen::Vector3f::UnitZ())).matrix());

        Eigen::Matrix4f prev = Eigen::Matrix4f::Identity();
        q.normalize();
        prev.block<3,3>(0, 0) = q.matrix();
        prev.block<3,1>(0, 3) = Eigen::Vector3f(t_prev.x, t_prev.y, t_prev.z);

        pcl::transformPointCloud(*nearHistorySurfKeyFrameCloudDS, *nearHistorySurfKeyFrameCloudDS, prev.inverse());

        for(int i = 0; i < nearHistorySurfKeyFrameCloudDS->size(); i++)
        {
            PointType p;
            p.x = nearHistorySurfKeyFrameCloudDS->points[i].z; 
            p.y = nearHistorySurfKeyFrameCloudDS->points[i].x; 
            p.z = nearHistorySurfKeyFrameCloudDS->points[i].y;

            nearHistorySurfKeyFrameCloudDS->points[i] = p;
        }
	// publish history near key frames
        if (pubHistoryKeyFrames.getNumSubscribers() != 0){
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*nearHistorySurfKeyFrameCloudDS, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "map";
            pubHistoryKeyFrames.publish(cloudMsgTemp);
        }
        if(nearHistorySurfKeyFrameCloudDS->size() <= 100) {
            std::cout << "dian tai shao" << std::endl;
            std::cout << nearHistorySurfKeyFrameCloudDS->size()  << " " << latestSurfKeyFrameCloud->size() << std::endl;
            return false;}


        return true;
    }


    void performLoopClosure(){

        if (cloudKeyPoses3D->points.empty() == true)
            return;
	// try to find close key frame if there are any
        if (potentialLoopFlag == false){

            if (detectLoopClosure() == true){
                ROS_INFO("detect loop!");
                potentialLoopFlag = true; // find some key frames that is old enough or close enough for loop closure
                timeSaveFirstCurrentScanForLoopClosure = timeLaserOdometry;
            }
            if (potentialLoopFlag == false)
                return;
        }
	// reset the flag first no matter icp successes or not
        potentialLoopFlag = false;

        // std::lock_guard<std::mutex> lock(mtx);


    // 测试一下ndt算法来做配准
        // pcl::NormalDistributionsTransform<PointType, PointType> icp;

        // // Setting scale dependent NDT parameters
        // // Setting minimum transformation difference for termination condition.
        // icp.setTransformationEpsilon (0.01);
        // // Setting maximum step size for More-Thuente line search.
        // icp.setStepSize (0.1);
        // //Setting Resolution of icp grid structure (VoxelGridCovariance).
        // icp.setResolution (1.0);

        // // Setting max number of registration iterations.
        // icp.setMaximumIterations (100);

	// ICP Settings
        // cout << "icpppppp" << matchBias << std::endl;

        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(100);//100
        icp.setMaximumIterations(100);// 200
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);
    // Align clouds

        icp.setInputSource(latestSurfKeyFrameCloud);
        icp.setInputTarget(nearHistorySurfKeyFrameCloudDS);

        Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();
        cout << "match bias:" << matchBias << std::endl;
        guess.block<3,3>(0,0) = Eigen::AngleAxisf(M_PI * matchBias / 180, Eigen::Vector3f::UnitZ()).matrix();
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        // icp.align(*unused_result);
        icp.align(*unused_result, guess);
        std::cout << "icp score: " << icp.getFitnessScore() << std::endl;
        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
        {
            if(loop_buffers.empty())
            {
                loop_buffers.push_back(std::make_pair(latestFrameIDLoopCloure, closestHistoryFrameID));
                // icp_buf.push_back(icp.getFitnessScore());

            }else
            {
                if(std::abs(latestFrameIDLoopCloure - loop_buffers.back().first) <= 5
                 && closestHistoryFrameID - loop_buffers.back().second <= 5
                 && closestHistoryFrameID - loop_buffers.back().second >=-5)
                 {
                    loop_buffers.push_back(std::make_pair(latestFrameIDLoopCloure, closestHistoryFrameID));
                    // icp_buf.push_back(icp.getFitnessScore());
                 }
                else 
                {
                    loop_buffers.clear();
                    // loop_buffers.push_back(std::make_pair(latestFrameIDLoopCloure, closestHistoryFrameID));
                }

            }

            if(loop_buffers.size() <= 10) 
                return;       

            // std::ofstream ofs;
            // ofs.open("/media/horizon/结构件验收数据/loops/" + std::to_string(loop_id) + ".txt");
            // latestSurfKeyFrameCloud->clear();
            // nearHistorySurfKeyFrameCloudDS->clear();
            // *latestSurfKeyFrameCloud += *cornerCloudKeyFrames[latestFrameIDLoopCloure];
            // *latestSurfKeyFrameCloud += *surfCloudKeyFrames[latestFrameIDLoopCloure];
            // *nearHistorySurfKeyFrameCloudDS += *cornerCloudKeyFrames[closestHistoryFrameID];
            // *nearHistorySurfKeyFrameCloudDS += *surfCloudKeyFrames[closestHistoryFrameID];
            // pcl::io::savePCDFileBinary("/media/horizon/结构件验收数据/loops/" + std::to_string(loop_id) + "_1.pcd", *latestSurfKeyFrameCloud);
            // pcl::io::savePCDFileBinary("/media/horizon/结构件验收数据/loops/" + std::to_string(loop_id++) + "_2.pcd", *nearHistorySurfKeyFrameCloudDS);

            // ofs << latestFrameIDLoopCloure << " " << closestHistoryFrameID << " " << icp.getFitnessScore() << std::endl;
            // ofs << icp.getFinalTransformation() << std::endl;     
        }

        //连续几帧都是回环才认为是回环
        // if(loop_check.empty())
        //     loop_check.push_back(std::make_pair(latestFrameIDLoopCloure, closestHistoryFrameID));
        // else
        // {
        //     if(std::abs(latestFrameIDLoopCloure - loop_check.back().first) <= 5
        //         && closestHistoryFrameID - loop_check.back().second <= 5
        //         && closestHistoryFrameID - loop_check.back().second >=-5)
        //         {
        //             loop_check.push_back(std::make_pair(latestFrameIDLoopCloure, closestHistoryFrameID));
        //         // icp_buf.push_back(icp.getFitnessScore());
        //         }
        //     else loop_check.clear();
        // }
        // if(loop_check.size() <= 1) return ;


        float noiseScore;
        

        if(loop_buffers.size() <= 10)
        {
            noiseScore = icp.getFitnessScore();
        }
        else
        { 
            if(icp.getFitnessScore() <= 5)
            { 
                    noiseScore = 0.5;
                    loop_buffers.clear();
                    std::cout <<"lopppppppppppppppppppppp--------" << std::endl;
            }
            else
            {
                loop_buffers.clear();
                return ;
            }
        }
      
        ROS_INFO("add a factor to graph! %f ", icp.getFitnessScore());

        visualization_msgs::Marker loop_marker;
        loop_marker.header.frame_id = "map";
        loop_marker.header.stamp = ros::Time::now();

        loop_marker.ns = "loop";
        loop_marker.id = 2;
        loop_marker.type = 4;
        loop_marker.action = 0;
        loop_marker.pose.position.x = 0;
        loop_marker.pose.position.y = 0;
        loop_marker.pose.position.z = 0;
        loop_marker.pose.orientation.x = 0;
        loop_marker.pose.orientation.y = 0;
        loop_marker.pose.orientation.z = 0;
        loop_marker.pose.orientation.w = 1;

        loop_marker.scale.x = 1;
        loop_marker.scale.y = 1;
        loop_marker.scale.z = 1;

        loop_marker.color.r = 1;
        loop_marker.color.g = 0;
        loop_marker.color.b = 0;
        loop_marker.color.a = 1;

        loop_marker.lifetime = ros::Duration();
        geometry_msgs::Point p1, p2;
        p1.x = cloudKeyPoses6D->points[latestFrameIDLoopCloure].z;
        p1.y = cloudKeyPoses6D->points[latestFrameIDLoopCloure].x;
        p1.z = cloudKeyPoses6D->points[latestFrameIDLoopCloure].y;
        p2.x = cloudKeyPoses6D->points[closestHistoryFrameID].z;
        p2.y = cloudKeyPoses6D->points[closestHistoryFrameID].x;
        p2.z = cloudKeyPoses6D->points[closestHistoryFrameID].y;

        loop_marker.points.push_back(p1);
        loop_marker.points.push_back(p2);
        loop_pub_.publish(loop_marker);
	// publish corrected cloud
        if (pubIcpKeyFrames.getNumSubscribers() != 0){
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud (*latestSurfKeyFrameCloud, *closed_cloud, icp.getFinalTransformation());
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*closed_cloud, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "camera_init";
            pubIcpKeyFrames.publish(cloudMsgTemp);
        }   
	/*
        	get pose constraint
        	*/
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionCameraFrame;
        correctionCameraFrame = icp.getFinalTransformation(); // get transformation in camera frame (because points are in camera frame)
        pcl::getTranslationAndEulerAngles(correctionCameraFrame, x, y, z, roll, pitch, yaw);
    //     Eigen::Affine3f correctionLidarFrame = pcl::getTransformation(z, x, y, yaw, roll, pitch);
	// // transform from world origin to wrong pose
    //     Eigen::Affine3f tWrong = pclPointToAffine3fCameraToLidar(cloudKeyPoses6D->points[latestFrameIDLoopCloure]);
	// // transform from world origin to corrected pose
    //     Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong; // pre-multiplying -> successive rotation about a fixed frame
    //     pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        // gtsam::Pose3 poseTo = pclPointTogtsamPose3(cloudKeyPoses6D->points[closestHistoryFrameID]);
        gtsam::Vector Vector6(6);
        
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        constraintNoise = noiseModel::Diagonal::Variances(Vector6);
	/* 
        	add constraints
        	*/
        std::lock_guard<std::mutex> lock(mtx);
        gtSAMgraph.add(BetweenFactor<Pose3>(closestHistoryFrameID,latestFrameIDLoopCloure, poseFrom, constraintNoise));
        // gtSAMgraph.add(BetweenFactor<Pose3>(latestFrameIDLoopCloure, closestHistoryFrameID, poseFrom.between(poseTo), constraintNoise));
        isam->update(gtSAMgraph);
        isam->update();
        // isam->update();
        gtSAMgraph.resize(0);

        last_knn_search = latestFrameIDLoopCloure;




        aLoopIsClosed = true;
        

    }

    Pose3 pclPointTogtsamPose3(PointTypePose thisPoint){ // camera frame to lidar frame
    	return Pose3(Rot3::RzRyRx(double(thisPoint.yaw), double(thisPoint.roll), double(thisPoint.pitch)),
                           Point3(double(thisPoint.z),   double(thisPoint.x),    double(thisPoint.y)));
    }

    Eigen::Affine3f pclPointToAffine3fCameraToLidar(PointTypePose thisPoint){ // camera frame to lidar frame
    	return pcl::getTransformation(thisPoint.z, thisPoint.x, thisPoint.y, thisPoint.yaw, thisPoint.roll, thisPoint.pitch);
    }

    void extractSurroundingKeyFrames(){

        if (cloudKeyPoses3D->points.empty() == true)
            return;	
		
	if (loopClosureEnableFlag == true){
	    // only use recent key poses for graph building
            if (recentCornerCloudKeyFrames.size() < surroundingKeyframeSearchNum){ // queue is not full (the beginning of mapping or a loop is just closed)
                // clear recent key frames queue
		recentCornerCloudKeyFrames. clear();
                recentSurfCloudKeyFrames.   clear();
                recentOutlierCloudKeyFrames.clear();
                int numPoses = cloudKeyPoses3D->points.size();
                for (int i = numPoses-1; i >= 0; --i){
                    int thisKeyInd = (int)cloudKeyPoses3D->points[i].intensity;
                    PointTypePose thisTransformation = cloudKeyPoses6D->points[thisKeyInd];
                    updateTransformPointCloudSinCos(&thisTransformation);
		    // extract surrounding map
                    recentCornerCloudKeyFrames. push_front(transformPointCloud(cornerCloudKeyFrames[thisKeyInd]));
                    recentSurfCloudKeyFrames.   push_front(transformPointCloud(surfCloudKeyFrames[thisKeyInd]));
                    recentOutlierCloudKeyFrames.push_front(transformPointCloud(outlierCloudKeyFrames[thisKeyInd]));
                    if (recentCornerCloudKeyFrames.size() >= surroundingKeyframeSearchNum)
                        break;
                }
            }else{  // queue is full, pop the oldest key frame and push the latest key frame
                if (latestFrameID != cloudKeyPoses3D->points.size() - 1){  // if the robot is not moving, no need to update recent frames

                    recentCornerCloudKeyFrames. pop_front();
                    recentSurfCloudKeyFrames.   pop_front();
                    recentOutlierCloudKeyFrames.pop_front();
		    // push latest scan to the end of queue
                    latestFrameID = cloudKeyPoses3D->points.size() - 1;
                    PointTypePose thisTransformation = cloudKeyPoses6D->points[latestFrameID];
                    updateTransformPointCloudSinCos(&thisTransformation);
                    recentCornerCloudKeyFrames. push_back(transformPointCloud(cornerCloudKeyFrames[latestFrameID]));
                    recentSurfCloudKeyFrames.   push_back(transformPointCloud(surfCloudKeyFrames[latestFrameID]));
                    recentOutlierCloudKeyFrames.push_back(transformPointCloud(outlierCloudKeyFrames[latestFrameID]));
                }
            }

            for (int i = 0; i < recentCornerCloudKeyFrames.size(); ++i){
                *laserCloudCornerFromMap += *recentCornerCloudKeyFrames[i];
                *laserCloudSurfFromMap   += *recentSurfCloudKeyFrames[i];
                *laserCloudSurfFromMap   += *recentOutlierCloudKeyFrames[i];
            }
	}else{
            surroundingKeyPoses->clear();
            surroundingKeyPosesDS->clear();
	    // extract all the nearby key poses and downsample them
	    kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D);
	    kdtreeSurroundingKeyPoses->radiusSearch(currentRobotPosPoint, (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis, 0);
	    for (int i = 0; i < pointSearchInd.size(); ++i)
                surroundingKeyPoses->points.push_back(cloudKeyPoses3D->points[pointSearchInd[i]]);
	    downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
	    downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);
	    // delete key frames that are not in surrounding region
            int numSurroundingPosesDS = surroundingKeyPosesDS->points.size();
            for (int i = 0; i < surroundingExistingKeyPosesID.size(); ++i){
                bool existingFlag = false;
                for (int j = 0; j < numSurroundingPosesDS; ++j){
                    if (surroundingExistingKeyPosesID[i] == (int)surroundingKeyPosesDS->points[j].intensity){
                        existingFlag = true;
                        break;
                    }
                }
                if (existingFlag == false){
                    surroundingExistingKeyPosesID.   erase(surroundingExistingKeyPosesID.   begin() + i);
                    surroundingCornerCloudKeyFrames. erase(surroundingCornerCloudKeyFrames. begin() + i);
                    surroundingSurfCloudKeyFrames.   erase(surroundingSurfCloudKeyFrames.   begin() + i);
                    surroundingOutlierCloudKeyFrames.erase(surroundingOutlierCloudKeyFrames.begin() + i);
                    --i;
                }
            }
	    // add new key frames that are not in calculated existing key frames
            for (int i = 0; i < numSurroundingPosesDS; ++i) {
                bool existingFlag = false;
                for (auto iter = surroundingExistingKeyPosesID.begin(); iter != surroundingExistingKeyPosesID.end(); ++iter){
                    if ((*iter) == (int)surroundingKeyPosesDS->points[i].intensity){
                        existingFlag = true;
                        break;
                    }
                }
                if (existingFlag == true){
                    continue;
                }else{
                    int thisKeyInd = (int)surroundingKeyPosesDS->points[i].intensity;
                    PointTypePose thisTransformation = cloudKeyPoses6D->points[thisKeyInd];
                    updateTransformPointCloudSinCos(&thisTransformation);
                    surroundingExistingKeyPosesID.   push_back(thisKeyInd);
                    surroundingCornerCloudKeyFrames. push_back(transformPointCloud(cornerCloudKeyFrames[thisKeyInd]));
                    surroundingSurfCloudKeyFrames.   push_back(transformPointCloud(surfCloudKeyFrames[thisKeyInd]));
                    surroundingOutlierCloudKeyFrames.push_back(transformPointCloud(outlierCloudKeyFrames[thisKeyInd]));
                }
            }

            for (int i = 0; i < surroundingExistingKeyPosesID.size(); ++i) {
                *laserCloudCornerFromMap += *surroundingCornerCloudKeyFrames[i];
                *laserCloudSurfFromMap   += *surroundingSurfCloudKeyFrames[i];
                *laserCloudSurfFromMap   += *surroundingOutlierCloudKeyFrames[i];
            }
	}
	// Downsample the surrounding corner key frames (or map)
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->points.size();
	// Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->points.size();
    }

    void downsampleCurrentScan(){

        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->points.size();

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->points.size();

        laserCloudOutlierLastDS->clear();
        downSizeFilterOutlier.setInputCloud(laserCloudOutlierLast);
        downSizeFilterOutlier.filter(*laserCloudOutlierLastDS);
        laserCloudOutlierLastDSNum = laserCloudOutlierLastDS->points.size();

        laserCloudSurfTotalLast->clear();
        laserCloudSurfTotalLastDS->clear();
        *laserCloudSurfTotalLast += *laserCloudSurfLastDS;
        *laserCloudSurfTotalLast += *laserCloudOutlierLastDS;
        downSizeFilterSurf.setInputCloud(laserCloudSurfTotalLast);
        downSizeFilterSurf.filter(*laserCloudSurfTotalLastDS);
        laserCloudSurfTotalLastDSNum = laserCloudSurfTotalLastDS->points.size();
    }

    void cornerOptimization(int iterCount){

        updatePointAssociateToMapSinCos();
        for (int i = 0; i < laserCloudCornerLastDSNum; i++) {
            pointOri = laserCloudCornerLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel);
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
            
            if (pointSearchSqDis[4] < 1.0) {
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++) {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5;

                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                cv::eigen(matA1, matD1, matV1);

                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {

                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                                    * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                                    * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))
                                    * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float ld2 = a012 / l12;

                    float s = 1 - 0.9 * fabs(ld2);

                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;

                    if (s > 0.1) {
                        laserCloudOri->push_back(pointOri);
                        coeffSel->push_back(coeff);
                    }
                }
            }
        }
    }

    void surfOptimization(int iterCount){
        updatePointAssociateToMapSinCos();
        for (int i = 0; i < laserCloudSurfTotalLastDSNum; i++) {
            pointOri = laserCloudSurfTotalLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel); 
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            if (pointSearchSqDis[4] < 1.0) {
                for (int j = 0; j < 5; j++) {
                    matA0.at<float>(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0.at<float>(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0.at<float>(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }
                cv::solve(matA0, matB0, matX0, cv::DECOMP_QR);

                float pa = matX0.at<float>(0, 0);
                float pb = matX0.at<float>(1, 0);
                float pc = matX0.at<float>(2, 0);
                float pd = 1;

                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                            + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    if (s > 0.1) {
                        laserCloudOri->push_back(pointOri);
                        coeffSel->push_back(coeff);
                    }
                }
            }
        }
    }

    // bool LMOptimization(int iterCount, gtsam::Vector6 LMScore){
    double LMOptimization(int iterCount){
        float srx = sin(transformTobeMapped[0]);
        float crx = cos(transformTobeMapped[0]);
        float sry = sin(transformTobeMapped[1]);
        float cry = cos(transformTobeMapped[1]);
        float srz = sin(transformTobeMapped[2]);
        float crz = cos(transformTobeMapped[2]);

        int laserCloudSelNum = laserCloudOri->points.size();
        if (laserCloudSelNum < 50) {
            return false;
        }

        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
        for (int i = 0; i < laserCloudSelNum; i++) {
            pointOri = laserCloudOri->points[i];
            coeff = coeffSel->points[i];

            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x 
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;

            matA.at<float>(i, 0) = arx;
            matA.at<float>(i, 1) = ary;
            matA.at<float>(i, 2) = arz;
            matA.at<float>(i, 3) = coeff.x;
            matA.at<float>(i, 4) = coeff.y;
            matA.at<float>(i, 5) = coeff.z;
            matB.at<float>(i, 0) = -coeff.intensity;
        }
        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        if (iterCount == 0) {
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate) {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        // LMScore << 
        //     abs(matX.at<float>(3, 0) * 100), abs(matX.at<float>(4, 0) * 100), abs(matX.at<float>(5, 0) * 100), 
        //     abs(pcl::rad2deg(matX.at<float>(0, 0))), abs(pcl::rad2deg(matX.at<float>(1, 0))), abs(pcl::rad2deg(matX.at<float>(2, 0)));
        double LMNoise = sqrt(pow(deltaR,2) + pow(deltaT, 2));
        
        if (deltaR < 0.05 && deltaT < 0.05) {
            return LMNoise;
        }
        return -1;
    }

    void scan2MapOptimization(){

        if (laserCloudCornerFromMapDSNum > 10 && laserCloudSurfFromMapDSNum > 100) {

            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            for (int iterCount = 0; iterCount < 10; iterCount++) {

                laserCloudOri->clear();
                coeffSel->clear();

                cornerOptimization(iterCount);
                surfOptimization(iterCount);
                
                LMNoise = LMOptimization(iterCount);
                if (LMNoise != -1)
                    break;              
            }

            transformUpdate();
        }
    }


    void saveKeyFramesAndFactor(){

        currentRobotPosPoint.x = transformAftMapped[3];
        currentRobotPosPoint.y = transformAftMapped[4];
        currentRobotPosPoint.z = transformAftMapped[5];

        bool saveThisKeyFrame = true;
        if (sqrt((previousRobotPosPoint.x-currentRobotPosPoint.x)*(previousRobotPosPoint.x-currentRobotPosPoint.x)
                +(previousRobotPosPoint.y-currentRobotPosPoint.y)*(previousRobotPosPoint.y-currentRobotPosPoint.y)
                +(previousRobotPosPoint.z-currentRobotPosPoint.z)*(previousRobotPosPoint.z-currentRobotPosPoint.z)) <1){//1 for 16 0.3 or 1
            saveThisKeyFrame = false;
        }

        

        if (saveThisKeyFrame == false && !cloudKeyPoses3D->points.empty())
        	return;

        previousRobotPosPoint = currentRobotPosPoint;
	/**
         * update grsam graph
         */
        
        // double LMNoise = sqrt(pow(LMScore[0],2) + pow(LMScore[1],2) + pow(LMScore[2],2) + pow(LMScore[3],2) + pow(LMScore[4], 2) + pow(LMScore[5],2));
        gtsam::Vector Vector6(6);
        if (LMNoise <= 1e-6)
            LMNoise = 1e-6;
        Vector6 << LMNoise, LMNoise, LMNoise, LMNoise, LMNoise, LMNoise;
        odometryNoise = noiseModel::Diagonal::Variances(Vector6);

        if (cloudKeyPoses3D->points.empty()){
            gtSAMgraph.add(PriorFactor<Pose3>(0, Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0], transformTobeMapped[1]),
                                                       		 Point3(transformTobeMapped[5], transformTobeMapped[3], transformTobeMapped[4])), priorNoise));
            initialEstimate.insert(0, Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0], transformTobeMapped[1]),
                                                  Point3(transformTobeMapped[5], transformTobeMapped[3], transformTobeMapped[4])));
            for (int i = 0; i < 6; ++i)
            	transformLast[i] = transformTobeMapped[i];
        }
        else{
            gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(transformLast[2], transformLast[0], transformLast[1]),
                                                Point3(transformLast[5], transformLast[3], transformLast[4]));
            gtsam::Pose3 poseTo   = Pose3(Rot3::RzRyRx(transformAftMapped[2], transformAftMapped[0], transformAftMapped[1]),
                                                Point3(transformAftMapped[5], transformAftMapped[3], transformAftMapped[4]));
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->points.size()-1, cloudKeyPoses3D->points.size(), poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->points.size(), Pose3(Rot3::RzRyRx(transformAftMapped[2], transformAftMapped[0], transformAftMapped[1]),
                                                                     		   Point3(transformAftMapped[5], transformAftMapped[3], transformAftMapped[4])));
        }
	/**
         * update iSAM
         */
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();
        
        gtSAMgraph.resize(0);
	initialEstimate.clear();

	/**
         * save key poses
         */
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);

        thisPose3D.x = latestEstimate.translation().y();
        thisPose3D.y = latestEstimate.translation().z();
        thisPose3D.z = latestEstimate.translation().x();
        thisPose3D.intensity = cloudKeyPoses3D->points.size(); // this can be used as index
        cloudKeyPoses3D->push_back(thisPose3D);

        {
            sensor_msgs::PointCloud2 msg;
            pcl::toROSMsg(*originLidar, msg);
            msg.header.frame_id = std::to_string(cloudKeyPoses6D->size()); // TODO: maybe need another index rule
            msg.header.stamp = originTime; // ros::Time::now();
            pubOriginLidar.publish(msg);
        }
        {
            cv_bridge::CvImage msg;
            msg.image = originImage;
            msg.encoding = sensor_msgs::image_encodings::RGB8;
            msg.header.frame_id = std::to_string(cloudKeyPoses6D->size());
            msg.header.stamp = originTime;
            pubOriginImage.publish(msg.toImageMsg());
        }

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity; // this can be used as index
        thisPose6D.roll  = latestEstimate.rotation().pitch();
        thisPose6D.pitch = latestEstimate.rotation().yaw();
        thisPose6D.yaw   = latestEstimate.rotation().roll(); // in camera frame
        thisPose6D.time = timeLaserOdometry;
        cloudKeyPoses6D->push_back(thisPose6D);

        // nav_msgs::Path

        geometry_msgs::PoseStamped curr_pose;
        curr_pose.header.frame_id = "camera_init";
        curr_pose.pose.position.x = thisPose6D.x;
        curr_pose.pose.position.y = thisPose6D.y;
        curr_pose.pose.position.z = thisPose6D.z;

        // geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw
        //                           (thisPose6D.yaw, -thisPose6D.roll, -thisPose6D.pitch);
        Rot3 r1 = Rot3::Rx(thisPose6D.roll);
        Rot3 r2 = Rot3::Ry(thisPose6D.pitch);
        Rot3 r3 = Rot3::Rz(thisPose6D.yaw);
        curr_pose.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        // Rot3 r = r2*r1*r3;
        // Rot3 rr = r2*r1*r3;
        Rot3 r  = r2 * r1 * r3;
        Quaternion q(r.matrix());
        // Quaternion qq(rr.matrix());
        // std::cout << q1.x() << " " << -q1.z() << " " <<q1.x() << " "<<q1.w() <<std::endl;
        // std::cout << q.coeffs ()  << std::endl;
        // std::cout << qq.coeffs ()  << std::endl;
        curr_pose.pose.orientation.x = 0;
        curr_pose.pose.orientation.y = 0;
        curr_pose.pose.orientation.z = 0;
        curr_pose.pose.orientation.w = 1;

        path_msg.poses.push_back(curr_pose);
	/**
         * save updated transform
         */
        if (cloudKeyPoses3D->points.size() > 1){
            transformAftMapped[0] = latestEstimate.rotation().pitch();
            transformAftMapped[1] = latestEstimate.rotation().yaw();
            transformAftMapped[2] = latestEstimate.rotation().roll();
            transformAftMapped[3] = latestEstimate.translation().y();
            transformAftMapped[4] = latestEstimate.translation().z();
            transformAftMapped[5] = latestEstimate.translation().x();

            for (int i = 0; i < 6; ++i){
            	transformLast[i] = transformAftMapped[i];
            	transformTobeMapped[i] = transformAftMapped[i];
            }
        }

        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisOutlierKeyFrame(new pcl::PointCloud<PointType>());

        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);
        pcl::copyPointCloud(*laserCloudOutlierLastDS, *thisOutlierKeyFrame);

        if(newcloudsave && timefullcloud-timeLaserOdometry < 0.05)
        {
            // pcl::PointCloud<PointType>::Ptr cloud1(new pcl::PointCloud<PointType>);
            // pcl::copyPointCloud(*cloud_to_save, *cloud1);

            // keyframes_save.push_back(cloud1);        

            // // pcl::io::savePCDFileBinary("/media/yingwang/DATADisk/dp/dp" + std::to_string(current_frame_id++) + ".pcd", *cloud_to_save);
            newcloudsave = false;
            cv::Mat1b m = LidarIris::GetIris(*velodyne_points);

            // LidarIris::FeatureDesc fd = iris.UpdateFrame(m, cloudKeyPoses3D->points.size() - 1);
            LidarIris::FeatureDesc fd = iris.GetFeature(m);
            fd.index = cloudKeyPoses3D->points.size() - 1;
            // mats.push_back(m.clone());
            fds.push_back(fd);
            //// std::cout << m.size() << " " << fd.index << std::endl;
            //// std::cout << " push keyframe" << std::endl;

        }

        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);
        outlierCloudKeyFrames.push_back(thisOutlierKeyFrame);

        // std::cout << timeimg << " " << timeLaserCloudCornerLast << std::endl;
        
        // if(newimg && timeimg - timeLaserCloudCornerLast < 0.05)
        // if(false)
        // {
        //     wy::KeyFrame* kf = new wy::KeyFrame(img, porb_vocabulary, porb_extractor, cloudKeyPoses3D->points.size() - 1);
        //     kf->extracORBFeaturs();
        //     kf->computeBoW();
        //     kf->mTimeStamp = timeLaserOdometry;
        //     pkfdb->add(kf);
        //     vpkfs.push_back(kf);
        //     // qpkfs.push(kf);
        //     // pcurrkf = kf;
        //     newimg = false;
        //     ROS_INFO("Add an image to db");

        //     if(image_pub_.getNumSubscribers () )
        //     {
        //         sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", color_).toImageMsg();
        //         image_pub_.publish(msg);
        //     }
        //     // std::cout << testnum++ << std::endl;

        // }

        if(newgps)
        {
            gps_p <<"T," << latitude << "," << longtitude <<","<<alt << std::endl;
            newgps = false;
        }
    }

    void poseCov() {
        posewithCovariance.data.clear();
        pcl::PointCloud<pcl::PointXYZI> tempCloud;
        int numPoses = isamCurrentEstimate.size();
        for (int i = 0; i < numPoses; ++i){
            pcl::PointXYZI temp;
            temp.x = cloudKeyPoses3D->points[i].x;
            temp.y = cloudKeyPoses3D->points[i].y;
            temp.z = cloudKeyPoses3D->points[i].z;
            temp.intensity = isam->marginalCovariance(i)(0,0);
            tempCloud.push_back(temp);
        }
        pcl::toROSMsg(tempCloud, posewithCovariance);
    }

    void correctPoses(){
    	if (aLoopIsClosed == true){
            recentCornerCloudKeyFrames. clear();
            recentSurfCloudKeyFrames.   clear();
            recentOutlierCloudKeyFrames.clear();
	        // update key poses
            int numPoses = isamCurrentEstimate.size();

            path_msg.poses.clear();
        
            for (int i = 0; i < numPoses; ++i){
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().z();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().x();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();
                cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                PointTypePose thisPose6D = cloudKeyPoses6D->points[i];

                geometry_msgs::PoseStamped curr_pose;
                curr_pose.header.frame_id = "camera_init";
                curr_pose.pose.position.x = thisPose6D.x;
                curr_pose.pose.position.y = thisPose6D.y;
                curr_pose.pose.position.z = thisPose6D.z;

                tf::Quaternion quat = tf::createQuaternionFromRPY(thisPose6D.pitch, thisPose6D.roll, thisPose6D.yaw);
                curr_pose.pose.orientation.x = 0;
                curr_pose.pose.orientation.y = 0;
                curr_pose.pose.orientation.z = 0;
                curr_pose.pose.orientation.w = 1;

                path_msg.poses.push_back(curr_pose);
            }
        }
    }

    void clearCloud(){
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear();  
        laserCloudCornerFromMapDS->clear();
        laserCloudSurfFromMapDS->clear();   
    }

    void run(){

        if (newLaserCloudCornerLast  && std::abs(timeLaserCloudCornerLast  - timeLaserOdometry) < 0.005 &&
            newLaserCloudSurfLast    && std::abs(timeLaserCloudSurfLast    - timeLaserOdometry) < 0.005 &&
            newLaserCloudOutlierLast && std::abs(timeLaserCloudOutlierLast - timeLaserOdometry) < 0.005 &&
            newLaserOdometry)
        {

            newLaserCloudCornerLast = false; newLaserCloudSurfLast = false; newLaserCloudOutlierLast = false; newLaserOdometry = false;

            std::lock_guard<std::mutex> lock(mtx);

            if (timeLaserOdometry - timeLastProcessing >= mappingProcessInterval) {

                timeLastProcessing = timeLaserOdometry;

                transformAssociateToMap();

                extractSurroundingKeyFrames();

                downsampleCurrentScan();

                scan2MapOptimization();

                saveKeyFramesAndFactor();

                correctPoses();

                poseCov();

                publishTF();

                publishKeyPosesAndFrames();

                clearCloud();
            }

            // if(cloudKeyPoses6D->size() >= 550)
            // 会消耗大量的时间
            if(b_save_map)
            {
                saveMap();
                ros::shutdown();
            }
        }
        
    }


    // 添加 GPS 数据到因子图优化
    void addGPSFactor()
    {
        std::deque<int> gpsQueue;
        if(gpsQueue.empty())
        {
            return;
        }

        if(cloudKeyPoses3D->points.empty())
        {
            
        }
    }





    void saveMap()
        {
        globalMapKeyFrames->clear();

        pcl::PointCloud<PointType>::Ptr wy_map(new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr wy_mapDS(new pcl::PointCloud<PointType>);
        std::cout << cornerCloudKeyFrames.size() << std::endl;
        std::cout << cloudKeyPoses3D->points.size() << std::endl;
        std::cout << cloudKeyPoses6D->points.size() << std::endl;
        std::cout << keyframes_save.size() << std::endl;


        pcl::io::savePCDFileBinary(fileDirectory + "lego_trajectory.pcd", *cloudKeyPoses3D);
        std::cout << "lego_trajectory.pcd saved!" << std::endl;

        std::ofstream of(fileDirectory + "lego_trajectory.txt");
        
        for(int i = 0; i < cloudKeyPoses6D->size(); i++)
        {
            of.setf(ios::fixed,ios::floatfield);
            of.precision(0);
            of << cloudKeyPoses6D->points[i].time << " ";
            of.precision(5);




            of << cloudKeyPoses6D->points[i].x << " ";
            of << cloudKeyPoses6D->points[i].y << " ";
            of << cloudKeyPoses6D->points[i].z << " ";
            // of << cloudKeyPoses6D->points[i].intensity << " ";

            Eigen::Matrix3f rotation_matrix = Eigen::AngleAxisf(cloudKeyPoses6D->points[i].pitch, Eigen::Vector3f::UnitY()).matrix()
                                        * Eigen::AngleAxisf(cloudKeyPoses6D->points[i].roll, Eigen::Vector3f::UnitX()).matrix()
                                        * Eigen::AngleAxisf(cloudKeyPoses6D->points[i].yaw, Eigen::Vector3f::UnitZ()).matrix();
            Eigen::Quaternionf q_f=Eigen::Quaternionf(rotation_matrix);
            q_f.normalize();
            // of << cloudKeyPoses6D->points[i].roll << " ";
            // of << cloudKeyPoses6D->points[i].pitch << " ";
            // of << cloudKeyPoses6D->points[i].yaw << " ";

            of << q_f.x()<< " ";
            of << q_f.y() << " ";
            of << q_f.z() << " ";
            of << q_f.w() << std::endl;
            
        }
        std::cout << "lego_trajectory.txt save successful" << std::endl;
        of.close();

        for(int i = 0; i < cloudKeyPoses6D->points.size(); ++i)
        {
            *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[i],   &cloudKeyPoses6D->points[i]);
        }
        pcl::io::savePCDFileBinary(fileDirectory + "lego_corner.pcd", *globalMapKeyFrames);
        *wy_map += *globalMapKeyFrames;
        globalMapKeyFrames->clear();

        for(int i = 0; i < cloudKeyPoses6D->points.size(); i++)
        {
			*globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
            // *globalMapKeyFrames += *transformPointCloud(outlierCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
        }

        pcl::io::savePCDFileBinary(fileDirectory + "lego_surf.pcd", *globalMapKeyFrames);
        *wy_map += *globalMapKeyFrames;

        globalMapKeyFrames->clear();

        for(int i = 0; i < cloudKeyPoses6D->points.size(); i++)
        {
			// *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
            *globalMapKeyFrames += *transformPointCloud(outlierCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
        }

        *wy_map += *globalMapKeyFrames;

        // for(int i = 0; i < keyframes_save.size(); i++)
        // {
        //     pcl::io::savePCDFileBinary(fileDirectory+"keyframes/" + std::to_string(i+1) + ".pcd", *keyframes_save[i]);
        // }

        downSizeFilterGlobalMapKeyFrames.setInputCloud(wy_map);
        downSizeFilterGlobalMapKeyFrames.filter(*wy_mapDS);
        pcl::io::savePCDFileBinary(fileDirectory + "lego_finalCloud.pcd", *wy_mapDS);
        std::cout << "finalCloud save successful" << std::endl;

        // 局部地图生成　用于后面二维障碍图　整合杨航的代码

        Eigen::Isometry3f initTrans = Eigen::Isometry3f::Identity();
        Eigen::Isometry3f odomTrans = Eigen::Isometry3f::Identity();

        int k = 0;
        int submapNum = 0;
        pcl::PointCloud<pcl::PointXYZI>::Ptr inputCloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr subMapCloud(new pcl::PointCloud<pcl::PointXYZI>);
        std::ofstream fout;
        fout.open(fileDirectory + "submap/trajectory.txt");
        for(int i = 0; i < keyframes_save.size(); i++)
        {
            Eigen::Matrix3f rotation_matrix = Eigen::AngleAxisf(cloudKeyPoses6D->points[i].pitch, Eigen::Vector3f::UnitY()).matrix()
                                        * Eigen::AngleAxisf(cloudKeyPoses6D->points[i].roll, Eigen::Vector3f::UnitX()).matrix()
                                        * Eigen::AngleAxisf(cloudKeyPoses6D->points[i].yaw, Eigen::Vector3f::UnitZ()).matrix();
            pcl::copyPointCloud(*keyframes_save[i], *inputCloud);
            CoordinateTransform(inputCloud);
            ExactLowerScan(inputCloud);
            std::cout << k << " ";

            if(k == 100 || i == keyframes_save.size() -1)
            {
                cout << "saved an submap : " << submapNum << endl;
                cout << "savePCDFileBinary return value : ";
                cout << pcl::io::savePCDFileBinary(fileDirectory + "submap/" + std::to_string(submapNum++) + ".pcd", *subMapCloud);
                cout << endl;
                subMapCloud->clear();
                k = 0;
                // initTrans = Eigen::Isometry3d::Identity();
                // initTrans.prerotate(rotation_matrix);
                // initTrans.pretranslate(pose);

                
                fout << cloudKeyPoses6D->points[i-1].x << " ";
                fout << cloudKeyPoses6D->points[i-1].y << " ";
                fout << cloudKeyPoses6D->points[i-1].z << " ";
                fout << cloudKeyPoses6D->points[i-1].intensity << " ";
                fout << cloudKeyPoses6D->points[i-1].roll << " ";
                fout << cloudKeyPoses6D->points[i-1].pitch << " ";
                fout << cloudKeyPoses6D->points[i-1].yaw << " ";
                fout << cloudKeyPoses6D->points[i-1].time << std::endl;
    
            }

        Eigen::Vector3f pose(cloudKeyPoses6D->points[i].x, cloudKeyPoses6D->points[i].y, cloudKeyPoses6D->points[i].z);
        odomTrans = Eigen::Isometry3f::Identity();
        odomTrans.prerotate(rotation_matrix);
        odomTrans.pretranslate(pose);
        // cout << "odomTrans: \n" << odomTrans.matrix() << endl;
        // pcl::transformPointCloud(*inputCloud, *inputCloud, (odomTrans.inverse()*initTrans).matrix());
        pcl::transformPointCloud(*inputCloud, *inputCloud, odomTrans.matrix());

        *subMapCloud += *inputCloud;
        k++;
        }

        std::cout << "all submap saved" << std::endl;

        std::vector<std::vector<int>> MapGridArray;
        MapGridArray.resize(9);

        pcl::VoxelGrid<PointT> downSizeFilter;
        downSizeFilter.setLeafSize(0.05, 0.05, 0.05);
        PointCloud::Ptr originalSubMap (new PointCloud());
        PointCloud::Ptr downSampledCloud (new PointCloud());
        pcl::PointCloud<PointL>::Ptr downSampledCloudL (new pcl::PointCloud<PointL>());
        pcl::PointCloud<PointL>::Ptr inputCloud1(new pcl::PointCloud<PointL>());
        pcl::PointCloud<PointL>::Ptr labelCloud(new pcl::PointCloud<PointL>());
        pcl::PointCloud<PointL>::Ptr fullResMap(new pcl::PointCloud<PointL>());
        pcl::PointCloud<PointL>::Ptr worldMap(new pcl::PointCloud<PointL>());

        GroundPlaneFit gpt;
        std::ifstream fin(fileDirectory + "trajectory.txt", std::ios::in);

        for(int i = 0; i < submapNum; i++)
        {
            pcl::io::loadPCDFile(fileDirectory + "submap/" + std::to_string(i) + ".pcd", *originalSubMap);
            downSampledCloud->clear();
            // cout << "origin cloud size: " << originalSubMap->size() << " ";
            downSizeFilter.setInputCloud(originalSubMap);
            downSizeFilter.filter(*downSampledCloud);

            downSampledCloudL->clear();
                PointL point;

                for (size_t i = 0; i < downSampledCloud->points.size(); i++)
                {
                    point.x = downSampledCloud->points[i].x;
                    point.y = downSampledCloud->points[i].y;
                    point.z = downSampledCloud->points[i].z;
                    point.intensity = downSampledCloud->points[i].intensity;
                    point.ring = int(downSampledCloud->points[i].intensity);
                    point.label = 0;// 0 means uncluster
                    downSampledCloudL->points.push_back(point);
                }
                copyPointCloud(*downSampledCloudL, *labelCloud);


            for(int j = 0; j < 100; j++)
            {

                double x,y,z,w;
                fin >> x >> y >> z >> w; // w是每帧的idx
                Eigen::Vector3d pose(x, y, z);
                fin >> x >> y >> z >> w; // w是每帧的时间
                if(fin.eof()) break;

                // 哇,这个巨坑. loam里面从odom坐标变换到世界坐标系中是按ZXY顺序旋转的,但是旋转矩阵的乘法是相反的, Rot_y*Rot_x*Rot_z*X
                Eigen::Matrix3d rotation_matrix = Eigen::AngleAxisd(y, Eigen::Vector3d::UnitY()).matrix()
                                                * Eigen::AngleAxisd(x, Eigen::Vector3d::UnitX()).matrix()
                                                * Eigen::AngleAxisd(z, Eigen::Vector3d::UnitZ()).matrix();


                // For mark ground points and hold all points
                

                Eigen::Isometry3d odomTrans = Eigen::Isometry3d::Identity();

                copyPointCloud(*downSampledCloudL, *inputCloud1);

                odomTrans = Eigen::Isometry3d::Identity();
                odomTrans.prerotate(rotation_matrix);
                odomTrans.pretranslate(pose);
                // cout << "k: " << k << endl;
                transformPointCloud(inputCloud1, odomTrans.inverse());

                gpt.extract_road_from_submap_(inputCloud1);
                for (int i = 0; i < labelCloud->size(); i++)
                {
                    if(inputCloud1->points[i].label>0)
                        labelCloud->points[i].label = inputCloud1->points[i].label; //inputCloud->points[i].label;
                }
                
            }
                *worldMap += *labelCloud;


        }
        cout << worldMap->size() << std::endl;
        pcl::io::savePCDFileBinary(fileDirectory + "submap/worldMap" + ".pcd", *worldMap);
        int max_z, max_x, min_z, min_x;
        max_z = max_x = min_z = min_x = 0;
        for (auto &pt : *worldMap)
        {
            if(pt.z<min_z)
                min_z = pt.z;
            if(pt.z>max_z)
                max_z = pt.z;
            if (pt.x < min_x)
                min_x = pt.x;
            if(pt.x>max_x)
                max_x = pt.x;
        }
        printf("point range: %d %d %d %d \n", max_x, min_x, max_z, min_z);
        int width = (max_x - min_x + 5) / 5 * 20 * 5;
        int height = (max_z - min_z + 5) / 5 * 20 * 5;
        cout << height << " " << width << endl;
        int offsetx = max_x + 1, offsety = max_z + 1;
        cout << "offsetx: " << offsetx << ", offsety: " << offsety << endl;

        // cv::Mat roadcurb(height, width, CV_8UC1, Scalar(0));
        // for (int i = 0; i < worldMap->size();i++)
        // {
        //     if(worldMap->points[i].label==0u)
        //     {
        //         int x = (worldMap->points[i].x + offsetx) / 0.05;
        //         int y = (worldMap->points[i].z + offsety) / 0.05;
        //         if (x >= 0 && x < width && y >= 0 && y < height)
        //         {
        //             roadcurb.at<uchar>(y, x) +=20;
        //         }
        //     }
        // }
        // cv::threshold(roadcurb,roadcurb,70,255,CV_THRESH_BINARY);
        // cv::imwrite(inputpath + "result/roadcurb.bmp", roadcurb);
        cv::Mat roadMap(height, width, CV_8UC1, Scalar(0));
        for (int i = 0; i < worldMap->size(); i++)
        {
            if(worldMap->points[i].label!=0)
            {
                int x = (offsetx - worldMap->points[i].x) / 0.05;
                int y = (offsety - worldMap->points[i].z ) / 0.05;
                if (x >= 0 && x < width && y >= 0 && y < height)
                {
                    roadMap.at<uchar>(y, x) = 255;
                }
            }
        }
        // cv::imwrite(fileDirectory + "result/roadMapOrg.bmp", roadMap);
        cv::imwrite(fileDirectory + "result/roadMapOrg.jpg", roadMap);

        // for (int i = 0; i < roadMap.rows;i++)
        // {
        //     for (int j = 0; j < roadMap.cols;j++)
        //     {
        //         if(roadMap.at<uchar>(i, j) == 255 && roadcurb.at<uchar>(i, j)==255)
        //         {
        //             roadMap.at<uchar>(i, j) = 0;
        //         }
        //     }
        // }
        cv::Mat filledImg(height, width, CV_8UC1, Scalar(0));
        fillHole(roadMap, filledImg, Point(offsetx, offsety));
        //cv::Mat filledImg(height, width, CV_8UC1, Scalar(0));
        fillHole(roadMap, filledImg, Point(offsetx, offsety));
        int dilation_size = 50;//需要根据最终地图效果更改　一般20 kitti50
        Mat dilation = getStructuringElement(MORPH_RECT, Size(2 * dilation_size + 1, 2 * dilation_size + 1), Point(dilation_size, dilation_size)); // MORPH_RECT / MORPH_CROSS / MORPH_ELLIPSE
        cv::dilate(filledImg, filledImg, dilation);
        int erosion_size = 50;//需要根据最终地图效果更改　一般20 kitti50
        Mat erosion = getStructuringElement(MORPH_RECT, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size)); // MORPH_RECT / MORPH_CROSS / MORPH_ELLIPSE
        cv::erode(filledImg, filledImg, erosion);
        cv::imwrite(fileDirectory + "result/" + to_string(offsetx * 20) + "_" + to_string(offsety * 20) + ".jpg", filledImg);
        // cv::imwrite(fileDirectory + "result/" + to_string(offsetx * 20) + "_" + to_string(offsety * 20) + ".bmp", filledImg);
        }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lego_loam");

    ROS_INFO("\033[1;32m---->\033[0m Map Optimization Started.");

    // ORB_SLAM2::ORBVocabulary* porbvocabulary = new ORB_SLAM2::ORBVocabulary();
    
    // std::string strVocFile("/home/yingwang/ros_ws/lego_ws/src/LeGO-LOAM/LeGO-LOAM/ORB_SLAM2/build/ORBvoc.txt");
    // std::string strSettingFile("/home/yingwang/ros_ws/lego_ws/src/LeGO-LOAM/LeGO-LOAM/ORB_SLAM2/build/KITTI04-12.yaml");

    // bool bvocload = false;
    // if(has_suffix(strVocFile, ".txt"))
    //     bvocload = porbvocabulary->loadFromTextFile(strVocFile);
    // else
    //     bvocload = false;


    // if(!bvocload)
    // {
    //     cerr << "Wrong path to vocabulary. " << endl;
    //     cerr << "Falied to open at: " << strVocFile << endl;
    //     exit(-1);
    // }
    // cout << "Vocabulary loaded!" << endl << endl;

    // cv::FileStorage fSettings(strSettingFile, cv::FileStorage::READ);
    // int nFeatures = fSettings["ORBextractor.nFeatures"];
    // float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    // int nLevels = fSettings["ORBextractor.nLevels"];
    // int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    // int fMinThFAST = fSettings["ORBextractor.minThFAST"];


    // ORB_SLAM2::ORBextractor* porbextractor = new ORB_SLAM2::ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    mapOptimization MO;

    std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

    ros::Rate rate(200);
    while (ros::ok())
    {
        ros::spinOnce();
        MO.run();
        rate.sleep();
    }
    // MO.saveMap();

    cout <<"join 1" << std::endl;
    loopthread.join();
    visualizeMapThread.join();
    cout << "join 2" << std::endl;

    return 0;
}
