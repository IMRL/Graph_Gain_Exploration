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

#include "utility.h"
#include <opencv2/opencv.hpp>


// #define SAVE_EVERY_SACN

#ifdef SAVE_EVERY_SACN
    #include <pcl-1.7/pcl/io/pcd_io.h>
    int kframeid = 1;
#endif

class ImageProjection{
private:

    ros::NodeHandle nh;

    ros::Subscriber subLaserCloud;
    
    ros::Publisher pubFullCloud;
    ros::Publisher pubFullInfoCloud;

    ros::Publisher pubGroundCloud;
    ros::Publisher pubSegmentedCloud;
    ros::Publisher pubSegmentedCloudPure;
    ros::Publisher pubSegmentedCloudInfo;
    ros::Publisher pubOutlierCloud;

    pcl::PointCloud<PointType>::Ptr laserCloudIn;

    pcl::PointCloud<PointType>::Ptr fullCloud; // projected velodyne raw cloud, but saved in the form of 1-D matrix
    pcl::PointCloud<PointType>::Ptr fullInfoCloud; // same as fullCloud, but with intensity - range

    pcl::PointCloud<PointType>::Ptr groundCloud;
    pcl::PointCloud<PointType>::Ptr segmentedCloud;
    pcl::PointCloud<PointType>::Ptr segmentedCloudPure;
    pcl::PointCloud<PointType>::Ptr outlierCloud;

    pcl::PointCloud<PointType>::Ptr velodyne_points;
    ros::Publisher pubVelodyne;

    // 潘多拉点云
    // pcl::PointCloud<PPoint>::Ptr pando_cloud;
    pcl::PointCloud<PointType>::Ptr pando_cloud;

    PointType nanPoint; // fill in fullCloud at each iteration

    cv::Mat rangeMat; // range matrix for range image
    cv::Mat labelMat; // label matrix for segmentaiton marking
    cv::Mat groundMat; // ground matrix for ground cloud marking

    cv::Mat depthMat;// 16* 360
    int labelCount;

    float startOrientation;
    float endOrientation;

    cloud_msgs::cloud_info segMsg; // info of segmented cloud
    std_msgs::Header cloudHeader;

    std::vector<std::pair<uint8_t, uint8_t> > neighborIterator; // neighbor iterator for segmentaiton process

    uint16_t *allPushedIndX; // array for tracking points of a segmented object
    uint16_t *allPushedIndY;

    uint16_t *queueIndX; // array for breadth-first search process of segmentation
    uint16_t *queueIndY;

public:
    ImageProjection():
        nh("~"){

        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 1, &ImageProjection::cloudHandler, this);

        pubFullCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_projected", 1);
        pubFullInfoCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_info", 1);

        pubGroundCloud = nh.advertise<sensor_msgs::PointCloud2> ("/ground_cloud", 1);
        pubSegmentedCloud = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud", 1);
        pubSegmentedCloudPure = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud_pure", 1);
        pubSegmentedCloudInfo = nh.advertise<cloud_msgs::cloud_info> ("/segmented_cloud_info", 1);
        pubOutlierCloud = nh.advertise<sensor_msgs::PointCloud2> ("/outlier_cloud", 1);

        pubVelodyne = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_points_2", 1);
        nanPoint.x = std::numeric_limits<float>::quiet_NaN();
        nanPoint.y = std::numeric_limits<float>::quiet_NaN();
        nanPoint.z = std::numeric_limits<float>::quiet_NaN();
        nanPoint.intensity = -1;

        allocateMemory();
        resetParameters();
    }

    void allocateMemory(){

        // pando_cloud.reset(new pcl::PointCloud<PPoint>());
        pando_cloud.reset(new pcl::PointCloud<PointType>());

        laserCloudIn.reset(new pcl::PointCloud<PointType>());

        fullCloud.reset(new pcl::PointCloud<PointType>());
        fullInfoCloud.reset(new pcl::PointCloud<PointType>());

        groundCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloudPure.reset(new pcl::PointCloud<PointType>());
        outlierCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);
        fullInfoCloud->points.resize(N_SCAN*Horizon_SCAN);

        segMsg.startRingIndex.assign(N_SCAN, 0);
        segMsg.endRingIndex.assign(N_SCAN, 0);

        segMsg.segmentedCloudGroundFlag.assign(N_SCAN*Horizon_SCAN, false);
        segMsg.segmentedCloudColInd.assign(N_SCAN*Horizon_SCAN, 0);
        segMsg.segmentedCloudRange.assign(N_SCAN*Horizon_SCAN, 0);

        std::pair<int8_t, int8_t> neighbor;
        neighbor.first = -1; neighbor.second =  0; neighborIterator.push_back(neighbor);
        neighbor.first =  0; neighbor.second =  1; neighborIterator.push_back(neighbor);
        neighbor.first =  0; neighbor.second = -1; neighborIterator.push_back(neighbor);
        neighbor.first =  1; neighbor.second =  0; neighborIterator.push_back(neighbor);

        allPushedIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        allPushedIndY = new uint16_t[N_SCAN*Horizon_SCAN];

        queueIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        queueIndY = new uint16_t[N_SCAN*Horizon_SCAN];

        velodyne_points.reset(new pcl::PointCloud<PointType>());
    }

    void resetParameters(){
        laserCloudIn->clear();
        groundCloud->clear();
        segmentedCloud->clear();
        segmentedCloudPure->clear();
        outlierCloud->clear();

        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
        groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
        labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));
        // depthMat = cv::Mat(16, 360, CV_32F, cv::Scalar::all(FLT_MAX));
        labelCount = 1;

        std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
        std::fill(fullInfoCloud->points.begin(), fullInfoCloud->points.end(), nanPoint);
    }

    ~ImageProjection(){}

    void copyPointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        cloudHeader = laserCloudMsg->header;
        cloudHeader.stamp = ros::Time::now(); // Ouster lidar users may need to uncomment this line
        // recopy
        // pcl::PointCloud<pcl::PointXYZ>::Ptr temp(new pcl::PointCloud<pcl::PointXYZ>);
        // pcl::fromROSMsg(*laserCloudMsg, *temp);
        // for(const auto &p: *temp){
        //     PointType pi;
        //     pi.x = p.x;
        //     pi.y = p.y;
        //     pi.z = p.z;
        //     pi.intensity = 0;
        //     laserCloudIn->push_back(pi);
        //     pando_cloud->push_back(pi);
        // }
        pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn);
        pcl::fromROSMsg(*laserCloudMsg, *pando_cloud);

        // Remove Nan points

        // 屎一样的数据要转换一下 wy_add 2020-7-28
        
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);

        // Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        // // Eigen::Matrix3f R = (Eigen::AngleAxisf(-M_PI/2, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(-M_PI, Eigen::Vector3f::UnitY())).matrix();
        // // Eigen::Matrix3f R = Eigen::AngleAxisf(M_PI/2, Eigen::Vector3f::UnitZ()).matrix();
        // Eigen::Matrix3f R = Eigen::AngleAxisf(-M_PI/2, Eigen::Vector3f::UnitZ()).matrix();
        // T.block(0,0,3,3) = R;

        // pcl::transformPointCloud(*laserCloudIn, *laserCloudIn, T);
        // pcl::transformPointCloud(*pando_cloud, *pando_cloud, T);

        velodyne_points->clear();
        *velodyne_points = *laserCloudIn;
    }
    
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        // 1. Convert ros message to pcl point cloud
        copyPointCloud(laserCloudMsg);

        // 2. Start and end angle of a scan
        findStartEndAngle();

        // 3. Range image projection
        projectPointCloud();
        // 4. Mark ground points
        groundRemoval();
        // 5. Point cloud segmentation
        cloudSegmentation();
        // 6. Publish all clouds
        publishCloud();
        // 7. Reset parameters for next iteration
        resetParameters();
    }

    void findStartEndAngle(){
        // start and end orientation of this cloud
        segMsg.startOrientation = -atan2(laserCloudIn->points[0].y, laserCloudIn->points[0].x);
        segMsg.endOrientation   = -atan2(laserCloudIn->points[laserCloudIn->points.size() - 1].y,
                                                     laserCloudIn->points[laserCloudIn->points.size() - 2].x) + 2 * M_PI;
        if (segMsg.endOrientation - segMsg.startOrientation > 3 * M_PI) {
            segMsg.endOrientation -= 2 * M_PI;
        } else if (segMsg.endOrientation - segMsg.startOrientation < M_PI)
            segMsg.endOrientation += 2 * M_PI;
        segMsg.orientationDiff = segMsg.endOrientation - segMsg.startOrientation;
    }

    void projectPointCloud(){
        // range image projection
        float verticalAngle, horizonAngle, range;
        size_t rowIdn, columnIdn, index, cloudSize; 
        PointType thisPoint;

        cloudSize = laserCloudIn->points.size();

        for (size_t i = 0; i < cloudSize; ++i){

            thisPoint.x = pando_cloud->points[i].x;
            thisPoint.y = pando_cloud->points[i].y;
            thisPoint.z = pando_cloud->points[i].z;
            // find the row and column index in the iamge for this point
            verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
            rowIdn = (verticalAngle + ang_bottom) / ang_res_y;
            // rowIdn = 39- pando_cloud->points[i].ring;

            // // for 64->16 
            // if (rowIdn <= 0 || rowIdn > 64 || rowIdn%4 != 0)
            //     continue;
            // rowIdn = rowIdn/ 4 - 1;

            // 64 or 16
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
            float horizonAngle1 = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI + 180;
            int columnIdn1 = round(horizonAngle1/ang_res_x);

            columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
            if (range < 1) //wy add 
            // if(range < 2)
                continue;
            
            rangeMat.at<float>(rowIdn, columnIdn1) = range;


            // if(columnIdn % 5 == 0)
            // {
            //     int col = (int)columnIdn / 5;
            //     depthMat.at<float>(rowIdn, col) = range;
            // }

            thisPoint.intensity = (float)rowIdn + (float)columnIdn1 / 10000.0;

            index = columnIdn1  + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
            fullInfoCloud->points[index] = thisPoint;
            fullInfoCloud->points[index].intensity = range; // the corresponding range of a point is saved as "intensity"
        }
    }


    void groundRemoval(){
        size_t lowerInd, upperInd;
        float diffX, diffY, diffZ, angle;
        // groundMat
        // -1, no valid info to check if ground of not
        //  0, initial value, after validation, means not ground
        //  1, ground
        for (size_t j = 0; j < Horizon_SCAN; ++j){
            for (size_t i = 0; i < groundScanInd; ++i){

                lowerInd = j + ( i )*Horizon_SCAN;
                upperInd = j + (i+1)*Horizon_SCAN;

                if (fullCloud->points[lowerInd].intensity == -1 ||
                    fullCloud->points[upperInd].intensity == -1){
                    // no info to check, invalid points
                    groundMat.at<int8_t>(i,j) = -1;
                    continue;
                }
                    
                diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
                diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
                diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;

                angle = atan2(diffZ, sqrt(diffX*diffX + diffY*diffY) ) * 180 / M_PI;

                if (abs(angle - sensorMountAngle) <= 10){//10
                    groundMat.at<int8_t>(i,j) = 1;
                    groundMat.at<int8_t>(i+1,j) = 1;
                }
            }
        }
        // extract ground cloud (groundMat == 1)
        // mark entry that doesn't need to label (ground and invalid point) for segmentation
        // note that ground remove is from 0~N_SCAN-1, need rangeMat for mark label matrix for the 16th scan
        for (size_t i = 0; i < N_SCAN; ++i){
            for (size_t j = 0; j < Horizon_SCAN; ++j){
                if (groundMat.at<int8_t>(i,j) == 1 || rangeMat.at<float>(i,j) == FLT_MAX){
                    labelMat.at<int>(i,j) = -1;
                }
            }
        }
        if (pubGroundCloud.getNumSubscribers() != 0){
            for (size_t i = 0; i <= groundScanInd; ++i){
                for (size_t j = 0; j < Horizon_SCAN; ++j){
                    if (groundMat.at<int8_t>(i,j) == 1)
                        groundCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                }
            }
        }
    }

    void cloudSegmentation(){
        // segmentation process
        for (size_t i = 0; i < N_SCAN; ++i)
            for (size_t j = 0; j < Horizon_SCAN; ++j)
                if (labelMat.at<int>(i,j) == 0)
                    labelComponents(i, j);

        int sizeOfSegCloud = 0;
        // extract segmented cloud for lidar odometry
        for (size_t i = 0; i < N_SCAN; ++i) {

            segMsg.startRingIndex[i] = sizeOfSegCloud-1 + 5;

            for (size_t j = 0; j < Horizon_SCAN; ++j) {
                if (labelMat.at<int>(i,j) > 0 || groundMat.at<int8_t>(i,j) == 1){
                    // outliers that will not be used for optimization (always continue)
                    if (labelMat.at<int>(i,j) == 999999){
                        if (i > groundScanInd && j % 5 == 0){
                            outlierCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                            continue;
                        }else{
                            continue;
                        }
                    }
                    // majority of ground points are skipped
                    if (groundMat.at<int8_t>(i,j) == 1){
                        if (j%5!=0 && j>5 && j<Horizon_SCAN-5)
                            continue;
                    }
                    // mark ground points so they will not be considered as edge features later
                    segMsg.segmentedCloudGroundFlag[sizeOfSegCloud] = (groundMat.at<int8_t>(i,j) == 1);
                    // mark the points' column index for marking occlusion later
                    segMsg.segmentedCloudColInd[sizeOfSegCloud] = j;
                    // save range info
                    segMsg.segmentedCloudRange[sizeOfSegCloud]  = rangeMat.at<float>(i,j);
                    // save seg cloud
                    segmentedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of seg cloud
                    ++sizeOfSegCloud;
                }
            }

            segMsg.endRingIndex[i] = sizeOfSegCloud-1 - 5;
        }
        
        // extract segmented cloud for visualization
        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
            for (size_t i = 0; i < N_SCAN; ++i){
                for (size_t j = 0; j < Horizon_SCAN; ++j){
                    if (labelMat.at<int>(i,j) > 0 && labelMat.at<int>(i,j) != 999999){
                        segmentedCloudPure->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                        segmentedCloudPure->points.back().intensity = labelMat.at<int>(i,j);
                    }
                }
            }
        }
    }

    void labelComponents(int row, int col){
        // use std::queue std::vector std::deque will slow the program down greatly
        float d1, d2, alpha, angle;
        int fromIndX, fromIndY, thisIndX, thisIndY; 
        bool lineCountFlag[N_SCAN] = {false};

        queueIndX[0] = row;
        queueIndY[0] = col;
        int queueSize = 1;
        int queueStartInd = 0;
        int queueEndInd = 1;

        allPushedIndX[0] = row;
        allPushedIndY[0] = col;
        int allPushedIndSize = 1;
        
        while(queueSize > 0){
            // Pop point
            fromIndX = queueIndX[queueStartInd];
            fromIndY = queueIndY[queueStartInd];
            --queueSize;
            ++queueStartInd;
            // Mark popped point
            labelMat.at<int>(fromIndX, fromIndY) = labelCount;
            // Loop through all the neighboring grids of popped grid
            for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter){
                // new index
                thisIndX = fromIndX + (*iter).first;
                thisIndY = fromIndY + (*iter).second;
                // index should be within the boundary
                if (thisIndX < 0 || thisIndX >= N_SCAN)
                    continue;
                // at range image margin (left or right side)
                if (thisIndY < 0)
                    thisIndY = Horizon_SCAN - 1;
                if (thisIndY >= Horizon_SCAN)
                    thisIndY = 0;
                // prevent infinite loop (caused by put already examined point back)
                if (labelMat.at<int>(thisIndX, thisIndY) != 0)
                    continue;

                d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY), 
                              rangeMat.at<float>(thisIndX, thisIndY));
                d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY), 
                              rangeMat.at<float>(thisIndX, thisIndY));

                if ((*iter).first == 0)
                    alpha = segmentAlphaX;
                else
                    alpha = segmentAlphaY;

                angle = atan2(d2*sin(alpha), (d1 -d2*cos(alpha)));

                if (angle > segmentTheta){

                    queueIndX[queueEndInd] = thisIndX;
                    queueIndY[queueEndInd] = thisIndY;
                    ++queueSize;
                    ++queueEndInd;

                    labelMat.at<int>(thisIndX, thisIndY) = labelCount;
                    lineCountFlag[thisIndX] = true;

                    allPushedIndX[allPushedIndSize] = thisIndX;
                    allPushedIndY[allPushedIndSize] = thisIndY;
                    ++allPushedIndSize;
                }
            }
        }

        // check if this segment is valid
        bool feasibleSegment = false;
        if (allPushedIndSize >= 30)
            feasibleSegment = true;
        else if (allPushedIndSize >= segmentValidPointNum){
            int lineCount = 0;
            for (size_t i = 0; i < N_SCAN; ++i)
                if (lineCountFlag[i] == true)
                    ++lineCount;
            if (lineCount >= segmentValidLineNum)
                feasibleSegment = true;            
        }
        // segment is valid, mark these points
        if (feasibleSegment == true){
            ++labelCount;
        }else{ // segment is invalid, mark these points
            for (size_t i = 0; i < allPushedIndSize; ++i){
                labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;
            }
        }
    }

    
    void publishCloud(){
        // 1. Publish Seg Cloud Info
        segMsg.header = cloudHeader;
        pubSegmentedCloudInfo.publish(segMsg);
        // 2. Publish clouds
        sensor_msgs::PointCloud2 laserCloudTemp;
        pcl::toROSMsg(*velodyne_points, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubVelodyne.publish(laserCloudTemp);

        pcl::toROSMsg(*outlierCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubOutlierCloud.publish(laserCloudTemp);
        // segmented cloud with ground
        pcl::toROSMsg(*segmentedCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubSegmentedCloud.publish(laserCloudTemp);
        // projected full cloud
        if (pubFullCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*fullCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullCloud.publish(laserCloudTemp);
        }
        // original dense ground cloud
        if (pubGroundCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*groundCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubGroundCloud.publish(laserCloudTemp);
        }
        // segmented cloud without ground
        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
            pcl::toROSMsg(*segmentedCloudPure, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubSegmentedCloudPure.publish(laserCloudTemp);
        }
        // projected full cloud info
        if (pubFullInfoCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*fullInfoCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullInfoCloud.publish(laserCloudTemp);
        }


        // for(int r = 0; r < depthMat.rows; r++)
        // {
        //     for(int c = 0; c < depthMat.cols; c++)
        //     {
        //         if(depthMat.at<float>(r,c) == FLT_MAX)
        //             depthMat.at<float>(r,c) = 0;
        //     }
        // }

        //save depth image 
        // cv::Mat depth_img, re_depth_img;
        // depthMat.convertTo(depth_img, CV_8U);
        // // cv::equalizeHist(rangeMat, rangeMat);
        // cv::flip(depth_img, depth_img, 0);
        // cv::normalize(depth_img, depth_img,0, 255, cv::NORM_MINMAX);
        // cv::imwrite("/media/yingwang/DATADisk/range_image/" + std::to_string(cloudHeader.stamp.toSec()) +"_1.png", depth_img);
        // cv::resize(depth_img, re_depth_img, cv::Size(depth_img.cols, 4*depth_img.rows));
        // cv::imwrite("/media/yingwang/DATADisk/range_image/" + std::to_string(cloudHeader.stamp.toSec()) +"_2.png", re_depth_img);
        // cv::Mat depth_image;
        // depth_image = cv::imread("/media/horizon/data/range/" + std::to_string(cloudHeader.stamp.toSec()) +".png", cv::IMREAD_ANYDEPTH | cv::IMREAD_UNCHANGED);
        // cout << depth_image << std::endl;

        // std::ofstream ofs;
        // ofs.open("/media/horizon/data/range/" + std::to_string(cloudHeader.stamp.toSec()) +".txt");

        // for(int r = 0; r < rangeMat.rows; r++)
        // {
        //     for(int c = 0; c < rangeMat.cols; c++)
        //     {

        //         if(rangeMat.at<float>(r,c) == FLT_MAX)
        //             ofs << 0 << " ";
        //         else 
        //             ofs << rangeMat.at<float>(r,c) << " ";
        //     }
        //     ofs << std::endl;
        // }
        // pcl::io::savePCDFileBinary("/media/horizon/data/range/" + std::to_string(cloudHeader.stamp.toSec()) + ".pcd", *fullCloud);

        #ifdef SAVE_EVERY_SACN
        std::cout << "------- " << kframeid << " -------------" << std::endl;
        pcl::io::savePCDFileBinary("/media/yingwang/DATADisk/dp/dp" + to_string(kframeid++) + ".pcd", *fullCloud);
        #endif 
    }
};




int main(int argc, char** argv){

    ros::init(argc, argv, "lego_loam");
    
    ImageProjection IP;

    ROS_INFO("\033[1;32m---->\033[0m Image Projection Started.");

    ros::spin();
    return 0;
}
