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

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam/nonlinear/ISAM2.h>

#include <visualization_msgs/Marker.h>
#include <fstream>

#include <nav_msgs/Path.h>

#include <Eigen/Dense>

#include "LidarIris.h"
#include "tic_toc.h"

using namespace gtsam;

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <std_msgs/Empty.h>

Eigen::Isometry3d makeIsometry(const PointTypePose &pointPose)
{
    geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw(pointPose.yaw, -pointPose.roll, -pointPose.pitch);

    Eigen::Quaterniond q(geoQuat.w, -geoQuat.y, -geoQuat.z, geoQuat.x);
    q = q * Eigen::AngleAxisd(-M_PI_2, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(-M_PI_2, Eigen::Vector3d::UnitX());
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translate(Eigen::Vector3d(pointPose.x, pointPose.y, pointPose.z));
    pose.rotate(q);
    pose = Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitY()) * pose;

    return pose;
}

gtsam::Pose3 makeGtPose(const PointTypePose &pointPose)
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(pointPose.yaw, pointPose.roll, pointPose.pitch), gtsam::Point3(pointPose.z, pointPose.x, pointPose.y));
}

class submapOptimization
{
private:
    // common down-sampler
    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterOutlier;

    // cloud input data
    bool newDataInput;
    double timeNewDataInput;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;      // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS;    // downsampled corner featuer set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;        // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS;      // downsampled surf featuer set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudOutlierLast;     // outlier set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudOutlierLastDS;   // downsampled outlier set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfTotalLast;   // surf total set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfTotalLastDS; // downsampled surf total set from odoOptimization
    // down-sampled num
    int laserCloudCornerLastDSNum;
    int laserCloudSurfLastDSNum;
    int laserCloudOutlierLastDSNum;
    int laserCloudSurfTotalLastDSNum;

    // near-map extract state
    int latestFrameID;
    // near-map deque
    std::deque<pcl::PointCloud<PointType>::Ptr> recentCornerCloudKeyFrames;
    std::deque<pcl::PointCloud<PointType>::Ptr> recentSurfCloudKeyFrames;
    std::deque<pcl::PointCloud<PointType>::Ptr> recentOutlierCloudKeyFrames;
    // near-map data
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;
    // near-map data down-sampled num
    int laserCloudCornerFromMapDSNum;
    int laserCloudSurfFromMapDSNum;
    // near-map data kd-tree
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;
    // front end match state
    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;
    // LM states
    bool isDegenerate;
    cv::Mat matP;

    // run state
    double timeLastProcessing;
    PointType previousRobotPosPoint;
    PointType currentRobotPosPoint;

    // pose states
    float transformLast[6];
    float transformSum[6]; // input-odometry
    float transformIncre[6];
    float transformTobeMapped[6];
    float transformBefMapped[6];
    float transformAftMapped[6];

    // global poses
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    // key frame data
    std::shared_ptr<std::vector<pcl::PointCloud<PointType>::Ptr>> cornerCloudKeyFrames;
    std::shared_ptr<std::vector<pcl::PointCloud<PointType>::Ptr>> surfCloudKeyFrames;
    std::shared_ptr<std::vector<pcl::PointCloud<PointType>::Ptr>> outlierCloudKeyFrames;

    // factor-graph
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2Params parameters;
    std::unique_ptr<ISAM2> isam;
    Values isamCurrentEstimate;
    // useful constants
    noiseModel::Diagonal::shared_ptr priorNoise;
    noiseModel::Diagonal::shared_ptr odometryNoise;
    noiseModel::Diagonal::shared_ptr constraintNoise;
    gtsam::Vector6 LMScore;
    double LMNoise;

    // submap infomation
    PointTypePose poseBase;
    bool aLoopIsClosed = false;

public:
    using Ptr = std::shared_ptr<submapOptimization>;

    submapOptimization()
    {
        parameters.relinearizeThreshold = 0.01;
        parameters.relinearizeSkip = 1;
        isam.reset(new ISAM2(parameters));

        downSizeFilterCorner.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilterSurf.setLeafSize(0.4, 0.4, 0.4);
        downSizeFilterOutlier.setLeafSize(0.4, 0.4, 0.4);

        allocateMemory();
    }

    void allocateMemory()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        newDataInput = false;
        timeNewDataInput = -1;
        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());      // corner feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>());    // downsampled corner featuer set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());        // surf feature set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>());      // downsampled surf featuer set from odoOptimization
        laserCloudOutlierLast.reset(new pcl::PointCloud<PointType>());     // outlier set from odoOptimization
        laserCloudOutlierLastDS.reset(new pcl::PointCloud<PointType>());   // downsampled outlier set from odoOptimization
        laserCloudSurfTotalLast.reset(new pcl::PointCloud<PointType>());   // surf feature set from odoOptimization
        laserCloudSurfTotalLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

        laserCloudCornerLastDSNum = 0;
        laserCloudSurfLastDSNum = 0;
        laserCloudOutlierLastDSNum = 0;
        laserCloudSurfTotalLastDSNum = 0;

        latestFrameID = 0;

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        laserCloudCornerFromMapDSNum = 0;
        laserCloudSurfFromMapDSNum = 0;

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        isDegenerate = false;
        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));

        timeLastProcessing = -1;

        for (int i = 0; i < 6; ++i)
        {
            transformLast[i] = 0;
            transformSum[i] = 0;
            transformIncre[i] = 0;
            transformTobeMapped[i] = 0;
            transformBefMapped[i] = 0;
            transformAftMapped[i] = 0;
        }

        cornerCloudKeyFrames.reset(new std::vector<pcl::PointCloud<PointType>::Ptr>{});
        surfCloudKeyFrames.reset(new std::vector<pcl::PointCloud<PointType>::Ptr>{});
        outlierCloudKeyFrames.reset(new std::vector<pcl::PointCloud<PointType>::Ptr>{});

        gtsam::Vector Vector6(6);
        Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-6;
        priorNoise = noiseModel::Diagonal::Variances(Vector6);
        odometryNoise = noiseModel::Diagonal::Variances(Vector6);

        poseBase = PointTypePose{};
    }

    void dataInput(const sensor_msgs::PointCloud2ConstPtr &corner, const sensor_msgs::PointCloud2ConstPtr &surf, const sensor_msgs::PointCloud2ConstPtr &outlier, const nav_msgs::Odometry::ConstPtr &odom)
    {
        laserCloudCornerLast->clear();
        pcl::fromROSMsg(*corner, *laserCloudCornerLast);
        laserCloudSurfLast->clear();
        pcl::fromROSMsg(*surf, *laserCloudSurfLast);
        laserCloudOutlierLast->clear();
        pcl::fromROSMsg(*outlier, *laserCloudOutlierLast);

        double roll, pitch, yaw;
        geometry_msgs::Quaternion geoQuat = odom->pose.pose.orientation;
        tf::Matrix3x3(tf::Quaternion(geoQuat.z, -geoQuat.x, -geoQuat.y, geoQuat.w)).getRPY(roll, pitch, yaw);
        transformSum[0] = -pitch;
        transformSum[1] = -yaw;
        transformSum[2] = roll;
        transformSum[3] = odom->pose.pose.position.x;
        transformSum[4] = odom->pose.pose.position.y;
        transformSum[5] = odom->pose.pose.position.z;

        newDataInput = true;
        timeNewDataInput = odom->header.stamp.toSec();
    }

    void transformAssociateToMap()
    {
        float x1 = cos(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) - sin(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);
        float y1 = transformBefMapped[4] - transformSum[4];
        float z1 = sin(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) + cos(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);

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

        float srx = -sbcx * (salx * sblx + calx * cblx * salz * sblz + calx * calz * cblx * cblz) - cbcx * sbcy * (calx * calz * (cbly * sblz - cblz * sblx * sbly) - calx * salz * (cbly * cblz + sblx * sbly * sblz) + cblx * salx * sbly) - cbcx * cbcy * (calx * salz * (cblz * sbly - cbly * sblx * sblz) - calx * calz * (sbly * sblz + cbly * cblz * sblx) + cblx * cbly * salx);
        transformTobeMapped[0] = -asin(srx);

        float srycrx = sbcx * (cblx * cblz * (caly * salz - calz * salx * saly) - cblx * sblz * (caly * calz + salx * saly * salz) + calx * saly * sblx) - cbcx * cbcy * ((caly * calz + salx * saly * salz) * (cblz * sbly - cbly * sblx * sblz) + (caly * salz - calz * salx * saly) * (sbly * sblz + cbly * cblz * sblx) - calx * cblx * cbly * saly) + cbcx * sbcy * ((caly * calz + salx * saly * salz) * (cbly * cblz + sblx * sbly * sblz) + (caly * salz - calz * salx * saly) * (cbly * sblz - cblz * sblx * sbly) + calx * cblx * saly * sbly);
        float crycrx = sbcx * (cblx * sblz * (calz * saly - caly * salx * salz) - cblx * cblz * (saly * salz + caly * calz * salx) + calx * caly * sblx) + cbcx * cbcy * ((saly * salz + caly * calz * salx) * (sbly * sblz + cbly * cblz * sblx) + (calz * saly - caly * salx * salz) * (cblz * sbly - cbly * sblx * sblz) + calx * caly * cblx * cbly) - cbcx * sbcy * ((saly * salz + caly * calz * salx) * (cbly * sblz - cblz * sblx * sbly) + (calz * saly - caly * salx * salz) * (cbly * cblz + sblx * sbly * sblz) - calx * caly * cblx * sbly);
        transformTobeMapped[1] = atan2(srycrx / cos(transformTobeMapped[0]),
                                       crycrx / cos(transformTobeMapped[0]));

        float srzcrx = (cbcz * sbcy - cbcy * sbcx * sbcz) * (calx * salz * (cblz * sbly - cbly * sblx * sblz) - calx * calz * (sbly * sblz + cbly * cblz * sblx) + cblx * cbly * salx) - (cbcy * cbcz + sbcx * sbcy * sbcz) * (calx * calz * (cbly * sblz - cblz * sblx * sbly) - calx * salz * (cbly * cblz + sblx * sbly * sblz) + cblx * salx * sbly) + cbcx * sbcz * (salx * sblx + calx * cblx * salz * sblz + calx * calz * cblx * cblz);
        float crzcrx = (cbcy * sbcz - cbcz * sbcx * sbcy) * (calx * calz * (cbly * sblz - cblz * sblx * sbly) - calx * salz * (cbly * cblz + sblx * sbly * sblz) + cblx * salx * sbly) - (sbcy * sbcz + cbcy * cbcz * sbcx) * (calx * salz * (cblz * sbly - cbly * sblx * sblz) - calx * calz * (sbly * sblz + cbly * cblz * sblx) + cblx * cbly * salx) + cbcx * cbcz * (salx * sblx + calx * cblx * salz * sblz + calx * calz * cblx * cblz);
        transformTobeMapped[2] = atan2(srzcrx / cos(transformTobeMapped[0]),
                                       crzcrx / cos(transformTobeMapped[0]));

        x1 = cos(transformTobeMapped[2]) * transformIncre[3] - sin(transformTobeMapped[2]) * transformIncre[4];
        y1 = sin(transformTobeMapped[2]) * transformIncre[3] + cos(transformTobeMapped[2]) * transformIncre[4];
        z1 = transformIncre[5];

        x2 = x1;
        y2 = cos(transformTobeMapped[0]) * y1 - sin(transformTobeMapped[0]) * z1;
        z2 = sin(transformTobeMapped[0]) * y1 + cos(transformTobeMapped[0]) * z1;

        transformTobeMapped[3] = transformAftMapped[3] - (cos(transformTobeMapped[1]) * x2 + sin(transformTobeMapped[1]) * z2);
        transformTobeMapped[4] = transformAftMapped[4] - y2;
        transformTobeMapped[5] = transformAftMapped[5] - (-sin(transformTobeMapped[1]) * x2 + cos(transformTobeMapped[1]) * z2);
    }

    void extractSurroundingKeyFrames()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        if (loopClosureEnableFlag == true)
        {
            // only use recent key poses for graph building
            if (recentCornerCloudKeyFrames.size() < surroundingKeyframeSearchNum)
            { // queue is not full (the beginning of mapping or a loop is just closed)
                // clear recent key frames queue
                recentCornerCloudKeyFrames.clear();
                recentSurfCloudKeyFrames.clear();
                recentOutlierCloudKeyFrames.clear();
                int numPoses = cloudKeyPoses3D->points.size();
                for (int i = numPoses - 1; i >= 0; --i)
                {
                    int thisKeyInd = (int)cloudKeyPoses3D->points[i].intensity;
                    PointTypePose thisTransformation = cloudKeyPoses6D->points[thisKeyInd];
                    updateTransformPointCloudSinCos(&thisTransformation);
                    // extract surrounding map
                    recentCornerCloudKeyFrames.push_front(transformPointCloud(cornerCloudKeyFrames->at(thisKeyInd)));
                    recentSurfCloudKeyFrames.push_front(transformPointCloud(surfCloudKeyFrames->at(thisKeyInd)));
                    recentOutlierCloudKeyFrames.push_front(transformPointCloud(outlierCloudKeyFrames->at(thisKeyInd)));
                    if (recentCornerCloudKeyFrames.size() >= surroundingKeyframeSearchNum)
                        break;
                }
            }
            else
            { // queue is full, pop the oldest key frame and push the latest key frame
                if (latestFrameID != cloudKeyPoses3D->points.size() - 1)
                { // if the robot is not moving, no need to update recent frames
                    recentCornerCloudKeyFrames.pop_front();
                    recentSurfCloudKeyFrames.pop_front();
                    recentOutlierCloudKeyFrames.pop_front();
                    // push latest scan to the end of queue
                    latestFrameID = cloudKeyPoses3D->points.size() - 1;
                    PointTypePose thisTransformation = cloudKeyPoses6D->points[latestFrameID];
                    updateTransformPointCloudSinCos(&thisTransformation);
                    recentCornerCloudKeyFrames.push_back(transformPointCloud(cornerCloudKeyFrames->at(latestFrameID)));
                    recentSurfCloudKeyFrames.push_back(transformPointCloud(surfCloudKeyFrames->at(latestFrameID)));
                    recentOutlierCloudKeyFrames.push_back(transformPointCloud(outlierCloudKeyFrames->at(latestFrameID)));
                }
            }

            for (int i = 0; i < recentCornerCloudKeyFrames.size(); ++i)
            {
                *laserCloudCornerFromMap += *recentCornerCloudKeyFrames[i];
                *laserCloudSurfFromMap += *recentSurfCloudKeyFrames[i];
                *laserCloudSurfFromMap += *recentOutlierCloudKeyFrames[i];
            }
        }
#if 0 // must use loop-clousre mode
        else
        {
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
            for (int i = 0; i < surroundingExistingKeyPosesID.size(); ++i)
            {
                bool existingFlag = false;
                for (int j = 0; j < numSurroundingPosesDS; ++j)
                {
                    if (surroundingExistingKeyPosesID[i] == (int)surroundingKeyPosesDS->points[j].intensity)
                    {
                        existingFlag = true;
                        break;
                    }
                }
                if (existingFlag == false)
                {
                    surroundingExistingKeyPosesID.erase(surroundingExistingKeyPosesID.begin() + i);
                    surroundingCornerCloudKeyFrames.erase(surroundingCornerCloudKeyFrames.begin() + i);
                    surroundingSurfCloudKeyFrames.erase(surroundingSurfCloudKeyFrames.begin() + i);
                    surroundingOutlierCloudKeyFrames.erase(surroundingOutlierCloudKeyFrames.begin() + i);
                    --i;
                }
            }
            // add new key frames that are not in calculated existing key frames
            for (int i = 0; i < numSurroundingPosesDS; ++i)
            {
                bool existingFlag = false;
                for (auto iter = surroundingExistingKeyPosesID.begin(); iter != surroundingExistingKeyPosesID.end(); ++iter)
                {
                    if ((*iter) == (int)surroundingKeyPosesDS->points[i].intensity)
                    {
                        existingFlag = true;
                        break;
                    }
                }
                if (existingFlag == true)
                {
                    continue;
                }
                else
                {
                    int thisKeyInd = (int)surroundingKeyPosesDS->points[i].intensity;
                    PointTypePose thisTransformation = cloudKeyPoses6D->points[thisKeyInd];
                    updateTransformPointCloudSinCos(&thisTransformation);
                    surroundingExistingKeyPosesID.push_back(thisKeyInd);
                    surroundingCornerCloudKeyFrames.push_back(transformPointCloud(cornerCloudKeyFrames[thisKeyInd]));
                    surroundingSurfCloudKeyFrames.push_back(transformPointCloud(surfCloudKeyFrames[thisKeyInd]));
                    surroundingOutlierCloudKeyFrames.push_back(transformPointCloud(outlierCloudKeyFrames[thisKeyInd]));
                }
            }

            for (int i = 0; i < surroundingExistingKeyPosesID.size(); ++i)
            {
                *laserCloudCornerFromMap += *surroundingCornerCloudKeyFrames[i];
                *laserCloudSurfFromMap += *surroundingSurfCloudKeyFrames[i];
                *laserCloudSurfFromMap += *surroundingOutlierCloudKeyFrames[i];
            }
        }
#endif
        // Downsample the surrounding corner key frames (or map)
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->points.size();
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->points.size();
    }

    void downsampleCurrentScan()
    {

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

    void cornerOptimization(int iterCount)
    {
        updatePointAssociateToMapSinCos();
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {
            PointType pointOri = laserCloudCornerLastDS->points[i];
            PointType pointSel;
            pointAssociateToMap(&pointOri, &pointSel);
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            if (pointSearchSqDis[4] < 1.0)
            {
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++)
                {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5;
                cy /= 5;
                cz /= 5;

                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++)
                {
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax;
                    a12 += ax * ay;
                    a13 += ax * az;
                    a22 += ay * ay;
                    a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5;
                a12 /= 5;
                a13 /= 5;
                a22 /= 5;
                a23 /= 5;
                a33 /= 5;

                cv::Mat1f matA1(3, 3);
                cv::Mat1f matD1(1, 3);
                cv::Mat1f matV1(3, 3);
                matA1.at<float>(0, 0) = a11;
                matA1.at<float>(0, 1) = a12;
                matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12;
                matA1.at<float>(1, 1) = a22;
                matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13;
                matA1.at<float>(2, 1) = a23;
                matA1.at<float>(2, 2) = a33;

                cv::eigen(matA1, matD1, matV1);

                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1))
                {

                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    float a012 = sqrt(((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) + ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) + ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));

                    float l12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));

                    float la = ((y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) + (z1 - z2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) - (z1 - z2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) + (y1 - y2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;

                    float ld2 = a012 / l12;

                    float s = 1 - 0.9 * fabs(ld2);

                    PointType coeff;
                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;

                    if (s > 0.1)
                    {
                        laserCloudOri->push_back(pointOri);
                        coeffSel->push_back(coeff);
                    }
                }
            }
        }
    }

    void surfOptimization(int iterCount)
    {
        updatePointAssociateToMapSinCos();
        for (int i = 0; i < laserCloudSurfTotalLastDSNum; i++)
        {
            PointType pointOri = laserCloudSurfTotalLastDS->points[i];
            PointType pointSel;
            pointAssociateToMap(&pointOri, &pointSel);
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            if (pointSearchSqDis[4] < 1.0)
            {
                cv::Mat1f matA0 = cv::Mat1f::zeros(5, 3);
                cv::Mat1f matB0 = cv::Mat1f(5, 1, -1);
                cv::Mat1f matX0 = cv::Mat1f::zeros(3, 1);
                for (int j = 0; j < 5; j++)
                {
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
                pa /= ps;
                pb /= ps;
                pc /= ps;
                pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++)
                {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2)
                    {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid)
                {
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                    PointType coeff;
                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    if (s > 0.1)
                    {
                        laserCloudOri->push_back(pointOri);
                        coeffSel->push_back(coeff);
                    }
                }
            }
        }
    }

    bool LMOptimization(int iterCount)
    {
        float srx = sin(transformTobeMapped[0]);
        float crx = cos(transformTobeMapped[0]);
        float sry = sin(transformTobeMapped[1]);
        float cry = cos(transformTobeMapped[1]);
        float srz = sin(transformTobeMapped[2]);
        float crz = cos(transformTobeMapped[2]);

        int laserCloudSelNum = laserCloudOri->points.size();
        if (laserCloudSelNum < 50)
        {
            return false;
        }

        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
        for (int i = 0; i < laserCloudSelNum; i++)
        {
            PointType pointOri = laserCloudOri->points[i];
            PointType coeff = coeffSel->points[i];

            float arx = (crx * sry * srz * pointOri.x + crx * crz * sry * pointOri.y - srx * sry * pointOri.z) * coeff.x + (-srx * srz * pointOri.x - crz * srx * pointOri.y - crx * pointOri.z) * coeff.y + (crx * cry * srz * pointOri.x + crx * cry * crz * pointOri.y - cry * srx * pointOri.z) * coeff.z;

            float ary = ((cry * srx * srz - crz * sry) * pointOri.x + (sry * srz + cry * crz * srx) * pointOri.y + crx * cry * pointOri.z) * coeff.x + ((-cry * crz - srx * sry * srz) * pointOri.x + (cry * srz - crz * srx * sry) * pointOri.y - crx * sry * pointOri.z) * coeff.z;

            float arz = ((crz * srx * sry - cry * srz) * pointOri.x + (-cry * crz - srx * sry * srz) * pointOri.y) * coeff.x + (crx * crz * pointOri.x - crx * srz * pointOri.y) * coeff.y + ((sry * srz + cry * crz * srx) * pointOri.x + (crz * sry - cry * srx * srz) * pointOri.y) * coeff.z;

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

        if (iterCount == 0)
        {
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--)
            {
                if (matE.at<float>(0, i) < eignThre[i])
                {
                    for (int j = 0; j < 6; j++)
                    {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                }
                else
                {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate)
        {
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

        if (deltaR < 0.05 && deltaT < 0.05)
        {   
            LMNoise = sqrt(pow(deltaR,2) + pow(deltaT, 2)) * 1e-4;
            // LMScore << 
            //     abs(matX.at<float>(3, 0) * 100), abs(matX.at<float>(4, 0) * 100), abs(matX.at<float>(5, 0) * 100), 
            //     abs(pcl::rad2deg(matX.at<float>(0, 0))), abs(pcl::rad2deg(matX.at<float>(1, 0))), abs(pcl::rad2deg(matX.at<float>(2, 0)));
            gtsam::Vector Vector6(6);
            if (LMNoise <= 1e-6)
                LMNoise = 1e-6;
            Vector6 << LMNoise, LMNoise, LMNoise, LMNoise, LMNoise, LMNoise;
            // ROS_WARN_STREAM(LMNoise);
            // Vector6 << 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-4;
            odometryNoise = noiseModel::Diagonal::Variances(Vector6);
            return true;
        }
        return false;
    }

    void transformUpdate()
    {
        for (int i = 0; i < 6; i++)
        {
            transformBefMapped[i] = transformSum[i];
            transformAftMapped[i] = transformTobeMapped[i];
        }
    }

    void scan2MapOptimization()
    {
        if (laserCloudCornerFromMapDSNum > 10 && laserCloudSurfFromMapDSNum > 100)
        {

            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            for (int iterCount = 0; iterCount < 10; iterCount++)
            {

                laserCloudOri->clear();
                coeffSel->clear();

                cornerOptimization(iterCount);
                surfOptimization(iterCount);
                
                if (LMOptimization(iterCount) == true)
                    break;
            }

            transformUpdate();
        }
    }

    bool testKeyFrame()
    {
        currentRobotPosPoint.x = transformAftMapped[3];
        currentRobotPosPoint.y = transformAftMapped[4];
        currentRobotPosPoint.z = transformAftMapped[5];

        bool saveThisKeyFrame = true;
        if (sqrt((previousRobotPosPoint.x - currentRobotPosPoint.x) * (previousRobotPosPoint.x - currentRobotPosPoint.x) + (previousRobotPosPoint.y - currentRobotPosPoint.y) * (previousRobotPosPoint.y - currentRobotPosPoint.y) + (previousRobotPosPoint.z - currentRobotPosPoint.z) * (previousRobotPosPoint.z - currentRobotPosPoint.z)) < 1)
        { // 1 for 16 0.3 or 1
            saveThisKeyFrame = false;
        }

        if (saveThisKeyFrame == false && !cloudKeyPoses3D->points.empty())
            return false;
        return true;
    }

    void saveKeyFramesAndFactor()
    {
        previousRobotPosPoint = currentRobotPosPoint;
        /**
         * update grsam graph
         */
        if (cloudKeyPoses3D->points.empty())
        {
            gtSAMgraph.add(PriorFactor<Pose3>(0, Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0], transformTobeMapped[1]), Point3(transformTobeMapped[5], transformTobeMapped[3], transformTobeMapped[4])), priorNoise));
            initialEstimate.insert(0, Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0], transformTobeMapped[1]),
                                            Point3(transformTobeMapped[5], transformTobeMapped[3], transformTobeMapped[4])));
            for (int i = 0; i < 6; ++i)
                transformLast[i] = transformTobeMapped[i];
        }
        else
        {
            gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(transformLast[2], transformLast[0], transformLast[1]),
                                          Point3(transformLast[5], transformLast[3], transformLast[4]));
            gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(transformAftMapped[2], transformAftMapped[0], transformAftMapped[1]),
                                        Point3(transformAftMapped[5], transformAftMapped[3], transformAftMapped[4]));
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->points.size() - 1, cloudKeyPoses3D->points.size(), poseFrom.between(poseTo), odometryNoise));
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
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size() - 1);

        thisPose3D.x = latestEstimate.translation().y();
        thisPose3D.y = latestEstimate.translation().z();
        thisPose3D.z = latestEstimate.translation().x();
        thisPose3D.intensity = cloudKeyPoses3D->points.size(); // this can be used as index
        cloudKeyPoses3D->push_back(thisPose3D);

        // TODO:
        // {
        //     sensor_msgs::PointCloud2 msg;
        //     pcl::toROSMsg(*originLidar, msg);
        //     msg.header.frame_id = std::to_string(cloudKeyPoses6D->size()); // TODO: maybe need another index rule
        //     msg.header.stamp = originTime;                                 // ros::Time::now();
        //     pubOriginLidar.publish(msg);
        // }

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity; // this can be used as index
        thisPose6D.roll = latestEstimate.rotation().pitch();
        thisPose6D.pitch = latestEstimate.rotation().yaw();
        thisPose6D.yaw = latestEstimate.rotation().roll(); // in camera frame
        thisPose6D.time = timeNewDataInput;
        cloudKeyPoses6D->push_back(thisPose6D);

        /**
         * save updated transform
         */
        if (cloudKeyPoses3D->points.size() > 1)
        {
            transformAftMapped[0] = latestEstimate.rotation().pitch();
            transformAftMapped[1] = latestEstimate.rotation().yaw();
            transformAftMapped[2] = latestEstimate.rotation().roll();
            transformAftMapped[3] = latestEstimate.translation().y();
            transformAftMapped[4] = latestEstimate.translation().z();
            transformAftMapped[5] = latestEstimate.translation().x();

            for (int i = 0; i < 6; ++i)
            {
                transformLast[i] = transformAftMapped[i];
                transformTobeMapped[i] = transformAftMapped[i];
            }
        }

        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisOutlierKeyFrame(new pcl::PointCloud<PointType>());

        pcl::copyPointCloud(*laserCloudCornerLastDS, *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS, *thisSurfKeyFrame);
        pcl::copyPointCloud(*laserCloudOutlierLastDS, *thisOutlierKeyFrame);

        cornerCloudKeyFrames->push_back(thisCornerKeyFrame);
        surfCloudKeyFrames->push_back(thisSurfKeyFrame);
        outlierCloudKeyFrames->push_back(thisOutlierKeyFrame);

        // TODO: data for loop-closure
#if 0
        if (newcloudsave && timefullcloud - timeLaserOdometry < 0.05)
        {
            newcloudsave = false;
            cv::Mat1b m = LidarIris::GetIris(*velodyne_points);

            LidarIris::FeatureDesc fd = iris.GetFeature(m);
            fd.index = cloudKeyPoses3D->points.size() - 1;
            fds.push_back(fd);
        }
#endif
        if (cloudKeyPoses6D->size() == 1)
        {
            poseBase = cloudKeyPoses6D->back();
            // std::cout << "init-base\n"
            //           << makeIsometry(poseBase).matrix() << std::endl;
            ROS_INFO_STREAM("submap start");
        }
    }

    void correctPoses()
    {
        if (aLoopIsClosed == true)
        {
            recentCornerCloudKeyFrames.clear();
            recentSurfCloudKeyFrames.clear();
            recentOutlierCloudKeyFrames.clear();
            // update key poses
            int numPoses = isamCurrentEstimate.size();

            for (int i = 0; i < numPoses; ++i)
            {
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().z();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().x();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();
                cloudKeyPoses6D->points[i].yaw = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
            }

            aLoopIsClosed = false;
        }
        return;
    }

    void updatePoseBase(const Pose3 &newBase)
    {
        // std::cout << "old-base\n"
        //           << makeIsometry(poseBase).matrix() << std::endl;

        auto oldPose = Pose3(Rot3::RzRyRx(poseBase.yaw, poseBase.roll, poseBase.pitch), Point3(poseBase.z, poseBase.x, poseBase.y));
        auto delta = newBase * oldPose.inverse();
        poseBase.x = newBase.translation().y();
        poseBase.y = newBase.translation().z();
        poseBase.z = newBase.translation().x();
        poseBase.roll = newBase.rotation().pitch();
        poseBase.pitch = newBase.rotation().yaw();
        poseBase.yaw = newBase.rotation().roll();

        // std::cout << "new-base\n"
        //           << makeIsometry(poseBase).matrix() << std::endl;

        int numPoses = isamCurrentEstimate.size();
        for (int i = 0; i < numPoses; i++)
        {
            auto &pose3d = cloudKeyPoses3D->points[i];
            auto &pose6d = cloudKeyPoses6D->points[i];

            auto pose = delta * Pose3(Rot3::RzRyRx(pose6d.yaw, pose6d.roll, pose6d.pitch), Point3(pose6d.z, pose6d.x, pose6d.y));

            pose3d.x = pose.translation().y();
            pose3d.y = pose.translation().z();
            pose3d.z = pose.translation().x();

            pose6d.x = pose3d.x;
            pose6d.y = pose3d.y;
            pose6d.z = pose3d.z;
            pose6d.roll = pose.rotation().pitch();
            pose6d.pitch = pose.rotation().yaw();
            pose6d.yaw = pose.rotation().roll();
        }


        {
            auto newPose = makeGtPose(cloudKeyPoses6D->back());
            transformAftMapped[0] = newPose.rotation().pitch();
            transformAftMapped[1] = newPose.rotation().yaw();
            transformAftMapped[2] = newPose.rotation().roll();
            transformAftMapped[3] = newPose.translation().y();
            transformAftMapped[4] = newPose.translation().z();
            transformAftMapped[5] = newPose.translation().x();

            for (int i = 0; i < 6; ++i)
            {
                transformLast[i] = transformAftMapped[i];
                transformTobeMapped[i] = transformAftMapped[i];
            }
        }
    }

    void clearCloud()
    {
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear();
        laserCloudCornerFromMapDS->clear();
        laserCloudSurfFromMapDS->clear();
    }

    bool run()
    {
        if (newDataInput)
        {
            newDataInput = false;

            // std::lock_guard<std::mutex> lock(mtx);

            if (timeNewDataInput - timeLastProcessing >= mappingProcessInterval)
            {
                timeLastProcessing = timeNewDataInput;

                // prepare data
                transformAssociateToMap();
                extractSurroundingKeyFrames();
                downsampleCurrentScan();

                // frontend
                scan2MapOptimization();

                // try add key-frame & backend
                bool newKeyFrame = testKeyFrame();

                if (newKeyFrame)
                    saveKeyFramesAndFactor();

                // extract pose from graph
                correctPoses();

                return newKeyFrame;
            }
        }
        return false;
    }

    bool locateRun()
    {
        if (newDataInput)
        {
            newDataInput = false;

            if (timeNewDataInput - timeLastProcessing >= mappingProcessInterval)
            {
                timeLastProcessing = timeNewDataInput;

                // prepare data
                transformAssociateToMap();
                extractSurroundingKeyFrames();
                downsampleCurrentScan();

                // frontend
                scan2MapOptimization();

                // try add key-frame & backend
                return testKeyFrame();
            }
        }
        return false;
    }

    void keyFrameStage()
    {
        saveKeyFramesAndFactor();

        correctPoses();

        return;
    }

    void addOnly(const Ptr &from)
    {
        timeNewDataInput = from->timeNewDataInput;

        timeLastProcessing = from->timeLastProcessing;
        previousRobotPosPoint = from->previousRobotPosPoint;
        currentRobotPosPoint = from->currentRobotPosPoint;

        for (int i = 0; i < 6; i++)
        {
            transformLast[i] = from->transformLast[i];
            transformSum[i] = from->transformSum[i];
            transformIncre[i] = from->transformIncre[i];
            transformTobeMapped[i] = from->transformTobeMapped[i];
            transformBefMapped[i] = from->transformBefMapped[i];
            transformAftMapped[i] = from->transformAftMapped[i];
        }

        pcl::copyPointCloud(*from->laserCloudCornerLastDS, *laserCloudCornerLastDS);
        pcl::copyPointCloud(*from->laserCloudSurfLastDS, *laserCloudSurfLastDS);
        pcl::copyPointCloud(*from->laserCloudOutlierLastDS, *laserCloudOutlierLastDS);

        LMNoise = from->LMNoise;
        // ROS_WARN_STREAM("active:" << LMNoise);
        gtsam::Vector Vector6(6);
        Vector6 << LMNoise, LMNoise, LMNoise, LMNoise, LMNoise, LMNoise;
        odometryNoise = noiseModel::Diagonal::Variances(Vector6);
    }

    int freezeFrameId = -1;

    void freeze()
    {
        if (freezeFrameId < 0)
        {
            freezeFrameId = cloudKeyPoses3D->size() - 1;
        }
    }

    std::tuple<nav_msgs::Odometry, tf::StampedTransform> publishTF()
    {
        geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw(transformAftMapped[2], -transformAftMapped[0], -transformAftMapped[1]);

        nav_msgs::Odometry odomAftMapped;
        odomAftMapped.header.stamp = ros::Time().fromSec(timeNewDataInput);
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
        // pubOdomAftMapped.publish(odomAftMapped);

        tf::StampedTransform aftMappedTrans;
        aftMappedTrans.stamp_ = ros::Time().fromSec(timeNewDataInput);
        aftMappedTrans.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
        aftMappedTrans.setOrigin(tf::Vector3(transformAftMapped[3], transformAftMapped[4], transformAftMapped[5]));
        // tfBroadcaster.sendTransform(aftMappedTrans);

        return {odomAftMapped, aftMappedTrans};
    }

    void resetAll()
    {
        // cloud input data
        newDataInput = false;
        timeNewDataInput = -1;
        laserCloudCornerLast->clear();
        laserCloudCornerLastDS->clear();
        laserCloudSurfLast->clear();
        laserCloudSurfLastDS->clear();
        laserCloudOutlierLast->clear();
        laserCloudOutlierLastDS->clear();
        laserCloudSurfTotalLast->clear();
        laserCloudSurfTotalLastDS->clear();
        // down-sampled num
        laserCloudCornerLastDSNum = 0;
        laserCloudSurfLastDSNum = 0;
        laserCloudOutlierLastDSNum = 0;
        laserCloudSurfTotalLastDSNum = 0;

        // near-map extract state
        latestFrameID = 0;
        // near-map deque
        recentCornerCloudKeyFrames.clear();
        recentSurfCloudKeyFrames.clear();
        recentOutlierCloudKeyFrames.clear();
        // near-map data
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear();
        laserCloudCornerFromMapDS->clear();
        laserCloudSurfFromMapDS->clear();
        // near-map data down-sampled num
        laserCloudCornerFromMapDSNum = 0;
        laserCloudSurfFromMapDSNum = 0;
        // front end match state
        laserCloudOri->clear();
        coeffSel->clear();
        // LM states
        isDegenerate = false;
        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));

        // run state
        timeLastProcessing = -1;
        previousRobotPosPoint = PointType{};
        currentRobotPosPoint = PointType{};

        // pose states
        for (int i = 0; i < 6; ++i)
        {
            transformLast[i] = 0;
            transformSum[i] = 0;
            transformIncre[i] = 0;
            transformTobeMapped[i] = 0;
            transformBefMapped[i] = 0;
            transformAftMapped[i] = 0;
        }

        // global poses
        cloudKeyPoses3D->clear();
        cloudKeyPoses6D->clear();
        // key frame data
        cornerCloudKeyFrames->clear();
        surfCloudKeyFrames->clear();
        outlierCloudKeyFrames->clear();

        // factor-graph
        gtSAMgraph = NonlinearFactorGraph{};
        initialEstimate.clear();
        optimizedEstimate.clear();
        isam.reset(new ISAM2(parameters));
        isamCurrentEstimate.clear();

        poseBase = PointTypePose{};
        aLoopIsClosed = false;
    }

private:
    float cRoll, sRoll, cPitch, sPitch, cYaw, sYaw, tX, tY, tZ;
    float ctRoll, stRoll, ctPitch, stPitch, ctYaw, stYaw, tInX, tInY, tInZ;

    void updatePointAssociateToMapSinCos()
    {
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

    void pointAssociateToMap(PointType const *const pi, PointType *const po)
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

    void updateTransformPointCloudSinCos(PointTypePose *tIn)
    {
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

    // utils
    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn)
    {
        // !!! DO NOT use pcl for point cloud transformation, results are not accurate
        // Reason: unkown
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;
        PointType pointTo;

        int cloudSize = cloudIn->points.size();
        cloudOut->resize(cloudSize);

        for (int i = 0; i < cloudSize; ++i)
        {

            pointFrom = &cloudIn->points[i];
            float x1 = ctYaw * pointFrom->x - stYaw * pointFrom->y;
            float y1 = stYaw * pointFrom->x + ctYaw * pointFrom->y;
            float z1 = pointFrom->z;

            float x2 = x1;
            float y2 = ctRoll * y1 - stRoll * z1;
            float z2 = stRoll * y1 + ctRoll * z1;

            pointTo.x = ctPitch * x2 + stPitch * z2 + tInX;
            pointTo.y = y2 + tInY;
            pointTo.z = -stPitch * x2 + ctPitch * z2 + tInZ;
            pointTo.intensity = pointFrom->intensity;

            cloudOut->points[i] = pointTo;
        }
        return cloudOut;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;
        PointType pointTo;

        int cloudSize = cloudIn->points.size();
        cloudOut->resize(cloudSize);

        for (int i = 0; i < cloudSize; ++i)
        {

            pointFrom = &cloudIn->points[i];
            float x1 = cos(transformIn->yaw) * pointFrom->x - sin(transformIn->yaw) * pointFrom->y;
            float y1 = sin(transformIn->yaw) * pointFrom->x + cos(transformIn->yaw) * pointFrom->y;
            float z1 = pointFrom->z;

            float x2 = x1;
            float y2 = cos(transformIn->roll) * y1 - sin(transformIn->roll) * z1;
            float z2 = sin(transformIn->roll) * y1 + cos(transformIn->roll) * z1;

            pointTo.x = cos(transformIn->pitch) * x2 + sin(transformIn->pitch) * z2 + transformIn->x;
            pointTo.y = y2 + transformIn->y;
            pointTo.z = -sin(transformIn->pitch) * x2 + cos(transformIn->pitch) * z2 + transformIn->z;
            pointTo.intensity = pointFrom->intensity;

            cloudOut->points[i] = pointTo;
        }
        return cloudOut;
    }

    friend class Node;
};

struct LoopState
{
    std::mutex loopMutex;
    int lastSearchedFrame = -1;
    int lastLoopDetectSuccess = -1;
    int closestHistoryFrameID;
    int latestFrameIDLoopCloure;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;
    // point cloud for match
    pcl::PointCloud<PointType>::Ptr latestKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr nearHistoryKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr nearHistoryKeyFrameCloudDS;
    std::vector<std::pair<int, int>> loopBuffers;
};

class Node
{
    // static constexpr int SubmapSequenceSize = 20; // only for debug
    static constexpr int MinSubmapSequenceSize = 10;

    static constexpr int SubmapOverlapSize = 5;
    int switchNew = 0;

    // working submap state
    submapOptimization::Ptr lastSubmap, activeSubmap;
    size_t submap_index = 0;

    // loop-closure
    LoopState localLoop;
    LoopState globalLoop;
    // loop-closure data
    std::vector<std::vector<LidarIris::FeatureDesc>> featureDescList;
    int matchBias;
    LidarIris iris = LidarIris(4, 18, 1.6, 0.75, 50);
    pcl::VoxelGrid<PointType> downSizeFilterHistoryKeyFrames; // for histor key frames of loop closure
    pcl::PointCloud<PointType>::Ptr totalCloudKeyPoses3D;
    std::vector<int> remapIndex;
    std::vector<pcl::PointCloud<PointTypePose>::Ptr> cloudKeyPoses6D;
    std::vector<std::shared_ptr<std::vector<pcl::PointCloud<PointType>::Ptr>>> cornerCloudKeyFrames;
    std::vector<std::shared_ptr<std::vector<pcl::PointCloud<PointType>::Ptr>>> surfCloudKeyFrames;
    // cached global loop
    std::deque<std::tuple<int, int, gtsam::Pose3, noiseModel::Diagonal::shared_ptr>> graphCache;

    // factor-graph
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    std::unique_ptr<ISAM2> isam;
    Values isamCurrentEstimate;
    // useful constants
    noiseModel::Diagonal::shared_ptr priorNoise;
    noiseModel::Diagonal::shared_ptr odometryNoise;
    // noiseModel::Diagonal::shared_ptr constraintNoise;
    // gtsam::Vector6 LMScore;
    // double LMNoise;

    ros::NodeHandle nh;
    // data input
    message_filters::Subscriber<sensor_msgs::PointCloud2> subLaserCloudCornerLast;
    message_filters::Subscriber<sensor_msgs::PointCloud2> subLaserCloudSurfLast;
    message_filters::Subscriber<sensor_msgs::PointCloud2> subOutlierCloudLast;
    message_filters::Subscriber<nav_msgs::Odometry> subLaserOdometry;
    using TimePolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2, sensor_msgs::PointCloud2, nav_msgs::Odometry>;
    std::shared_ptr<message_filters::Synchronizer<TimePolicy>> sync;
    // tranform publisher
    ros::Publisher pubOdomAftMapped;
    tf::TransformBroadcaster tfBroadcaster;

    // indexed points & poses
    ros::Subscriber subOriginLidar;
    ros::Publisher pubOriginLidar;
    ros::Publisher pubGlobalPoses;
    ros::Publisher pubPoseCov;
    pcl::PointCloud<PointType>::Ptr originLidar;
    ros::Time originTime;

    // switch next submap
    ros::Subscriber subNextSubmap;
    bool nextSubmap;

    // debug infos
    ros::Publisher pubKeyPoses;
    ros::Publisher pubKeyPosesLast;
    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubRegisteredCloud;

public:
    Node() : nh("~")
    {
        lastSubmap.reset(new submapOptimization{});
        activeSubmap.reset(new submapOptimization{});

        localLoop.kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        localLoop.latestKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        localLoop.nearHistoryKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        localLoop.nearHistoryKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());

        globalLoop.kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        globalLoop.latestKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        globalLoop.nearHistoryKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        globalLoop.nearHistoryKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());

        featureDescList.push_back({});
        downSizeFilterHistoryKeyFrames.setLeafSize(0.4, 0.4, 0.4); // for histor key frames of loop closure
        totalCloudKeyPoses3D.reset(new pcl::PointCloud<PointType>);

        cloudKeyPoses6D.push_back(activeSubmap->cloudKeyPoses6D);
        cornerCloudKeyFrames.push_back(activeSubmap->cornerCloudKeyFrames);
        surfCloudKeyFrames.push_back(activeSubmap->surfCloudKeyFrames);

        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.01;
        parameters.relinearizeSkip = 1;
        isam.reset(new ISAM2(parameters));

        gtsam::Vector Vector6(6);
        Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-6;
        priorNoise = noiseModel::Diagonal::Variances(Vector6);
        Vector6 << 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-4;
        odometryNoise = noiseModel::Diagonal::Variances(Vector6);

        subLaserCloudCornerLast.subscribe(nh, "/laser_cloud_corner_last", 1);
        subLaserCloudSurfLast.subscribe(nh, "/laser_cloud_surf_last", 1);
        subOutlierCloudLast.subscribe(nh, "/outlier_cloud_last", 1);
        subLaserOdometry.subscribe(nh, "/laser_odom_to_init", 1);
        sync.reset(new message_filters::Synchronizer<TimePolicy>(TimePolicy(2), subLaserCloudCornerLast, subLaserCloudSurfLast, subOutlierCloudLast, subLaserOdometry));
        sync->registerCallback(boost::bind(&Node::dataHandler, this, _1, _2, _3, _4));

        pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 5);

        subOriginLidar = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 2, &Node::originLidarHandler, this);
        pubOriginLidar = nh.advertise<sensor_msgs::PointCloud2>("/points_indexed", 2);
        pubGlobalPoses = nh.advertise<nav_msgs::Path>("/global_poses", 2);
        originLidar.reset(new pcl::PointCloud<PointType>);

        subNextSubmap = nh.subscribe<std_msgs::Empty>("/next_submap", 2, &Node::nextSubmapHandler, this);
        nextSubmap = false;

        pubKeyPoses = nh.advertise<sensor_msgs::PointCloud2>("/key_pose_origin", 2);
        pubKeyPosesLast = nh.advertise<sensor_msgs::PointCloud2>("/key_pose_origin_last", 2);
        pubRecentKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/recent_cloud", 2);
        pubRegisteredCloud = nh.advertise<sensor_msgs::PointCloud2>("/registered_cloud", 2);

        pubPoseCov = nh.advertise<sensor_msgs::PointCloud2>("/pose_with_cov", 2);
    }

    void nextSubmapHandler(const std_msgs::EmptyConstPtr &msg)
    {
        nextSubmap = true;
    }

    void dataHandler(const sensor_msgs::PointCloud2ConstPtr &corner, const sensor_msgs::PointCloud2ConstPtr &surf, const sensor_msgs::PointCloud2ConstPtr &outlier, const nav_msgs::Odometry::ConstPtr &odom)
    {
        {
            std::lock_guard<std::mutex> lock(localLoop.loopMutex);
            bool addNewKeyFrame = false;
            if (switchNew > 0)
            {
                lastSubmap->dataInput(corner, surf, outlier, odom);

                if (lastSubmap->locateRun())
                {
                    addNewKeyFrame = true;
                    switchNew -= 1;

                    {
                        // for debug viz
                        activeSubmap->transformAssociateToMap();
                        activeSubmap->extractSurroundingKeyFrames();
                    }

                    activeSubmap->addOnly(lastSubmap);
                    lastSubmap->keyFrameStage();
                    activeSubmap->keyFrameStage();

                    {
                        // debug viz
                        if (pubKeyPosesLast.getNumSubscribers() != 0)
                        {
                            sensor_msgs::PointCloud2 cloudMsgTemp;
                            pcl::PointCloud<PointType>::Ptr vizPoses(new pcl::PointCloud<PointType>);
                            for (int i = 0; i < lastSubmap->freezeFrameId; i++)
                                vizPoses->push_back(lastSubmap->cloudKeyPoses3D->at(i));
                            pcl::toROSMsg(*vizPoses, cloudMsgTemp);
                            cloudMsgTemp.header.stamp = ros::Time::now();
                            cloudMsgTemp.header.frame_id = "camera_init";
                            pubKeyPosesLast.publish(cloudMsgTemp);
                        }
                    }
                }
                vizPerKeyFrame(addNewKeyFrame, true);

                activeSubmap->clearCloud();
                lastSubmap->clearCloud();
            }
            else
            {
                activeSubmap->dataInput(corner, surf, outlier, odom);

                if (activeSubmap->run())
                {
                    addNewKeyFrame = true;
                }
                vizPerKeyFrame(addNewKeyFrame, false);

                activeSubmap->clearCloud();
            }

            if (addNewKeyFrame)
            {
                pcl::PointCloud<pcl::PointXYZ>::Ptr irisLidar(new pcl::PointCloud<pcl::PointXYZ>);
                pcl::copyPointCloud(*originLidar, *irisLidar);
                cv::Mat1b m = LidarIris::GetIris(*irisLidar);
                LidarIris::FeatureDesc fd = iris.GetFeature(m);
                fd.index = activeSubmap->cloudKeyPoses3D->size() - 1;
                featureDescList.back().push_back(fd);

                // pub pose with cov
                // if (0)
                // if (pubPoseCov.getNumSubscribers() != 0)
                {
                    sensor_msgs::PointCloud2 posewithCovariance;
                    pcl::PointCloud<pcl::PointXYZI> tempCloud;
                    for (int i = 0; i < activeSubmap->cloudKeyPoses3D->size(); i++){
                        pcl::PointXYZI temp = activeSubmap->cloudKeyPoses3D->at(i);
                        temp.intensity = activeSubmap->isam->marginalCovariance(i)(0,0);
                        tempCloud.push_back(temp);
                    }
                    pcl::toROSMsg(tempCloud, posewithCovariance);
                    posewithCovariance.header.stamp = ros::Time::now();
                    posewithCovariance.header.frame_id = "camera_init";
                    pubPoseCov.publish(posewithCovariance);
                }

                publishOriginLidar();
                publishActivePoses();
            }
        }

        {
            // if (activeSubmap->cloudKeyPoses3D->size() == 100)
            if (nextSubmap && activeSubmap->cloudKeyPoses3D->size() >= MinSubmapSequenceSize)
            {
                nextSubmap = false;
                switchNewSubmap();
            }
        }
    }

    void localLoopClosureThread()
    {
        ros::Rate rate(1);
        while (ros::ok())
        {
            rate.sleep();
            localPerformLoopClosure();
        }
    }

    bool localDetectLoopClosure()
    {
        auto &loop = localLoop;
        const double deltaTimeThreshold = 15.0; // 60.0

        loop.latestKeyFrameCloud->clear();
        loop.nearHistoryKeyFrameCloud->clear();
        loop.nearHistoryKeyFrameCloudDS->clear();

        const auto &fds = featureDescList.back();
        if (fds.size() < SubmapOverlapSize || fds.size() <= 20) // TODO:
            return false;

        loop.closestHistoryFrameID = -1;

        LidarIris::FeatureDesc kf = fds.back();

        loop.latestFrameIDLoopCloure = kf.index;
        if (loop.latestFrameIDLoopCloure == loop.lastSearchedFrame)
            return false;
        loop.lastSearchedFrame = loop.latestFrameIDLoopCloure;

        // 根据位置来找回环
        if (loop.latestFrameIDLoopCloure - loop.lastLoopDetectSuccess < 10)
        // if(1)
        {
            std::vector<int> pointSearchIndLoop;
            std::vector<float> pointSearchSqDisLoop;
            loop.kdtreeHistoryKeyPoses->setInputCloud(activeSubmap->cloudKeyPoses3D);
            loop.kdtreeHistoryKeyPoses->radiusSearch(activeSubmap->currentRobotPosPoint, historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, surroundingKeyframeSearchRadius);

            loop.closestHistoryFrameID = -1;
            for (int i = 0; i < pointSearchIndLoop.size(); ++i)
            {
                int id = pointSearchIndLoop[i];
                if (abs(activeSubmap->cloudKeyPoses6D->points[id].time - activeSubmap->timeNewDataInput) > deltaTimeThreshold)
                {
                    loop.closestHistoryFrameID = id;
                    break;
                }
            }
            if (loop.closestHistoryFrameID == -1)
            {
                return false;
            }
        }
        // knn + lidariris找到回换
        else
        {
            float distance = 1;
            int index = -1;

            TicToc tt;
            std::vector<int> pointSearchIndLoop;
            std::vector<float> pointSearchSqDisLoop;

            loop.kdtreeHistoryKeyPoses->setInputCloud(activeSubmap->cloudKeyPoses3D);
            loop.kdtreeHistoryKeyPoses->radiusSearch(activeSubmap->currentRobotPosPoint, historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, surroundingKeyframeSearchRadius);

            loop.closestHistoryFrameID = -1;
            for (int i = 0; i < pointSearchIndLoop.size(); ++i)
            {
                int id = pointSearchIndLoop[i];
                if (abs(activeSubmap->cloudKeyPoses6D->points[id].time - activeSubmap->timeNewDataInput) > deltaTimeThreshold)
                {
                    int bias = 0;
                    float dis = iris.Compare(fds[id], kf, &bias);
                    if (dis < distance)
                    {
                        distance = dis;
                        matchBias = bias;
                        index = id;
                    }
                }
            }

            if (distance < 0.4)
            {
                loop.closestHistoryFrameID = index;
            }

            if (loop.closestHistoryFrameID == -1)
            {
                return false;
            }

            // 回环检测增加时间上的约束
            if (abs(activeSubmap->cloudKeyPoses6D->points[loop.closestHistoryFrameID].time - activeSubmap->cloudKeyPoses6D->points[loop.latestFrameIDLoopCloure].time) <= deltaTimeThreshold)
            {
                return false;
            }
        }
        //// ROS_INFO("detectLoopClosure() ok");

        *loop.latestKeyFrameCloud += *activeSubmap->cornerCloudKeyFrames->at(loop.latestFrameIDLoopCloure);
        *loop.latestKeyFrameCloud += *activeSubmap->surfCloudKeyFrames->at(loop.latestFrameIDLoopCloure);

        pcl::PointCloud<PointType>::Ptr hahaCloud(new pcl::PointCloud<PointType>());
        int cloudSize = loop.latestKeyFrameCloud->points.size();
        for (int i = 0; i < cloudSize; ++i)
        {
            if ((int)loop.latestKeyFrameCloud->points[i].intensity >= 0)
            {
                hahaCloud->push_back(loop.latestKeyFrameCloud->points[i]);
            }
        }
        loop.latestKeyFrameCloud->clear();
        *loop.latestKeyFrameCloud = *hahaCloud;
        for (int i = 0; i < loop.latestKeyFrameCloud->size(); i++)
        {
            PointType p;
            p.x = loop.latestKeyFrameCloud->points[i].z;
            p.y = loop.latestKeyFrameCloud->points[i].x;
            p.z = loop.latestKeyFrameCloud->points[i].y;

            loop.latestKeyFrameCloud->points[i] = p;
        }
        // save history near key frames
        for (int j = -historyKeyframeSearchNum; j <= historyKeyframeSearchNum; ++j)
        {
            if (loop.closestHistoryFrameID + j < 0 || loop.closestHistoryFrameID + j > loop.latestFrameIDLoopCloure)
                continue;
            *loop.nearHistoryKeyFrameCloud += *activeSubmap->transformPointCloud(activeSubmap->cornerCloudKeyFrames->at(loop.closestHistoryFrameID + j), &activeSubmap->cloudKeyPoses6D->points[loop.closestHistoryFrameID + j]);
            *loop.nearHistoryKeyFrameCloud += *activeSubmap->transformPointCloud(activeSubmap->surfCloudKeyFrames->at(loop.closestHistoryFrameID + j), &activeSubmap->cloudKeyPoses6D->points[loop.closestHistoryFrameID + j]);
        }
        downSizeFilterHistoryKeyFrames.setInputCloud(loop.nearHistoryKeyFrameCloud);
        downSizeFilterHistoryKeyFrames.filter(*loop.nearHistoryKeyFrameCloudDS);

        PointTypePose t_prev = activeSubmap->cloudKeyPoses6D->points[loop.closestHistoryFrameID];

        Eigen::Quaternionf q((Eigen::AngleAxisf(t_prev.pitch, Eigen::Vector3f::UnitY()) *
                              Eigen::AngleAxisf(t_prev.roll, Eigen::Vector3f::UnitX()) *
                              Eigen::AngleAxisf(t_prev.yaw, Eigen::Vector3f::UnitZ()))
                                 .matrix());

        Eigen::Matrix4f prev = Eigen::Matrix4f::Identity();
        q.normalize();
        prev.block<3, 3>(0, 0) = q.matrix();
        prev.block<3, 1>(0, 3) = Eigen::Vector3f(t_prev.x, t_prev.y, t_prev.z);

        pcl::transformPointCloud(*loop.nearHistoryKeyFrameCloudDS, *loop.nearHistoryKeyFrameCloudDS, prev.inverse());

        for (int i = 0; i < loop.nearHistoryKeyFrameCloudDS->size(); i++)
        {
            PointType p;
            p.x = loop.nearHistoryKeyFrameCloudDS->points[i].z;
            p.y = loop.nearHistoryKeyFrameCloudDS->points[i].x;
            p.z = loop.nearHistoryKeyFrameCloudDS->points[i].y;

            loop.nearHistoryKeyFrameCloudDS->points[i] = p;
        }
#if 0
        // publish history near key frames
        if (pubHistoryKeyFrames.getNumSubscribers() != 0)
        {
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*nearHistorySurfKeyFrameCloudDS, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "map";
            pubHistoryKeyFrames.publish(cloudMsgTemp);
        }
#endif
        if (loop.nearHistoryKeyFrameCloudDS->size() <= 100)
        {
            std::cout << "too less point for relocalization" << std::endl;
            std::cout << loop.nearHistoryKeyFrameCloudDS->size() << " " << loop.latestKeyFrameCloud->size() << std::endl;
            return false;
        }

        return true;
    }

    void localPerformLoopClosure()
    {
        std::lock_guard<std::mutex> lock(localLoop.loopMutex);
        if (activeSubmap->cloudKeyPoses3D->points.empty() == true)
            return;
        // try to find close key frame if there are any
        if (localDetectLoopClosure() == true)
        {
            ROS_INFO("local detect loop!");
        }
        else
        {
            return;
        }

        // ICP Settings

        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(100); // 100
        icp.setMaximumIterations(100);         // 200
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);
        // Align clouds

        icp.setInputSource(localLoop.latestKeyFrameCloud);
        icp.setInputTarget(localLoop.nearHistoryKeyFrameCloudDS);

        Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();
        ROS_INFO_STREAM("match bias:" << matchBias);
        guess.block<3, 3>(0, 0) = Eigen::AngleAxisf(M_PI * matchBias / 180, Eigen::Vector3f::UnitZ()).matrix();
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        // icp.align(*unused_result);
        icp.align(*unused_result, guess);
        ROS_INFO_STREAM("icp score: " << icp.getFitnessScore());
        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
        {
            if (localLoop.loopBuffers.empty())
            {
                localLoop.loopBuffers.push_back(std::make_pair(localLoop.latestFrameIDLoopCloure, localLoop.closestHistoryFrameID));
                // icp_buf.push_back(icp.getFitnessScore());
            }
            else
            {
                if (std::abs(localLoop.latestFrameIDLoopCloure - localLoop.loopBuffers.back().first) <= 5 && localLoop.closestHistoryFrameID - localLoop.loopBuffers.back().second <= 5 && localLoop.closestHistoryFrameID - localLoop.loopBuffers.back().second >= -5)
                {
                    localLoop.loopBuffers.push_back(std::make_pair(localLoop.latestFrameIDLoopCloure, localLoop.closestHistoryFrameID));
                }
                else
                {
                    localLoop.loopBuffers.clear();
                }
            }

            if (localLoop.loopBuffers.size() <= 10)
                return;
        }

        float noiseScore;

        if (localLoop.loopBuffers.size() <= 10)
        {
            noiseScore = icp.getFitnessScore();
        }
        else
        {
            if (icp.getFitnessScore() <= 5)
            {
                noiseScore = 0.5;
                localLoop.loopBuffers.clear();
                // std::cout << "--------loop--------" << std::endl;
            }
            else
            {
                localLoop.loopBuffers.clear();
                return;
            }
        }

        ROS_INFO("local add a factor to graph! %f ", icp.getFitnessScore());
        /*
        get pose constraint
        */
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionCameraFrame;
        correctionCameraFrame = icp.getFinalTransformation(); // get transformation in camera frame (because points are in camera frame)
        pcl::getTranslationAndEulerAngles(correctionCameraFrame, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        gtsam::Vector Vector6(6);

        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        activeSubmap->constraintNoise = noiseModel::Diagonal::Variances(Vector6);
        /*
        add constraints
        */
        {
            activeSubmap->gtSAMgraph.add(BetweenFactor<Pose3>(localLoop.closestHistoryFrameID, localLoop.latestFrameIDLoopCloure, poseFrom, activeSubmap->constraintNoise));
            activeSubmap->isam->update(activeSubmap->gtSAMgraph);
            activeSubmap->isam->update();
            activeSubmap->gtSAMgraph.resize(0);

            localLoop.lastLoopDetectSuccess = localLoop.latestFrameIDLoopCloure;
        }

        activeSubmap->aLoopIsClosed = true;
    }

    void globalLoopClosureThread()
    {
        ros::Rate rate(1);
        while (ros::ok())
        {
            rate.sleep();
            globalPerformLoopClosure();

            // try update
            {
                std::lock_guard<std::mutex> glock(globalLoop.loopMutex);
                std::lock_guard<std::mutex> llock(localLoop.loopMutex);
                bool added = false;
                while (graphCache.size() > 0)
                {
                    auto [from, to, between, noise] = graphCache.front();
                    auto [tsid, tfid] = unpackIndex(to);
                    if (tsid >= submap_index)
                        break;
                    added = true;
                    graphCache.pop_front();

                    auto [fsid, ffid] = unpackIndex(from);

                    auto Tfb = makeGtPose(cloudKeyPoses6D[fsid]->at(0));
                    auto Tfp = makeGtPose(cloudKeyPoses6D[fsid]->at(ffid));
                    auto Ttb = makeGtPose(cloudKeyPoses6D[tsid]->at(0));
                    auto Ttp = makeGtPose(cloudKeyPoses6D[tsid]->at(tfid));
                    Pose3 delta = Tfb.inverse() * Tfp * between * Ttp.inverse() * Ttb;

                    gtSAMgraph.add(BetweenFactor<Pose3>(fsid, tsid, delta, noise));
                }
                if (added)
                {
                    isam->update(gtSAMgraph);
                    isam->update();
                    gtSAMgraph.resize(0);

                    correctPoses();
                }
            }
        }
    }

    void correctPoses()
    {
        std::vector<Pose3> poses;
        isamCurrentEstimate = isam->calculateEstimate();
        for (int i = 0; i < submap_index; i++)
        {
            poses.push_back(isamCurrentEstimate.at<Pose3>(i));
        }
        isamCurrentEstimate.clear();

        int currentS = -1;
        gtsam::Pose3 delta;
        for (int i = 0; i < remapIndex.size(); i++)
        {
            auto [sid, fid] = unpackIndex(remapIndex[i]);
            if (sid > currentS)
            {
                auto poseBase = cloudKeyPoses6D[sid]->at(0);
                auto oldPose = Pose3(Rot3::RzRyRx(poseBase.yaw, poseBase.roll, poseBase.pitch), Point3(poseBase.z, poseBase.x, poseBase.y));
                delta = poses[sid] * oldPose.inverse();

                currentS = sid;
            }
            auto &pose3d = totalCloudKeyPoses3D->points[i];
            auto &pose6d = cloudKeyPoses6D[sid]->points[fid];

            auto pose = delta * Pose3(Rot3::RzRyRx(pose6d.yaw, pose6d.roll, pose6d.pitch), Point3(pose6d.z, pose6d.x, pose6d.y));

            pose3d.x = pose.translation().y();
            pose3d.y = pose.translation().z();
            pose3d.z = pose.translation().x();

            pose6d.x = pose3d.x;
            pose6d.y = pose3d.y;
            pose6d.z = pose3d.z;
            pose6d.roll = pose.rotation().pitch();
            pose6d.pitch = pose.rotation().yaw();
            pose6d.yaw = pose.rotation().roll();
        }

        if (cloudKeyPoses6D.back()->size() > 0)
        {
            ROS_INFO_STREAM(makeGtPose(activeSubmap->poseBase) << makeGtPose(cloudKeyPoses6D.back()->front()));
            activeSubmap->updatePoseBase(poses.back() * makeGtPose(lastSubmap->poseBase).inverse() * makeGtPose(activeSubmap->poseBase));
            activeSubmap->recentCornerCloudKeyFrames.clear();
            activeSubmap->recentSurfCloudKeyFrames.clear();
            activeSubmap->recentOutlierCloudKeyFrames.clear();
        }

        lastSubmap->updatePoseBase(poses.back());

        publishGlobalPoses();
    }

    int packIndex(int submap, int frame)
    {
        return (submap << 16) + frame;
    }

    std::tuple<int, int> unpackIndex(int index)
    {
        return {index >> 16, index & ((1 << 16) - 1)};
    }

    int indexDistance(int a, int b)
    {
        bool neg = false;
        if (a < b)
        {
            neg = true;
            std::swap(a, b);
        }
        auto [sa, fa] = unpackIndex(a);
        auto [sb, fb] = unpackIndex(b);
        int count = 0;
        while (sb < sa)
        {
            count += cloudKeyPoses6D[sb]->size() - fb;
            sb += 1;
            fb = 0;
        }
        count += fa - fb;
        if (neg)
            count = -count;
        return count;
    }

    bool globalDetectLoopClosure()
    {
        auto &loop = globalLoop;
        const double deltaTimeThreshold = 60.0;

        loop.latestKeyFrameCloud->clear();
        loop.nearHistoryKeyFrameCloud->clear();
        loop.nearHistoryKeyFrameCloudDS->clear();

        std::lock_guard<std::mutex> glock(loop.loopMutex);
        std::lock_guard<std::mutex> llock(localLoop.loopMutex);

        {
            if (featureDescList.back().size() == 0)
                return false;
            int count = 0;
            for (const auto &it : featureDescList)
                count += it.size();
            if (count < 20)
                return false;
        }

        loop.closestHistoryFrameID = -1;

        LidarIris::FeatureDesc kf = featureDescList.back().back();

        loop.latestFrameIDLoopCloure = packIndex(featureDescList.size() - 1, kf.index);
        if (loop.latestFrameIDLoopCloure == loop.lastSearchedFrame)
            return false;
        loop.lastSearchedFrame = loop.latestFrameIDLoopCloure;

        // 根据位置来找回环
        if (loop.lastLoopDetectSuccess >= 0 && indexDistance(loop.latestFrameIDLoopCloure, loop.lastLoopDetectSuccess) < 50)
        // if(1)
        {
            std::vector<int> pointSearchIndLoop;
            std::vector<float> pointSearchSqDisLoop;
            loop.kdtreeHistoryKeyPoses->setInputCloud(totalCloudKeyPoses3D);
            loop.kdtreeHistoryKeyPoses->radiusSearch(activeSubmap->currentRobotPosPoint, historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, surroundingKeyframeSearchRadius);

            loop.closestHistoryFrameID = -1;
            for (int i = 0; i < pointSearchIndLoop.size(); ++i)
            {
                int id = remapIndex[pointSearchIndLoop[i]];
                auto [sid, fid] = unpackIndex(id);
                if (sid == submap_index)
                    continue;
                if (abs(cloudKeyPoses6D[sid]->points[fid].time - activeSubmap->timeNewDataInput) > deltaTimeThreshold)
                {
                    loop.closestHistoryFrameID = id;
                    break;
                }
            }
            if (loop.closestHistoryFrameID == -1)
            {
                return false;
            }
        }
        // knn + lidariris找到回换
        else
        {
            float distance = 1;
            int index = -1;

            TicToc tt;
            std::vector<int> pointSearchIndLoop;
            std::vector<float> pointSearchSqDisLoop;

            loop.kdtreeHistoryKeyPoses->setInputCloud(totalCloudKeyPoses3D);
            loop.kdtreeHistoryKeyPoses->radiusSearch(activeSubmap->currentRobotPosPoint, historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, surroundingKeyframeSearchRadius);

            loop.closestHistoryFrameID = -1;
            for (int i = 0; i < pointSearchIndLoop.size(); ++i)
            {
                int id = remapIndex[pointSearchIndLoop[i]];
                auto [sid, fid] = unpackIndex(id);
                if (sid == submap_index)
                    continue;
                if (abs(cloudKeyPoses6D[sid]->points[fid].time - activeSubmap->timeNewDataInput) > deltaTimeThreshold)
                {
                    int bias = 0;
                    float dis = iris.Compare(featureDescList[sid][fid], kf, &bias);
                    if (dis < distance)
                    {
                        distance = dis;
                        matchBias = bias;
                        index = id;
                    }
                }
            }

            if (distance < 0.4)
            {
                loop.closestHistoryFrameID = index;
            }

            if (loop.closestHistoryFrameID == -1)
            {
                return false;
            }

            // 回环检测增加时间上的约束
            auto [closeSid, closeFid] = unpackIndex(loop.closestHistoryFrameID);
            auto [latestSid, latestFid] = unpackIndex(loop.latestFrameIDLoopCloure);
            if (abs(cloudKeyPoses6D[closeSid]->points[closeFid].time - cloudKeyPoses6D[latestSid]->points[latestFid].time) <= deltaTimeThreshold)
            {
                return false;
            }
        }
        //// ROS_INFO("detectLoopClosure() ok");
        auto [latestSid, latestFid] = unpackIndex(loop.latestFrameIDLoopCloure);
        auto [closeSid, closeFid] = unpackIndex(loop.closestHistoryFrameID);

        *loop.latestKeyFrameCloud += *cornerCloudKeyFrames[latestSid]->at(latestFid);
        *loop.latestKeyFrameCloud += *surfCloudKeyFrames[latestSid]->at(latestFid);

        pcl::PointCloud<PointType>::Ptr hahaCloud(new pcl::PointCloud<PointType>());
        int cloudSize = loop.latestKeyFrameCloud->points.size();
        for (int i = 0; i < cloudSize; ++i)
        {
            if ((int)loop.latestKeyFrameCloud->points[i].intensity >= 0)
            {
                hahaCloud->push_back(loop.latestKeyFrameCloud->points[i]);
            }
        }
        loop.latestKeyFrameCloud->clear();
        *loop.latestKeyFrameCloud = *hahaCloud;
        for (int i = 0; i < loop.latestKeyFrameCloud->size(); i++)
        {
            PointType p;
            p.x = loop.latestKeyFrameCloud->points[i].z;
            p.y = loop.latestKeyFrameCloud->points[i].x;
            p.z = loop.latestKeyFrameCloud->points[i].y;

            loop.latestKeyFrameCloud->points[i] = p;
        }
        // save history near key frames
        for (int j = -historyKeyframeSearchNum; j <= historyKeyframeSearchNum; ++j)
        {
            auto [sid, fid] = unpackIndex(loop.closestHistoryFrameID);
            fid += j;
            while (fid < 0)
            {
                sid -= 1;
                if (sid < 0)
                    break;
                fid + cloudKeyPoses6D[sid]->size();
            }
            if (sid < 0)
                continue;
            while (fid >= cloudKeyPoses6D[sid]->size())
            {
                fid -= cloudKeyPoses6D[sid]->size();
                sid += 1;
                if (sid >= cloudKeyPoses6D.size())
                    break;
            }
            if (sid >= cloudKeyPoses6D.size())
                continue;
            if (packIndex(sid, fid) > loop.latestFrameIDLoopCloure)
                continue;
            *loop.nearHistoryKeyFrameCloud += *activeSubmap->transformPointCloud(cornerCloudKeyFrames[sid]->at(fid), &cloudKeyPoses6D[sid]->points[fid]);
            *loop.nearHistoryKeyFrameCloud += *activeSubmap->transformPointCloud(surfCloudKeyFrames[sid]->at(fid), &cloudKeyPoses6D[sid]->points[fid]);
        }
        downSizeFilterHistoryKeyFrames.setInputCloud(loop.nearHistoryKeyFrameCloud);
        downSizeFilterHistoryKeyFrames.filter(*loop.nearHistoryKeyFrameCloudDS);

        PointTypePose t_prev = cloudKeyPoses6D[closeSid]->points[closeFid];

        Eigen::Quaternionf q((Eigen::AngleAxisf(t_prev.pitch, Eigen::Vector3f::UnitY()) *
                              Eigen::AngleAxisf(t_prev.roll, Eigen::Vector3f::UnitX()) *
                              Eigen::AngleAxisf(t_prev.yaw, Eigen::Vector3f::UnitZ()))
                                 .matrix());

        Eigen::Matrix4f prev = Eigen::Matrix4f::Identity();
        q.normalize();
        prev.block<3, 3>(0, 0) = q.matrix();
        prev.block<3, 1>(0, 3) = Eigen::Vector3f(t_prev.x, t_prev.y, t_prev.z);

        pcl::transformPointCloud(*loop.nearHistoryKeyFrameCloudDS, *loop.nearHistoryKeyFrameCloudDS, prev.inverse());

        for (int i = 0; i < loop.nearHistoryKeyFrameCloudDS->size(); i++)
        {
            PointType p;
            p.x = loop.nearHistoryKeyFrameCloudDS->points[i].z;
            p.y = loop.nearHistoryKeyFrameCloudDS->points[i].x;
            p.z = loop.nearHistoryKeyFrameCloudDS->points[i].y;

            loop.nearHistoryKeyFrameCloudDS->points[i] = p;
        }
#if 0
        // publish history near key frames
        if (pubHistoryKeyFrames.getNumSubscribers() != 0)
        {
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*nearHistorySurfKeyFrameCloudDS, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "map";
            pubHistoryKeyFrames.publish(cloudMsgTemp);
        }
#endif
        if (loop.nearHistoryKeyFrameCloudDS->size() <= 100)
        {
            std::cout << "too less point for relocalization" << std::endl;
            std::cout << loop.nearHistoryKeyFrameCloudDS->size() << " " << loop.latestKeyFrameCloud->size() << std::endl;
            return false;
        }

        return true;
    }

    void globalPerformLoopClosure()
    {
        int activeSubmap;
        if (totalCloudKeyPoses3D->points.empty() == true)
            return;
        // try to find close key frame if there are any
        if (globalDetectLoopClosure() == true)
        {
            ROS_INFO("global detect loop!");
        }
        else
        {
            return;
        }

        // ICP Settings

        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(100); // 100
        icp.setMaximumIterations(100);         // 200
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);
        // Align clouds

        icp.setInputSource(globalLoop.latestKeyFrameCloud);
        icp.setInputTarget(globalLoop.nearHistoryKeyFrameCloudDS);

        Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();
        ROS_INFO_STREAM("global match bias:" << matchBias);
        guess.block<3, 3>(0, 0) = Eigen::AngleAxisf(M_PI * matchBias / 180, Eigen::Vector3f::UnitZ()).matrix();
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        // icp.align(*unused_result);
        icp.align(*unused_result, guess);
        ROS_INFO_STREAM("global icp score: " << icp.getFitnessScore());
        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
        {
            if (globalLoop.loopBuffers.empty())
            {
                globalLoop.loopBuffers.push_back(std::make_pair(globalLoop.latestFrameIDLoopCloure, globalLoop.closestHistoryFrameID));
                // icp_buf.push_back(icp.getFitnessScore());
            }
            else
            {
                if (std::abs(indexDistance(globalLoop.latestFrameIDLoopCloure, globalLoop.loopBuffers.back().first)) <= 5 && std::abs(indexDistance(globalLoop.closestHistoryFrameID, globalLoop.loopBuffers.back().second)) <= 5)
                {
                    globalLoop.loopBuffers.push_back(std::make_pair(globalLoop.latestFrameIDLoopCloure, globalLoop.closestHistoryFrameID));
                }
                else
                {
                    globalLoop.loopBuffers.clear();
                }
            }

            if (globalLoop.loopBuffers.size() <= 10)
                return;
        }

        float noiseScore;

        if (globalLoop.loopBuffers.size() <= 10)
        {
            noiseScore = icp.getFitnessScore();
        }
        else
        {
            if (icp.getFitnessScore() <= 5)
            {
                noiseScore = 0.5;
                globalLoop.loopBuffers.clear();
                // ROS_INFO_STREAM("--------global loop--------");
            }
            else
            {
                globalLoop.loopBuffers.clear();
                return;
            }
        }

        ROS_INFO("global add a factor to graph! %f ", icp.getFitnessScore());
        /*
        get pose constraint
        */
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionCameraFrame;
        correctionCameraFrame = icp.getFinalTransformation(); // get transformation in camera frame (because points are in camera frame)
        pcl::getTranslationAndEulerAngles(correctionCameraFrame, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        gtsam::Vector Vector6(6);

        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);
        /*
        add constraints
        */
        {
            std::lock_guard<std::mutex> lock(globalLoop.loopMutex);
            graphCache.push_back({globalLoop.closestHistoryFrameID, globalLoop.latestFrameIDLoopCloure, poseFrom, constraintNoise});

            globalLoop.lastLoopDetectSuccess = globalLoop.latestFrameIDLoopCloure;
        }
    }

    void switchNewSubmap()
    {
        // wait once local loop-closure detect
        while (true)
        {
            {
                std::lock_guard<std::mutex> lock(localLoop.loopMutex);
                if (localLoop.lastSearchedFrame >= activeSubmap->cloudKeyPoses3D->size() - 1)
                    break;
            }
            std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
        }

        std::lock_guard<std::mutex> glock(globalLoop.loopMutex);
        std::lock_guard<std::mutex> llock(localLoop.loopMutex);

        switchNew = SubmapOverlapSize;

        cloudKeyPoses6D.pop_back();
        cornerCloudKeyFrames.pop_back();
        surfCloudKeyFrames.pop_back();
        pcl::PointCloud<PointTypePose>::Ptr localKeyPose6D(new pcl::PointCloud<PointTypePose>);
        pcl::copyPointCloud(*activeSubmap->cloudKeyPoses6D, *localKeyPose6D);
        cloudKeyPoses6D.push_back(localKeyPose6D);
        std::shared_ptr<std::vector<pcl::PointCloud<PointType>::Ptr>> localCornerCloudKeyFrames(new std::vector<pcl::PointCloud<PointType>::Ptr>{});
        std::shared_ptr<std::vector<pcl::PointCloud<PointType>::Ptr>> localSurfCloudKeyFrames(new std::vector<pcl::PointCloud<PointType>::Ptr>{});
        std::copy(activeSubmap->cornerCloudKeyFrames->begin(), activeSubmap->cornerCloudKeyFrames->end(), std::back_inserter(*localCornerCloudKeyFrames));
        std::copy(activeSubmap->surfCloudKeyFrames->begin(), activeSubmap->surfCloudKeyFrames->end(), std::back_inserter(*localSurfCloudKeyFrames));
        cornerCloudKeyFrames.push_back(localCornerCloudKeyFrames);
        surfCloudKeyFrames.push_back(localSurfCloudKeyFrames);
        for (int i = 0; i < activeSubmap->cloudKeyPoses3D->size(); i++)
        {
            const auto &p = activeSubmap->cloudKeyPoses3D->at(i);
            PointType totalP;
            totalP.getArray3fMap() = p.getArray3fMap();
            totalCloudKeyPoses3D->push_back(totalP);
            remapIndex.push_back(packIndex(submap_index, i));
        }

        activeSubmap->freeze();

        PointTypePose lastPose = lastSubmap->poseBase;
        PointTypePose currentPose = activeSubmap->poseBase;

        std::swap(activeSubmap, lastSubmap);
        activeSubmap->resetAll();
        submap_index += 1;

        cloudKeyPoses6D.push_back(activeSubmap->cloudKeyPoses6D);
        cornerCloudKeyFrames.push_back(activeSubmap->cornerCloudKeyFrames);
        surfCloudKeyFrames.push_back(activeSubmap->surfCloudKeyFrames);

        // should do after local opt finish
        addSubmapToGlobalGraph(lastPose, currentPose);

        correctPoses();
        // publishGlobalPoses();

        featureDescList.push_back({});
    }

    void addSubmapToGlobalGraph(const PointTypePose &lastPose, const PointTypePose &currentPose) // beforeLast, afterLast, real-cov
    {
        if (submap_index == 1)
        {
            gtSAMgraph.add(PriorFactor<Pose3>(0, Pose3(Rot3::RzRyRx(currentPose.yaw, currentPose.roll, currentPose.pitch), Point3(currentPose.z, currentPose.x, currentPose.y)), priorNoise));
            initialEstimate.insert(0, Pose3(Rot3::RzRyRx(currentPose.yaw, currentPose.roll, currentPose.pitch),
                                            Point3(currentPose.z, currentPose.x, currentPose.y)));
        }
        else
        {
            gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(lastPose.yaw, lastPose.roll, lastPose.pitch),
                                          Point3(lastPose.z, lastPose.x, lastPose.y));
            gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(currentPose.yaw, currentPose.roll, currentPose.pitch),
                                        Point3(currentPose.z, currentPose.x, currentPose.y));
            gtSAMgraph.add(BetweenFactor<Pose3>(submap_index - 2, submap_index - 1, poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(submap_index - 1, Pose3(Rot3::RzRyRx(currentPose.yaw, currentPose.roll, currentPose.pitch),
                                                           Point3(currentPose.z, currentPose.x, currentPose.y)));
        }
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();
        gtSAMgraph.resize(0);
        initialEstimate.clear();

        // isamCurrentEstimate = isam->calculateEstimate();
        // lastSubmap->updatePoseBase(isamCurrentEstimate.at<Pose3>(submap_index - 1));
        // isamCurrentEstimate.clear();
    }

    void publishGlobalPoses()
    {
        nav_msgs::Path poses;
        poses.header.frame_id = "camera_init";
        poses.header.stamp = ros::Time::now();
        for (size_t index = 0; index < cloudKeyPoses6D.size(); index++)
        {
            if (cloudKeyPoses6D[index]->size() == 0)
                continue;
            auto pose = makeIsometry(cloudKeyPoses6D[index]->at(0));

            const auto &translation = pose.translation();
            const auto &rotation = Eigen::Quaterniond(pose.rotation());
            geometry_msgs::PoseStamped poseMsg;
            poseMsg.header.frame_id = std::to_string(index);
            poseMsg.pose.position.x = translation.x();
            poseMsg.pose.position.y = translation.y();
            poseMsg.pose.position.z = translation.z();// + submap_index * 4; // only for debug
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
    }

    void originLidarHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
    {
        originLidar->clear();
        pcl::fromROSMsg(*msg, *originLidar);
        originTime = msg->header.stamp;
    }

    void publishOriginLidar()
    {
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(*originLidar, msg);
        msg.header.frame_id = std::to_string(submap_index) + ":" + std::to_string(activeSubmap->cloudKeyPoses6D->size() - 1);
        msg.header.stamp = originTime;
        pubOriginLidar.publish(msg);
    }

    void publishActivePoses()
    {
        nav_msgs::Path poses;
        poses.header.frame_id = "camera_init";
        poses.header.stamp = ros::Time::now();
        for (size_t index = 0; index < activeSubmap->cloudKeyPoses6D->size(); index++)
        {
            auto pose = makeIsometry(activeSubmap->cloudKeyPoses6D->points[index]);

            const auto &translation = pose.translation();
            const auto &rotation = Eigen::Quaterniond(pose.rotation());
            geometry_msgs::PoseStamped poseMsg;
            poseMsg.header.frame_id = std::to_string(submap_index) + ":" + std::to_string(index);
            poseMsg.pose.position.x = translation.x();
            poseMsg.pose.position.y = translation.y();
            poseMsg.pose.position.z = translation.z();
            poseMsg.pose.orientation.x = rotation.x();
            poseMsg.pose.orientation.y = rotation.y();
            poseMsg.pose.orientation.z = rotation.z();
            poseMsg.pose.orientation.w = rotation.w();
            poses.poses.push_back(poseMsg);
            //
#if 0
            tf::StampedTransform transform;
            transform.stamp_ = ros::Time().now();
            transform.frame_id_ = "global_map";
            transform.child_frame_id_ = poseMsg.header.frame_id;
            transform.setRotation(tf::Quaternion(rotation.x(), rotation.y(), rotation.z(), rotation.w()));
            transform.setOrigin(tf::Vector3(translation.x(), translation.y(), translation.z()));
            tfBroadcaster.sendTransform(transform);
#endif
        }
        pubGlobalPoses.publish(poses);
    }

private:
    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw = transformIn[2];
        return thisPose6D;
    }

    void vizPerKeyFrame(bool hasNewFrame, bool overlapStage)
    {
        // publish TF
        auto submap = activeSubmap;
        if (overlapStage)
            submap = lastSubmap;
        auto [odom, transform] = submap->publishTF();
        odom.header.frame_id = "camera_init";
        odom.child_frame_id = "aft_mapped";
        transform.frame_id_ = "camera_init";
        transform.child_frame_id_ = "aft_mapped";
        pubOdomAftMapped.publish(odom);
        tfBroadcaster.sendTransform(transform);

        if (!hasNewFrame)
            return;

        // publish viz
        if (pubKeyPoses.getNumSubscribers() != 0)
        {
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*activeSubmap->cloudKeyPoses3D, cloudMsgTemp);
            cloudMsgTemp.header.stamp = odom.header.stamp;
            cloudMsgTemp.header.frame_id = "camera_init";
            pubKeyPoses.publish(cloudMsgTemp);
        }
        
        if (pubRecentKeyFrames.getNumSubscribers() != 0)
        {
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*activeSubmap->laserCloudSurfFromMapDS, cloudMsgTemp);
            cloudMsgTemp.header.stamp = odom.header.stamp;
            cloudMsgTemp.header.frame_id = "camera_init";
            pubRecentKeyFrames.publish(cloudMsgTemp);
        }

        if (pubRegisteredCloud.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            PointTypePose thisPose6D = trans2PointTypePose(activeSubmap->transformTobeMapped);
            *cloudOut += *activeSubmap->transformPointCloud(activeSubmap->laserCloudCornerLastDS, &thisPose6D);
            *cloudOut += *activeSubmap->transformPointCloud(activeSubmap->laserCloudSurfTotalLast, &thisPose6D);

            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*cloudOut, cloudMsgTemp);
            cloudMsgTemp.header.stamp = odom.header.stamp;
            cloudMsgTemp.header.frame_id = "camera_init";
            pubRegisteredCloud.publish(cloudMsgTemp);
        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lego_loam");

    ROS_INFO("\033[1;32m---->\033[0m Map Optimization Started.");

    Node n;

    // std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
    // std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

    std::thread localLoopThread(&Node::localLoopClosureThread, &n);
    std::thread globalLoopThread(&Node::globalLoopClosureThread, &n);

    ros::spin();

    localLoopThread.join();
    globalLoopThread.join();
    // loopthread.join();
    // visualizeMapThread.join();

    return 0;
}
