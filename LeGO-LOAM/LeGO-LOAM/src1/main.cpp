#include <iostream>
#include <vector>
#include <string>
#include <pcl/io/pcd_io.h>
#include "LidarIris.h"

#include <boost/filesystem.hpp>
#include <ctime>

using namespace std;

void OneCoupleCompare(string cloudFileName1, string cloudFileName2)
{
    LidarIris iris(4, 18, 1.6, 0.75, 50);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud0(new pcl::PointCloud<pcl::PointXYZ>), cloud1(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile(cloudFileName1, *cloud0) == -1)
    {
        abort();
    }
    if (pcl::io::loadPCDFile(cloudFileName2, *cloud1) == -1)
    {
        abort();
    }
    int bias;
    auto dis = iris.Compare(iris.GetFeature(LidarIris::GetIris(*cloud0)), iris.GetFeature(LidarIris::GetIris(*cloud1)), &bias);
    cout << "try compare:" << endl
         << cloudFileName1 << endl
         << cloudFileName2 << endl;
    cout << "dis = " << dis << ", bias = " << bias << endl;
}

void MatixCompare(std::vector<string> files, cv::Mat1f &disMat, cv::Mat1s biasMat)
{
    LidarIris iris(4, 18, 1.6, 0.75, 50);
    disMat = cv::Mat1f::zeros(files.size(), files.size());
    biasMat = cv::Mat1f::zeros(files.size(), files.size());
    std::vector<cv::Mat1b> mats(files.size());
    for (int i = 0; i < files.size(); i++)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPCDFile(files[i], *cloud) == -1)
        {
            abort();
        }
        mats[i] = LidarIris::GetIris(*cloud);
    }
    std::vector<LidarIris::FeatureDesc> features(files.size());
    std::vector<std::vector<float>> vecs(files.size());
    //
    clock_t startTime = clock();
    //
    for (int i = 0; i < files.size(); i++)
    {
        features[i] = iris.GetFeature(mats[i], vecs[i]);
    }
    //
    int count = 0;
    for (int i = 0; i < files.size(); i++)
    {
        for (int j = i + 1; j < files.size(); j++)
        {
            int bias;
            float dis = iris.Compare(features[i], features[j]);
            disMat.at<float>(i, j) = dis;
            disMat.at<float>(j, i) = dis;
            biasMat.at<uint16_t>(i, j) = bias;
            biasMat.at<uint16_t>(j, i) = bias;
            count++;
        }
    }
    //
    clock_t endTime = clock();
    cout << (endTime - startTime) / (double)CLOCKS_PER_SEC / count << endl;
}

void FlowCompare(std::vector<string> files, std::vector<int> &matches)
{
    LidarIris iris(4, 18, 1.6, 0.75, 50);
    std::vector<cv::Mat1b> mats(files.size());
    for (int i = 0; i < files.size(); i++)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPCDFile(files[i], *cloud) == -1)
        {
            abort();
        }
        mats[i] = LidarIris::GetIris(*cloud);
    }
    matches = std::vector<int>(files.size());
    //
    clock_t startTime = clock();
    for (int i = 0; i < files.size(); i++)
    {
        iris.UpdateFrame(mats[i], i, nullptr, &matches[i]);
    }
    //
    clock_t endTime = clock();
    cout << (endTime - startTime) / (double)CLOCKS_PER_SEC / files.size() << endl;
}

namespace fs = boost::filesystem;

vector<string> GetFiles(fs::path full_path)
{
    vector<string> result;
    if (fs::exists(full_path))
    {
        fs::directory_iterator item_begin(full_path);
        fs::directory_iterator item_end;
        for (; item_begin != item_end; item_begin++)
        {
            if (!fs::is_directory(*item_begin))
            {
                result.push_back(item_begin->path().generic_string());
            }
        }
    }
    return result;
}

// #define MATRIX_TEST

int main(int argc, char *argv[])
{
#ifdef ONE_COUPLE_TEST
    OneCoupleCompare("/mnt/d/PointCloudData/20.pcd", "/mnt/d/PointCloudData/21.pcd");
    OneCoupleCompare("/mnt/d/PointCloudData/20.pcd", "/mnt/d/PointCloudData/87.pcd");
#elif (defined MATRIX_TEST)
    auto paths = GetFiles(argv[1]);
    cv::Mat1f dis;
    cv::Mat1b bias;
    MatixCompare(paths, dis, bias);
    cv::imwrite(string(argv[1]) + "/result/dis.bmp", dis * 255);
    cv::imwrite(string(argv[1]) + "/result/bias.bmp", bias);
#else
    auto paths = GetFiles(argv[1]);
    std::vector<int> matches;
    FlowCompare(paths, matches);
    for (auto it : matches)
    {
        cout << it << ",";
    }
    cout << endl;
#endif
    return 0;
}