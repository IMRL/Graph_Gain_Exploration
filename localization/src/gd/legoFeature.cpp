#include "legoFeature.h"

cv::Mat1f calcRangeMat(const cv::Mat3f &imagery)
{
    cv::Mat1f xyz[3];
    cv::split(imagery, xyz);
    cv::Mat1f range;
    cv::sqrt(xyz[0].mul(xyz[0]) + xyz[1].mul(xyz[1]) + xyz[2].mul(xyz[2]), range);
    return range;
}

std::vector<cv::Point> neighbors = {
    {0, -1},
    {0, 1},
    {1, 0},
    {-1, 0}};

void labelComponents(const cv::Mat1f &range, const cv::Mat1b &valid, const cv::Point &start, cv::Mat1b &labelMat)
{
    std::queue<cv::Point> q;
    q.push(start);
    cv::Mat1b tempMat = cv::Mat1b::zeros(labelMat.size());
    tempMat(start) = 255;
    int minY = start.y;
    int maxY = start.y;
    while (!q.empty())
    {
        cv::Point current = q.front();
        q.pop();
        for (const auto &offset : neighbors)
        {
            cv::Point next = current + offset;
            next.x = (next.x + labelMat.cols) % labelMat.cols;
            if (next.y < 0 || next.y >= labelMat.rows)
                continue;
            if (!valid(next) || labelMat(next) != 0 || tempMat(next) != 0)
                continue;
            float d1 = std::max(range(current), range(next));
            float d2 = std::min(range(current), range(next));
            float alpha = offset.x != 0 ? segAlphaX : segAlphaY;
            float angle = std::atan2(d2 * sin(alpha), d1 - d2 * cos(alpha));
            if (angle > segmentTheta)
            {
                q.push(next);
                tempMat(next) = 255;
                maxY = std::max(maxY, next.y);
                minY = std::min(minY, next.y);
            }
        }
    }
    if (cv::sum(tempMat)[0] <= LeastSegPoint || maxY - minY + 1 <= LeastSegLine)
    {
        // labelMat |= tempMat;
        tempMat &= 0x7f;
        // labelMat += tempMat * label;
        // return true;
    }
    labelMat |= tempMat;
    // return false;
}

cv::Mat1b legoFeature(const cv::Mat3f &imagery, const cv::Mat1b &mask, const cv::Mat1b &valid, float max_range)
{
    auto rangeMat = calcRangeMat(imagery);
    // cv::Mat1i label = cv::Mat1i::zeros(imagery.size());
    cv::Mat1b label = cv::Mat1b::zeros(imagery.size());
    // int currentLabel = 1;
    cv::Mat1b real_valid = valid & (mask != 255) & (rangeMat < max_range);

    for (int r = 0; r < imagery.rows; r++)
        for (int c = 0; c < imagery.cols; c++)
            if (real_valid(r, c) && label(r, c) == 0)
            {
                labelComponents(rangeMat, real_valid, {c, r}, label);
                // currentLabel += 1;
            }
    // return label != 0;
    return label == 255;
}
