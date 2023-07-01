#pragma once

#include <array>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/Vertices.h>

#ifndef _AZ_CONCAVE2D_HPP_
#define _AZ_CONCAVE2D_HPP_

// https://github.com/delfrrr/delaunator-cpp
#include "delaunator.hpp"

namespace concave2d
{

    struct Point
    {
        std::unordered_map<size_t, size_t> edge; // {point_id, edge_id}
    };

    struct Edge
    {
        std::vector<std::tuple<size_t, size_t>> tri; // {tri_id, edge_local_index}
        std::array<size_t, 2> point;
        double length;
        bool drop;
    };

    struct Triangle
    {
        std::array<size_t, 3> edge;
        bool drop;
    };

    template <typename PointT>
    std::vector<pcl::Vertices> concave2d(const pcl::PointCloud<PointT> &points, float alpha)
    {
        std::vector<Point> pointMeta;
        std::vector<Edge> edgeMeta;
        std::vector<Triangle> triMeta;

        auto pointSort = [&](const std::array<size_t, 3> &input) -> std::array<size_t, 4> // the more one for quick index
        {
            Eigen::Vector3d p0{points[input[0]].x, points[input[0]].y, 0.0};
            Eigen::Vector3d p1{points[input[1]].x, points[input[1]].y, 0.0};
            Eigen::Vector3d p2{points[input[2]].x, points[input[2]].y, 0.0};
            Eigen::Vector3d crossed = (p1 - p0).cross(p2 - p0);
            if (crossed.z() > 0)
            {
                return {input[0], input[1], input[2], input[0]};
            }
            else
            {
                return {input[0], input[2], input[1], input[0]};
            }
        };

        auto edgeLength = [&](Edge &edge)
        {
            auto p0 = points[edge.point[0]];
            auto p1 = points[edge.point[1]];
            edge.length = std::sqrt((p0.x - p1.x) * (p0.x - p1.x) + (p0.y - p1.y) * (p0.y - p1.y));
        };

        std::vector<double> linearData;
        for (const auto &p : points)
        {
            linearData.push_back(p.x);
            linearData.push_back(p.y);
            pointMeta.emplace_back();
        }
        delaunator::Delaunator d(linearData);

        // make indices
        for (std::size_t i = 0; i < d.triangles.size(); i += 3)
        {
            size_t triIndex = triMeta.size();
            Triangle tri;
            tri.drop = false;
            auto p = pointSort({d.triangles[i], d.triangles[i + 1], d.triangles[i + 2]});
            for (int l = 0; l < 3; l++)
            {
                if (pointMeta[p[l]].edge.find(p[l + 1]) == pointMeta[p[l]].edge.end())
                {
                    size_t edgeIndex = edgeMeta.size();
                    Edge edge;
                    edge.drop = false;
                    edge.point = {p[l], p[l + 1]};
                    edgeLength(edge);
                    edgeMeta.push_back(edge);
                    pointMeta[p[l]].edge[p[l + 1]] = edgeIndex;
                    pointMeta[p[l + 1]].edge[p[l]] = edgeIndex;
                }
                size_t edgeIndex = pointMeta[p[l]].edge[p[l + 1]];
                tri.edge[l] = edgeIndex;
                edgeMeta[edgeIndex].tri.push_back({triIndex, l});
            }
            triMeta.push_back(tri);
        }

        // analyze drop
        for (size_t i = 0; i < edgeMeta.size(); i++)
        {
            auto e = edgeMeta[i];
            if (e.length > alpha)
            {
                e.drop = true;
                for (const auto &t : e.tri)
                    triMeta[std::get<0>(t)].drop = true;
            }
        }

        // collect edge
        std::unordered_set<size_t> usefulEdge;
        for (size_t i = 0; i < triMeta.size(); i++)
        {
            if (!triMeta[i].drop)
            {
                for (const auto &e : triMeta[i].edge)
                {
                    if ((edgeMeta[e].tri.size() < 2) || (triMeta[std::get<0>(edgeMeta[e].tri[0])].drop != triMeta[std::get<0>(edgeMeta[e].tri[1])].drop))
                    {
                        usefulEdge.insert(e);
                    }
                }
            }
        }

        // make chains
        std::vector<std::vector<size_t>> chains;
        while (usefulEdge.size() > 0)
        {
            size_t cur = *usefulEdge.begin();
            usefulEdge.erase(cur);

            std::vector<size_t> chain{cur};

            bool finishChain = false;
            while (!finishChain)
            {
                // get the tri
                std::tuple<size_t, size_t> triInfo;
                for (const auto &t : edgeMeta[cur].tri)
                    if (!triMeta[std::get<0>(t)].drop)
                    {
                        triInfo = t;
                        break;
                    }

                while (true)
                {
                    // move next
                    size_t next = triMeta[std::get<0>(triInfo)].edge[(std::get<1>(triInfo) + 1) % 3];

                    // test stop
                    if (next == *chain.begin())
                    {
                        chains.push_back(chain);
                        finishChain = true;
                        break;
                    }

                    // test new chain
                    if (usefulEdge.find(next) != usefulEdge.end())
                    {
                        chain.push_back(next);
                        cur = next;
                        usefulEdge.erase(cur);
                        break;
                    }

                    // move to neighbor
                    for (const auto &tri : edgeMeta[next].tri)
                    {
                        if (std::get<0>(tri) != std::get<0>(triInfo))
                        {
                            triInfo = tri;
                            break;
                        }
                    }
                }
            }
        }

        // chains to polygons
        std::vector<pcl::Vertices> polygons;
        for (auto &chain : chains)
        {
            int N = chain.size();
            chain.push_back(chain[0]);
            pcl::Vertices polygon;
            for (size_t i = 0; i < N; i++)
            {
                auto cur = edgeMeta[chain[i]];
                auto next = edgeMeta[chain[i + 1]];
                if ((cur.point[0] == next.point[0]) || (cur.point[0] == next.point[1]))
                {
                    polygon.vertices.push_back(cur.point[1]);
                }
                else
                {
                    polygon.vertices.push_back(cur.point[0]);
                }
            }
            polygons.push_back(polygon);
        }
        return polygons;
    }

}

#endif // _AZ_CONCAVE2D_HPP_
