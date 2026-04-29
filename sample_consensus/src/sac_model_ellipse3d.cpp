/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2014-, Open Perception Inc.
 *
 *  All rights reserved
 */

#include <pcl/sample_consensus/impl/sac_model_ellipse3d.hpp>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

namespace pcl
{
  namespace internal
  {
    int
    optimizeModelCoefficientsEllipse3D (Eigen::VectorXf &coeff, const Eigen::ArrayXf &pts_x, const Eigen::ArrayXf &pts_y, const Eigen::ArrayXf &pts_z)
    {
      struct Ellipse3DOptimizationFunctor : pcl::Functor<double>
      {
        Ellipse3DOptimizationFunctor (const Eigen::ArrayXf& x, const Eigen::ArrayXf& y, const Eigen::ArrayXf& z) :
          pcl::Functor<double> (static_cast<int>(x.size ())), x_ (x), y_ (y), z_ (z) {}

        int operator() (const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
        {
          // c : Ellipse Center
          const Eigen::Vector3f c (static_cast<float>(x[0]), static_cast<float>(x[1]), static_cast<float>(x[2]));
          // a : Ellipse semi-major axis (X) length
          const float par_a (static_cast<float>(x [3]));
          // b : Ellipse semi-minor axis (Y) length
          const float par_b (static_cast<float>(x [4]));
          // n : Ellipse (Plane) Normal
          const Eigen::Vector3f n_axis = Eigen::Vector3f (static_cast<float>(x[5]), static_cast<float>(x[6]), static_cast<float>(x[7])).normalized ();
          // x : Ellipse (Plane) X-Axis
          Eigen::Vector3f x_ax = Eigen::Vector3f (static_cast<float>(x[8]), static_cast<float>(x[9]), static_cast<float>(x[10]));
          x_ax = (x_ax - x_ax.dot (n_axis) * n_axis).normalized ();
          // y : Ellipse (Plane) Y-Axis
          const Eigen::Vector3f y_ax = n_axis.cross (x_ax).normalized ();

          // Compute the rotation matrix and its transpose
          const Eigen::Matrix3f Rot = (Eigen::Matrix3f (3,3)
            << x_ax (0), y_ax(0), n_axis(0),
            x_ax (1), y_ax(1), n_axis(1),
            x_ax (2), y_ax(2), n_axis(2))
            .finished ();
          const Eigen::Matrix3f Rot_T = Rot.transpose ();

          const Eigen::VectorXf params = (Eigen::VectorXf (5) << par_a, par_b, 0.0f, 0.0f, 0.0f).finished ();
          for (int i = 0; i < values (); ++i)
          {
            const Eigen::Vector3f p (x_ [i], y_ [i], z_ [i]);
            const Eigen::Vector3f p_ = Rot_T * (p - c);
            float th_opt;
            // k : Point on Ellipse
            // Calculate the shortest distance from the point to the ellipse which is
            // given by the norm of a vector that is normal to the ellipse tangent
            // calculated at the point it intersects the tangent.
            fvec [i] = static_cast<double>(dvec2ellipse (params, p_ (0), p_ (1), th_opt).norm ());
          }
          return (0);
        }

        const Eigen::ArrayXf &x_, &y_, &z_;
      };

      Ellipse3DOptimizationFunctor functor (pts_x, pts_y, pts_z);
      Eigen::NumericalDiff<Ellipse3DOptimizationFunctor> num_diff (functor);
      Eigen::LevenbergMarquardt<Eigen::NumericalDiff<Ellipse3DOptimizationFunctor>, double> lm (num_diff);
      Eigen::VectorXd coeff_double = coeff.cast<double> ();
      const int info = lm.minimize (coeff_double);
      coeff = coeff_double.cast<float> ();
      
      const Eigen::Vector3f n_axis = Eigen::Vector3f (coeff[5], coeff[6], coeff[7]).normalized ();
      coeff[5] = n_axis[0];
      coeff[6] = n_axis[1];
      coeff[7] = n_axis[2];

      Eigen::Vector3f x_ax = Eigen::Vector3f (coeff[8], coeff[9], coeff[10]);
      x_ax = (x_ax - x_ax.dot (n_axis) * n_axis).normalized ();
      coeff[8] = x_ax[0];
      coeff[9] = x_ax[1];
      coeff[10] = x_ax[2];

      return info;
    }
  }
}

#ifndef PCL_NO_PRECOMPILE
#include <pcl/impl/instantiate.hpp>
#include <pcl/point_types.h>

// Instantiations of specific point types
#ifdef PCL_ONLY_CORE_POINT_TYPES
  PCL_INSTANTIATE(SampleConsensusModelEllipse3D, (pcl::PointXYZ)(pcl::PointXYZI)(pcl::PointXYZRGBA)(pcl::PointXYZRGB)(pcl::PointXYZRGBNormal))
#else
  PCL_INSTANTIATE(SampleConsensusModelEllipse3D, PCL_XYZ_POINT_TYPES)
#endif
#endif    // PCL_NO_PRECOMPILE
