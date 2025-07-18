/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *
 */

#pragma once

#include <type_traits>

namespace pcl
{
  /** \brief Helper functor structure for concatenate. 
    * \ingroup common
    */
  template<typename PointInT, typename PointOutT>
  struct NdConcatenateFunctor
  {
    using PodIn = typename traits::POD<PointInT>::type;
    using PodOut = typename traits::POD<PointOutT>::type;
    
    NdConcatenateFunctor (const PointInT &p1, PointOutT &p2)
      : p1_ (reinterpret_cast<const PodIn&> (p1))
      , p2_ (reinterpret_cast<PodOut&> (p2)) { }

    template<typename Key> inline void 
    operator () ()
    {
      // This sucks without Fusion :(
      //boost::fusion::at_key<Key> (p2_) = boost::fusion::at_key<Key> (p1_);
      using InT = typename pcl::traits::datatype<PointInT, Key>::type;
      using OutT = typename pcl::traits::datatype<PointOutT, Key>::type;
      *reinterpret_cast<OutT*>(reinterpret_cast<std::uint8_t*>(&p2_) + pcl::traits::offset<PointOutT, Key>::value) =
        *reinterpret_cast<const InT*>(reinterpret_cast<const std::uint8_t*>(&p1_) + pcl::traits::offset<PointInT, Key>::value);
    }

    private:
      const PodIn &p1_;
      PodOut &p2_;
  };
}
