/*
 *  Copyright 2008-2012 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/omp/detail/remove.h>
#include <thrust/system/detail/generic/remove.h>

namespace thrust
{
namespace system
{
namespace omp
{
namespace detail
{

template<typename System,
         typename ForwardIterator,
         typename Predicate>
  ForwardIterator remove_if(dispatchable<System> &system,
                            ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred)
{
  // omp prefers generic::remove_if to cpp::remove_if
  return thrust::system::detail::generic::remove_if(system, first, last, pred);
}


template<typename System,
         typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
  ForwardIterator remove_if(dispatchable<System> &system,
                            ForwardIterator first,
                            ForwardIterator last,
                            InputIterator stencil,
                            Predicate pred)
{
  // omp prefers generic::remove_if to cpp::remove_if
  return thrust::system::detail::generic::remove_if(system, first, last, stencil, pred);
}


template<typename System,
         typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(dispatchable<System> &system,
                                InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                Predicate pred)
{
  // omp prefers generic::remove_copy_if to cpp::remove_copy_if
  return thrust::system::detail::generic::remove_copy_if(system, first, last, result, pred);
}

template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(dispatchable<System> &system,
                                InputIterator1 first,
                                InputIterator1 last,
                                InputIterator2 stencil,
                                OutputIterator result,
                                Predicate pred)
{
  // omp prefers generic::remove_copy_if to cpp::remove_copy_if
  return thrust::system::detail::generic::remove_copy_if(system, first, last, stencil, result, pred);
}

} // end namespace detail
} // end namespace omp
} // end namespace system
} // end namespace thrust

