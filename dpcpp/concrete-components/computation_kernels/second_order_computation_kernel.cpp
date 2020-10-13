#include "second_order_computation_kernel.h"
#include <cmath>
#include <iostream>
#include <rtm-framework/skeleton/helpers/timer/timer.hpp>

//
// Created by mirnamoawad on 10/28/19.
//

#define fma(a, b, c) ((a) * (b) + (c))

SecondOrderComputationKernel::~SecondOrderComputationKernel() {
  cl::sycl::free((void *)d_coeff_x, in_queue->get_context());
  cl::sycl::free((void *)d_coeff_y, in_queue->get_context());
  cl::sycl::free((void *)d_coeff_z, in_queue->get_context());
  cl::sycl::free((void *)d_vertical, in_queue->get_context());
  cl::sycl::free((void *)d_front, in_queue->get_context());
  // time_out->close();
}

SecondOrderComputationKernel::SecondOrderComputationKernel() {
  this->boundary_manager = nullptr;
  this->grid = nullptr;
  this->parameters = nullptr;
}

template <bool is_2D, HALF_LENGTH half_length>
void SecondOrderComputationKernel::Computation_syclDevice(
		AcousticSecondGrid *grid, AcousticDpcComputationParameters *parameters) {


	AcousticDpcComputationParameters::device_queue->submit([&](handler &cgh) {
		const float *current = grid->pressure_current;
		float *next = grid->pressure_next;
		const float *prev = grid->pressure_previous;
		const float *vel = grid->window_velocity;
		const float *c_x = d_coeff_x;
		const float *c_z = d_coeff_z;
		const float c_xyz = coeff_xyz;
		const int *v = d_vertical;
		const size_t wnx = grid->window_size.window_nx;
		const size_t wnz = grid->window_size.window_nz;

		auto global_range = range<2>(wnx - 2 * half_length, wnz - 2 * half_length);
		auto local_range = range<2>(parameters->block_x, parameters->block_z);
		auto global_offset = id<2>(half_length, half_length);
		auto global_nd_range = nd_range<2>(global_range, local_range, global_offset);

		cgh.parallel_for<class secondOrderComputationKernel>(
				global_nd_range, [=](nd_item<2> it) {
			int x = it.get_global_id(0);
			int z = it.get_global_id(1);

			int idx = wnx * z + x;

			float value = current[idx] * c_xyz;

			value = fma(current[idx - 1] + current[idx + 1], c_x[0], value);
			value = fma(current[idx - v[0]] + current[idx + v[0]], c_z[0], value);

			if (half_length > 1) {
				value = fma(current[idx - 2] + current[idx + 2], c_x[1], value);
				value = fma(current[idx - v[1]] + current[idx + v[1]], c_z[1], value);
			}
			if (half_length > 2) {
				value = fma(current[idx - 3] + current[idx + 3], c_x[2], value);
				value = fma(current[idx - 4] + current[idx + 4], c_x[3], value);
				value = fma(current[idx - v[2]] + current[idx + v[2]], c_z[2], value);
				value = fma(current[idx - v[3]] + current[idx + v[3]], c_z[3], value);
			}
			if (half_length > 4) {
				value = fma(current[idx - 5] + current[idx + 5], c_x[4], value);
				value = fma(current[idx - 6] + current[idx + 6], c_x[5], value);
				value = fma(current[idx - v[4]] + current[idx + v[4]], c_z[4], value);
				value = fma(current[idx - v[5]] + current[idx + v[5]], c_z[5], value);
			}
			if (half_length > 6) {
				value = fma(current[idx - 7] + current[idx + 7], c_x[6], value);
				value = fma(current[idx - 8] + current[idx + 8], c_x[7], value);
				value = fma(current[idx - v[6]] + current[idx + v[6]], c_z[6], value);
				value = fma(current[idx - v[7]] + current[idx + v[7]], c_z[7], value);
			}

			next[idx] = (2 * current[idx]) - prev[idx] + (vel[idx] * value);

		});
	});

	AcousticDpcComputationParameters::device_queue->wait();
}

void SecondOrderComputationKernel::Step() {
  // Pre-compute the coefficients for each direction.
  int half_length = parameters->half_length;

  int size = (grid->original_dimensions.nx - 2 * half_length) *
             (grid->original_dimensions.nz - 2 * half_length);

  // General note: floating point operations for forward is the same as backward
  // (calculated below are for forward). number of floating point operations for
  // the computation kernel in 2D for the half_length loop:6*k,where K is the
  // half_length 5 floating point operations outside the half_length loop Total
  // = 6*K+5 =6*K+5
  int flops_per_second = 6 * half_length + 5;

  // Take a step in time.
  Timer *timer = Timer::getInstance();
  timer->_start_timer_for_kernel("ComputationKernel::kernel", size, 4,
                                 true, flops_per_second);
  if ((grid->grid_size.ny) == 1) {
    switch (parameters->half_length) {
    case O_2:
      Computation_syclDevice<true, O_2>(grid, parameters);
      break;
    case O_4:
      Computation_syclDevice<true, O_4>(grid, parameters);
      break;
    case O_8:
      Computation_syclDevice<true, O_8>(grid, parameters);
      break;
    case O_12:
      Computation_syclDevice<true, O_12>(grid, parameters);
      break;
    case O_16:
      Computation_syclDevice<true, O_16>(grid, parameters);
      break;
    }
  } else {
    /*
    switch (parameters->half_length) {
        case O_2:
            Computation_syclDevice<false, O_2>(grid, parameters);
            break;
        case O_4:
            Computation_syclDevice<false, O_4>(grid, parameters);
            break;
        case O_8:
            Computation_syclDevice<false, O_8>(grid, parameters);
            break;
        case O_12:
            Computation_syclDevice<false, O_12>(grid, parameters);
            break;
        case O_16:
            Computation_syclDevice<false, O_16>(grid, parameters);
            break;
    }
     */
    std::cout << "3D not supported" << std::endl;
  }
  // Swap pointers : Next to current, current to prev and unwanted prev to next
  // to be overwritten.
  if (grid->pressure_previous == grid->pressure_next) {
    // two pointers case : curr becomes both next and prev, while next becomes
    // current.
    grid->pressure_previous = grid->pressure_current;
    grid->pressure_current = grid->pressure_next;
    grid->pressure_next = grid->pressure_previous;
  } else {
    // three pointers : normal swapping between the three pointers.
    float *temp = grid->pressure_next;
    grid->pressure_next = grid->pressure_previous;
    grid->pressure_previous = grid->pressure_current;
    grid->pressure_current = temp;
  }
  timer->stop_timer("ComputationKernel::kernel");
  timer->start_timer("BoundaryManager::ApplyBoundary");
  if (this->boundary_manager != nullptr) {
    this->boundary_manager->ApplyBoundary(0);
  }
  timer->stop_timer("BoundaryManager::ApplyBoundary");
}

void SecondOrderComputationKernel::FirstTouch(float *ptr, uint nx, uint nz,
		uint ny) {
	uint half_length = parameters->half_length;


	AcousticDpcComputationParameters::device_queue->submit([&](handler &cgh){

		auto global_range = range<2>(nx - 2 * half_length, nz - 2 * half_length);
		auto local_range = range<2>(parameters->block_x, parameters->block_z);
		auto global_offset = id<2>(half_length, half_length);
		auto global_nd_range = nd_range<2>(global_range, local_range, global_offset);

		cgh.parallel_for<class secondOrderComputationKernel>(
				global_nd_range, [=](nd_item<2> it) {

			int x = it.get_global_id(0);
			int z = it.get_global_id(1);

			int idx = nx * z + x;

			ptr[idx] = 0.0f;

		});
	});
	AcousticDpcComputationParameters::device_queue->wait();
}

void SecondOrderComputationKernel::SetComputationParameters(
    ComputationParameters *parameters) {
  this->parameters = (AcousticDpcComputationParameters *)(parameters);
  if (this->parameters == nullptr) {
    std::cout << "Not a compatible computation parameters : "
                 "expected AcousticDpcComputationParameters"
              << std::endl;
    exit(-1);
  }
}

void SecondOrderComputationKernel::SetGridBox(GridBox *grid_box) {
  ;
  this->grid = (AcousticSecondGrid *)(grid_box);
  if (this->grid == nullptr) {
    std::cout << "Not a compatible gridbox : "
                 "expected AcousticSecondGrid"
              << std::endl;
    exit(-1);
  }
  in_queue = AcousticDpcComputationParameters::device_queue;
  int wnx = grid->window_size.window_nx;
  int wny = grid->window_size.window_ny;
  int wnz = grid->window_size.window_nz;
  float dx = grid->cell_dimensions.dx;
  float dy;
  float dz = grid->cell_dimensions.dz;
  float dx2 = 1 / (dx * dx);
  float dy2;
  float dz2 = 1 / (dz * dz);
  float *coeff = parameters->second_derivative_fd_coeff;
  bool is_2D = wny == 1;
  int wnxnz = wnx * wnz;

  if (!is_2D) {
    dy = grid->cell_dimensions.dy;
    dy2 = 1 / (dy * dy);
  }
  int hl = parameters->half_length;
  int array_length = sizeof(float) * hl;
  float coeff_x[hl];
  float coeff_y[hl];
  float coeff_z[hl];
  int vertical[hl];
  int front[hl];
  d_coeff_x = (float *)cl::sycl::malloc_device(
      array_length,
      AcousticDpcComputationParameters::device_queue->get_device(),
      AcousticDpcComputationParameters::device_queue->get_context());
  d_coeff_y = (float *)cl::sycl::malloc_device(
      array_length,
      AcousticDpcComputationParameters::device_queue->get_device(),
      AcousticDpcComputationParameters::device_queue->get_context());
  d_coeff_z = (float *)cl::sycl::malloc_device(
      array_length,
      AcousticDpcComputationParameters::device_queue->get_device(),
      AcousticDpcComputationParameters::device_queue->get_context());
  d_vertical = (int *)cl::sycl::malloc_device(
      hl * sizeof(int),
      AcousticDpcComputationParameters::device_queue->get_device(),
      AcousticDpcComputationParameters::device_queue->get_context());
  d_front = (int *)cl::sycl::malloc_device(
      hl * sizeof(int),
      AcousticDpcComputationParameters::device_queue->get_device(),
      AcousticDpcComputationParameters::device_queue->get_context());
  for (int i = 0; i < hl; i++) {
    coeff_x[i] = coeff[i + 1] * dx2;
    coeff_z[i] = coeff[i + 1] * dz2;
    vertical[i] = (i + 1) * (wnx);

    if (!is_2D) {
      coeff_y[i] = coeff[i + 1] * dy2;
      front[i] = (i + 1) * wnxnz;
    }
  }
  if (is_2D) {
    coeff_xyz = coeff[0] * (dx2 + dz2);
  } else {
    coeff_xyz = coeff[0] * (dx2 + dy2 + dz2);
  }
  AcousticDpcComputationParameters::device_queue->submit(
      [&](handler &cgh) { cgh.memcpy(d_coeff_x, coeff_x, array_length); });
  AcousticDpcComputationParameters::device_queue->submit(
      [&](handler &cgh) { cgh.memcpy(d_coeff_z, coeff_z, array_length); });
  AcousticDpcComputationParameters::device_queue->submit([&](handler &cgh) {
    cgh.memcpy(d_vertical, vertical, hl * sizeof(int));
  });
  if (!is_2D) {
    AcousticDpcComputationParameters::device_queue->submit(
        [&](handler &cgh) { cgh.memcpy(d_coeff_y, coeff_y, array_length); });
    AcousticDpcComputationParameters::device_queue->submit(
        [&](handler &cgh) { cgh.memcpy(d_front, front, hl * sizeof(int)); });
  }
  AcousticDpcComputationParameters::device_queue->wait();
}
