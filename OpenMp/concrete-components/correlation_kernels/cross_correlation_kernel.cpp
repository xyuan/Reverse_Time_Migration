//
// Created by mirnamoawad on 10/30/19.
//
#include "cross_correlation_kernel.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <skeleton/helpers/memory_allocation/memory_allocator.h>

using namespace std;
#define EPSILON 1e-20

template <bool is_2D, CompensationType comp>
void Correlation(float *out, GridBox *in_1, GridBox *in_2,
                 AcousticOmpComputationParameters *parameters,
				 float* source_illumination, float* receiver_illumination) {

  GridBox *in_grid_1 = in_1;
  GridBox *in_grid_2 = in_2;
  int nx = in_2->grid_size.nx;
  int ny = in_2->grid_size.ny;
  int nz = in_2->grid_size.nz;
  int wnx = in_2->window_size.window_nx;
  int wny = in_2->window_size.window_ny;
  int wnz = in_2->window_size.window_nz;
  float *frame_1 = in_grid_1->pressure_current;
  float *frame_2 = in_grid_2->pressure_current;
  float *curr_1;
  float *curr_2;
  float *source_i;
  float *receive_i;
  float *output = out;
  float *curr_o;
  uint offset = parameters->half_length;
  int nxEnd = wnx - offset;
  int nyEnd;
  int nzEnd = wnz - offset;
  int y_start;
  if (!is_2D) {
    y_start = offset;
    nyEnd = wny - offset;
  } else {
    y_start = 0;
    nyEnd = 1;
  }
#pragma omp parallel default(shared)
  {
    const uint block_x = parameters->block_x;
    const uint block_y = parameters->block_y;
    const uint block_z = parameters->block_z;

#pragma omp for schedule(static, 1) collapse(2)
    for (int by = y_start; by < nyEnd; by += block_y) {
      for (int bz = offset; bz < nzEnd; bz += block_z) {
        for (int bx = offset; bx < nxEnd; bx += block_x) {

          int izEnd = fmin(bz + block_z, nzEnd);
          int iyEnd = fmin(by + block_y, nyEnd);
          int ixEnd = fmin(block_x, nxEnd - bx);

          for (int iy = by; iy < iyEnd; ++iy) {
            for (int iz = bz; iz < izEnd; ++iz) {
              uint b_offset = iy * wnx * wnz + iz * wnx + bx;
              curr_1 = frame_1 + b_offset;
              curr_2 = frame_2 + b_offset;
              curr_o = output + b_offset;
              source_i = source_illumination + b_offset;
              receive_i = receiver_illumination + b_offset;

#pragma vector aligned
#pragma ivdep
              for (int ix = 0; ix < ixEnd; ++ix) {
                float value;

                value = curr_1[ix] * curr_2[ix];
                curr_o[ix] += value;

                if(comp == SOURCE_COMPENSATION){
                	source_i[ix] += curr_1[ix] * curr_1[ix];
                }
                else if(comp == RECEIVER_COMPENSATION){
                	receive_i[ix] += curr_2[ix] * curr_2[ix];
                }
                else if(comp == COMBINED_COMPENSATION){
                	source_i[ix] += curr_1[ix] * curr_1[ix];
                	receive_i[ix] += curr_2[ix] * curr_2[ix];
                }
              }
            }
          }
        }
      }
    }
  }
}

void CrossCorrelationKernel ::Correlate(GridBox *in_1) {
	if (grid->grid_size.ny == 1) {
		switch(compensation_type){

		case NO_COMPENSATION:
			Correlation<true, NO_COMPENSATION>(this->shot_correlation, in_1, grid, parameters, source_illumination, receiver_illumination);
			break;

		case SOURCE_COMPENSATION:
			Correlation<true, SOURCE_COMPENSATION>(this->shot_correlation, in_1, grid, parameters, source_illumination, receiver_illumination);
			break;

		case RECEIVER_COMPENSATION:
			Correlation<true, RECEIVER_COMPENSATION>(this->shot_correlation, in_1, grid, parameters, source_illumination, receiver_illumination);
			break;

		case COMBINED_COMPENSATION:
			Correlation<true, COMBINED_COMPENSATION>(this->shot_correlation, in_1, grid, parameters, source_illumination, receiver_illumination);
			break;

		}
	} else {
		switch(compensation_type){

		case NO_COMPENSATION:
			Correlation<false, NO_COMPENSATION>(this->shot_correlation, in_1, grid, parameters, source_illumination, receiver_illumination);
			break;

		case SOURCE_COMPENSATION:
			Correlation<false, SOURCE_COMPENSATION>(this->shot_correlation, in_1, grid, parameters, source_illumination, receiver_illumination);
			break;

		case RECEIVER_COMPENSATION:
			Correlation<false, RECEIVER_COMPENSATION>(this->shot_correlation, in_1, grid, parameters, source_illumination, receiver_illumination);
			break;

		case COMBINED_COMPENSATION:
			Correlation<false, COMBINED_COMPENSATION>(this->shot_correlation, in_1, grid, parameters, source_illumination, receiver_illumination);
			break;

		}
	}
}

void CrossCorrelationKernel ::Stack() {
  int wnx = grid->window_size.window_nx;
  int wny = grid->window_size.window_ny;
  int wnz = grid->window_size.window_nz;
  int nx = grid->grid_size.nx;
  int ny = grid->grid_size.ny;
  int nz = grid->grid_size.nz;
  float *in = this->shot_correlation;
  float *out = this->total_correlation + grid->window_size.window_start.x + grid->window_size.window_start.z * nx
          + grid->window_size.window_start.y * nx * nz;
  float *in_src = this->source_illumination;
  float *out_src = this->total_source_illumination + grid->window_size.window_start.x + grid->window_size.window_start.z * nx
          + grid->window_size.window_start.y * nx * nz;
  float *in_rcv = this->receiver_illumination;
  float *out_rcv = this->total_receiver_illumination + grid->window_size.window_start.x + grid->window_size.window_start.z * nx
          + grid->window_size.window_start.y * nx * nz;
  float *input;
  float *output;
  float *input_src;
  float *output_src;
  float *input_rcv;
  float *output_rcv;
  uint block_x = parameters->block_x;
  uint block_z = parameters->block_z;
  uint block_y = parameters->block_y;
  uint offset = parameters->half_length + parameters->boundary_length;
  int nxEnd = wnx - offset;
  int nyEnd;
  int nzEnd = wnz - offset;
  int y_start;
  if (ny > 1) {
    y_start = offset;
    nyEnd = wny - offset;
  } else {
    y_start = 0;
    nyEnd = 1;
  }
#pragma omp parallel for schedule(static, 1) collapse(2)
  for (int by = y_start; by < nyEnd; by += block_y) {
    for (int bz = offset; bz < nzEnd; bz += block_z) {
      for (int bx = offset; bx < nxEnd; bx += block_x) {

        int izEnd = fmin(bz + block_z, nzEnd);
        int iyEnd = fmin(by + block_y, nyEnd);
        int ixEnd = fmin(bx + block_x, nxEnd);

        for (int iy = by; iy < iyEnd; iy++) {
          for (int iz = bz; iz < izEnd; iz++) {
            uint offset_w = iy * wnx * nz + iz * wnx;
            uint offset = iy * nx * nz + iz * nx;
            input = in + offset_w;
            output = out + offset;
            input_src = in_src + offset_w;
            output_src = out_src + offset;
            input_rcv = in_rcv + offset_w;
            output_rcv = out_rcv + offset;
#pragma ivdep
#pragma vector aligned
            for (int ix = bx; ix < ixEnd; ix++) {
              output[ix] += input[ix];
              output_rcv[ix] += input_rcv[ix];
              output_src[ix] += input_src[ix];
            }
          }
        }
      }
    }
  }
}

CrossCorrelationKernel ::~CrossCorrelationKernel() {
  mem_free((void *)shot_correlation);
  mem_free((void *)total_correlation);
  mem_free((void *)source_illumination);
  mem_free((void *)receiver_illumination);
  mem_free((void *)total_source_illumination);
  mem_free((void *)total_receiver_illumination);
}

void CrossCorrelationKernel::SetComputationParameters(
    ComputationParameters *parameters) {
  this->parameters = (AcousticOmpComputationParameters *)(parameters);
}

void CrossCorrelationKernel::SetCompensation(CompensationType c)
{
	compensation_type = c;
}

void CrossCorrelationKernel::SetGridBox(GridBox *grid_box) {
  this->grid = grid_box;
  shot_correlation = (float *)mem_allocate(
      sizeof(float),
      grid_box->window_size.window_nx * grid_box->window_size.window_nz * grid_box->window_size.window_ny,
      "shot_correlation", parameters->half_length);
  total_correlation = (float *)mem_allocate(
      sizeof(float),
      grid_box->grid_size.nx * grid_box->grid_size.nz * grid_box->grid_size.ny,
      "stacked_shot_correlation");
  num_bytes = grid_box->grid_size.nx * grid_box->grid_size.nz *
              grid_box->grid_size.ny * sizeof(float);
  memset(total_correlation, 0, num_bytes);

  source_illumination = (float *)mem_allocate(
      sizeof(float),
      grid_box->window_size.window_nx * grid_box->window_size.window_nz * grid_box->window_size.window_ny,
      "source_illumination", parameters->half_length);
  receiver_illumination = (float *)mem_allocate(\
      sizeof(float),
      grid_box->window_size.window_nx * grid_box->window_size.window_nz * grid_box->window_size.window_ny,
      "receiver_illumination", parameters->half_length);
  total_source_illumination = (float *)mem_allocate(
      sizeof(float),
      grid_box->grid_size.nx * grid_box->grid_size.nz * grid_box->grid_size.ny,
      "stacked_source_illumination");
  memset(total_source_illumination, 0, num_bytes);

  total_receiver_illumination = (float *)mem_allocate(
      sizeof(float),
      grid_box->grid_size.nx * grid_box->grid_size.nz * grid_box->grid_size.ny,
      "stacked_receiver_illumination");
  memset(total_receiver_illumination, 0, num_bytes);
}

CrossCorrelationKernel::CrossCorrelationKernel() {}

void CrossCorrelationKernel::ResetShotCorrelation() {
  uint window_bytes = sizeof(float) * grid->window_size.window_nx *
          grid->window_size.window_nz * grid->window_size.window_ny;
  memset(shot_correlation, 0, window_bytes);
  memset(source_illumination, 0, window_bytes);
  memset(receiver_illumination, 0, window_bytes);
}

float *CrossCorrelationKernel::GetShotCorrelation() {
  return this->shot_correlation;
}

float *CrossCorrelationKernel::GetStackedShotCorrelation() {
  return this->total_correlation;
}

MigrationData *CrossCorrelationKernel::GetMigrationData() {

	float* target;
	switch(compensation_type)
	{
	case NO_COMPENSATION:
		target = this->total_correlation;
		break;
	case SOURCE_COMPENSATION:
		target = GetSourceCompensationCorrelation();
		break;
	case RECEIVER_COMPENSATION:
		target = GetReceiverCompensationCorrelation();
		break;
	case COMBINED_COMPENSATION:
		target = GetCombinedCompensationCorrelation();
		break;
	default:
		target = GetCombinedCompensationCorrelation();
		break;
	}
	return new MigrationData(
      grid->grid_size.nx, grid->grid_size.nz, grid->grid_size.ny, grid->nt,
      grid->cell_dimensions.dx, grid->cell_dimensions.dz,
      grid->cell_dimensions.dy, grid->dt, target);
}

float *CrossCorrelationKernel::GetSourceCompensationCorrelation()
{
	uint size = grid->grid_size.nx * grid->grid_size.nz * grid->grid_size.ny;
	source_illumination_compensation = new float[size];
	for(int i = 0; i < size; i++)
	{
            source_illumination_compensation[i] = total_correlation[i] / (total_source_illumination[i] + EPSILON);
	}

	return source_illumination_compensation;
}

float *CrossCorrelationKernel::GetReceiverCompensationCorrelation()
{
	uint size = grid->grid_size.nx * grid->grid_size.nz * grid->grid_size.ny;
	receiver_illumination_compensation = new float[size];
	for(int i = 0; i < size; i++)
	{
		    receiver_illumination_compensation[i] = total_correlation[i] / (total_receiver_illumination[i] + EPSILON);
	}

	return receiver_illumination_compensation;
}

float *CrossCorrelationKernel::GetCombinedCompensationCorrelation()
{
	uint size = grid->grid_size.nx * grid->grid_size.nz * grid->grid_size.ny;
	combined_illumination_compensation = new float[size];
	for(int i = 0; i < size; i++)
	{
		combined_illumination_compensation[i] = total_correlation[i] / sqrt(total_source_illumination[i] * total_receiver_illumination[i]  + EPSILON);
	}

	return combined_illumination_compensation;
}
