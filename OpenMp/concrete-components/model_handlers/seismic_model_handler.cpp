//
// Created by ingy on 1/26/20.
//

#include "seismic_model_handler.h"
#include <bits/stdc++.h>
#include <concrete-components/data_units/staggered_grid.h>
#include <seismic-io-framework/datatypes.h>
#include <skeleton/helpers/timer/timer.hpp>


SeismicModelHandler::SeismicModelHandler(bool is_staggered) {
  sio = new SeIO();
  IO = new SEGYIOManager();
  this->is_staggered = is_staggered;
}

void SeismicModelHandler ::GetSuitableDt(int ny, float dx, float dz, float dy,
                                         float *dt, float *coeff, int max,
                                         int half_length, float dt_relax) {

  // Calculate dt through finite difference stability equation
  float dxSquare = 1 / (dx * dx);
  float dySquare;

  if (ny != 1)
    dySquare = 1 / (dy * dy);
  else // case of 2D
    dySquare = 0.0;

  float dzSquare = 1 / (dz * dz);

  float distanceM = 1 / (sqrtf(dxSquare + dySquare + dzSquare));

  float a1 = 4; // the sum of absolute value of weights for second derivative if
                // du per dt. we use second order so the weights are (-1,2,1)
  float a2 = 0;

  for (int i = 1; i <= half_length; i++) {
    a2 += fabs(coeff[i]);
  }
  a2 *= 2.0;
  a2 += fabs(coeff[0]); // the sum of absolute values for second derivative id
                        // du per dx ( one dimension only )

  *dt = ((sqrtf(a1 / a2)) * distanceM) / max * dt_relax;
}

void SeismicModelHandler ::PreprocessModel(
    ComputationKernel *computational_kernel) {

  int nx = grid_box->window_size.window_nx;
  int nz = grid_box->window_size.window_nz;
  int ny = grid_box->window_size.window_ny;

  unsigned int model_size = nx * nz * ny;
  // allocating and zeroing prev, curr, and next pressure
  float *curr = (float *)mem_allocate(sizeof(float), model_size, "curr",
                                      parameters->half_length, 32);
  grid_box->pressure_current = curr;
  computational_kernel->FirstTouch(curr, nx, nz, ny);
  memset(curr, 0, sizeof(float) * model_size);

  if (is_staggered) {
    StaggeredGrid *grid_box = (StaggeredGrid *)this->grid_box;
    float *particle_vel_x =
        (float *)mem_allocate(sizeof(float), model_size, "particle_vel_x",
                              parameters->half_length, 16);
    grid_box->particle_velocity_x_current = particle_vel_x;
    computational_kernel->FirstTouch(particle_vel_x, nx, nz, ny);
    memset(particle_vel_x, 0, sizeof(float) * model_size);

    float *particle_vel_z =
        (float *)mem_allocate(sizeof(float), model_size, "particle_vel_z",
                              parameters->half_length, 48);
    grid_box->particle_velocity_z_current = particle_vel_z;
    computational_kernel->FirstTouch(particle_vel_x, nx, nz, ny);
    memset(particle_vel_x, 0, sizeof(float) * model_size);

    if (ny > 1) {
      float *particle_vel_y =
          (float *)mem_allocate(sizeof(float), model_size, "particle_vel_y",
                                parameters->half_length, 64);
      grid_box->particle_velocity_y_current = particle_vel_y;
      computational_kernel->FirstTouch(particle_vel_y, nx, nz, ny);
      memset(particle_vel_y, 0, sizeof(float) * model_size);
    }

    float *next = grid_box->pressure_current;
    grid_box->pressure_next = next;
    computational_kernel->FirstTouch(next, nx, nz, ny);
    memset(next, 0, sizeof(float) * model_size);
  } else {
    AcousticSecondGrid *grid_box = (AcousticSecondGrid *)this->grid_box;
    float *prev = (float *)mem_allocate(sizeof(float), model_size, "prev",
                                        parameters->half_length, 16);
    grid_box->pressure_previous = prev;
    computational_kernel->FirstTouch(prev, nx, nz, ny);
    memset(prev, 0, sizeof(float) * model_size);

    float *next = prev;
    grid_box->pressure_next = next;
    computational_kernel->FirstTouch(next, nx, nz, ny);
    memset(next, 0, sizeof(float) * model_size);
  }
  float dt = grid_box->dt;
  float dt2 = grid_box->dt * grid_box->dt;
  float *velocity_values = grid_box->velocity;
  int full_nx = grid_box->grid_size.nx;
  int full_nx_nz = grid_box->grid_size.nx * grid_box->grid_size.nz;
  if (is_staggered) {
    StaggeredGrid *grid_box = (StaggeredGrid *)this->grid_box;
    float *density_values = grid_box->density;
    // Preprocess the velocity model by calculating the dt * c2 * density
    // component of the wave equation.
#pragma omp parallel default(shared)
    {
#pragma omp for schedule(static) collapse(3)
      for (int y = 0; y < grid_box->grid_size.ny; ++y) {
        for (int z = 0; z < grid_box->grid_size.nz; ++z) {
          for (int x = 0; x < grid_box->grid_size.nx; ++x) {
            float value = velocity_values[y * full_nx_nz + z * full_nx + x];
            int offset = y * full_nx_nz + z * full_nx + x;
            velocity_values[offset] =
                value * value * dt * density_values[offset];
            if (density_values[offset] != 0) {
              density_values[offset] = dt / density_values[offset];
            }
          }
        }
      }
    }
  } else {
    // Preprocess the velocity model by calculating the dt2 * c2 component of
    // the wave equation.
#pragma omp parallel default(shared)
    {
#pragma omp for schedule(static) collapse(3)
      for (int y = 0; y < grid_box->grid_size.ny; ++y) {
        for (int z = 0; z < grid_box->grid_size.nz; ++z) {
          for (int x = 0; x < grid_box->grid_size.nx; ++x) {
            float value = velocity_values[y * full_nx_nz + z * full_nx + x];
            velocity_values[y * full_nx_nz + z * full_nx + x] =
                value * value * dt2;
          }
        }
      }
    }
  }
}

void SeismicModelHandler::SetComputationParameters(
    ComputationParameters *parameters) {
  this->parameters = parameters;
}

void SeismicModelHandler::SetGridBox(GridBox *grid_box) {
  this->grid_box = grid_box;
}

GridBox *
SeismicModelHandler::ReadModel(vector<string> filenames,
                               ComputationKernel *computational_kernel) {

  string file_name = filenames[0];
  Timer *timer = Timer::getInstance();
  timer->start_timer("IO::ReadVelocityFromSegyFile");
  IO->ReadVelocityDataFromFile(file_name, "CSR", this->sio);
  timer->stop_timer("IO::ReadVelocityFromSegyFile");
  GridBox *grid;
  if (is_staggered) {
    grid = (GridBox *)mem_allocate(sizeof(StaggeredGrid), 1, "StaggeredGrid");
    string d_file_name = filenames[1];
    timer->start_timer("IO::ReadDensityFromSegyFile");
    IO->ReadDensityDataFromFile(d_file_name, "CSR", this->sio); // case density
    timer->stop_timer("IO::ReadDensityFromSegyFile");
  } else {
    grid = (GridBox *)mem_allocate(sizeof(AcousticSecondGrid), 1, "GridBox");
  }
  int nx, ny, nz;

  nx = grid->grid_size.nx = sio->DM.nx + 2 * parameters->boundary_length +
                            2 * parameters->half_length;
  nz = grid->grid_size.nz = sio->DM.nz + 2 * parameters->boundary_length +
                            2 * parameters->half_length;

  if (sio->DM.ny > 1) {
    ny = grid->grid_size.ny = sio->DM.ny + 2 * parameters->boundary_length +
                              2 * parameters->half_length;
  } else {
    ny = grid->grid_size.ny = 1;
  }

  float dx, dy, dz, dt;

  dx = grid->cell_dimensions.dx = sio->DM.dx;
  dz = grid->cell_dimensions.dz = sio->DM.dz;
  dy = grid->cell_dimensions.dy = sio->DM.dy;

  int last = sio->Velocity.size() - 1;

  grid->reference_point.x =
      sio->Velocity.at(0).TraceMetaData.source_location_x <
              sio->Velocity.at(last).TraceMetaData.source_location_x
          ? sio->Velocity.at(0).TraceMetaData.source_location_x
          : sio->Velocity.at(last).TraceMetaData.source_location_x;

  grid->reference_point.z =
      sio->Velocity.at(0).TraceMetaData.source_location_z <
              sio->Velocity.at(last).TraceMetaData.source_location_z
          ? sio->Velocity.at(0).TraceMetaData.source_location_z
          : sio->Velocity.at(last).TraceMetaData.source_location_z;

  grid->reference_point.y =
      sio->Velocity.at(0).TraceMetaData.source_location_y <
              sio->Velocity.at(last).TraceMetaData.source_location_y
          ? sio->Velocity.at(0).TraceMetaData.source_location_y
          : sio->Velocity.at(last).TraceMetaData.source_location_y;

  cout << "refrence x " << grid->reference_point.x << "refrence z "
       << grid->reference_point.z << "refrence y " << grid->reference_point.y
       << endl;
  unsigned int model_size = nx * nz * ny;

    // if there is no window then window size equals full model size
    if (parameters->use_window) {
        grid->window_size.window_start.x = 0;
        grid->window_size.window_start.z = 0;
        grid->window_size.window_start.y = 0;
        if (parameters->left_window == 0 && parameters->right_window == 0) {
            grid->window_size.window_nx = nx;
        } else {
            grid->window_size.window_nx = std::min(
                    parameters->left_window + parameters->right_window + 1 + 2 * parameters->boundary_length + 2 * parameters->half_length,
                    nx);
        }
        if (parameters->depth_window == 0) {
            grid->window_size.window_nz = nz;
        } else {
            grid->window_size.window_nz = std::min(parameters->depth_window + 2 * parameters->boundary_length + 2 * parameters->half_length, nz);
        }
        if ((parameters->front_window == 0 && parameters->back_window == 0) || ny == 1) {
            grid->window_size.window_ny = ny;
        } else {
            grid->window_size.window_ny = std::min(parameters->front_window + parameters->back_window + 1 + 2 * parameters->boundary_length + 2 * parameters->half_length, ny);
        }
    } else {
        grid->window_size.window_start.x = 0;
        grid->window_size.window_start.z = 0;
        grid->window_size.window_start.y = 0;
        grid->window_size.window_nx = nx;
        grid->window_size.window_nz = nz;
        grid->window_size.window_ny = ny;
    }

  float *velocity = (float *)mem_allocate(sizeof(float), model_size, "velocity",
                                          parameters->half_length, 0);
  computational_kernel->FirstTouch(velocity, grid->grid_size.nx,
                                   grid->grid_size.nz, grid->grid_size.ny);
  memset(velocity, 0, sizeof(float) * model_size);

  int offset = parameters->boundary_length + parameters->half_length;
  int offset_y =
      ny > 1 ? parameters->boundary_length + parameters->half_length : 0;
  float max = 0;
  for (unsigned int k = offset_y; k < ny - offset_y; k++) {
    for (unsigned int i = offset; i < nx - offset; i++) {
      for (unsigned int j = offset; j < nz - offset; j++) {
        float temp_velocity = velocity[k * nx * nz + j * nx + i] =
            sio->Velocity.at(i - offset).TraceData[j - offset];
        if (temp_velocity > max) {
          max = temp_velocity;
        }
      }
    }
  }

  if (is_staggered) {
    StaggeredGrid *s_grid = (StaggeredGrid *)grid;
    float *density = (float *)mem_allocate(sizeof(float), model_size, "density",
                                           parameters->half_length, 0);
    computational_kernel->FirstTouch(density, grid->grid_size.nx,
                                     grid->grid_size.nz, grid->grid_size.ny);
    memset(density, 0, sizeof(float) * model_size);

    for (unsigned int k = offset_y; k < ny - offset_y; k++) {
      for (unsigned int i = offset; i < nx - offset; i++) {
        for (unsigned int j = offset; j < nz - offset; j++) {
          density[k * nx * nz + j * nx + i] =
              sio->Density.at(i - offset).TraceData[j - offset];
        }
      }
    }
    s_grid->density = density;
    if (parameters->use_window) {
        s_grid->window_density = (float *)mem_allocate(sizeof(float),
                grid->window_size.window_nx * grid->window_size.window_nz * grid->window_size.window_ny,
                "window_density", parameters->half_length, 0);
        computational_kernel->FirstTouch(s_grid->window_density, grid->window_size.window_nx,
                grid->window_size.window_nz, grid->window_size.window_ny);
        memset(s_grid->window_density, 0,
                sizeof(float) * grid->window_size.window_nx * grid->window_size.window_nz * grid->window_size.window_ny);
    } else {
        s_grid->window_density = s_grid->density;
    }
  }

  grid->velocity = velocity;
  if (parameters->use_window) {
      grid->window_velocity = (float *)mem_allocate(sizeof(float),
              grid->window_size.window_nx * grid->window_size.window_nz * grid->window_size.window_ny,
              "window_velocity", parameters->half_length, 0);
      computational_kernel->FirstTouch(grid->window_velocity, grid->window_size.window_nx,
              grid->window_size.window_nz, grid->window_size.window_ny);
      memset(grid->window_velocity, 0,
              sizeof(float) * grid->window_size.window_nx * grid->window_size.window_nz * grid->window_size.window_ny);
  } else {
      grid->window_velocity = grid->velocity;
  }

  GetSuitableDt(ny, dx, dz, dy, &grid->dt,
                parameters->second_derivative_fd_coeff, max,
                parameters->half_length, parameters->dt_relax);
  free(sio);
  free(IO);
  return grid;
}

SeismicModelHandler ::~SeismicModelHandler() = default;

void SeismicModelHandler::SetupWindow() {
    if (parameters->use_window) {
        uint wnx = grid_box->window_size.window_nx;
        uint wnz = grid_box->window_size.window_nz;
        uint wny = grid_box->window_size.window_ny;
        uint nx = grid_box->grid_size.nx;
        uint nz = grid_box->grid_size.nz;
        uint ny = grid_box->grid_size.ny;
        uint sx = grid_box->window_size.window_start.x;
        uint sz = grid_box->window_size.window_start.z;
        uint sy = grid_box->window_size.window_start.y;
        uint offset = parameters->half_length + parameters->boundary_length;
        uint start_x = offset;
        uint end_x = wnx - offset;
        uint start_z = offset;
        uint end_z = wnz - offset;
        uint start_y = 0;
        uint end_y = 1;
        if (ny != 1) {
            start_y = offset;
            end_y = wny - offset;
        }
        for (uint iy = start_y; iy < end_y; iy++) {
            for (uint iz = start_z; iz < end_z; iz++) {
                for (uint ix = start_x; ix < end_x; ix++) {
                    uint offset_window = iy * wnx * wnz + iz * wnx + ix;
                    uint offset_full = (iy + sy) * nx * nz + (iz + sz) * nx + ix + sx;
                    grid_box->window_velocity[offset_window] = grid_box->velocity[offset_full];
                }
            }
        }
        if (is_staggered) {
            StaggeredGrid *s_grid = (StaggeredGrid *)grid_box;
            for (uint iy = start_y; iy < end_y; iy++) {
                for (uint iz = start_z; iz < end_z; iz++) {
                    for (uint ix = start_x; ix < end_x; ix++) {
                        uint offset_window = iy * wnx * wnz + iz * wnx + ix;
                        uint offset_full = (iy + sy) * nx * nz + (iz + sz) * nx + ix + sx;
                        s_grid->window_density[offset_window] = s_grid->density[offset_full];
                    }
                }
            }
        }
    }
}
