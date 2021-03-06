//
// Created by amr on 25/01/2020.
//

#include "correlation_kernel_parser.h"

CorrelationKernel *
parse_correlation_kernel_acoustic_iso_openmp_second(ConfigMap map) {
  CorrelationKernel *correlation_kernel = nullptr;
  if (map.find("correlation-kernel") == map.end()) {
    cout << "No entry for correlation-kernel key : supported values [ "
            "cross-correlation ]"
         << std::endl;
    cout << "Terminating..." << std::endl;
    exit(0);
  } else if (map["correlation-kernel"] == "cross-correlation") {
    correlation_kernel = new CrossCorrelationKernel();
    cout << "Using Cross-Correlation as correlation kernel..." << std::endl;
  } else {
    cout << "Invalid value for correlation-kernel key : supported values [ "
            "cross-correlation ]"
         << std::endl;
    cout << "Terminating..." << std::endl;
    exit(0);
  }

  if (map.find("correlation-kernel.compensation") == map.end()) {
      cout << "No entry for correlation-kernel.compensation key : supported values [ "
              "no | source | receiver | combined ]"
           << std::endl;
      cout << "Terminating..." << std::endl;
      exit(0);
    } else if (map["correlation-kernel.compensation"] == "no") {
    	correlation_kernel->SetCompensation(NO_COMPENSATION);
    	cout << "No illumination compensation is requested" << std::endl;
    } else if (map["correlation-kernel.compensation"] == "source") {
    	correlation_kernel->SetCompensation(SOURCE_COMPENSATION);
    	cout << "Applying source illumination compensation" << std::endl;
    } else if (map["correlation-kernel.compensation"] == "receiver") {
    	correlation_kernel->SetCompensation(RECEIVER_COMPENSATION);
    	cout << "Applying receiver illumination compensation" << std::endl;
    } else if (map["correlation-kernel.compensation"] == "combined") {
    	correlation_kernel->SetCompensation(COMBINED_COMPENSATION);
    	cout << "Applying combined illumination compensation" << std::endl;
    } else {
      cout << "Invalid value for correlation-kernel.compensation key : supported values [ "
              "no | source | receiver | combined  ]"
           << std::endl;
      cout << "Terminating..." << std::endl;
      exit(0);
    }

  return correlation_kernel;
}
