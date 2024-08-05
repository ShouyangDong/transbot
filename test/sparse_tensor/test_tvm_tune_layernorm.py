import logging
import tempfile

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm.meta_schedule.schedule_rule import ApplyCustomRule
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir.schedule import BlockRV, Schedule

logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)
extern "C" void layernorm_kernel(float* input,   // shape=[1, 4, 128]
                                  float* gamma,   // shape=[128]
                                  float* beta,    // shape=[128]
                                  float* output)  // shape=[1, 4, 128]
{
  for (int i_seq = 0; i_seq < 4; i_seq++) {
    float mean = 0.0;
    float variance = 0.0;
    float diff[128];
    // Calculate mean
    for (int i_mean = 0; i_mean < 128; i_mean++) {
      mean += input[i_seq * 128 + i_mean];
    }
    mean /= 128;
    // Calculate variance
    for (int i_diff = 0; i_diff < 128; i_diff++) {
      diff[i_diff] = input[i_seq * 128 + i_diff] - mean;
    }

    for (int i_pow = 0; i_pow < 128; i_pow++) {
      diff[i_pow] = diff[i_pow] * diff[i_pow];
    }
    for (int i_var = 0; i_var < 128; i_var++) {
      variance += diff[i_var];
    }
    variance = sqrt(variance / 128);

    // Normalize input
    for (int i_norm = 0; i_norm < 128; i_norm++) {
      diff[i_norm] = (input[i_seq * 128 + i_norm] - mean);
    }

    for (int i_mul = 0; i_mul < 128; i_mul++) {
      diff[i_mul] = diff[i_mul] * gamma[i_mul];
    }

    for (int i_div = 0; i_div < 128; i_div++) {
      diff[i_div] = diff[i_div] / (variance + 1e-5f);
    }

    for (int i_bet = 0; i_bet < 128; i_bet++) {
      output[i_seq * 128 + i_bet] = diff[i_bet] + beta[i_bet];
    }
  }
}
