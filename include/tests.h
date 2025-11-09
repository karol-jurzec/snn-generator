#include "models/model_base.h"
#include "network.h"

#ifndef TESTS_H
#define TESTS_H


void stmnist_test();
void nmnist_test();
void test_nmnist_pruning();
void test_stmnist_pruning();
void study_stmnist_threshold_impact();
void test_stmnist_bidirectional_pruning();
void test_nmnist_bidirectional_pruning();

#endif // TESTS_H
