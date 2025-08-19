// include/utils/perf.h
#ifndef PERF_H
#define PERF_H

#include <stddef.h>

void perf_enable(int enabled);
int perf_enabled(void);

void perf_mark_start(const char* name);
void perf_mark_end(const char* name);

void perf_add_metric(const char* name, double value);

size_t perf_mem_rss_bytes(void);

void perf_report(void);
void perf_reset(void);

#endif