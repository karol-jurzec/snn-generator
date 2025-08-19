// src/utils/perf.c
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#endif

#define PERF_MAX_ENTRIES 64

typedef struct {
	char name[64];
	int is_metric;              /* 0=stage, 1=metric */
	/* stage fields */
	double total_sec;
	int count;
	double start_sec;
	int active;
	size_t last_mem_before;
	size_t last_mem_after;
	size_t max_mem_after;
	/* metric fields */
	double sum_value;
	double min_value;
	double max_value;
} PerfEntry;

static PerfEntry g_entries[PERF_MAX_ENTRIES];
static int g_entries_count = 0;
static int g_enabled = 0;

static double perf_now(void) {
#ifdef _WIN32
	LARGE_INTEGER freq, counter;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&counter);
	return (double)counter.QuadPart / (double)freq.QuadPart;
#else
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec * 1e-6;
#endif
}

void perf_enable(int enabled) { g_enabled = enabled ? 1 : 0; }
int perf_enabled(void) { return g_enabled; }

static PerfEntry* perf_find_or_create(const char* name, int is_metric) {
	for (int i = 0; i < g_entries_count; ++i) {
		if (strcmp(g_entries[i].name, name) == 0 && g_entries[i].is_metric == is_metric) {
			return &g_entries[i];
		}
	}
	if (g_entries_count >= PERF_MAX_ENTRIES) return NULL;
	PerfEntry* e = &g_entries[g_entries_count++];
	memset(e, 0, sizeof(*e));
	strncpy(e->name, name, sizeof(e->name) - 1);
	e->is_metric = is_metric;
	if (is_metric) {
		e->min_value = 1e300;
		e->max_value = -1e300;
	}
	return e;
}

size_t perf_mem_rss_bytes(void) {
#ifdef _WIN32
	PROCESS_MEMORY_COUNTERS pmc;
	if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
		return (size_t)pmc.WorkingSetSize;
	}
	return 0;
#else
	return 0;
#endif
}

void perf_mark_start(const char* name) {
	if (!g_enabled) return;
	PerfEntry* e = perf_find_or_create(name, 0);
	if (!e || e->active) return;
	e->active = 1;
	e->start_sec = perf_now();
	e->last_mem_before = perf_mem_rss_bytes();
}

void perf_mark_end(const char* name) {
	if (!g_enabled) return;
	PerfEntry* e = perf_find_or_create(name, 0);
	if (!e || !e->active) return;
	double dt = perf_now() - e->start_sec;
	e->total_sec += dt;
	e->count += 1;
	e->active = 0;
	e->last_mem_after = perf_mem_rss_bytes();
	if (e->last_mem_after > e->max_mem_after) e->max_mem_after = e->last_mem_after;
}

void perf_add_metric(const char* name, double value) {
	if (!g_enabled) return;
	PerfEntry* e = perf_find_or_create(name, 1);
	if (!e) return;
	e->sum_value += value;
	if (value < e->min_value) e->min_value = value;
	if (value > e->max_value) e->max_value = value;
	e->count += 1;
}

static void print_bytes(size_t bytes) {
	const double kb = 1024.0;
	const double mb = kb * 1024.0;
	const double gb = mb * 1024.0;
	if (bytes >= (size_t)gb) {
		printf("%.2f GiB", (double)bytes / gb);
	} else if (bytes >= (size_t)mb) {
		printf("%.2f MiB", (double)bytes / mb);
	} else if (bytes >= (size_t)kb) {
		printf("%.2f KiB", (double)bytes / kb);
	} else {
		printf("%zu B", bytes);
	}
}

void perf_report(void) {
	if (!g_enabled) return;

	printf("=== Performance Report ===\n");
	for (int i = 0; i < g_entries_count; ++i) {
		PerfEntry* e = &g_entries[i];
		if (!e->is_metric) {
			double avg = e->count ? (e->total_sec / e->count) : 0.0;
			printf("- Stage: %s | runs: %d | total: %.6f s | avg: %.6f s | mem(before->after, peak): ",
				e->name, e->count, e->total_sec, avg);
			print_bytes(e->last_mem_before);
			printf(" -> ");
			print_bytes(e->last_mem_after);
			printf(", peak ");
			print_bytes(e->max_mem_after);
			printf("\n");
		}
	}
	for (int i = 0; i < g_entries_count; ++i) {
		PerfEntry* e = &g_entries[i];
		if (e->is_metric) {
			double avg = e->count ? (e->sum_value / e->count) : 0.0;
			printf("- Metric: %s | n: %d | avg: %.6f | min: %.6f | max: %.6f\n",
				e->name, e->count, avg, e->min_value, e->max_value);
		}
	}
	printf("==========================\n");
}

void perf_reset(void) {
	memset(g_entries, 0, sizeof(g_entries));
	g_entries_count = 0;
}