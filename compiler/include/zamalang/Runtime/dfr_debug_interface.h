#ifndef ZAMALANG_DRF_DEBUG_INTERFACE_H
#define ZAMALANG_DRF_DEBUG_INTERFACE_H

#include <stdint.h>
#include <unistd.h>

extern "C" {
size_t _dfr_debug_get_node_id();
size_t _dfr_debug_get_worker_id();
void _dfr_debug_print_task(const char *name, int inputs, int outputs);
void _dfr_print_debug(size_t val);
}
#endif
