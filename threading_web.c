#include "crux_threading.h"
#include "extension.h"
#include "kernel_types.h"
#include "logging.h"
#include "memory.h"
#include "threading_api.h"
#include <emscripten.h>
// The threading support probe (emscripten_has_threading_support) + the logical
// core count (emscripten_num_logical_cores) live in the threading sub-header.
#include <emscripten/threading.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

// The Emscripten threading PRIMITIVE: emscripten's stable pthreads support
// (SharedArrayBuffer + Atomics) is the backing — the same pthread API the posix
// driver speaks, with the web-specific facts:
//
// 1. THE PRE-RUN POOL. emscripten's pthread_create that needs a fresh Web
//    Worker cannot complete synchronously (creating the worker requires
//    returning to the event loop) — so a pthread must ALREADY exist when
//    Thread_Create runs: the -sPTHREAD_POOL_SIZE linker flag (baked by the
//    project's [platform.web] link_options, the recipe channel) pre-creates
//    them at page preRun, before main(). A pthread_create drawn on that pre-run
//    pool is synchronous. The pool's SIZE is the page's own SharedArrayBuffer
//    fact, expressed at page init (the expression in the recipe):
//    cross-origin isolated (COOP + COEP headers) -> hardwareConcurrency
//    workers; NOT isolated -> 0 (a pre-run spawn on a non-isolated page throws
//    DataCloneError transferring the WebAssembly.Memory and stalls module init
//    on the "loading-workers" dependency BEFORE main runs — the probe in
//    Thread_Create is too late for that path, so the size is
//    runtime-conditional in the recipe). No arbitrary cap is invented: the
//    measured per-page Web Worker limit (backlog 0037, this Firefox build,
//    2026-10-31) is 512, far above any realistic hardwareConcurrency, so the
//    pre-run pool = hardwareConcurrency is safe.
//
// 2. THE NAMED DEGRADE. Without SharedArrayBuffer (the page is not
//    cross-origin isolated), no real thread can exist — pthread_create would
//    fail EAGAIN. So this primitive DEGRADES: Thread_Create runs the function
//    INLINE on the calling thread and returns an already-finished handle (a
//    later Join is a no-op), and Thread_Concurrency reads 0 — the fact a
//    worker pool (the thread_pool standalone) keys its serial path off. One
//    -pthread build therefore serves both pages: threaded when isolated,
//    serial otherwise.
//
// The concurrency fact (Thread_Concurrency) is what the pool sizes off — NOT
// the core count. On an isolated page it is the pre-run pool size
// (hardwareConcurrency); on a non-isolated page it is 0. The hardware's core
// counts (logical/physical) are reported separately and are ALWAYS available
// (they are navigator facts, not thread facts).

// The opaque handle: the platform thread id (a Web Worker, wrapped in
// emscripten's pthread_t), wrapped so the caller tracks an opaque pointer it
// passes back (never dereferenced).
struct CruxThread {
  pthread_t handle;
};

// The unit of work the thread consumes (fn + arg), owned by the thread after
// Create — kept separate from the handle so the caller can free the handle
// without a (possibly still-running) thread's work item going stale.
typedef struct TwbWork {
  void (*fn)(void *arg);
  void *arg;
} TwbWork;

// The shared sentinel: an already-finished (inline) unit of work. The no-SAB
// degrade returns it; a Join/Detach on it is a no-op (there is no real thread
// to wait on or reclaim, and the sentinel is static — never freed).
static struct CruxThread s_web_finished = {0};

// The thread's entry: run the work, then reclaim the work item (the handle is
// the caller's; the thread never frees it).
static void *twb_thread_main(void *arg) {
  TwbWork *work = (TwbWork *)arg;
  work->fn(work->arg);
  Memory_Free(work);
  return NULL;
}

CruxThread *Thread_Create(void (*func)(void *arg), void *arg) {
  if (func == NULL) {
    return NULL;
  }
  // The named degradation: no SharedArrayBuffer on the page -> no real thread
  // can exist. Run the unit of work inline and return the already-finished
  // sentinel (a Join/Detach on it is a no-op). The fact that no real threads
  // exist is Thread_Concurrency returning 0 (not a NULL here).
  if (!emscripten_has_threading_support()) {
    func(arg);
    return &s_web_finished;
  }
  CruxThread *thread = (CruxThread *)Memory_Calloc(1, sizeof *thread);
  TwbWork *work = (TwbWork *)Memory_Calloc(1, sizeof *work);
  if (thread == NULL || work == NULL) {
    if (thread != NULL) {
      Memory_Free(thread);
    }
    if (work != NULL) {
      Memory_Free(work);
    }
    // An allocation failure is a NAMED refusal (NULL, per the contract) — do
    // NOT run inline: the page IS isolated, so the caller may be a pool
    // creating a worker whose loop is infinite, and inline would block the
    // creating thread forever (the same reason the pthread_create refusal
    // below returns NULL).
    LogError("[threading_web] Create: allocation failed (handle or work item)");
    return NULL;
  }
  work->fn = func;
  work->arg = arg;
  if (pthread_create(&thread->handle, NULL, twb_thread_main, work) != 0) {
    // A refused pthread (no pre-run worker left, or the SAB went away). Do
    // NOT degrade to inline here: the caller may be a pool creating a worker
    // whose loop is infinite, and running it inline would block the creating
    // thread forever. Return NULL — the pool refuses to boot named (a named
    // log); a direct primitive caller handles the NULL.
    Memory_Free(thread);
    Memory_Free(work);
    LogError("[threading_web] Create: pthread_create refused (no pre-run worker available)");
    return NULL;
  }
  return thread;
}

void Thread_Join(CruxThread *thread) {
  if (thread == NULL || thread == &s_web_finished) {
    return; // no-op: the sentinel is shared (an inline-degraded unit of work)
  }
  pthread_join(thread->handle, NULL); // the thread reclaimed its work item on exit
  Memory_Free(thread);                // the caller owns the handle (the thread never frees it)
}

void Thread_Detach(CruxThread *thread) {
  if (thread == NULL || thread == &s_web_finished) {
    return; // no-op: the sentinel is shared
  }
  pthread_detach(thread->handle); // the thread's work item reclaims itself on exit
  Memory_Free(thread);            // the caller owns the handle; the thread no longer needs the id
}

// The hardware's LOGICAL core count: navigator.hardwareConcurrency — a
// navigator fact, ALWAYS available (even without SharedArrayBuffer).
int Thread_Count_Logical(void) {
  int count = emscripten_num_logical_cores();
  return (count > 0) ? count : 1;
}

// The hardware's PHYSICAL core count. On web, physical is NOT a hard fact
// (no API) — this is an HONEST best-effort estimate: logical / 2, the
// "probably physical" SMT heuristic (2:1 on the common SMT desktop/laptop
// silicon). It is documented as an estimate, not a fact; a later refinement
// (an async Client Hints platform probe) would use all N on Apple Silicon
// (no SMT, 1:1) and keep the /2 elsewhere.
int Thread_Count_Physical(void) {
  int logical = Thread_Count_Logical();
  int physical = logical / 2;
  return (physical > 0) ? physical : 1;
}

// The HARD CEILING on concurrent real threads, right now — the fact the pool
// keys its size + its serial degradation off. Isolated (SharedArrayBuffer
// present): the pre-run Web Worker pool size — the recipe's PTHREAD_POOL_SIZE
// value, hardwareConcurrency with the SAME ||4 floor the recipe bakes. NOT
// isolated: 0 — no real threads are possible, so a pool keyed on this runs
// serial.
int Thread_Concurrency(void) {
  if (!emscripten_has_threading_support()) {
    return 0;
  }
  int cores = emscripten_num_logical_cores();
  return (cores > 0) ? cores : 4; // the recipe's ||4 floor, so the ceiling == the pre-run pool
}

// The served struct — valid from load, before Construct. The package's
// identity is the manifest's (the implementation targets carry no baked
// identity defines — the org's convention, the time_posix shape).
static ThreadingAPI g_threading_api = {
    .version = {1, 0, 0},
    .Thread_Create = Thread_Create,
    .Thread_Join = Thread_Join,
    .Thread_Detach = Thread_Detach,
    .Thread_Count_Logical = Thread_Count_Logical,
    .Thread_Count_Physical = Thread_Count_Physical,
    .Thread_Concurrency = Thread_Concurrency,
};

static bool threading_web_construct(void) {
  Engine_RegisterConcern(CRUX_CONCERN_THREADING);
  return true;
}

static void *threading_web_get_api(void) {
  return &g_threading_api;
}

ExtensionInterface g_extension_threading_web = {
    "threading_web",
    {1, 0, 0},
    threading_web_construct,
    NULL,
    NULL,
    NULL,
    NULL,
    threading_web_get_api,
};
