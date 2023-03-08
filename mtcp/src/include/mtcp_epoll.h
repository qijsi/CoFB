#ifndef MTCP_EPOLL_H
#define MTCP_EPOLL_H

#include "mtcp_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PKT_LEN 1400

/*----------------------------------------------------------------------------*/
enum mtcp_epoll_op
{
	MTCP_EPOLL_CTL_ADD = 1, 
	MTCP_EPOLL_CTL_DEL = 2, 
	MTCP_EPOLL_CTL_MOD = 3, 
};
/*----------------------------------------------------------------------------*/
enum mtcp_event_type
{
	MTCP_EPOLLNONE	= 0x000, 
	MTCP_EPOLLIN	= 0x001, 
	MTCP_EPOLLPRI	= 0x002,
	MTCP_EPOLLOUT	= 0x004,
	MTCP_EPOLLRDNORM	= 0x040, 
	MTCP_EPOLLRDBAND	= 0x080, 
	MTCP_EPOLLWRNORM	= 0x100, 
	MTCP_EPOLLWRBAND	= 0x200, 
	MTCP_EPOLLMSG		= 0x400, 
	MTCP_EPOLLERR		= 0x008,
	MTCP_EPOLLHUP		= 0x010,
	MTCP_EPOLLRDHUP 	= 0x2000,
	MTCP_EPOLLONESHOT	= (1 << 30), 
	MTCP_EPOLLET		= (1 << 31)
};
/*----------------------------------------------------------------------------*/
typedef union mtcp_epoll_data
{
	void *ptr;
	int sockid;
	uint32_t u32;
	uint64_t u64;
	struct timeval start, end;
} mtcp_epoll_data_t;
/*----------------------------------------------------------------------------*/
struct mtcp_epoll_event
{
	uint32_t events;
	mtcp_epoll_data_t data;
};
/*----------------------------------------------------------------------------*/
struct server_vars {
  char request[PKT_LEN];
  int recv_len;
  int request_len;
  long int total_read, total_sent;
  uint8_t done;
  uint8_t rsp_sent;
  uint8_t keep_alive;
};
/*----------------------------------------------------------------------------*/
struct thread_context {
  mctx_t mctx;
  int ep;
  struct server_vars *svars;
//  model_t model;
  int listener;
  struct mtcp_epoll_event *events;
};

/*----------------------------------------------------------------------------*/
int 
mtcp_epoll_create(mctx_t mctx, int size);
/*----------------------------------------------------------------------------*/
int
mtcp_epoll_create1(mctx_t mctx, int flags);
/*----------------------------------------------------------------------------*/
int 
mtcp_epoll_ctl(mctx_t mctx, int epid, 
		int op, int sockid, struct mtcp_epoll_event *event);
/*----------------------------------------------------------------------------*/
int 
mtcp_epoll_wait(mctx_t mctx, int epid, 
		struct mtcp_epoll_event *events, int maxevents, int timeout);
/*----------------------------------------------------------------------------*/
char * 
EventToString(uint32_t event);
/*----------------------------------------------------------------------------*/

#ifdef __cplusplus
};
#endif

#endif /* MTCP_EPOLL_H */
