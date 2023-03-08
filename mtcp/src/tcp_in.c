#include "tcp_in.h"

#include <assert.h>
#include <inttypes.h>
#include <time.h>

#include "clock.h"
#include "debug.h"
#include "eventpoll.h"
#include "ip_in.h"
#include "tcp_out.h"
#include "tcp_ring_buffer.h"
#include "tcp_util.h"
#include "timer.h"
#if USE_CCP
#include "ccp.h"
#endif

#include <pthread.h>
#include <rte_eth_ctrl.h>
#include <rte_ethdev.h>

#include "tcp_stream.h"

//#define MAX(a, b) ((a) > (b) ? (a) : (b))
//#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define VERIFY_RX_CHECKSUM TRUE
#define RECOVERY_AFTER_LOSS TRUE
#define SELECTIVE_WRITE_EVENT_NOTIFY TRUE

extern pthread_spinlock_t fdir_lock;
extern uint64_t fdir_group[MAX_NIC][GROUP];  // 8000 / 64 = 125 < 128;
extern uint64_t fdir_elem[MAX_NIC][ELEM];    // Each bit for fdir_id in a group

// static uint16_t soft_id = 1;

static uint64_t num_conn;

/*----------------------------------------------------------------------------*/
static inline int FilterSYNPacket(mtcp_manager_t mtcp, uint32_t ip,
                                  uint16_t port) {
  struct sockaddr_in *addr;
  struct tcp_listener *listener;

  /* TODO: This listening logic should be revised */

  /* if not the address we want, drop */
  listener = (struct tcp_listener *)ListenerHTSearch(mtcp->listeners, &port);
  if (listener == NULL) return FALSE;

  addr = &listener->socket->saddr;

  if (addr->sin_port == port) {
    if (addr->sin_addr.s_addr != INADDR_ANY) {
      if (ip == addr->sin_addr.s_addr) {
        return TRUE;
      }
      return FALSE;
    } else {
      int i;

      for (i = 0; i < CONFIG.eths_num; i++) {
        if (ip == CONFIG.eths[i].ip_addr) {
          return TRUE;
        }
      }
      return FALSE;
    }
  }

  return FALSE;
}
/*----------------------------------------------------------------------------*/
static inline tcp_stream *HandlePassiveOpen(mtcp_manager_t mtcp,
                                            uint32_t cur_ts,
                                            const struct iphdr *iph,
                                            const struct tcphdr *tcph,
                                            uint32_t seq, uint16_t window) {
  tcp_stream *cur_stream = NULL;

  /* create new stream and add to flow hash table */
  cur_stream = CreateTCPStream(mtcp, NULL, MTCP_SOCK_STREAM, iph->daddr,
                               tcph->dest, iph->saddr, tcph->source);
  if (!cur_stream) {
    TRACE_ERROR("INFO: Could not allocate tcp_stream!\n");
    fprintf(stdout, "Could not allocate tcp_stream\n");
    fflush(stdout);
    return FALSE;
  }
  cur_stream->rcvvar->irs = seq;
  cur_stream->sndvar->peer_wnd = window;
  cur_stream->rcv_nxt = cur_stream->rcvvar->irs;
  cur_stream->sndvar->cwnd = 1;
  ParseTCPOptions(cur_stream, cur_ts, (uint8_t *)tcph + TCP_HEADER_LEN,
                  (tcph->doff << 2) - TCP_HEADER_LEN);

  return cur_stream;
}
/*----------------------------------------------------------------------------*/
static inline int HandleActiveOpen(mtcp_manager_t mtcp, tcp_stream *cur_stream,
                                   uint32_t cur_ts, struct tcphdr *tcph,
                                   uint32_t seq, uint32_t ack_seq,
                                   uint16_t window) {
  cur_stream->rcvvar->irs = seq;
  cur_stream->snd_nxt = ack_seq;
  cur_stream->sndvar->peer_wnd = window;
  cur_stream->rcvvar->snd_wl1 = cur_stream->rcvvar->irs - 1;
  cur_stream->rcv_nxt = cur_stream->rcvvar->irs + 1;
  cur_stream->rcvvar->last_ack_seq = ack_seq;
  ParseTCPOptions(cur_stream, cur_ts, (uint8_t *)tcph + TCP_HEADER_LEN,
                  (tcph->doff << 2) - TCP_HEADER_LEN);
  cur_stream->sndvar->cwnd = ((cur_stream->sndvar->cwnd == 1)
                                  ? (cur_stream->sndvar->mss * TCP_INIT_CWND)
                                  : cur_stream->sndvar->mss);
  cur_stream->sndvar->ssthresh = cur_stream->sndvar->mss * 10;
  UpdateRetransmissionTimer(mtcp, cur_stream, cur_ts);

  return TRUE;
}
/*----------------------------------------------------------------------------*/
/* ValidateSequence: validates sequence number of the segment                 */
/* Return: TRUE if acceptable, FALSE if not acceptable                        */
/*----------------------------------------------------------------------------*/
static inline int ValidateSequence(mtcp_manager_t mtcp, tcp_stream *cur_stream,
                                   uint32_t cur_ts, struct tcphdr *tcph,
                                   uint32_t seq, uint32_t ack_seq,
                                   int payloadlen) {
  /* Protect Against Wrapped Sequence number (PAWS) */
  if (!tcph->rst && cur_stream->saw_timestamp) {
    struct tcp_timestamp ts;

    if (!ParseTCPTimestamp(cur_stream, &ts, (uint8_t *)tcph + TCP_HEADER_LEN,
                           (tcph->doff << 2) - TCP_HEADER_LEN)) {
      /* if there is no timestamp */
      /* TODO: implement here */
      TRACE_DBG("No timestamp found.\n");
      return FALSE;
    }

    /* RFC1323: if SEG.TSval < TS.Recent, drop and send ack */
    if (TCP_SEQ_LT(ts.ts_val, cur_stream->rcvvar->ts_recent)) {
      /* TODO: ts_recent should be invalidated
                       before timestamp wraparound for long idle flow */
      TRACE_DBG(
          "PAWS Detect wrong timestamp. "
          "seq: %u, ts_val: %u, prev: %u\n",
          seq, ts.ts_val, cur_stream->rcvvar->ts_recent);
      EnqueueACK(mtcp, cur_stream, cur_ts, ACK_OPT_NOW);
      return FALSE;
    } else {
      /* valid timestamp */
      if (TCP_SEQ_GT(ts.ts_val, cur_stream->rcvvar->ts_recent)) {
        TRACE_TSTAMP(
            "Timestamp update. cur: %u, prior: %u "
            "(time diff: %uus)\n",
            ts.ts_val, cur_stream->rcvvar->ts_recent,
            TS_TO_USEC(cur_ts - cur_stream->rcvvar->ts_last_ts_upd));
        cur_stream->rcvvar->ts_last_ts_upd = cur_ts;
      }

      cur_stream->rcvvar->ts_recent = ts.ts_val;
      cur_stream->rcvvar->ts_lastack_rcvd = ts.ts_ref;
    }
  }

  /* TCP sequence validation */
  if (!TCP_SEQ_BETWEEN(seq + payloadlen, cur_stream->rcv_nxt,
                       cur_stream->rcv_nxt + cur_stream->rcvvar->rcv_wnd)) {
    /* if RST bit is set, ignore the segment */
    if (tcph->rst) {
      fprintf(stdout, "%s %d lcore:%d sport:%u rst\n", __FILE__, __LINE__,
              mtcp->ctx->cpu, ntohs(cur_stream->dport));
      fflush(stdout);

      return FALSE;
    }

    if (cur_stream->state == TCP_ST_ESTABLISHED) {
      /* check if it is to get window advertisement */
      /*
              fprintf(stdout, "%s %d lcore:%d sport:%u ValidateSequence, seq:%u
         payloadlen:%d nxt:%u wnd:%u\n", \
                              __FILE__, __LINE__, mtcp->ctx->cpu,
         ntohs(cur_stream->dport), \ seq, payloadlen, cur_stream->rcv_nxt,
         cur_stream->rcvvar->rcv_wnd); fflush(stdout);
              */

      if (seq + 1 == cur_stream->rcv_nxt) {
#if 0
				TRACE_DBG("Window update request. (seq: %u, rcv_wnd: %u)\n", 
						seq, cur_stream->rcvvar->rcv_wnd);
				fprintf(stdout, "%s %d lcore:%d sport:%u ValidateSequence, seq:%u payloadlen:%d nxt:%u wnd:%u\n", \
						__FILE__, __LINE__, mtcp->ctx->cpu, ntohs(cur_stream->dport), seq, \
						payloadlen, cur_stream->rcv_nxt, cur_stream->rcvvar->rcv_wnd);
				fflush(stdout);

#endif
        EnqueueACK(mtcp, cur_stream, cur_ts, ACK_OPT_AGGREGATE);
        return FALSE;
      }

      if (TCP_SEQ_LEQ(seq, cur_stream->rcv_nxt)) {
        /*
        fprintf(stdout, "%s %d lcore:%d sport:%u ValidateSequence, seq:%u
        payloadlen:%d nxt:%u\n", \
                        __FILE__, __LINE__, mtcp->ctx->cpu,
        ntohs(cur_stream->dport), seq, payloadlen, cur_stream->rcv_nxt);
        fflush(stdout);
        */
        EnqueueACK(mtcp, cur_stream, cur_ts, ACK_OPT_AGGREGATE);
      } else {
        /*
        fprintf(stdout, "%s %d lcore:%d sport:%u ValidateSequence, seq:%u
        payloadlen:%d nxt:%u\n", \
                        __FILE__, __LINE__, mtcp->ctx->cpu,
        ntohs(cur_stream->dport), seq, payloadlen, cur_stream->rcv_nxt);

        fflush(stdout);
        */
        EnqueueACK(mtcp, cur_stream, cur_ts, ACK_OPT_NOW);
      }
    } else {
      if (cur_stream->state == TCP_ST_TIME_WAIT) {
        TRACE_DBG("Stream %d: tw expire update to %u\n", cur_stream->id,
                  cur_stream->rcvvar->ts_tw_expire);
        AddtoTimewaitList(mtcp, cur_stream, cur_ts);
      }

      AddtoControlList(mtcp, cur_stream, cur_ts);

      /*
      if (cur_stream->state < TCP_ST_LAST_ACK) {
      fprintf(stdout, "%s %d lcore:%d sport:%d state:%d AddtoControlList, req:%u
      payloadlen:%u rcv_nxt:%u rcv_wnd:%u\n", \
                      __FILE__, __LINE__, mtcp->ctx->cpu,
      ntohs(cur_stream->dport), cur_stream->state, \ seq, payloadlen,
      cur_stream->rcv_nxt, cur_stream->rcvvar->rcv_wnd); fflush(stdout);
      }
      */
    }
    return FALSE;
  }

  return TRUE;
}
/*----------------------------------------------------------------------------*/
static inline void NotifyConnectionReset(mtcp_manager_t mtcp,
                                         tcp_stream *cur_stream) {
  TRACE_DBG("Stream %d: Notifying connection reset.\n", cur_stream->id);
  /* TODO: implement this function */
  /* signal to user "connection reset" */
  AddEpollEvent(mtcp->ep, MTCP_EVENT_QUEUE, cur_stream->socket, MTCP_EPOLLERR);
}
/*----------------------------------------------------------------------------*/
static inline int ProcessRST(mtcp_manager_t mtcp, tcp_stream *cur_stream,
                             uint32_t ack_seq) {
  /* TODO: we need reset validation logic */
  /* the sequence number of a RST should be inside window */
  /* (in SYN_SENT state, it should ack the previous SYN */

  int ret;
  int group, group_idx, elem_idx;
  TRACE_DBG("Stream %d: TCP RESET (%s)\n", cur_stream->id,
            TCPStateToString(cur_stream));
#if DUMP_STREAM
  DumpStream(mtcp, cur_stream);
#endif

  if (cur_stream->state <= TCP_ST_SYN_SENT) {
    /* not handled here */
    return FALSE;
  }

  if (cur_stream->state == TCP_ST_SYN_RCVD) {
    if (ack_seq == cur_stream->snd_nxt) {
      cur_stream->state = TCP_ST_CLOSED;
      cur_stream->close_reason = TCP_RESET;
      printf("destroy %u in processRST(syn_rcvd)\n", ntohs(cur_stream->dport));
      DestroyTCPStream(mtcp, cur_stream);
      group = (cur_stream->soft_id / 64) / 64;
      group_idx = (cur_stream->soft_id / 64) % 64;
      elem_idx = cur_stream->soft_id % 64;

      pthread_spin_lock(&fdir_lock);
      ret = fdir_del_perfect_filter(cur_stream->nic_id, cur_stream->soft_id,
                                    mtcp->ctx->cpu, cur_stream->saddr,
                                    cur_stream->sport, cur_stream->daddr,
                                    cur_stream->dport);
      if(!ret)
      num_conn--;
      fdir_group[cur_stream->nic_id][group] |= ((uint64_t)1 << group_idx);
      fdir_elem[cur_stream->nic_id][group * 64 + group_idx] |=
          ((uint64_t)1 << elem_idx);
      pthread_spin_unlock(&fdir_lock);
      if (ret < 0)
        fprintf(stderr,
                "[thread: %d sport:%u] failed to delete signature filter:%s\n",
                mtcp->ctx->cpu, htons(cur_stream->dport), strerror(-ret));
      else
        printf("%s %d del filter saddr:0x%x sport:%u\n", __FILE__, __LINE__,
               htonl(cur_stream->daddr), htons(cur_stream->dport));
    }
    return TRUE;
  }

  /* if the application is already closed the connection,
     just destroy the it */
  if (cur_stream->state == TCP_ST_FIN_WAIT_1 ||
      cur_stream->state == TCP_ST_FIN_WAIT_2 ||
      cur_stream->state == TCP_ST_LAST_ACK ||
      cur_stream->state == TCP_ST_CLOSING ||
      cur_stream->state == TCP_ST_TIME_WAIT) {
    cur_stream->state = TCP_ST_CLOSED;
    cur_stream->close_reason = TCP_ACTIVE_CLOSE;

    group = (cur_stream->soft_id / 64) / 64;
    group_idx = (cur_stream->soft_id / 64) % 64;
    elem_idx = cur_stream->soft_id % 64;

    pthread_spin_lock(&fdir_lock);
    ret = fdir_del_perfect_filter(cur_stream->nic_id, cur_stream->soft_id,
                                  mtcp->ctx->cpu, cur_stream->saddr,
                                  cur_stream->sport, cur_stream->daddr,
                                  cur_stream->dport);
    if(!ret)
    num_conn--;
    fdir_group[cur_stream->nic_id][group] |= ((uint64_t)1 << group_idx);
    fdir_elem[cur_stream->nic_id][group * 64 + group_idx] |=
        ((uint64_t)1 << elem_idx);
    pthread_spin_unlock(&fdir_lock);
    if (ret < 0)
      fprintf(stderr,
              "[thread: %d sport:%u] failed to delete perfect filter:%s\n",
              mtcp->ctx->cpu, htons(cur_stream->dport), strerror(-ret));
    else
      printf("%s %d del filter saddr:0x%x sport:%u\n", __FILE__, __LINE__,
             htonl(cur_stream->daddr), htons(cur_stream->dport));
    DestroyTCPStream(mtcp, cur_stream);
    return TRUE;
  }

  if (cur_stream->state >= TCP_ST_ESTABLISHED &&
      cur_stream->state <= TCP_ST_CLOSE_WAIT) {
    /* ESTABLISHED, FIN_WAIT_1, FIN_WAIT_2, CLOSE_WAIT */
    /* TODO: flush all the segment queues */
    // NotifyConnectionReset(mtcp, cur_stream);
    // qi
    cur_stream->state = TCP_ST_CLOSED;
    cur_stream->close_reason = TCP_RESET;
    /*
    ret = fdir_del_signature_filter(0, mtcp->ctx->cpu, htonl(cur_stream->saddr),
    \ htons(cur_stream->sport), htonl(cur_stream->daddr),
    htons(cur_stream->dport));

    ret = fdir_del_signature_filter(0, mtcp->ctx->cpu, cur_stream->saddr, \
                    cur_stream->sport, cur_stream->daddr, cur_stream->dport);

    if (ret < 0)
            fprintf(stderr, "[thread: %d sport:%u] failed to delete signature
    filter:%s\n", mtcp->ctx->cpu, htons(cur_stream->dport), strerror(-ret));

    fprintf(stdout, "[thread: %d sport:%u ack_seq:%u] process RST\n", \
                    mtcp->ctx->cpu, htons(cur_stream->dport), ack_seq);
    fflush(stdout);
                    */

    fprintf(stdout, "destroy %u in processRST\n", ntohs(cur_stream->dport));
    fflush(stdout);

    group = (cur_stream->soft_id / 64) / 64;
    group_idx = (cur_stream->soft_id / 64) % 64;
    elem_idx = cur_stream->soft_id % 64;

    pthread_spin_lock(&fdir_lock);
    ret = fdir_del_perfect_filter(cur_stream->nic_id, cur_stream->soft_id,
                                  mtcp->ctx->cpu, cur_stream->saddr,
                                  cur_stream->sport, cur_stream->daddr,
                                  cur_stream->dport);
    if(!ret)
    num_conn--;
    fdir_group[cur_stream->nic_id][group] |= ((uint64_t)1 << group_idx);
    fdir_elem[cur_stream->nic_id][group * 64 + group_idx] |=
        ((uint64_t)1 << elem_idx);
    pthread_spin_unlock(&fdir_lock);
    if (ret < 0)
      fprintf(stderr,
              "[thread: %d sport:%u] failed to delete perfect filter:%s\n",
              mtcp->ctx->cpu, htons(cur_stream->dport), strerror(-ret));
    else
      printf("%s %d del filter saddr:0x%x sport:%u\n", __FILE__, __LINE__,
             htonl(cur_stream->daddr), htons(cur_stream->dport));

    fprintf(stdout, "[thread: %d sport:%u ack_seq:%u] process RST\n",
            mtcp->ctx->cpu, htons(cur_stream->dport), ack_seq);
    fflush(stdout);
    DestroyTCPStream(mtcp, cur_stream);

    // qi-add
    if (cur_stream->state == TCP_ST_ESTABLISHED) {
      fprintf(stdout,
              "[thread: %d sport:%u ack_seq:%u]  NotifyConnectionReset\n",
              mtcp->ctx->cpu, htons(cur_stream->dport), ack_seq);
      fflush(stdout);
      NotifyConnectionReset(mtcp, cur_stream);
    }
  }

  if (!(cur_stream->sndvar->on_closeq || cur_stream->sndvar->on_closeq_int ||
        cur_stream->sndvar->on_resetq || cur_stream->sndvar->on_resetq_int)) {
    // cur_stream->state = TCP_ST_CLOSED;
    // DestroyTCPStream(mtcp, cur_stream);
    cur_stream->state = TCP_ST_CLOSE_WAIT;
    cur_stream->close_reason = TCP_RESET;
    fprintf(stdout, "[thread: %d sport:%u ack_seq:%u]   RaiseCloseEvent\n",
            mtcp->ctx->cpu, htons(cur_stream->dport), ack_seq);
    fflush(stdout);
    RaiseCloseEvent(mtcp, cur_stream);
  }

  return TRUE;
}
/*----------------------------------------------------------------------------*/
inline void EstimateRTT(mtcp_manager_t mtcp, tcp_stream *cur_stream,
                        uint32_t mrtt) {
  /* This function should be called for not retransmitted packets */
  /* TODO: determine tcp_rto_min */
#define TCP_RTO_MIN 0
  long m = mrtt;
  uint32_t tcp_rto_min = TCP_RTO_MIN;
  struct tcp_recv_vars *rcvvar = cur_stream->rcvvar;

  if (m == 0) {
    m = 1;
  }
  if (rcvvar->srtt != 0) {
    /* rtt = 7/8 rtt + 1/8 new */
    m -= (rcvvar->srtt >> 3);
    rcvvar->srtt += m;
    if (m < 0) {
      m = -m;
      m -= (rcvvar->mdev >> 2);
      if (m > 0) {
        m >>= 3;
      }
    } else {
      m -= (rcvvar->mdev >> 2);
    }
    rcvvar->mdev += m;
    if (rcvvar->mdev > rcvvar->mdev_max) {
      rcvvar->mdev_max = rcvvar->mdev;
      if (rcvvar->mdev_max > rcvvar->rttvar) {
        rcvvar->rttvar = rcvvar->mdev_max;
      }
    }
    if (TCP_SEQ_GT(cur_stream->sndvar->snd_una, rcvvar->rtt_seq)) {
      if (rcvvar->mdev_max < rcvvar->rttvar) {
        rcvvar->rttvar -= (rcvvar->rttvar - rcvvar->mdev_max) >> 2;
      }
      rcvvar->rtt_seq = cur_stream->snd_nxt;
      rcvvar->mdev_max = tcp_rto_min;
    }
  } else {
    /* fresh measurement */
    rcvvar->srtt = m << 3;
    rcvvar->mdev = m << 1;
    rcvvar->mdev_max = rcvvar->rttvar = MAX(rcvvar->mdev, tcp_rto_min);
    rcvvar->rtt_seq = cur_stream->snd_nxt;
  }

  TRACE_RTT(
      "mrtt: %u (%uus), srtt: %u (%ums), mdev: %u, mdev_max: %u, "
      "rttvar: %u, rtt_seq: %u\n",
      mrtt, mrtt * TIME_TICK, rcvvar->srtt, TS_TO_MSEC((rcvvar->srtt) >> 3),
      rcvvar->mdev, rcvvar->mdev_max, rcvvar->rttvar, rcvvar->rtt_seq);
}

/*----------------------------------------------------------------------------*/
static inline void ProcessACK(mtcp_manager_t mtcp, tcp_stream *cur_stream,
                              uint32_t cur_ts, struct tcphdr *tcph,
                              uint32_t seq, uint32_t ack_seq, uint16_t window,
                              int payloadlen) {
  struct tcp_send_vars *sndvar = cur_stream->sndvar;
  uint32_t cwindow, cwindow_prev;
  uint32_t rmlen;
  uint32_t snd_wnd_prev;
  uint32_t right_wnd_edge;
  uint8_t dup;
  int ret;

  cwindow = window;
  if (!tcph->syn) {
    cwindow = cwindow << sndvar->wscale_peer;
  }
  right_wnd_edge = sndvar->peer_wnd + cur_stream->rcvvar->snd_wl2;

  /* If ack overs the sending buffer, return */
  if (cur_stream->state == TCP_ST_FIN_WAIT_1 ||
      cur_stream->state == TCP_ST_FIN_WAIT_2 ||
      cur_stream->state == TCP_ST_CLOSING ||
      cur_stream->state == TCP_ST_CLOSE_WAIT ||
      cur_stream->state == TCP_ST_LAST_ACK) {
    if (sndvar->is_fin_sent && ack_seq == sndvar->fss + 1) {
#if PROFILING
      fprintf(stdout,
              "%s %d lcore:%d sport:%u ack_seq:%u cksum:0x%x in processack for "
              "LAST_ACK\n",
              __FILE__, __LINE__, mtcp->ctx->cpu, ntohs(cur_stream->dport),
              ack_seq, ntohs(tcph->check));
      fflush(stdout);
#endif
      ack_seq--;
    }
  }

  if (TCP_SEQ_GT(ack_seq, sndvar->sndbuf->head_seq + sndvar->sndbuf->len)) {
    TRACE_DBG(
        "Stream %d (%s): invalid acknologement. "
        "ack_seq: %u, possible max_ack_seq(head_seq:%u + len:%u): %u\n",
        cur_stream->id, TCPStateToString(cur_stream), ack_seq,
        sndvar->sndbuf->head_seq, sndvar->sndbuf->len,
        sndvar->sndbuf->head_seq + sndvar->sndbuf->len);
    return;
  }

  /* Update window */
  if (TCP_SEQ_LT(cur_stream->rcvvar->snd_wl1, seq) ||
      (cur_stream->rcvvar->snd_wl1 == seq &&
       TCP_SEQ_LT(cur_stream->rcvvar->snd_wl2, ack_seq)) ||
      (cur_stream->rcvvar->snd_wl2 == ack_seq && cwindow > sndvar->peer_wnd)) {
    cwindow_prev = sndvar->peer_wnd;
    sndvar->peer_wnd = cwindow;
    cur_stream->rcvvar->snd_wl1 = seq;
    cur_stream->rcvvar->snd_wl2 = ack_seq;
#if 0
		TRACE_CLWND("Window update. "
				"ack: %u, peer_wnd: %u, snd_nxt-snd_una: %u\n", 
				ack_seq, cwindow, cur_stream->snd_nxt - sndvar->snd_una);
#endif
    if (cwindow_prev < cur_stream->snd_nxt - sndvar->snd_una &&
        sndvar->peer_wnd >= cur_stream->snd_nxt - sndvar->snd_una) {
      TRACE_CLWND(
          "%u Broadcasting client window update! "
          "ack_seq: %u, peer_wnd: %u (before: %u), "
          "(snd_nxt - snd_una: %u)\n",
          cur_stream->id, ack_seq, sndvar->peer_wnd, cwindow_prev,
          cur_stream->snd_nxt - sndvar->snd_una);
      RaiseWriteEvent(mtcp, cur_stream);
    }
  }

  /* Check duplicated ack count */
  /* Duplicated ack if
     1) ack_seq is old
     2) payload length is 0.
     3) advertised window not changed.
     4) there is outstanding unacknowledged data
     5) ack_seq == snd_una
   */

  dup = FALSE;
  if (TCP_SEQ_LT(ack_seq, cur_stream->snd_nxt)) {
    if (ack_seq == cur_stream->rcvvar->last_ack_seq && payloadlen == 0) {
      if (cur_stream->rcvvar->snd_wl2 + sndvar->peer_wnd == right_wnd_edge) {
        if (cur_stream->rcvvar->dup_acks + 1 > cur_stream->rcvvar->dup_acks) {
          cur_stream->rcvvar->dup_acks++;
#if USE_CCP
          ccp_record_event(mtcp, cur_stream, EVENT_DUPACK,
                           (cur_stream->snd_nxt - ack_seq));
#endif
        }
        dup = TRUE;
      }
    }
  }
  if (!dup) {
#if USE_CCP
    if (cur_stream->rcvvar->dup_acks >= 3) {
      TRACE_DBG(
          "passed dup_acks, ack=%u, snd_nxt=%u, last_ack=%u len=%u wl2=%u "
          "peer_wnd=%u right=%u\n",
          ack_seq - sndvar->iss, cur_stream->snd_nxt - sndvar->iss,
          cur_stream->rcvvar->last_ack_seq - sndvar->iss, payloadlen,
          cur_stream->rcvvar->snd_wl2 - sndvar->iss,
          sndvar->peer_wnd / sndvar->mss, right_wnd_edge - sndvar->iss);
    }
#endif
    cur_stream->rcvvar->dup_acks = 0;
    cur_stream->rcvvar->last_ack_seq = ack_seq;
  }
#if USE_CCP
  if (cur_stream->wait_for_acks) {
    TRACE_DBG("got ack, but waiting to send... ack=%u, snd_next=%u cwnd=%u\n",
              ack_seq - sndvar->iss, cur_stream->snd_nxt - sndvar->iss,
              sndvar->cwnd / sndvar->mss);
  }
#endif
  /* Fast retransmission */
  if (dup && cur_stream->rcvvar->dup_acks == 3) {
    TRACE_LOSS("Triple duplicated ACKs!! ack_seq: %u\n", ack_seq);
    TRACE_CCP("tridup ack %u (%u)!\n", ack_seq - cur_stream->sndvar->iss,
              ack_seq);
    if (TCP_SEQ_LT(ack_seq, cur_stream->snd_nxt)) {
      TRACE_LOSS("Reducing snd_nxt from %u to %u\n",
                 cur_stream->snd_nxt - sndvar->iss,
                 ack_seq - cur_stream->sndvar->iss);

#if RTM_STAT
      sndvar->rstat.tdp_ack_cnt++;
      sndvar->rstat.tdp_ack_bytes += (cur_stream->snd_nxt - ack_seq);
#endif

#if USE_CCP
      ccp_record_event(mtcp, cur_stream, EVENT_TRI_DUPACK, ack_seq);
#endif
      if (ack_seq != sndvar->snd_una) {
        TRACE_DBG(
            "ack_seq and snd_una mismatch on tdp ack. "
            "ack_seq: %u, snd_una: %u\n",
            ack_seq, sndvar->snd_una);
      }
#if USE_CCP
      sndvar->missing_seq = ack_seq;
#else
      cur_stream->snd_nxt = ack_seq;
#endif
    }

    /* update congestion control variables */
    /* ssthresh to half of min of cwnd and peer wnd */
    sndvar->ssthresh = MIN(sndvar->cwnd, sndvar->peer_wnd) / 2;
    if (sndvar->ssthresh < 2 * sndvar->mss) {
      sndvar->ssthresh = 2 * sndvar->mss;
    }
    sndvar->cwnd = sndvar->ssthresh + 3 * sndvar->mss;

    TRACE_CONG("fast retrans: cwnd = ssthresh(%u)+3*mss = %u\n",
               sndvar->ssthresh / sndvar->mss, sndvar->cwnd / sndvar->mss);

    /* count number of retransmissions */
    if (sndvar->nrtx < TCP_MAX_RTX) {
      sndvar->nrtx++;
    } else {
      TRACE_DBG("Exceed MAX_RTX.\n");
    }

    fprintf(stdout, "%s %d lcore:%d sport:%u to AddtoSendList\n", __FILE__,
            __LINE__, mtcp->ctx->cpu, ntohs(cur_stream->dport));
    fflush(stdout);
    AddtoSendList(mtcp, cur_stream);

  } else if (cur_stream->rcvvar->dup_acks > 3) {
    /* Inflate congestion window until before overflow */
    if ((uint32_t)(sndvar->cwnd + sndvar->mss) > sndvar->cwnd) {
      sndvar->cwnd += sndvar->mss;
      TRACE_CONG("Dupack cwnd inflate. cwnd: %u, ssthresh: %u\n", sndvar->cwnd,
                 sndvar->ssthresh);
    }
  }

#if TCP_OPT_SACK_ENABLED
  ParseSACKOption(cur_stream, ack_seq, (uint8_t *)tcph + TCP_HEADER_LEN,
                  (tcph->doff << 2) - TCP_HEADER_LEN);
#endif /* TCP_OPT_SACK_ENABLED */

#if RECOVERY_AFTER_LOSS
#if USE_CCP
  /* updating snd_nxt (when recovered from loss) */
  if (TCP_SEQ_GT(ack_seq, cur_stream->snd_nxt) ||
      (cur_stream->wait_for_acks &&
       TCP_SEQ_GT(ack_seq, cur_stream->seq_at_last_loss)
#if TCP_OPT_SACK_ENABLED
       && cur_stream->rcvvar->sacked_pkts == 0
#endif
       ))
#else
  if (TCP_SEQ_GT(ack_seq, cur_stream->snd_nxt))
#endif /* USE_CCP */
  {
#if RTM_STAT
    sndvar->rstat.ack_upd_cnt++;
    sndvar->rstat.ack_upd_bytes += (ack_seq - cur_stream->snd_nxt);
#endif
    // fast retransmission exit: cwnd=ssthresh
    cur_stream->sndvar->cwnd = cur_stream->sndvar->ssthresh;

    TRACE_LOSS("Updating snd_nxt from %u to %u\n", cur_stream->snd_nxt,
               ack_seq);
#if USE_CCP
    cur_stream->wait_for_acks = FALSE;
#endif
    cur_stream->snd_nxt = ack_seq;
    TRACE_DBG("Sending again..., ack_seq=%u sndlen=%u cwnd=%u\n",
              ack_seq - sndvar->iss, sndvar->sndbuf->len,
              sndvar->cwnd / sndvar->mss);
    if (sndvar->sndbuf->len == 0) {
      fprintf(stdout, "%s %d lcore:%d sport:%u RemovefromSendList\n", __FILE__,
              __LINE__, mtcp->ctx->cpu, ntohs(cur_stream->dport));
      fflush(stdout);

      RemoveFromSendList(mtcp, cur_stream);
    } else {
      fprintf(stdout, "%s %d lcore:%d sport:%u to AddtoSendList\n", __FILE__,
              __LINE__, mtcp->ctx->cpu, ntohs(cur_stream->dport));
      fflush(stdout);

      AddtoSendList(mtcp, cur_stream);
    }
  }
#endif /* RECOVERY_AFTER_LOSS */

  rmlen = ack_seq - sndvar->sndbuf->head_seq;
  uint16_t packets = rmlen / sndvar->eff_mss;
  if (packets * sndvar->eff_mss > rmlen) {
    packets++;
  }

#if USE_CCP
  ccp_cong_control(mtcp, cur_stream, ack_seq, rmlen, packets);
#else
  // log_cwnd_rtt(cur_stream);
#endif

  /* If ack_seq is previously acked, return */
  if (TCP_SEQ_GEQ(sndvar->sndbuf->head_seq, ack_seq)) {
    return;
  }

  /* Remove acked sequence from send buffer */
  if (rmlen > 0) {
    /* Routine goes here only if there is new payload (not retransmitted) */

    /* Estimate RTT and calculate rto */
    if (cur_stream->saw_timestamp) {
      EstimateRTT(mtcp, cur_stream,
                  cur_ts - cur_stream->rcvvar->ts_lastack_rcvd);
      sndvar->rto =
          (cur_stream->rcvvar->srtt >> 3) + cur_stream->rcvvar->rttvar;

      /*
      fprintf(stdout, "%s %d lcore:%d sport:%u sndvar->rto:%u\n", \
                      __FILE__, __LINE__, mtcp->ctx->cpu,
      ntohs(cur_stream->dport), sndvar->rto); fflush(stdout);
      */

      assert(sndvar->rto > 0);
    } else {
      // TODO: Need to implement timestamp estimation without timestamp
      TRACE_RTT("NOT IMPLEMENTED.\n");
    }

    // TODO CCP should comment this out?
    /* Update congestion control variables */
    if (cur_stream->state >= TCP_ST_ESTABLISHED) {
      if (sndvar->cwnd < sndvar->ssthresh) {
        if ((sndvar->cwnd + sndvar->mss) > sndvar->cwnd) {
          sndvar->cwnd += (sndvar->mss * packets);
        }
        TRACE_CONG("slow start cwnd: %u, ssthresh: %u\n", sndvar->cwnd,
                   sndvar->ssthresh);
      } else {
        uint32_t new_cwnd =
            sndvar->cwnd + packets * sndvar->mss * sndvar->mss / sndvar->cwnd;
        if (new_cwnd > sndvar->cwnd) {
          sndvar->cwnd = new_cwnd;
        }
        // TRACE_CONG("congestion avoidance cwnd: %u, ssthresh: %u\n",
        //		sndvar->cwnd, sndvar->ssthresh);
      }
    }

    if (SBUF_LOCK(&sndvar->write_lock)) {
      if (errno == EDEADLK) perror("ProcessACK: write_lock blocked\n");
      assert(0);
    }
    ret = SBRemove(mtcp->rbm_snd, sndvar->sndbuf, rmlen);
    sndvar->snd_una = ack_seq;
    snd_wnd_prev = sndvar->snd_wnd;
    sndvar->snd_wnd = sndvar->sndbuf->size - sndvar->sndbuf->len;

    /* If there was no available sending window */
    /* notify the newly available window to application */
#if SELECTIVE_WRITE_EVENT_NOTIFY
    if (snd_wnd_prev <= 0) {
#endif /* SELECTIVE_WRITE_EVENT_NOTIFY */
      RaiseWriteEvent(mtcp, cur_stream);
#if SELECTIVE_WRITE_EVENT_NOTIFY
    }
#endif /* SELECTIVE_WRITE_EVENT_NOTIFY */

    SBUF_UNLOCK(&sndvar->write_lock);

#if PROFILINGRTO
    fprintf(stdout, "%s %d lcore:%d sport:%u UpdateRetransmissionTimer\n",
            __FILE__, __LINE__, mtcp->ctx->cpu, ntohs(cur_stream->dport));
    fflush(stdout);
#endif

    UpdateRetransmissionTimer(mtcp, cur_stream, cur_ts);
  }

  UNUSED(ret);
}
/*----------------------------------------------------------------------------*/
/* ProcessTCPPayload: merges TCP payload using receive ring buffer            */
/* Return: TRUE (1) in normal case, FALSE (0) if immediate ACK is required    */
/* CAUTION: should only be called at ESTABLISHED, FIN_WAIT_1, FIN_WAIT_2      */
/*----------------------------------------------------------------------------*/
static inline int ProcessTCPPayload(mtcp_manager_t mtcp, tcp_stream *cur_stream,
                                    uint32_t cur_ts, uint8_t *payload,
                                    uint32_t seq, int payloadlen) {
  struct tcp_recv_vars *rcvvar = cur_stream->rcvvar;
  uint32_t prev_rcv_nxt;
  int ret;

  /* if seq and segment length is lower than rcv_nxt, ignore and send ack */
  if (TCP_SEQ_LT(seq + payloadlen, cur_stream->rcv_nxt)) {
    return FALSE;
  }
  /* if payload exceeds receiving buffer, drop and send ack */
  if (TCP_SEQ_GT(seq + payloadlen, cur_stream->rcv_nxt + rcvvar->rcv_wnd)) {
    return FALSE;
  }

  /* allocate receive buffer if not exist */
  if (!rcvvar->rcvbuf) {
    rcvvar->rcvbuf = RBInit(mtcp->rbm_rcv, rcvvar->irs + 1);
    if (!rcvvar->rcvbuf) {
      TRACE_ERROR("Stream %d: Failed to allocate receive buffer.\n",
                  cur_stream->id);
      cur_stream->state = TCP_ST_CLOSED;
      cur_stream->close_reason = TCP_NO_MEM;
      RaiseErrorEvent(mtcp, cur_stream);

      return ERROR;
    }
  }

  if (SBUF_LOCK(&rcvvar->read_lock)) {
    if (errno == EDEADLK) perror("ProcessTCPPayload: read_lock blocked\n");
    assert(0);
  }

  prev_rcv_nxt = cur_stream->rcv_nxt;
  ret =
      RBPut(mtcp->rbm_rcv, rcvvar->rcvbuf, payload, (uint32_t)payloadlen, seq);
  if (ret < 0) {
    TRACE_ERROR("Cannot merge payload. reason: %d\n", ret);
  }

  cur_stream->recv_len += payloadlen;

#if 0
  if (cur_stream->recv_len > 602112) {
    gettimeofday(&cur_stream->recvtime[1], NULL);
    fprintf(stdout, "worker:%u sport:%u recv len:%u latency:%f ms\n", mtcp->ctx->cpu, ntohs(cur_stream->dport), cur_stream->recv_len, \
    (cur_stream->recvtime[1].tv_sec - cur_stream->recvtime[0].tv_sec)*1000+(cur_stream->recvtime[1].tv_usec - cur_stream->recvtime[0].tv_usec)/1000.0);
    fflush(stdout);
  }
#endif

  /* discard the buffer if the state is FIN_WAIT_1 or FIN_WAIT_2,
     meaning that the connection is already closed by the application */
  if (cur_stream->state == TCP_ST_FIN_WAIT_1 ||
      cur_stream->state == TCP_ST_FIN_WAIT_2) {
    RBRemove(mtcp->rbm_rcv, rcvvar->rcvbuf, rcvvar->rcvbuf->merged_len,
             AT_MTCP);
  }

  cur_stream->rcv_nxt = rcvvar->rcvbuf->head_seq + rcvvar->rcvbuf->merged_len;

  rcvvar->rcv_wnd = rcvvar->rcvbuf->size - rcvvar->rcvbuf->merged_len;

#if 1
  if (rcvvar->rcv_wnd < 1448) {
    fprintf(stdout, "rcvwnd:%u =rcvbuf->size:%u - rcvbuf->merged_len:%u\n",
            rcvvar->rcv_wnd, rcvvar->rcvbuf->size, rcvvar->rcvbuf->merged_len);
    fflush(stdout);
  }
#endif

  SBUF_UNLOCK(&rcvvar->read_lock);

  if (TCP_SEQ_LEQ(cur_stream->rcv_nxt, prev_rcv_nxt)) {
#if 0
	  fprintf(stdout, "worker:%u, ip:%x port:%u packet lost, seq:%u rcv_nxt:%u buf->fctx->seq:%u merged_len:%u in_buff:%u\n", mtcp->ctx->cpu, ntohl(cur_stream->daddr), ntohs(cur_stream->dport), \
        seq, prev_rcv_nxt, cur_stream->rcvvar->rcvbuf->fctx->seq, \
        cur_stream->rcvvar->rcvbuf->merged_len, getseq(cur_stream->rcvvar->rcvbuf, prev_rcv_nxt));
	  fflush(stdout);
#endif
#if 0
    struct rte_eth_stats stats;
    rte_eth_stats_get(cur_stream->nic_id, &stats);
    fprintf(stdout, "worker:%u imissed:%lu ierrors:%lu nobuf:%lu\n",
            mtcp->ctx->cpu, stats.imissed, stats.ierrors, stats.rx_nombuf);
    fflush(stdout);
#endif
    /* There are some lost packets */

#if 1
    fprintf(stdout,
            "worker:%u lost pkt, rcv:%u, head_seq:%u merged_len:%u, "
            "payloadlen:%u\n",
            mtcp->ctx->cpu, cur_stream->rcv_nxt, rcvvar->rcvbuf->head_seq,
            rcvvar->rcvbuf->merged_len, payloadlen);
    fflush(stdout);
#endif
    return FALSE;
  }

  TRACE_EPOLL(
      "Stream %d data arrived. "
      "len: %d, ET: %u, IN: %u, OUT: %u\n",
      cur_stream->id, payloadlen,
      cur_stream->socket ? cur_stream->socket->epoll & MTCP_EPOLLET : 0,
      cur_stream->socket ? cur_stream->socket->epoll & MTCP_EPOLLIN : 0,
      cur_stream->socket ? cur_stream->socket->epoll & MTCP_EPOLLOUT : 0);

  if (cur_stream->state == TCP_ST_ESTABLISHED) {
    RaiseReadEvent(mtcp, cur_stream);
  }

  return TRUE;
}
/*----------------------------------------------------------------------------*/
static inline tcp_stream *CreateNewFlowHTEntry(
    mtcp_manager_t mtcp, uint32_t cur_ts, const struct iphdr *iph, int ip_len,
    const struct tcphdr *tcph, uint32_t seq, uint32_t ack_seq, int payloadlen,
    uint16_t window) {
  tcp_stream *cur_stream;
  int ret;

  if (tcph->syn && !tcph->ack) {
    /* handle the SYN */
    ret = FilterSYNPacket(mtcp, iph->daddr, tcph->dest);
    if (!ret) {
      TRACE_DBG("Refusing SYN packet.\n");
      fprintf(stdout, "Refusing SYN packet.\n");
      fflush(stdout);

#ifdef DBGMSG
      DumpIPPacket(mtcp, iph, ip_len);
#endif
      fprintf(stdout, "file %s line %d\n", __FILE__, __LINE__);
      fflush(stdout);
      SendTCPPacketStandalone(mtcp, iph->daddr, tcph->dest, iph->saddr,
                              tcph->source, 0, seq + payloadlen + 1, 0,
                              TCP_FLAG_RST | TCP_FLAG_ACK, NULL, 0, cur_ts, 0);

      return NULL;
    }

    /* now accept the connection */
    cur_stream = HandlePassiveOpen(mtcp, cur_ts, iph, tcph, seq, window);
    if (!cur_stream) {
      TRACE_DBG("Not available space in flow pool.\n");
#ifdef DBGMSG
      DumpIPPacket(mtcp, iph, ip_len);
#endif
      fprintf(stdout, "file %s line %d\n", __FILE__, __LINE__);
      fflush(stdout);

      SendTCPPacketStandalone(mtcp, iph->daddr, tcph->dest, iph->saddr,
                              tcph->source, 0, seq + payloadlen + 1, 0,
                              TCP_FLAG_RST | TCP_FLAG_ACK, NULL, 0, cur_ts, 0);

      return NULL;
    }

    return cur_stream;
  } else if (tcph->rst) {
    TRACE_DBG("Reset packet comes\n");
#ifdef DBGMSG
    DumpIPPacket(mtcp, iph, ip_len);
#endif
    /* for the reset packet, just discard */
    return NULL;
  } else {
    TRACE_DBG("Weird packet comes.\n");

#ifdef DBGMSG
    DumpIPPacket(mtcp, iph, ip_len);
#endif
    /* TODO: for else, discard and send a RST */
    /* if the ACK bit is off, respond with seq 0:
       <SEQ=0><ACK=SEG.SEQ+SEG.LEN><CTL=RST,ACK>
       else (ACK bit is on):
       <SEQ=SEG.ACK><CTL=RST>
       */
    if (tcph->ack) {
#if 0
      struct rte_eth_fdir_stats fdir_stat;
      struct rte_mbuf *tmp_mbuf;
      uint16_t hash, id;

      tmp_mbuf =
          (struct rte_mbuf *)((unsigned char *)tcph - sizeof(struct iphdr) -
                              sizeof(struct ethhdr) - RTE_PKTMBUF_HEADROOM -
                              sizeof(struct rte_mbuf));
      hash = tmp_mbuf->hash.fdir.hash;
      id = tmp_mbuf->hash.fdir.id;

      fdir_stat = fdir_retrieve_stats(0);
      TRACE_DBG("weired ack packet\n");
      fprintf(stdout,
              "weired ack, [thread:%d] collision:%" PRIu32 " add: %" PRIu64
              " f_add: %" PRIu64
              " \
					guarant_cnt: %" PRIu32
              " best_cnt: %" PRIu32 "\n",
              mtcp->ctx->cpu, fdir_stat.collision, fdir_stat.add,
              fdir_stat.f_add, fdir_stat.guarant_cnt, fdir_stat.best_cnt);

      fprintf(stdout,
              "thread:%d saddr:0x%x sport:%u, daddr:0x%x dport:%u [seq:%u "
              "ack:%u cksum:0x%x cksum:0x%x hash:0x%x id:0x%x]\n",
              mtcp->ctx->cpu, ntohl(iph->saddr), ntohs(tcph->source),
              ntohl(iph->daddr), ntohs(tcph->dest), (tcph->seq),
              ntohl(tcph->ack_seq), ntohs(tcph->check), tcph->check, hash, id);
      fflush(stdout);

#endif

      SendTCPPacketStandalone(mtcp, iph->daddr, tcph->dest, iph->saddr,
                              tcph->source, ack_seq, 0, 0, TCP_FLAG_RST, NULL,
                              0, cur_ts, 0);
    } else {
      fprintf(stdout, "file %s line %d\n", __FILE__, __LINE__);
      fflush(stdout);

      SendTCPPacketStandalone(mtcp, iph->daddr, tcph->dest, iph->saddr,
                              tcph->source, 0, seq + payloadlen, 0,
                              TCP_FLAG_RST | TCP_FLAG_ACK, NULL, 0, cur_ts, 0);
    }
    return NULL;
  }
}
/*----------------------------------------------------------------------------*/
static __thread int conn;



static inline void Handle_TCP_ST_LISTEN(mtcp_manager_t mtcp, uint32_t cur_ts,
                                        tcp_stream *cur_stream,
                                        struct tcphdr *tcph) {
  int ret;

  int group_id;
  int group_idx;
  int elem_idx;
  int soft_id;
  // struct rte_eth_fdir_stats fdir_stat;
  if (tcph->syn) {
    if (cur_stream->state == TCP_ST_LISTEN) {
      cur_stream->rcv_nxt++;

      cur_stream->state = TCP_ST_SYN_RCVD;
      TRACE_STATE("Stream %d: TCP_ST_SYN_RCVD\n", cur_stream->id);
      AddtoControlList(mtcp, cur_stream, cur_ts);

      pthread_spin_lock(&fdir_lock);
      for (group_id = 0; group_id < GROUP; group_id++) {
        group_idx =
            __builtin_ffsll(fdir_group[cur_stream->nic_id][group_id]) - 1;
        if (group_idx < 0) continue;
        elem_idx =
            __builtin_ffsll(
                fdir_elem[cur_stream->nic_id][group_id * 64 + group_idx]) -
            1;
        if (group_idx >= 0 && elem_idx >= 0) break;
      }

      assert(elem_idx >= 0 && "elem_idx > 0");

      fdir_elem[cur_stream->nic_id][group_id * 64 + group_idx] &=
          ~((uint64_t)1 << elem_idx);

      if (__builtin_ffsll(
              fdir_elem[cur_stream->nic_id][group_id * 64 + group_idx]) == 0)
        fdir_group[cur_stream->nic_id][group_id] &= ~((uint64_t)1 << group_idx);

      soft_id = (group_id * 64 + group_idx) * 64 + elem_idx;

#if 0
        fprintf(stdout, "nic:%d group:%d group_idx:%d elem_idx:%d soft_id:%d\n", \
        cur_stream->nic_id, group_id, group_idx, elem_idx, soft_id);
        fflush(stdout);
#endif

      cur_stream->soft_id = soft_id;
      gettimeofday(&cur_stream->flow_start_time, NULL);

#if 0
      fprintf(stdout, "existing flow num: %lu\n", num_conn);
      fflush(stdout);
#endif      
      ret = fdir_add_perfect_filter(cur_stream->nic_id, soft_id, mtcp->ctx->cpu,
                                    cur_stream->saddr, cur_stream->sport,
                                    cur_stream->daddr, cur_stream->dport);
      if (!ret)
      num_conn++;
#if 0
      fdir_stat = fdir_retrieve_stats(cur_stream->nic_id);
      fprintf(stdout," guarant_cnt: %" PRIu32
              " best_cnt: %" PRIu32 "\n", fdir_stat.guarant_cnt, fdir_stat.best_cnt);
#endif
      
      pthread_spin_unlock(&fdir_lock);
      //      soft_id=RTE_MAX((soft_id + 1)%7000, 1);

      // fprintf(stdout, "worker %d recv %d conn\n", mtcp->ctx->cpu, ++conn);
      //    fflush(stdout);

      if (ret < 0) {
        fprintf(stdout,
                "[thread:%d sport:%u] failed to add perfect filter: %s\n",
                mtcp->ctx->cpu, ntohs(cur_stream->dport), strerror(-ret));
        fflush(stdout);
      } else {
#if 0
        fprintf(stdout, "[thread:%d sport:%u, soft_id:%u] add perfect filter\n",
                mtcp->ctx->cpu, ntohs(cur_stream->dport), soft_id);
        fflush(stdout);
#endif
      }
    }
  } else {
    CTRACE_ERROR(
        "Stream %d (TCP_ST_LISTEN): "
        "Packet without SYN.\n",
        cur_stream->id);
  }
}
/*----------------------------------------------------------------------------*/
static inline void Handle_TCP_ST_SYN_SENT(mtcp_manager_t mtcp, uint32_t cur_ts,
                                          tcp_stream *cur_stream,
                                          const struct iphdr *iph,
                                          struct tcphdr *tcph, uint32_t seq,
                                          uint32_t ack_seq, int payloadlen,
                                          uint16_t window) {
  int ret;
  int group, group_idx, elem_idx;
  /* when active open */
  if (tcph->ack) {
    /* filter the unacceptable acks */
    if (TCP_SEQ_LEQ(ack_seq, cur_stream->sndvar->iss) ||
        TCP_SEQ_GT(ack_seq, cur_stream->snd_nxt)) {
      if (!tcph->rst) {
        fprintf(stdout, "file %s line %d\n", __FILE__, __LINE__);
        fflush(stdout);

        SendTCPPacketStandalone(mtcp, iph->daddr, tcph->dest, iph->saddr,
                                tcph->source, ack_seq, 0, 0, TCP_FLAG_RST, NULL,
                                0, cur_ts, 0);
      }
      return;
    }
    /* accept the ack */
    cur_stream->sndvar->snd_una++;
  }

  if (tcph->rst) {
    if (tcph->ack) {
      cur_stream->state = TCP_ST_CLOSE_WAIT;
      cur_stream->close_reason = TCP_RESET;
      if (cur_stream->socket) {
        RaiseErrorEvent(mtcp, cur_stream);
      } else {
        DestroyTCPStream(mtcp, cur_stream);
        group = (cur_stream->soft_id / 64) / 64;
        group_idx = (cur_stream->soft_id / 64) % 64;
        elem_idx = cur_stream->soft_id % 64;

        pthread_spin_lock(&fdir_lock);
        #if 0
        fprintf(stdout, "existing flow num: %lu, del one flow\n", num_conn);
        fflush(stdout);
        #endif
        ret = fdir_del_perfect_filter(cur_stream->nic_id, cur_stream->soft_id,
                                      mtcp->ctx->cpu, cur_stream->saddr,
                                      cur_stream->sport, cur_stream->daddr,
                                      cur_stream->dport);

        fdir_group[cur_stream->nic_id][group] |= ((uint64_t)1 << group_idx);
        fdir_elem[cur_stream->nic_id][group * 64 + group_idx] |=
            ((uint64_t)1 << elem_idx);
        if (!ret)
        num_conn--;
        pthread_spin_unlock(&fdir_lock);
        if (ret < 0)
          fprintf(stderr,
                  "[thread: %d sport:%u] failed to delete perfect filter:%s\n",
                  mtcp->ctx->cpu, htons(cur_stream->dport), strerror(-ret));
        else
          printf("%s %d del filter saddr:0x%x sport:%u\n", __FILE__, __LINE__,
                 htonl(cur_stream->daddr), htons(cur_stream->dport));

        fprintf(stdout, "[thread: %d sport:%u ack_seq:%u] process RST\n",
                mtcp->ctx->cpu, htons(cur_stream->dport), ack_seq);
        fflush(stdout);
      }
    }
    return;
  }

  if (tcph->syn) {
    if (tcph->ack) {
      int ret = HandleActiveOpen(mtcp, cur_stream, cur_ts, tcph, seq, ack_seq,
                                 window);
      if (!ret) {
        return;
      }

      cur_stream->sndvar->nrtx = 0;
      cur_stream->rcv_nxt = cur_stream->rcvvar->irs + 1;
      RemoveFromRTOList(mtcp, cur_stream);
      cur_stream->state = TCP_ST_ESTABLISHED;
      TRACE_STATE("Stream %d: TCP_ST_ESTABLISHED\n", cur_stream->id);

      if (cur_stream->socket) {
        RaiseWriteEvent(mtcp, cur_stream);
      } else {
        TRACE_STATE("Stream %d: ESTABLISHED, but no socket\n", cur_stream->id);
        fprintf(stdout, "file %s line %d\n", __FILE__, __LINE__);
        fflush(stdout);

        SendTCPPacketStandalone(mtcp, iph->daddr, tcph->dest, iph->saddr,
                                tcph->source, 0, seq + payloadlen + 1, 0,
                                TCP_FLAG_RST | TCP_FLAG_ACK, NULL, 0, cur_ts,
                                0);
        cur_stream->close_reason = TCP_ACTIVE_CLOSE;

        group = (cur_stream->soft_id / 64) / 64;
        group_idx = (cur_stream->soft_id / 64) % 64;
        elem_idx = cur_stream->soft_id % 64;

        pthread_spin_lock(&fdir_lock);
        #if 0
        fprintf(stdout, "existing flow num: %lu, del one flow\n", num_conn);
        fflush(stdout);
        #endif
        ret = fdir_del_perfect_filter(cur_stream->nic_id, cur_stream->soft_id,
                                      mtcp->ctx->cpu, cur_stream->saddr,
                                      cur_stream->sport, cur_stream->daddr,
                                      cur_stream->dport);

        fdir_group[cur_stream->nic_id][group] |= ((uint64_t)1 << group_idx);
        fdir_elem[cur_stream->nic_id][group * 64 + group_idx] |=
            ((uint64_t)1 << elem_idx);
        if(!ret)
        num_conn--;
        pthread_spin_unlock(&fdir_lock);
        if (ret < 0)
          fprintf(
              stderr,
              "[thread: %d sport:%u] failed to delete signature filter:%s\n",
              mtcp->ctx->cpu, htons(cur_stream->dport), strerror(-ret));
        else
          printf("%s %d del filter saddr:0x%x sport:%u\n", __FILE__, __LINE__,
                 htonl(cur_stream->daddr), htons(cur_stream->dport));

        fprintf(stdout, "[thread: %d sport:%u ack_seq:%u] process RST\n",
                mtcp->ctx->cpu, htons(cur_stream->dport), ack_seq);
        fflush(stdout);
        DestroyTCPStream(mtcp, cur_stream);
        return;
      }
      AddtoControlList(mtcp, cur_stream, cur_ts);
      if (CONFIG.tcp_timeout > 0) AddtoTimeoutList(mtcp, cur_stream);

    } else {
      cur_stream->state = TCP_ST_SYN_RCVD;
      TRACE_STATE("Stream %d: TCP_ST_SYN_RCVD\n", cur_stream->id);
      cur_stream->snd_nxt = cur_stream->sndvar->iss;
      AddtoControlList(mtcp, cur_stream, cur_ts);
    }
  }
}
/*----------------------------------------------------------------------------*/
static inline void Handle_TCP_ST_SYN_RCVD(mtcp_manager_t mtcp, uint32_t cur_ts,
                                          tcp_stream *cur_stream,
                                          struct tcphdr *tcph,
                                          uint32_t ack_seq) {
  struct tcp_send_vars *sndvar = cur_stream->sndvar;
  int ret;
  if (tcph->ack) {
    struct tcp_listener *listener;
    uint32_t prior_cwnd;
    /* check if ACK of SYN */
    if (ack_seq != sndvar->iss + 1) {
      CTRACE_ERROR(
          "Stream %d (TCP_ST_SYN_RCVD): "
          "weird ack_seq: %u, iss: %u\n",
          cur_stream->id, ack_seq, sndvar->iss);
      TRACE_DBG(
          "Stream %d (TCP_ST_SYN_RCVD): "
          "weird ack_seq: %u, iss: %u\n",
          cur_stream->id, ack_seq, sndvar->iss);
      return;
    }

    sndvar->snd_una++;
    cur_stream->snd_nxt = ack_seq;
    prior_cwnd = sndvar->cwnd;
    sndvar->cwnd =
        ((prior_cwnd == 1) ? (sndvar->mss * TCP_INIT_CWND) : sndvar->mss);
    TRACE_DBG("sync_recvd: updating cwnd from %u to %u\n", prior_cwnd,
              sndvar->cwnd);

    // UpdateRetransmissionTimer(mtcp, cur_stream, cur_ts);
    sndvar->nrtx = 0;
    cur_stream->rcv_nxt = cur_stream->rcvvar->irs + 1;
    RemoveFromRTOList(mtcp, cur_stream);

    cur_stream->state = TCP_ST_ESTABLISHED;
    TRACE_STATE("Stream %d: TCP_ST_ESTABLISHED\n", cur_stream->id);
    gettimeofday(&cur_stream->recvtime[0], NULL);
    /* update listening socket */
    listener =
        (struct tcp_listener *)ListenerHTSearch(mtcp->listeners, &tcph->dest);

    ret = StreamEnqueue(listener->acceptq, cur_stream);
    if (ret < 0) {
      TRACE_ERROR(
          "Stream %d: Failed to enqueue to "
          "the listen backlog!\n",
          cur_stream->id);
      cur_stream->close_reason = TCP_NOT_ACCEPTED;
      cur_stream->state = TCP_ST_CLOSED;
      TRACE_STATE("Stream %d: TCP_ST_CLOSED\n", cur_stream->id);
      AddtoControlList(mtcp, cur_stream, cur_ts);
    }
    // TRACE_DBG("Stream %d inserted into acceptq.\n", cur_stream->id);
    if (CONFIG.tcp_timeout > 0) AddtoTimeoutList(mtcp, cur_stream);

    /* raise an event to the listening socket */
    if (listener->socket && (listener->socket->epoll & MTCP_EPOLLIN)) {
      AddEpollEvent(mtcp->ep, MTCP_EVENT_QUEUE, listener->socket, MTCP_EPOLLIN);
    }

  } else {
    TRACE_DBG("Stream %d (TCP_ST_SYN_RCVD): No ACK.\n", cur_stream->id);
    /* retransmit SYN/ACK */
    cur_stream->snd_nxt = sndvar->iss;
    AddtoControlList(mtcp, cur_stream, cur_ts);
  }
}
/*----------------------------------------------------------------------------*/
static inline void Handle_TCP_ST_ESTABLISHED(mtcp_manager_t mtcp,
                                             uint32_t cur_ts,
                                             tcp_stream *cur_stream,
                                             struct tcphdr *tcph, uint32_t seq,
                                             uint32_t ack_seq, uint8_t *payload,
                                             int payloadlen, uint16_t window) {
#if 0
	printf("tcp flag: %d src:%x tcp payloadlen: %d\n", \
	tcph->ack<<4|tcph->psh<<3|tcph->fin, tcph->source, payloadlen);
#endif
int ret;
 int group, group_idx, elem_idx;

  if (tcph->syn) {
    TRACE_DBG(
        "Stream %d (TCP_ST_ESTABLISHED): weird SYN. "
        "seq: %u, expected: %u, ack_seq: %u, expected: %u\n",
        cur_stream->id, seq, cur_stream->rcv_nxt, ack_seq, cur_stream->snd_nxt);
    cur_stream->snd_nxt = ack_seq;
    AddtoControlList(mtcp, cur_stream, cur_ts);
    return;
  }

  if (payloadlen > 0) {
    if (ProcessTCPPayload(mtcp, cur_stream, cur_ts, payload, seq, payloadlen)) {
      /* if return is TRUE, send ACK */
      EnqueueACK(mtcp, cur_stream, cur_ts, ACK_OPT_AGGREGATE);
    } else {
      EnqueueACK(mtcp, cur_stream, cur_ts, ACK_OPT_NOW);
    }
  }

  if (tcph->ack) {
    if (cur_stream->sndvar->sndbuf) {
      ProcessACK(mtcp, cur_stream, cur_ts, tcph, seq, ack_seq, window,
                 payloadlen);
    }
  }

  if (tcph->fin) {
    /* process the FIN only if the sequence is valid */
    /* FIN packet is allowed to push payload (should we check for PSH flag)? */
    if (seq + payloadlen == cur_stream->rcv_nxt) {
      cur_stream->state = TCP_ST_CLOSE_WAIT;
      TRACE_STATE("Stream %d: TCP_ST_CLOSE_WAIT", cur_stream->id);
      cur_stream->rcv_nxt++;

#if 0
      fprintf(stdout,
              "%s %d lcore:%d sport:%u process fin: seq:%u ack:%u "
              "payloadlen:%d rcv_nxt:%u to AddtoControlList\n",
              __FILE__, __LINE__, mtcp->ctx->cpu, ntohs(cur_stream->dport), seq,
              ack_seq, payloadlen, cur_stream->rcv_nxt);
      fflush(stdout);
#endif

      group = (cur_stream->soft_id / 64) / 64;
      group_idx = (cur_stream->soft_id / 64) % 64;
      elem_idx = cur_stream->soft_id % 64;

      pthread_spin_lock(&fdir_lock);
      #if 0
      fprintf(stdout, "existing flow num: %lu, del one flow\n", num_conn);
      fflush(stdout);
      #endif
      ret = fdir_del_perfect_filter(cur_stream->nic_id, cur_stream->soft_id,
                                    mtcp->ctx->cpu, cur_stream->saddr,
                                    cur_stream->sport, cur_stream->daddr,
                                    cur_stream->dport);
      if(!ret)
      num_conn--;
#if 0
      fprintf(stdout, "del, soft id:%d fdir_group:%d group_idx:%d elem_idx:%d\n",
              cur_stream->soft_id, group, group_idx, elem_idx);
      fflush(stdout);
#endif
      fdir_group[cur_stream->nic_id][group] |= ((uint64_t)1 << group_idx);
      fdir_elem[cur_stream->nic_id][group * 64 + group_idx] |=
          ((uint64_t)1 << elem_idx);
      pthread_spin_unlock(&fdir_lock);

      AddtoControlList(mtcp, cur_stream, cur_ts);

      /* notify FIN to application */
      // RaiseReadEvent(mtcp, cur_stream);
    } else {
      EnqueueACK(mtcp, cur_stream, cur_ts, ACK_OPT_NOW);
      return;
    }
  }
}
/*----------------------------------------------------------------------------*/
static inline void Handle_TCP_ST_CLOSE_WAIT(mtcp_manager_t mtcp,
                                            uint32_t cur_ts,
                                            tcp_stream *cur_stream,
                                            struct tcphdr *tcph, uint32_t seq,
                                            uint32_t ack_seq, int payloadlen,
                                            uint16_t window) {
  if (TCP_SEQ_LT(seq, cur_stream->rcv_nxt)) {
    TRACE_DBG(
        "Stream %d (TCP_ST_CLOSE_WAIT): "
        "weird seq: %u, expected: %u\n",
        cur_stream->id, seq, cur_stream->rcv_nxt);

    fprintf(stdout,
            "%s %d lcore:%d sport:%u in Handle_TCP_ST_CLOSE_WAIT "
            "AddtoControlList\n",
            __FILE__, __LINE__, mtcp->ctx->cpu, ntohs(cur_stream->dport));
    fflush(stdout);

    AddtoControlList(mtcp, cur_stream, cur_ts);
    return;
  }

  fprintf(stdout, "%s %d lcore:%d sport:%u in Handle_TCP_ST_CLOSE_WAIT\n",
          __FILE__, __LINE__, mtcp->ctx->cpu, ntohs(cur_stream->dport));
  fflush(stdout);

  if (cur_stream->sndvar->sndbuf) {
    ProcessACK(mtcp, cur_stream, cur_ts, tcph, seq, ack_seq, window,
               payloadlen);
  }
}
/*----------------------------------------------------------------------------*/
static inline void Handle_TCP_ST_LAST_ACK(mtcp_manager_t mtcp, uint32_t cur_ts,
                                          const struct iphdr *iph, int ip_len,
                                          tcp_stream *cur_stream,
                                          struct tcphdr *tcph, uint32_t seq,
                                          uint32_t ack_seq, int payloadlen,
                                          uint16_t window) {
  int ret;
  int group, group_idx, elem_idx;
  if (TCP_SEQ_LT(seq, cur_stream->rcv_nxt)) {
    fprintf(stdout, "%s %d lcore:%d sport:%u error in Handle_TCP_ST_LAST_ACK\n",
            __FILE__, __LINE__, mtcp->ctx->cpu, ntohs(cur_stream->dport));
    fflush(stdout);

    TRACE_DBG(
        "Stream %d (TCP_ST_LAST_ACK): "
        "weird seq: %u, expected: %u\n",
        cur_stream->id, seq, cur_stream->rcv_nxt);
    return;
  }

  if (tcph->ack) {
    if (cur_stream->sndvar->sndbuf) {
      ProcessACK(mtcp, cur_stream, cur_ts, tcph, seq, ack_seq, window,
                 payloadlen);
    }

    if (!cur_stream->sndvar->is_fin_sent) {
      /* the case that FIN is not sent yet */
      /* this is not ack for FIN, ignore */
      TRACE_DBG(
          "Stream %d (TCP_ST_LAST_ACK): "
          "No FIN sent yet.\n",
          cur_stream->id);
#ifdef DBGMSG
      DumpIPPacket(mtcp, iph, ip_len);
#endif
#if DUMP_STREAM
      DumpStream(mtcp, cur_stream);
      DumpControlList(mtcp, mtcp->n_sender[0]);
#endif
      return;
    }

    /* check if ACK of FIN */
    if (ack_seq == cur_stream->sndvar->fss + 1) {
      cur_stream->sndvar->snd_una++;
      UpdateRetransmissionTimer(mtcp, cur_stream, cur_ts);
      cur_stream->state = TCP_ST_CLOSED;
      cur_stream->close_reason = TCP_PASSIVE_CLOSE;
      TRACE_STATE("Stream %d: TCP_ST_CLOSED\n", cur_stream->id);
#if PROFILING
      fprintf(stdout,
              "%s %d lcore:%d sport:%d ackseq:%u cksum:0x%x fss:%u snd_nxt:%u "
              "DestroyTCPStream\n",
              __FILE__, __LINE__, mtcp->ctx->cpu, ntohs(cur_stream->dport),
              ack_seq, ntohs(tcph->check), cur_stream->sndvar->fss,
              cur_stream->snd_nxt);
      fflush(stdout);
#endif

      group = (cur_stream->soft_id / 64) / 64;
      group_idx = (cur_stream->soft_id / 64) % 64;
      elem_idx = cur_stream->soft_id % 64;

      pthread_spin_lock(&fdir_lock);
      #if 0
      fprintf(stdout, "existing flow num: %lu, del one flow\n", num_conn);
      fflush(stdout);
      #endif
      ret = fdir_del_perfect_filter(cur_stream->nic_id, cur_stream->soft_id,
                                    mtcp->ctx->cpu, cur_stream->saddr,
                                    cur_stream->sport, cur_stream->daddr,
                                    cur_stream->dport);
      if(!ret)
      num_conn--;
#if 0
      fprintf(stdout, "del, fdir_group:%d group_idx:%d elem_idx:%d\n", group,
              group_idx, elem_idx);
      fflush(stdout);
#endif
      fdir_group[cur_stream->nic_id][group] |= ((uint64_t)1 << group_idx);
      fdir_elem[cur_stream->nic_id][group * 64 + group_idx] |=
          ((uint64_t)1 << elem_idx);
      pthread_spin_unlock(&fdir_lock);
      if (ret < 0)
        fprintf(stderr,
                "[thread: %d sport:%u] failed to delete signature filter:%s\n",
                mtcp->ctx->cpu, htons(cur_stream->dport), strerror(-ret));
      else {
        fprintf(stdout, "%s %d del filter saddr:0x%x sport:%u\n", __FILE__,
                __LINE__, htonl(cur_stream->daddr), htons(cur_stream->dport));
        fflush(stdout);
      }
      /*
      ret = fdir_del_signature_filter(0, mtcp->ctx->cpu, cur_stream->saddr, \
              cur_stream->sport, cur_stream->daddr, cur_stream->dport);
              if (ret < 0)
              fprintf(stderr, "[thread: %d sport:%u] failed to delete signature
      filter:%s\n", mtcp->ctx->cpu, htons(cur_stream->dport), strerror(-ret));
              else
                      printf("%s %d del filter saddr:0x%x sport:%u\n", __FILE__,
      __LINE__,\ htonl(cur_stream->daddr), htons(cur_stream->dport));
                                      */

    } else {
      TRACE_DBG(
          "Stream %d (TCP_ST_LAST_ACK): Not ACK of FIN. "
          "ack_seq: %u, expected: %u, snd_nxt:%u\n",
          cur_stream->id, ack_seq, cur_stream->sndvar->fss + 1,
          cur_stream->snd_nxt);
      // cur_stream->id, ack_seq, cur_stream->sndvar->fss + 1);
      // cur_stream->snd_nxt = cur_stream->sndvar->fss;
#if PROFILING
      fprintf(stdout,
              "%s %d lcore:%d sport:%u (TCP_ST_LAST_ACK): Not ACK of FIN. "
              "ack_seq: %u, expected: %u, snd_nxt:%u fss:%u \n",
              __FILE__, __LINE__, mtcp->ctx->cpu, ntohs(cur_stream->dport),
              ack_seq, cur_stream->sndvar->fss + 1, cur_stream->snd_nxt,
              cur_stream->sndvar->fss);
      fflush(stdout);
#endif

      // AddtoControlList(mtcp, cur_stream, cur_ts);
    }
    DestroyTCPStream(mtcp, cur_stream);
  } else {
    CTRACE_ERROR("Stream %d (TCP_ST_LAST_ACK): No ACK\n", cur_stream->id);
    // cur_stream->snd_nxt = cur_stream->sndvar->fss;
    fprintf(
        stdout,
        "%s %d lcore:%d sport:%u in Handle_TCP_ST_LAST_ACK AddtoControlList\n",
        __FILE__, __LINE__, mtcp->ctx->cpu, ntohs(cur_stream->dport));
    fflush(stdout);

    AddtoControlList(mtcp, cur_stream, cur_ts);
  }
}
/*----------------------------------------------------------------------------*/
static inline void Handle_TCP_ST_FIN_WAIT_1(mtcp_manager_t mtcp,
                                            uint32_t cur_ts,
                                            tcp_stream *cur_stream,
                                            struct tcphdr *tcph, uint32_t seq,
                                            uint32_t ack_seq, uint8_t *payload,
                                            int payloadlen, uint16_t window) {
  if (TCP_SEQ_LT(seq, cur_stream->rcv_nxt)) {
    TRACE_DBG(
        "Stream %d (TCP_ST_LAST_ACK): "
        "weird seq: %u, expected: %u\n",
        cur_stream->id, seq, cur_stream->rcv_nxt);

    AddtoControlList(mtcp, cur_stream, cur_ts);
    return;
  }

  if (tcph->ack) {
    if (cur_stream->sndvar->sndbuf) {
      ProcessACK(mtcp, cur_stream, cur_ts, tcph, seq, ack_seq, window,
                 payloadlen);
    }

    if (cur_stream->sndvar->is_fin_sent &&
        ack_seq == cur_stream->sndvar->fss + 1) {
      cur_stream->sndvar->snd_una = ack_seq;
      if (TCP_SEQ_GT(ack_seq, cur_stream->snd_nxt)) {
        TRACE_DBG("Stream %d: update snd_nxt to %u\n", cur_stream->id, ack_seq);
        cur_stream->snd_nxt = ack_seq;
      }
      // cur_stream->sndvar->snd_una++;
      // UpdateRetransmissionTimer(mtcp, cur_stream, cur_ts);
      cur_stream->sndvar->nrtx = 0;
      RemoveFromRTOList(mtcp, cur_stream);
      cur_stream->state = TCP_ST_FIN_WAIT_2;
      TRACE_STATE("Stream %d: TCP_ST_FIN_WAIT_2\n", cur_stream->id);
    }

  } else {
    TRACE_DBG("Stream %d: does not contain an ack!\n", cur_stream->id);
    return;
  }

  if (payloadlen > 0) {
    if (ProcessTCPPayload(mtcp, cur_stream, cur_ts, payload, seq, payloadlen)) {
      /* if return is TRUE, send ACK */
      EnqueueACK(mtcp, cur_stream, cur_ts, ACK_OPT_AGGREGATE);
    } else {
      EnqueueACK(mtcp, cur_stream, cur_ts, ACK_OPT_NOW);
    }
  }

  if (tcph->fin) {
    /* process the FIN only if the sequence is valid */
    /* FIN packet is allowed to push payload (should we check for PSH flag)? */
    if (seq + payloadlen == cur_stream->rcv_nxt) {
      cur_stream->rcv_nxt++;

      if (cur_stream->state == TCP_ST_FIN_WAIT_1) {
        cur_stream->state = TCP_ST_CLOSING;
        TRACE_STATE("Stream %d: TCP_ST_CLOSING\n", cur_stream->id);

      } else if (cur_stream->state == TCP_ST_FIN_WAIT_2) {
        cur_stream->state = TCP_ST_TIME_WAIT;
        TRACE_STATE("Stream %d: TCP_ST_TIME_WAIT\n", cur_stream->id);
        AddtoTimewaitList(mtcp, cur_stream, cur_ts);
      }
      AddtoControlList(mtcp, cur_stream, cur_ts);
    }
  }
}
/*----------------------------------------------------------------------------*/
static inline void Handle_TCP_ST_FIN_WAIT_2(mtcp_manager_t mtcp,
                                            uint32_t cur_ts,
                                            tcp_stream *cur_stream,
                                            struct tcphdr *tcph, uint32_t seq,
                                            uint32_t ack_seq, uint8_t *payload,
                                            int payloadlen, uint16_t window) {
  if (tcph->ack) {
    if (cur_stream->sndvar->sndbuf) {
      ProcessACK(mtcp, cur_stream, cur_ts, tcph, seq, ack_seq, window,
                 payloadlen);
    }
  } else {
    TRACE_DBG("Stream %d: does not contain an ack!\n", cur_stream->id);
    return;
  }

  if (payloadlen > 0) {
    if (ProcessTCPPayload(mtcp, cur_stream, cur_ts, payload, seq, payloadlen)) {
      /* if return is TRUE, send ACK */
      EnqueueACK(mtcp, cur_stream, cur_ts, ACK_OPT_AGGREGATE);
    } else {
      EnqueueACK(mtcp, cur_stream, cur_ts, ACK_OPT_NOW);
    }
  }

  if (tcph->fin) {
    /* process the FIN only if the sequence is valid */
    /* FIN packet is allowed to push payload (should we check for PSH flag)? */
    if (seq + payloadlen == cur_stream->rcv_nxt) {
      cur_stream->state = TCP_ST_TIME_WAIT;
      cur_stream->rcv_nxt++;
      TRACE_STATE("Stream %d: TCP_ST_TIME_WAIT\n", cur_stream->id);
      fprintf(stdout, "Stream %d: TCP_ST_TIME_WAIT\n", cur_stream->id);
      AddtoTimewaitList(mtcp, cur_stream, cur_ts);
      AddtoControlList(mtcp, cur_stream, cur_ts);
    }
#if 0
	} else {
		TRACE_DBG("Stream %d (TCP_ST_FIN_WAIT_2): No FIN. "
				"seq: %u, ack_seq: %u, snd_nxt: %u, snd_una: %u\n", 
				cur_stream->id, seq, ack_seq, 
				cur_stream->snd_nxt, cur_stream->sndvar->snd_una);
#if DBGMSG
		DumpIPPacket(mtcp, iph, ip_len);
#endif
#endif
  }
}
/*----------------------------------------------------------------------------*/
static inline void Handle_TCP_ST_CLOSING(mtcp_manager_t mtcp, uint32_t cur_ts,
                                         tcp_stream *cur_stream,
                                         struct tcphdr *tcph, uint32_t seq,
                                         uint32_t ack_seq, int payloadlen,
                                         uint16_t window) {
  if (tcph->ack) {
    if (cur_stream->sndvar->sndbuf) {
      ProcessACK(mtcp, cur_stream, cur_ts, tcph, seq, ack_seq, window,
                 payloadlen);
    }

    if (!cur_stream->sndvar->is_fin_sent) {
      TRACE_DBG(
          "Stream %d (TCP_ST_CLOSING): "
          "No FIN sent yet.\n",
          cur_stream->id);
      return;
    }

    // check if ACK of FIN
    if (ack_seq != cur_stream->sndvar->fss + 1) {
#if 0
			CTRACE_ERROR("Stream %d (TCP_ST_CLOSING): Not ACK of FIN. "
					"ack_seq: %u, snd_nxt: %u, snd_una: %u, fss: %u\n", 
					cur_stream->id, ack_seq, cur_stream->snd_nxt, 
					cur_stream->sndvar->snd_una, cur_stream->sndvar->fss);
			DumpIPPacketToFile(stderr, iph, ip_len);
			DumpStream(mtcp, cur_stream);
#endif
      // assert(0);
      /* if the packet is not the ACK of FIN, ignore */
      return;
    }

    cur_stream->sndvar->snd_una = ack_seq;
    cur_stream->snd_nxt = ack_seq;
    UpdateRetransmissionTimer(mtcp, cur_stream, cur_ts);

    cur_stream->state = TCP_ST_TIME_WAIT;
    TRACE_STATE("Stream %d: TCP_ST_TIME_WAIT\n", cur_stream->id);

    AddtoTimewaitList(mtcp, cur_stream, cur_ts);

  } else {
    CTRACE_ERROR("Stream %d (TCP_ST_CLOSING): Not ACK\n", cur_stream->id);
    return;
  }
}
/*----------------------------------------------------------------------------*/

//
int ProcessTCPPacket(mtcp_manager_t mtcp, uint32_t cur_ts, const int ifidx,
                     const struct iphdr *iph, int ip_len) {
  struct tcphdr *tcph = (struct tcphdr *)((u_char *)iph + (iph->ihl << 2));
  uint8_t *payload = (uint8_t *)tcph + (tcph->doff << 2);

  int payloadlen = ip_len - (payload - (u_char *)iph);
  // printf("payloadlen:%u\n", payloadlen);
  tcp_stream s_stream;
  tcp_stream *cur_stream = NULL;
  uint32_t seq = ntohl(tcph->seq);
  uint32_t ack_seq = ntohl(tcph->ack_seq);
  uint16_t window = ntohs(tcph->window);
  uint16_t check;
  int ret;
  int rc = -1;
  struct timeval now;

  /* Check ip packet invalidation */
  if (ip_len < ((iph->ihl + tcph->doff) << 2)) {
    return ERROR;
  }

#if VERIFY_RX_CHECKSUM
#ifndef DISABLE_HWCSUM
  if (mtcp->iom->dev_ioctl != NULL)
    rc = mtcp->iom->dev_ioctl(mtcp->ctx, ifidx, PKT_RX_TCP_CSUM, NULL);
#endif
  if (rc == -1) {
    check = TCPCalcChecksum((uint16_t *)tcph, (tcph->doff << 2) + payloadlen,
                            iph->saddr, iph->daddr);
    if (check) {
      TRACE_DBG(
          "Checksum Error: Original: 0x%04x, calculated: 0x%04x\n", tcph->check,
          TCPCalcChecksum((uint16_t *)tcph, (tcph->doff << 2) + payloadlen,
                          iph->saddr, iph->daddr));
      tcph->check = 0;
      return ERROR;
    }
  }
#endif

#if defined(NETSTAT) && defined(ENABLELRO)
  mtcp->nstat.rx_gdptbytes += payloadlen;
#endif /* NETSTAT */

  s_stream.saddr = iph->daddr;
  s_stream.sport = tcph->dest;
  s_stream.daddr = iph->saddr;
  s_stream.dport = tcph->source;

  if (!(cur_stream = StreamHTSearch(mtcp->tcp_flow_table, &s_stream))) {
    /* not found in flow table */
    cur_stream = CreateNewFlowHTEntry(mtcp, cur_ts, iph, ip_len, tcph, seq,
                                      ack_seq, payloadlen, window);
    if (!cur_stream) {
      return TRUE;
    }
  }

  /* Validate sequence. if not valid, ignore the packet */
  if (cur_stream->state > TCP_ST_SYN_RCVD) {
    ret = ValidateSequence(mtcp, cur_stream, cur_ts, tcph, seq, ack_seq,
                           payloadlen);
    if (!ret) {
      TRACE_DBG("Stream %d: Unexpected sequence: %u, expected: %u\n",
                cur_stream->id, seq, cur_stream->rcv_nxt);

#if PROFILING
      fprintf(stdout,
              "%s %d lcore:%d sport:%u: state:%d Unexpected sequence: %u, "
              "expected(rcv_nxt): %u, cksum:0x%x fss:%u\n",
              __FILE__, __LINE__, mtcp->ctx->cpu, ntohs(cur_stream->dport),
              cur_stream->state, seq, cur_stream->rcv_nxt, ntohs(tcph->check),
              cur_stream->sndvar->fss);
      fflush(stdout);
#endif

#ifdef DBGMSG
      DumpIPPacket(mtcp, iph, ip_len);
#endif
#ifdef DUMP_STREAM
      DumpStream(mtcp, cur_stream);
#endif
      return TRUE;
    }
  }

  /* Update receive window size */
  if (tcph->syn) {
    cur_stream->sndvar->peer_wnd = window;
  } else {
    cur_stream->sndvar->peer_wnd = (uint32_t)window
                                   << cur_stream->sndvar->wscale_peer;
  }

  cur_stream->last_active_ts = cur_ts;
  UpdateTimeoutList(mtcp, cur_stream);

  gettimeofday(&now, NULL);

#if 0
  if ((now.tv_sec - cur_stream->last_recv.tv_sec)*1000 + \
  (now.tv_usec - cur_stream->last_recv.tv_usec)/1000.0 >= 0.4 && (now.tv_sec - cur_stream->last_recv.tv_sec)*1000 + \
  (now.tv_usec - cur_stream->last_recv.tv_usec)/1000.0 < 3 && cur_stream->state == TCP_ST_ESTABLISHED)
  printf("worker:%u src_port:%u poll_gap:%lu active_poll_gap:%lu poll_time_gap:%f active_poll_time:%f poll_time_gap2:%f active_poll_time2:%f flow_recv_time:%f\n", \
  mtcp->ctx->cpu, ntohs(cur_stream->dport), (mtcp->ctx->poll_id - cur_stream->last_poll_id), \
  (mtcp->ctx->active_poll_id - cur_stream->last_active_poll_id), \
  (now.tv_sec - mtcp->ctx->last_poll_time_point[0].tv_sec)*1000 + \
  (now.tv_usec - mtcp->ctx->last_poll_time_point[0].tv_usec)/1000.0, \
  (now.tv_sec - mtcp->ctx->last_poll_time_point[1].tv_sec)*1000 + \
  (now.tv_usec - mtcp->ctx->last_poll_time_point[1].tv_usec)/1000.0, 
  (now.tv_sec - mtcp->ctx->last_last_poll_time_point[0].tv_sec)*1000 + \
  (now.tv_usec - mtcp->ctx->last_last_poll_time_point[0].tv_usec)/1000.0, \
  (now.tv_sec - mtcp->ctx->last_last_poll_time_point[1].tv_sec)*1000 + \
  (now.tv_usec - mtcp->ctx->last_last_poll_time_point[1].tv_usec)/1000.0, 
  (now.tv_sec - cur_stream->last_recv.tv_sec)*1000 + \
  (now.tv_usec - cur_stream->last_recv.tv_usec)/1000.0);
#endif

  cur_stream->last_recv = now;

  cur_stream->last_poll_id = mtcp->ctx->active_poll_id;
  cur_stream->last_active_poll_id = mtcp->ctx->active_poll_id;

  /* Process RST: process here only if state > TCP_ST_SYN_SENT */
  if (tcph->rst) {
    cur_stream->have_reset = TRUE;
    if (cur_stream->state > TCP_ST_SYN_SENT) {
      printf("in processRST state:%d sport:%u\n", cur_stream->state,
             ntohs(tcph->source));
      if (ProcessRST(mtcp, cur_stream, ack_seq)) {
        return TRUE;
      }
    }
  }

  switch (cur_stream->state) {
    case TCP_ST_LISTEN:
      cur_stream->nic_id = ifidx;
      Handle_TCP_ST_LISTEN(mtcp, cur_ts, cur_stream, tcph);
      break;

    case TCP_ST_SYN_SENT:
      Handle_TCP_ST_SYN_SENT(mtcp, cur_ts, cur_stream, iph, tcph, seq, ack_seq,
                             payloadlen, window);
      break;

    case TCP_ST_SYN_RCVD:
      /* SYN retransmit implies our SYN/ACK was lost. Resend */
      if (tcph->syn && seq == cur_stream->rcvvar->irs)
        Handle_TCP_ST_LISTEN(mtcp, cur_ts, cur_stream, tcph);
      else {
        Handle_TCP_ST_SYN_RCVD(mtcp, cur_ts, cur_stream, tcph, ack_seq);
        if (payloadlen > 0 && cur_stream->state == TCP_ST_ESTABLISHED) {
          Handle_TCP_ST_ESTABLISHED(mtcp, cur_ts, cur_stream, tcph, seq,
                                    ack_seq, payload, payloadlen, window);
        }
      }
      break;

    case TCP_ST_ESTABLISHED:
#if LATENCY_DUMP
      gettimeofday(&cur_stream->timestamp[0], NULL);
#endif
      Handle_TCP_ST_ESTABLISHED(mtcp, cur_ts, cur_stream, tcph, seq, ack_seq,
                                payload, payloadlen, window);
      break;

    case TCP_ST_CLOSE_WAIT:
      Handle_TCP_ST_CLOSE_WAIT(mtcp, cur_ts, cur_stream, tcph, seq, ack_seq,
                               payloadlen, window);
      break;

    case TCP_ST_LAST_ACK:
      Handle_TCP_ST_LAST_ACK(mtcp, cur_ts, iph, ip_len, cur_stream, tcph, seq,
                             ack_seq, payloadlen, window);
      break;

    case TCP_ST_FIN_WAIT_1:
      Handle_TCP_ST_FIN_WAIT_1(mtcp, cur_ts, cur_stream, tcph, seq, ack_seq,
                               payload, payloadlen, window);
      break;

    case TCP_ST_FIN_WAIT_2:
      Handle_TCP_ST_FIN_WAIT_2(mtcp, cur_ts, cur_stream, tcph, seq, ack_seq,
                               payload, payloadlen, window);
      break;

    case TCP_ST_CLOSING:
      Handle_TCP_ST_CLOSING(mtcp, cur_ts, cur_stream, tcph, seq, ack_seq,
                            payloadlen, window);
      break;

    case TCP_ST_TIME_WAIT:
      /* the only thing that can arrive in this state is a retransmission
         of the remote FIN. Acknowledge it, and restart the 2 MSL timeout */
      if (cur_stream->on_timewait_list) {
        RemoveFromTimewaitList(mtcp, cur_stream);
        AddtoTimewaitList(mtcp, cur_stream, cur_ts);
      }

      AddtoControlList(mtcp, cur_stream, cur_ts);
      break;

    case TCP_ST_CLOSED:
      break;
  }

  return TRUE;
}
