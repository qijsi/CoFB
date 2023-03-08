/* for io_module_func def'ns */
#include "io_module.h"
#ifndef DISABLE_DPDK
/* for mtcp related def'ns */
#include "mtcp.h"
/* for errno */
#include <errno.h>
/* for logging */
#include "debug.h"
/* for num_devices_* */
#include "config.h"
/* for rte_max_eth_ports */
#include <rte_common.h>
/* for rte_eth_rxconf */
#include <rte_ethdev.h>
/* for delay funcs */
#include <rte_cycles.h>
#include <rte_errno.h>


#define ENABLE_STATS_IOCTL 1
#ifdef ENABLE_STATS_IOCTL
/* for close */
#include <unistd.h>
/* for open */
#include <fcntl.h>
/* for ioctl */
#include <sys/ioctl.h>
#endif /* !ENABLE_STATS_IOCTL */
/* for ip pseudo-chksum */
#include <rte_ip.h>
#include <rte_tcp.h>
//#define IP_DEFRAG			1
#ifdef IP_DEFRAG
/* for ip defragging */
#include <rte_ip_frag.h>
#endif
/* for ioctl funcs */
#include <dpdk_iface_common.h>
/* for retrieving rte version(s) */
#include <rte_version.h>

#include "apc.h"
/*----------------------------------------------------------------------------*/
/* Essential macros */
#define MAX_RX_QUEUE_PER_LCORE MAX_CPUS
#define MAX_TX_QUEUE_PER_PORT MAX_CPUS

#ifdef ENABLELRO
#define BUF_SIZE 32768
#else
#define BUF_SIZE 2048
#endif /* !ENABLELRO */
#define MBUF_SIZE (BUF_SIZE + sizeof(struct rte_mbuf) + RTE_PKTMBUF_HEADROOM)
#define NB_MBUF 8192
#define MEMPOOL_CACHE_SIZE 256
#ifdef ENFORCE_RX_IDLE
#define RX_IDLE_ENABLE 1
#define RX_IDLE_TIMEOUT 1 /* in micro-seconds */
#endif

/*
 * RX and TX Prefetch, Host, and Write-back threshold values should be
 * carefully set for optimal performance. Consult the network
 * controller's datasheet and supporting DPDK documentation for guidance
 * on how these parameters should be set.
 */
#define RX_PTHRESH 128 /**< Default values of RX prefetch threshold reg. */
#define RX_HTHRESH 128 /**< Default values of RX host threshold reg. */
#define RX_WTHRESH 128 /**< Default values of RX write-back threshold reg. */

/*
 * These default values are optimized for use with the Intel(R) 82599 10 GbE
 * Controller and the DPDK ixgbe PMD. Consider using other values for other
 * network controllers and/or network drivers.
 */
#define TX_PTHRESH 36 /**< Default values of TX prefetch threshold reg. */
#define TX_HTHRESH 0  /**< Default values of TX host threshold reg. */
#define TX_WTHRESH 0  /**< Default values of TX write-back threshold reg. */

#define MAX_PKT_BURST 4096 /*128*/

/*
 * Configurable number of RX/TX ring descriptors
 */
#define RTE_DPLANE_MASTER_DESC 1024
#define RTE_TEST_RX_DESC_DEFAULT 4096
#define RTE_TEST_TX_DESC_DEFAULT 1024

/*
 * Ethernet frame overhead
 */

#define ETHER_IFG 12
#define ETHER_PREAMBLE 8
#if RTE_VERSION < RTE_VERSION_NUM(19, 8, 0, 0)
#define ETHER_OVR (ETHER_CRC_LEN + ETHER_PREAMBLE + ETHER_IFG)
#else
#define ETHER_OVR (RTE_ETHER_CRC_LEN + ETHER_PREAMBLE + ETHER_IFG)
#endif

#ifdef ENABLE_GDP
gdp_flowtable *flowtable;
priority_queue *queues;
#endif


static const uint16_t nb_rxd = RTE_TEST_RX_DESC_DEFAULT;
static const uint16_t nb_txd = RTE_TEST_TX_DESC_DEFAULT;
/*----------------------------------------------------------------------------*/
/* packet memory pools for storing packet bufs */
static struct rte_mempool *pktmbuf_pool[MAX_CPUS] = {NULL};


//#define DEBUG				1
#ifdef DEBUG
/* ethernet addresses of ports */
static struct ether_addr ports_eth_addr[RTE_MAX_ETHPORTS];
#endif

static struct rte_eth_dev_info dev_info[RTE_MAX_ETHPORTS];

static struct rte_eth_conf port_conf = {
    .lpbk_mode = 0,
    .rxmode =
        {
            .mq_mode = ETH_MQ_RX_NONE,
//   .mq_mode = ETH_MQ_RX_NONE,
#if RTE_VERSION < RTE_VERSION_NUM(19, 8, 0, 0)
            .max_rx_pkt_len = ETHER_MAX_LEN,
#else
            .max_rx_pkt_len = RTE_ETHER_MAX_LEN,
#endif

#if RTE_VERSION > RTE_VERSION_NUM(17, 8, 0, 0)
            .offloads = (
#if RTE_VERSION < RTE_VERSION_NUM(18, 5, 0, 0)
                DEV_RX_OFFLOAD_CRC_STRIP |
#endif /* !18.05 */
                DEV_RX_OFFLOAD_CHECKSUM
#ifdef ENABLELRO
                | DEV_RX_OFFLOAD_TCP_LRO
#endif
                ),
#endif /* !17.08 */
            .split_hdr_size = 0,
#if RTE_VERSION < RTE_VERSION_NUM(18, 5, 0, 0)
            .header_split = 0,   /**< Header Split disabled */
            .hw_ip_checksum = 1, /**< IP checksum offload enabled */
            .hw_vlan_filter = 0, /**< VLAN filtering disabled */
            .jumbo_frame = 0,    /**< Jumbo Frame Support disabled */
            .hw_strip_crc = 1,   /**< CRC stripped by hardware */
#endif                           /* !18.05 */
#ifdef ENABLELRO
            .enable_lro = 1, /**< Enable LRO */
#endif
        },
#if 0
	.rx_adv_conf = {
		.rss_conf = {
			.rss_key = 	NULL,
			.rss_hf = 	ETH_RSS_TCP | ETH_RSS_UDP |
					ETH_RSS_IP | ETH_RSS_L2_PAYLOAD
		},
	},
#endif
    .fdir_conf = {.mode = RTE_FDIR_MODE_PERFECT,
                  .pballoc = RTE_FDIR_PBALLOC_64K,
                  .status = RTE_FDIR_NO_REPORT_STATUS,
                  //.status = RTE_FDIR_REPORT_STATUS,
                  .mask.vlan_tci_mask = 0x0,
                  .mask.ipv4_mask.src_ip = 0xFFFFFFFF,
                  .mask.ipv4_mask.dst_ip = 0xFFFFFFFF,
                  .mask.src_port_mask = 0xFFFF,
                  .mask.dst_port_mask = 0xFFFF,
                  .mask.mac_addr_byte_mask = 0x0,
                  .mask.tunnel_type_mask = 0x0,
                  .mask.tunnel_id_mask = 0x0,
                  .drop_queue = 127,
                  .flex_conf.nb_payloads = 0x0,
                  .flex_conf.nb_flexmasks = 0x0},
    .txmode =
        {
            .mq_mode = ETH_MQ_TX_NONE,
#if RTE_VERSION >= RTE_VERSION_NUM(18, 5, 0, 0)
#if 0
               .offloads = (DEV_TX_OFFLOAD_IPV4_CKSUM |
                            DEV_TX_OFFLOAD_UDP_CKSUM | DEV_TX_OFFLOAD_TCP_CKSUM)
#endif
#endif
        },
};

static const struct rte_eth_rxconf rx_conf = {
    .rx_thresh =
        {
            .pthresh = RX_PTHRESH, /* RX prefetch threshold reg */
            .hthresh = RX_HTHRESH, /* RX host threshold reg */
            .wthresh = RX_WTHRESH, /* RX write-back threshold reg */
        },
    .rx_free_thresh = 128,
    //  .rx_free_thresh = 32,
    .rx_drop_en = 0};

static const struct rte_eth_txconf tx_conf = {
    .tx_thresh =
        {
            .pthresh = TX_PTHRESH, /* TX prefetch threshold reg */
            .hthresh = TX_HTHRESH, /* TX host threshold reg */
            .wthresh = TX_WTHRESH, /* TX write-back threshold reg */
        },
    .tx_free_thresh = 0, /* Use PMD default values */
    .tx_rs_thresh = 0,   /* Use PMD default values */
#if RTE_VERSION < RTE_VERSION_NUM(18, 5, 0, 0)
    /*
     * As the example won't handle mult-segments and offload cases,
     * set the flag by default.
     */
    .txq_flags = 0x0,
#endif
    .offloads = 0x10000};

struct mbuf_table {
  uint16_t len; /* length of queued packets */
  struct rte_mbuf *m_table[MAX_PKT_BURST];
};

struct dpdk_private_context {
  struct mbuf_table rmbufs[RTE_MAX_ETHPORTS];
  struct mbuf_table wmbufs[RTE_MAX_ETHPORTS];
  struct rte_mempool *pktmbuf_pool;

  struct rte_mbuf *pkts_burst[MAX_PKT_BURST];
#ifdef RX_IDLE_ENABLE
  uint8_t rx_idle;
#endif
#ifdef IP_DEFRAG
  struct rte_ip_frag_tbl *frag_tbl;
  struct rte_ip_frag_death_row death_row;
#endif
#ifdef ENABLELRO
  struct rte_mbuf *cur_rx_m;
#endif
#ifdef ENABLE_STATS_IOCTL
  int fd;
  uint32_t cur_ts;
#endif /* !ENABLE_STATS_IOCTL */
} __rte_cache_aligned;

#ifdef ENABLE_STATS_IOCTL
/**
 * stats struct passed on from user space to the driver
 */
struct stats_struct {
  uint64_t tx_bytes;
  uint64_t tx_pkts;
  uint64_t rx_bytes;
  uint64_t rx_pkts;
  uint64_t rmiss;
  uint64_t rerr;
  uint64_t terr;
  uint8_t qid;
  uint8_t dev;
};
#endif /* !ENABLE_STATS_IOCTL */

#ifdef IP_DEFRAG
/* Should be power of two. */
#define IP_FRAG_TBL_BUCKET_ENTRIES 16
#define RTE_LOGTYPE_IP_RSMBL RTE_LOGTYPE_USER1
#define MAX_FRAG_NUM RTE_LIBRTE_IP_FRAG_MAX_FRAG
#endif /* !IP_DEFRAG */
/*----------------------------------------------------------------------------*/
void dpdk_init_handle(struct mtcp_thread_context *ctxt) {
  struct dpdk_private_context *dpc;
  int i, j;
  char mempool_name[RTE_MEMPOOL_NAMESIZE];

  /* create and initialize private I/O module context */
  ctxt->io_private_context = calloc(1, sizeof(struct dpdk_private_context));
  if (ctxt->io_private_context == NULL) {
    TRACE_ERROR(
        "Failed to initialize ctxt->io_private_context: "
        "Can't allocate memory\n");
    exit(EXIT_FAILURE);
  }

  sprintf(mempool_name, "mbuf_pool-%d", ctxt->cpu);
  dpc = (struct dpdk_private_context *)ctxt->io_private_context;
  dpc->pktmbuf_pool = pktmbuf_pool[ctxt->cpu];

  /* set wmbufs correctly */
  fprintf(stdout, "num_devices_attached:%u\n", num_devices_attached);
  fflush(stdout);
  for (j = 0; j < num_devices_attached; j++) {
    /* Allocate wmbufs for each registered port */
    for (i = 0; i < MAX_PKT_BURST/2; i++) {
      dpc->wmbufs[j].m_table[i] = rte_pktmbuf_alloc(pktmbuf_pool[ctxt->cpu]);
      if (dpc->wmbufs[j].m_table[i] == NULL) {
        TRACE_ERROR("Failed to allocate %d:wmbuf[%d] on device %d!\n",
                    ctxt->cpu, i, j);
        fprintf(stdout, "Failed to allocate %d:wmbuf[%d] on device %d!\n",
                ctxt->cpu, i, j);
        fflush(stdout);

        exit(EXIT_FAILURE);
      }
    }
    /* set mbufs queue length to 0 to begin with */
    dpc->wmbufs[j].len = 0;
  }

#ifdef IP_DEFRAG
  int max_flows;
  int socket;
  uint64_t frag_cycles;

  max_flows = CONFIG.max_concurrency / CONFIG.num_cores;
  frag_cycles = (rte_get_tsc_hz() + MS_PER_S - 1) / MS_PER_S * max_flows;
  socket = rte_lcore_to_socket_id(ctxt->cpu);

  if ((dpc->frag_tbl =
           rte_ip_frag_table_create(max_flows, IP_FRAG_TBL_BUCKET_ENTRIES,
                                    max_flows, frag_cycles, socket)) == NULL) {
    RTE_LOG(ERR, IP_RSMBL,
            "ip_frag_tbl_create(%u) on "
            "lcore: %u for queue: %u failed\n",
            max_flows, ctxt->cpu, ctxt->cpu);
    exit(EXIT_FAILURE);
  }
#endif /* !IP_DEFRAG */

#ifdef ENABLE_STATS_IOCTL
  dpc->fd = open(DEV_PATH, O_RDWR);
  if (dpc->fd == -1) {
    TRACE_ERROR("Can't open " DEV_PATH
                " for context->cpu: %d! "
                "Are you using mlx4/mlx5 driver?\n",
                ctxt->cpu);
  }
#endif /* !ENABLE_STATS_IOCTL */
}
/*----------------------------------------------------------------------------*/
int dpdk_link_devices(struct mtcp_thread_context *ctxt) {
  /* linking takes place during mtcp_init() */

  return 0;
}

/*----------------------------------------------------------------------------*/
static int syn_op_filter(int port_id, uint16_t rx_queue, int op) {
  struct rte_eth_syn_filter sfilter;

  memset(&sfilter, 0, sizeof(sfilter));
  sfilter.hig_pri = 1;
  sfilter.queue = rx_queue;

  return rte_eth_dev_filter_ctrl(port_id, RTE_ETH_FILTER_SYN, op, &sfilter);
}

int syn_add_filter(int port_id, uint16_t rx_queue) {
  return syn_op_filter(port_id, rx_queue, RTE_ETH_FILTER_ADD);
}

static int fdir_op_perfect_filter(int port_id, uint16_t soft_id,
                                  uint16_t rx_queue, uint32_t dip,
                                  uint16_t dport, uint32_t sip, uint16_t sport,
                                  int op) {
  struct rte_eth_fdir_filter filter;
#if 0
  struct rte_eth_fdir_filter_info filter_info;
  memset(&filter_info, 0, sizeof(filter_info));
  filter_info.info_type = RTE_ETH_FDIR_FILTER_INPUT_SET_SELECT;
  filter_info.info.input_set_conf.flow_type = RTE_ETH_FLOW_NONFRAG_IPV4_TCP;
  filter_info.info.input_set_conf.field[0] = RTE_ETH_INPUT_SET_NONE;
  filter_info.info.input_set_conf.inset_size = 0;
  filter_info.info.input_set_conf.op = RTE_ETH_INPUT_SET_SELECT;
#endif

  memset(&filter, 0, sizeof(filter));
  filter.soft_id = soft_id;
  filter.input.flow_type = RTE_ETH_FLOW_NONFRAG_IPV4_TCP;
  filter.input.flow.ip4_flow.dst_ip = dip;
  filter.input.flow.ip4_flow.src_ip = sip;
  filter.input.flow.tcp4_flow.dst_port = dport;
  filter.input.flow.tcp4_flow.src_port = sport;
  filter.action.report_status = RTE_ETH_FDIR_REPORT_ID;
  // filter.action.report_status = RTE_ETH_FDIR_REPORT_ID_FLEX_4;
  filter.action.behavior = RTE_ETH_FDIR_ACCEPT;
  filter.action.rx_queue = rx_queue;

#if 0
  rte_eth_dev_filter_ctrl(port_id, RTE_ETH_FILTER_FDIR, RTE_ETH_FILTER_SET,
                          &filter_info);
#endif

  return rte_eth_dev_filter_ctrl(port_id, RTE_ETH_FILTER_FDIR, op, &filter);
}

int fdir_add_perfect_filter(int port_id, uint16_t soft_id, uint16_t rx_queue,
                            uint32_t dip, uint16_t dport, uint32_t sip,
                            uint16_t sport) {
  return fdir_op_perfect_filter(port_id, soft_id, rx_queue, dip, dport, sip,
                                sport, RTE_ETH_FILTER_ADD);
}

int fdir_del_perfect_filter(int port_id, uint16_t soft_id, uint16_t rx_queue,
                            uint32_t dip, uint16_t dport, uint32_t sip,
                            uint16_t sport) {
  return fdir_op_perfect_filter(port_id, soft_id, rx_queue, dip, dport, sip,
                                sport, RTE_ETH_FILTER_DELETE);
}

static int fdir_op_signature_filter(int port_id, uint16_t rx_queue,
                                    uint32_t dip, uint16_t dport, uint32_t sip,
                                    uint16_t sport, int op) {
  struct rte_eth_fdir_filter filter;

  memset(&filter, 0, sizeof(filter));

  filter.input.flow_type = RTE_ETH_FLOW_NONFRAG_IPV4_TCP;
  filter.input.flow.ip4_flow.dst_ip = dip;
  filter.input.flow.ip4_flow.src_ip = sip;
  filter.input.flow.tcp4_flow.dst_port = dport;
  filter.input.flow.tcp4_flow.src_port = sport;
  filter.action.report_status = RTE_ETH_FDIR_REPORT_ID;
  filter.action.behavior = RTE_ETH_FDIR_ACCEPT;
  filter.action.rx_queue = rx_queue;

  return rte_eth_dev_filter_ctrl(port_id, RTE_ETH_FILTER_FDIR, op, &filter);
}

static int fdir_flush_init(int port_id) {
  return rte_eth_dev_filter_ctrl(port_id, RTE_ETH_FILTER_FDIR,
                                 RTE_ETH_FILTER_FLUSH, NULL);
}

int fdir_add_signature_filter(int port_id, uint16_t rx_queue, uint32_t dip,
                              uint16_t dport, uint32_t sip, uint16_t sport) {
  return fdir_op_signature_filter(port_id, rx_queue, dip, dport, sip, sport,
                                  RTE_ETH_FILTER_ADD);
  /*
   struct rte_eth_fdir_filter filter;

   memset(&filter, 0, sizeof(filter));

   filter.input.flow_type = RTE_ETH_FLOW_NONFRAG_IPV4_TCP;
   filter.input.flow.ip4_flow.dst_ip = dip;
   filter.input.flow.ip4_flow.src_ip = sip;
   filter.input.flow.tcp4_flow.dst_port = dport;
   filter.input.flow.tcp4_flow.src_port = sport;
   filter.action.report_status = RTE_ETH_FDIR_REPORT_ID;
   filter.action.behavior = RTE_ETH_FDIR_ACCEPT;
   filter.action.rx_queue = rx_queue;

   return rte_eth_dev_filter_ctrl(port_id, RTE_ETH_FILTER_FDIR,
   RTE_ETH_FILTER_ADD, &filter);
   */
}

int fdir_del_signature_filter(int port_id, uint16_t rx_queue, uint32_t dip,
                              uint16_t dport, uint32_t sip, uint16_t sport) {
  return fdir_op_signature_filter(port_id, rx_queue, dip, dport, sip, sport,
                                  RTE_ETH_FILTER_DELETE);
}

struct rte_eth_fdir_stats fdir_retrieve_stats(int port_id) {
  struct rte_eth_fdir_stats fdir_stat;
  //	struct rte_eth_fdir_info fdir_info;

  //	memset(&fdir_info, 0, sizeof(fdir_info));
  //	rte_eth_dev_filter_ctrl(port_id, RTE_ETH_FILTER_FDIR,
  // RTE_ETH_FILTER_INFO, &fdir_info);
  memset(&fdir_stat, 0, sizeof(fdir_stat));
  rte_eth_dev_filter_ctrl(port_id, RTE_ETH_FILTER_FDIR, RTE_ETH_FILTER_STATS,
                          &fdir_stat);
  return fdir_stat;
}

#if 0
int fdir_add_signature_filter(int port_id, uint16_t rx_queue, uint32_t dip,
                              uint16_t dport, uint32_t sip, uint16_t sport) {
  int ret;
  struct rte_eth_fdir_filter filter;

  memset(&filter, 0, sizeof(filter));

  filter.input.flow_type = RTE_ETH_FLOW_NONFRAG_IPV4_TCP;
  filter.input.flow.ip4_flow.dst_ip = dip;
  filter.input.flow.ip4_flow.src_ip = sip;
  filter.input.flow.tcp4_flow.dst_port = dport;
  filter.input.flow.tcp4_flow.src_port = sport;
  filter.action.report_status = RTE_ETH_FDIR_REPORT_ID;
  filter.action.behavior = RTE_ETH_FDIR_ACCEPT;
  filter.action.rx_queue = rx_queue;

  ret = rte_eth_dev_filter_ctrl(port_id, RTE_ETH_FILTER_FDIR,
                                RTE_ETH_FILTER_ADD, &filter);
  return ret;
}
#endif

/*----------------------------------------------------------------------------*/
void dpdk_release_pkt(struct mtcp_thread_context *ctxt, int ifidx,
                      unsigned char *pkt_data, int len) {
  /*
   * do nothing over here - memory reclamation
   * will take place in dpdk_recv_pkts
   */
}
/*----------------------------------------------------------------------------*/
int dpdk_send_pkts(struct mtcp_thread_context *ctxt, int ifidx) {
  struct dpdk_private_context *dpc;

#ifdef NETSTAT
  mtcp_manager_t mtcp;
#endif
  int ret, i, portid = CONFIG.eths[ifidx].ifindex;

  dpc = (struct dpdk_private_context *)ctxt->io_private_context;
#ifdef NETSTAT
  mtcp = ctxt->mtcp_manager;
#endif
  ret = 0;

  /* if there are packets in the queue... flush them out to the wire */
  if (dpc->wmbufs[ifidx].len > /*= MAX_PKT_BURST*/ 0) {
    struct rte_mbuf **pkts;
#ifdef ENABLE_STATS_IOCTL
#ifdef NETSTAT
    struct rte_eth_stats stats;
    struct stats_struct ss;
#endif
#endif /* !ENABLE_STATS_IOCTL */
    int cnt = dpc->wmbufs[ifidx].len;
    pkts = dpc->wmbufs[ifidx].m_table;
#ifdef NETSTAT
    mtcp->nstat.tx_packets[ifidx] += cnt;
#ifdef ENABLE_STATS_IOCTL
    /* only pass stats after >= 1 sec interval */
    if (abs(mtcp->cur_ts - dpc->cur_ts) >= 1000 && likely(dpc->fd >= 0)) {
      /* rte_get_stats is global func, use only for 1 core */
      if (ctxt->cpu == 0) {
        rte_eth_stats_get(portid, &stats);
        ss.rmiss = stats.imissed;
        ss.rerr = stats.ierrors;
        ss.terr = stats.oerrors;
      } else
        ss.rmiss = ss.rerr = ss.terr = 0;

      ss.tx_pkts = mtcp->nstat.tx_packets[ifidx];
      ss.tx_bytes = mtcp->nstat.tx_bytes[ifidx];
      ss.rx_pkts = mtcp->nstat.rx_packets[ifidx];
      ss.rx_bytes = mtcp->nstat.rx_bytes[ifidx];
      ss.qid = ctxt->cpu;
      ss.dev = portid;
      /* pass the info now */
      if (ioctl(dpc->fd, SEND_STATS, &ss) == -1)
        TRACE_ERROR("Can't update iface stats!\n");
      dpc->cur_ts = mtcp->cur_ts;
      if (ctxt->cpu == 0) rte_eth_stats_reset(portid);
    }
#endif /* !ENABLE_STATS_IOCTL */
#endif
    do {
      /* tx cnt # of packets */
      ret = rte_eth_tx_burst(portid, ctxt->cpu, pkts, cnt);
      pkts += ret;
      cnt -= ret;
      /* if not all pkts were sent... then repeat the cycle */
    } while (cnt > 0);

    /* time to allocate fresh mbufs for the queue */
    for (i = 0; i < dpc->wmbufs[ifidx].len; i++) {
      dpc->wmbufs[ifidx].m_table[i] =
          rte_pktmbuf_alloc(pktmbuf_pool[ctxt->cpu]);
      /* error checking */
      if (unlikely(dpc->wmbufs[ifidx].m_table[i] == NULL)) {
        TRACE_ERROR("Failed to allocate %d:wmbuf[%d] on device %d!\n",
                    ctxt->cpu, i, ifidx);
        fprintf(stdout, "%s %d Failed to allocate %d:wmbuf[%d] on device %d!\n",
                __FILE__, __LINE__, ctxt->cpu, i, ifidx);
        fflush(stdout);

        exit(EXIT_FAILURE);
      }
    }
    /* reset the len of mbufs var after flushing of packets */
    dpc->wmbufs[ifidx].len = 0;
  }

  return ret;
}
/*----------------------------------------------------------------------------*/
uint8_t *dpdk_get_wptr(struct mtcp_thread_context *ctxt, int ifidx,
                       uint16_t pktsize) {
  struct dpdk_private_context *dpc;
#ifdef NETSTAT
  mtcp_manager_t mtcp;
#endif
  struct rte_mbuf *m;
  uint8_t *ptr;
  int len_of_mbuf;

  dpc = (struct dpdk_private_context *)ctxt->io_private_context;
#ifdef NETSTAT
  mtcp = ctxt->mtcp_manager;
#endif

  /* sanity check */
  if (unlikely(dpc->wmbufs[ifidx].len == MAX_PKT_BURST)) return NULL;

  len_of_mbuf = dpc->wmbufs[ifidx].len;
  m = dpc->wmbufs[ifidx].m_table[len_of_mbuf];

  /* retrieve the right write offset */
  ptr = (void *)rte_pktmbuf_mtod(m, struct ether_hdr *);
  m->pkt_len = m->data_len = pktsize;
  m->nb_segs = 1;
  m->next = NULL;

#ifdef NETSTAT
  mtcp->nstat.tx_bytes[ifidx] += pktsize + ETHER_OVR;
#endif

  /* increment the len_of_mbuf var */
  dpc->wmbufs[ifidx].len = len_of_mbuf + 1;

  return (uint8_t *)ptr;
}
/*----------------------------------------------------------------------------*/
static inline void free_pkts(struct rte_mbuf **mtable, unsigned len) {
  int i;

  /* free the freaking packets */
  for (i = 0; i < len; i++) {
    rte_pktmbuf_free(mtable[i]);
    RTE_MBUF_PREFETCH_TO_FREE(mtable[i + 1]);
  }
}

/*----------------------------------------------------------------------------*/
#if 0
static int GetCore(uint64_t x, int a[64]) {
  int i = 0;
  while (x) {
    a[i] = __builtin_ffsll(x) - 1;
    i++;
    x &= (x - 1);
  }
  return i;
}
#endif
/*----------------------------------------------------------------------------*/

static inline bool test_bit(int nr, uint64_t mask) {
  int value;
  asm volatile(
      "bt %2,%1\n\t"
      "sbbl %0,%0"
      : "=r"(value)
      : "m"(mask), "r"(nr));
  return !!(value);
}
/*----------------------------------------------------------------------------*/

// struct gro_status gro_port[RTE_MAX_ETHPORTS];
// uint8_t gro_flush_cycles = GRO_DEFAULT_FLUSH_CYCLES;

int32_t dpdk_recv_pkts(struct mtcp_thread_context *ctxt, int ifidx) {
  struct dpdk_private_context *dpc;
  int ret;
  static int lastpos = 1;
  int num = 0;
  struct rte_mbuf **cur_m;

  //*tmp_mbuf;
  int p = 0;

 dpc = (struct dpdk_private_context *)ctxt->io_private_context;

 

  if (dpc->rmbufs[ifidx].len != 0) {
    free_pkts(dpc->rmbufs[ifidx].m_table, dpc->rmbufs[ifidx].len);
    dpc->rmbufs[ifidx].len = 0;
  }

  // int portid = CONFIG.eths[ifidx].ifindex;
  ret = rte_eth_rx_burst(ifidx, ctxt->cpu, dpc->pkts_burst, MAX_PKT_BURST);
  ctxt->poll_id++;
  ctxt->last_last_poll_time_point[0] = ctxt->last_poll_time_point[0];
  ctxt->last_poll_time_point[0] = ctxt->poll_time_point[0];
  gettimeofday(&ctxt->poll_time_point[0], NULL);
#ifdef RX_IDLE_ENABLE
  dpc->rx_idle = (likely(ret != 0)) ? 0 : dpc->rx_idle + 1;
#endif

  if (ctxt->master) {
    if (ret) {
      cur_m = dpc->pkts_burst;
      p = lastpos;
      while (ret) {
        num = rte_ring_enqueue_burst(conn_ring[p], (void **)cur_m, 1, NULL);
        p++;
        p = RTE_MAX(1, p % CONFIG.num_cores);
        ret -= num;
        cur_m += num;
      }

      // lastpos = RTE_MAX(1, p % CONFIG.num_cores);
      lastpos = p;
    }
  } else {
    dpc->rmbufs[ifidx].len = ret;
    ctxt->active_poll_id++;
    ctxt->last_last_poll_time_point[1] = ctxt->last_poll_time_point[1];
    ctxt->last_poll_time_point[1] = ctxt->poll_time_point[1];
    gettimeofday(&ctxt->poll_time_point[1], NULL);


  }

  return ret;
}
/*----------------------------------------------------------------------------*/
#ifdef IP_DEFRAG
struct rte_mbuf *ip_reassemble(struct dpdk_private_context *dpc,
                               struct rte_mbuf *m) {
  struct ether_hdr *eth_hdr;
  struct rte_ip_frag_tbl *tbl;
  struct rte_ip_frag_death_row *dr;

  /* if packet is IPv4 */
  if (RTE_ETH_IS_IPV4_HDR(m->packet_type)) {
    struct ipv4_hdr *ip_hdr;

    eth_hdr = rte_pktmbuf_mtod(m, struct ether_hdr *);
    ip_hdr = (struct ipv4_hdr *)(eth_hdr + 1);

    /* if it is a fragmented packet, then try to reassemble. */
    if (rte_ipv4_frag_pkt_is_fragmented(ip_hdr)) {
      struct rte_mbuf *mo;

      tbl = dpc->frag_tbl;
      dr = &dpc->death_row;

      /* prepare mbuf: setup l2_len/l3_len. */
      m->l2_len = sizeof(*eth_hdr);
      m->l3_len = sizeof(*ip_hdr);

      /* process this fragment. */
      mo = rte_ipv4_frag_reassemble_packet(tbl, dr, m, rte_rdtsc(), ip_hdr);
      if (mo == NULL) /* no packet to send out. */
        return NULL;

      /* we have our packet reassembled. */
      if (mo != m) m = mo;
    }
  }

  /* if packet isn't IPv4, just accept it! */
  return m;
}
#endif
/*----------------------------------------------------------------------------*/
uint8_t *dpdk_get_mbuf(struct mtcp_thread_context *ctxt, int ifidx, int index) {
  struct dpdk_private_context *dpc;
  struct rte_mbuf *m;

  dpc = (struct dpdk_private_context *)ctxt->io_private_context;
  m = dpc->pkts_burst[index];

  /* verify checksum values from ol_flags */
  if ((m->ol_flags & (PKT_RX_L4_CKSUM_BAD | PKT_RX_IP_CKSUM_BAD)) != 0) {
    TRACE_ERROR(
        "%s(%p, %d, %d): mbuf with invalid checksum: "
        "%p(%lu);\n",
        __func__, ctxt, ifidx, index, m, m->ol_flags);
    return NULL;
  }

  dpc->rmbufs[ifidx].m_table[index] = m;
#ifdef ENABLELRO
  dpc->cur_rx_m = m;
#endif /* ENABLELRO */
  return (uint8_t *)m;
}
/*----------------------------------------------------------------------------*/
uint8_t *dpdk_get_rptr(struct mtcp_thread_context *ctxt, int ifidx, int index,
                       uint16_t *len) {
  struct dpdk_private_context *dpc;
  struct rte_mbuf *m;
  uint8_t *pktbuf=NULL;

#if 0
  struct ether_hdr *e_hdr;
  struct ipv4_hdr *ip_hdr;
  struct tcp_hdr *tcp_hdr;
#endif
  dpc = (struct dpdk_private_context *)ctxt->io_private_context;

  m = dpc->pkts_burst[index];

#if 0
  e_hdr = rte_pktmbuf_mtod(m, struct ether_hdr *);
  ip_hdr = (struct ipv4_hdr *)(e_hdr + 1);
  tcp_hdr = (struct tcp_hdr *)(ip_hdr + 1);
  fprintf(stdout, "worker %u index:%u nb_meg:%u saddr:%x sport:%u seq:%u\n", ctxt->cpu,
          index, m->nb_segs, ntohl(ip_hdr->src_addr), ntohs(tcp_hdr->src_port),
          ntohl(tcp_hdr->sent_seq));
  fflush(stdout);
#endif

#ifdef IP_DEFRAG
  m = ip_reassemble(dpc, m);
#endif

#ifdef ENABLE_GRO
  *len = 0;
  struct rte_mbuf *m_t = m;
  uint8_t *to;

  struct rte_mbuf *ml = rte_pktmbuf_alloc(pktmbuf_pool_large[ctxt->cpu]);
  if (ml==NULL) {
    fprintf(stderr, "worker:%u mbuf_large allocate failed\n", ctxt->cpu);
  }
 else {
  to  = rte_pktmbuf_mtod(ml, uint8_t *);

  while (m_t != NULL) {
    rte_prefetch0(rte_pktmbuf_mtod(m_t, void *));
    memcpy(to+*len, rte_pktmbuf_mtod(m_t, uint8_t *), m_t->data_len);
    *len += m_t->data_len;
    m_t = m_t->next;
  }
  pktbuf = to;
  dpc->rmbufs_large[ifidx].m_table[index] = ml;
 // dpc->rmbufs_large[ifidx].len = *len;
 }
#else
  *len = m->pkt_len;
  rte_prefetch0(rte_pktmbuf_mtod(m, void *));
  pktbuf = rte_pktmbuf_mtod(m, uint8_t *);

  /* enqueue the pkt ptr in mbuf */
  dpc->rmbufs[ifidx].m_table[index] = m;
#endif
  /* verify checksum values from ol_flags */
  if ((m->ol_flags & (PKT_RX_L4_CKSUM_BAD | PKT_RX_IP_CKSUM_BAD)) != 0) {
    TRACE_ERROR(
        "%s(%p, %d, %d): mbuf with invalid checksum: "
        "%p(0x%lx);\n",
        __func__, ctxt, ifidx, index, m, m->ol_flags);
    pktbuf = NULL;
  }
#ifdef ENABLELRO
  dpc->cur_rx_m = m;
#endif /* ENABLELRO */

  return pktbuf;
}

#if 0
uint8_t **dpdk_get_rptrv(struct mtcp_thread_context *ctxt, int ifidx, int index,
                       uint16_t *len) {
  struct dpdk_private_context *dpc;
  struct rte_mbuf *m, *mm;
  uint8_t **pktbuf;

  *len=0;
  int i=0;
#if 0
  struct ether_hdr *e_hdr;
  struct ipv4_hdr *ip_hdr;
  struct tcp_hdr *tcp_hdr;
#endif
  dpc = (struct dpdk_private_context *)ctxt->io_private_context;

  mm = m = dpc->pkts_burst[index];

#if 0
  e_hdr = rte_pktmbuf_mtod(m, struct ether_hdr *);
  ip_hdr = (struct ipv4_hdr *)(e_hdr + 1);
  tcp_hdr = (struct tcp_hdr *)(ip_hdr + 1);
  fprintf(stdout, "worker %u index:%u nb_meg:%u saddr:%x sport:%u seq:%u\n", ctxt->cpu,
          index, m->nb_segs, ntohl(ip_hdr->src_addr), ntohs(tcp_hdr->src_port),
          ntohl(tcp_hdr->sent_seq));
  fflush(stdout);
#endif

#ifdef IP_DEFRAG
  m = ip_reassemble(dpc, m);
#endif

while(mm!=NULL){
  *len +=mm->pkt_len;
  rte_prefetch0(rte_pktmbuf_mtod(mm, void *));
  pktbuf[i++] = rte_pktmbuf_mtod(mm, uint8_t *);
  mm=mm->next;
  } 
  /* enqueue the pkt ptr in mbuf */
  dpc->rmbufs[ifidx].m_table[index] = m;

  /* verify checksum values from ol_flags */
  if ((m->ol_flags & (PKT_RX_L4_CKSUM_BAD | PKT_RX_IP_CKSUM_BAD)) != 0) {
    TRACE_ERROR(
        "%s(%p, %d, %d): mbuf with invalid checksum: "
        "%p(0x%lx);\n",
        __func__, ctxt, ifidx, index, m, m->ol_flags);
    pktbuf = NULL;
  }
#ifdef ENABLELRO
  dpc->cur_rx_m = m;
#endif /* ENABLELRO */

  return pktbuf;
}
#endif
/*----------------------------------------------------------------------------*/
int32_t dpdk_select(struct mtcp_thread_context *ctxt) {
#ifdef RX_IDLE_ENABLE
  struct dpdk_private_context *dpc;

  dpc = (struct dpdk_private_context *)ctxt->io_private_context;
  if (dpc->rx_idle > RX_IDLE_THRESH) {
    dpc->rx_idle = 0;
    usleep(RX_IDLE_TIMEOUT);
  }
#endif
  return 0;
}
/*----------------------------------------------------------------------------*/
void dpdk_destroy_handle(struct mtcp_thread_context *ctxt) {
  struct dpdk_private_context *dpc;
  int i;

  dpc = (struct dpdk_private_context *)ctxt->io_private_context;

  /* free wmbufs */
  for (i = 0; i < num_devices_attached; i++)
    free_pkts(dpc->wmbufs[i].m_table, MAX_PKT_BURST);

#ifdef ENABLE_STATS_IOCTL
  /* free fd */
  if (dpc->fd >= 0) close(dpc->fd);
#endif /* !ENABLE_STATS_IOCTL */

  /* free it all up */
  free(dpc);
}
/*----------------------------------------------------------------------------*/
static void check_all_ports_link_status(uint8_t port_num, uint32_t port_mask) {
#define CHECK_INTERVAL 100 /* 100ms */
#define MAX_CHECK_TIME 90  /* 9s (90 * 100ms) in total */

  uint8_t portid, count, all_ports_up, print_flag = 0;
  struct rte_eth_link link;

  printf("\nChecking link status");
  fflush(stdout);
  for (count = 0; count <= MAX_CHECK_TIME; count++) {
    all_ports_up = 1;
    for (portid = 0; portid < port_num; portid++) {
      if ((port_mask & (1 << portid)) == 0) continue;
      memset(&link, 0, sizeof(link));
      rte_eth_link_get_nowait(portid, &link);
      /* print link status if flag set */
      if (print_flag == 1) {
        if (link.link_status)
          printf(
              "Port %d Link Up - speed %u "
              "Mbps - %s\n",
              (uint8_t)portid, (unsigned)link.link_speed,
              (link.link_duplex == ETH_LINK_FULL_DUPLEX) ? ("full-duplex")
                                                         : ("half-duplex\n"));
        else
          printf("Port %d Link Down\n", (uint8_t)portid);
        continue;
      }
      /* clear all_ports_up flag if any link down */
      if (link.link_status == 0) {
        all_ports_up = 0;
        break;
      }
    }
    /* after finally printing all link status, get out */
    if (print_flag == 1) break;

    if (all_ports_up == 0) {
      printf(".");
      fflush(stdout);
      rte_delay_ms(CHECK_INTERVAL);
    }

    /* set the print_flag if all ports up or timeout */
    if (all_ports_up == 1 || count == (MAX_CHECK_TIME - 1)) {
      print_flag = 1;
      printf("done\n");
    }
  }
}
/*----------------------------------------------------------------------------*/
void dpdk_load_module(void) {
  int portid, rxlcore_id, ret;
  /* for Ethernet flow control settings */
  struct rte_eth_fc_conf fc_conf;
  /* setting the rss key */
  int promiscuous_mode;


  if (!CONFIG.multi_process ||
      (CONFIG.multi_process && CONFIG.multi_process_is_master)) {
    for (rxlcore_id = 0; rxlcore_id < CONFIG.num_cores; rxlcore_id++) {
      char name[RTE_MEMPOOL_NAMESIZE];
      uint32_t nb_mbuf;
      sprintf(name, "mbuf_pool-%d", rxlcore_id);
      if (rxlcore_id == 0)
        nb_mbuf = (CONFIG.num_cores - 1) * NB_MBUF;
      else
        nb_mbuf = 16 * NB_MBUF;
#ifdef IP_DEFRAG
      int max_flows;

      // max_flows = CONFIG.max_concurrency / CONFIG.num_cores;
      // for fps, worker = num_cores - 1
      max_flows = CONFIG.max_concurrency / (CONFIG.num_cores - 1);

      /*
       * At any given moment up to <max_flows * (MAX_FRAG_NUM)>
       * mbufs could be stored int the fragment table.
       * Plus, each TX queue can hold up to <max_flows> packets.
       */

      if (rxlcore_id == 0)
        nb_mbuf =
            RTE_MAX(CONFIG.max_concurrency, 2UL * MAX_PKT_BURST) * MAX_FRAG_NUM;
      else
        nb_mbuf = RTE_MAX(max_flows, 2UL * MAX_PKT_BURST) * MAX_FRAG_NUM;

      /*
  nb_mbuf *= (port_conf.rxmode.max_rx_pkt_len + BUF_SIZE - 1) / BUF_SIZE;
  nb_mbuf += RTE_TEST_RX_DESC_DEFAULT + RTE_TEST_TX_DESC_DEFAULT;
      */

      nb_mbuf = RTE_MAX(nb_mbuf, (uint32_t)NB_MBUF);
#endif
      fprintf(stdout, "thread %d nb_mbuf %d\n", rxlcore_id, nb_mbuf);
      fflush(stdout);
      /* create the mbuf pools */
      pktmbuf_pool[rxlcore_id] = rte_mempool_create(
          name, nb_mbuf, MBUF_SIZE, MEMPOOL_CACHE_SIZE,
          sizeof(struct rte_pktmbuf_pool_private), rte_pktmbuf_pool_init, NULL,
          rte_pktmbuf_init, NULL, rte_socket_id(),
          MEMPOOL_F_SP_PUT | MEMPOOL_F_SC_GET);

      if (pktmbuf_pool[rxlcore_id] == NULL)
        rte_exit(EXIT_FAILURE, "Cannot init mbuf pool, errno: %d\n", rte_errno);

#ifdef ENABLE_GDP

InitPriorityQueue(queues[rxlcore_id]);
#endif

#ifdef GDP_GRO
flow_table[rxlcore_id] =
          CreateFDirHashtable(FDirHashFlow, FDirEqualFlow, NUM_BINS_FLOWS);

#endif

}
  /* Initialise each port */
  int i;
  for (i = 0; i < num_devices_attached; ++i) {
    /* get portid form the index of attached devices */
    portid = devices_attached[i];

    /* check port capabilities */
    rte_eth_dev_info_get(portid, &dev_info[portid]);
    /* init port */
    fflush(stdout);

    struct rte_eth_dev_info dev_info;
    rte_eth_dev_info_get(portid, &dev_info);
    port_conf.txmode.offloads = dev_info.tx_offload_capa;
    ret = rte_eth_dev_configure(portid, CONFIG.num_cores, CONFIG.num_cores,
                                &port_conf);
    if (ret < 0)
      rte_exit(EXIT_FAILURE,
               "Cannot configure device: err=%d, port=%u, cores: %d\n", ret,
               (unsigned)portid, CONFIG.num_cores);

    /* init one RX queue per CPU */
#ifdef DEBUG
    rte_eth_macaddr_get(portid, &ports_eth_addr[portid]);
#endif
    fprintf(stdout, "CONFIG.num_cores:%u\n", CONFIG.num_cores);
    fflush(stdout);
    ret = rte_eth_rx_queue_setup(portid, 0, RTE_DPLANE_MASTER_DESC,
                                 rte_eth_dev_socket_id(portid), &rx_conf,
                                 pktmbuf_pool[0]);

    for (rxlcore_id = 1; rxlcore_id < CONFIG.num_cores; rxlcore_id++) {
      ret = rte_eth_rx_queue_setup(portid, rxlcore_id, nb_rxd,
                                   rte_eth_dev_socket_id(portid), &rx_conf,
                                   pktmbuf_pool[rxlcore_id]);
      if (ret < 0)
        rte_exit(EXIT_FAILURE,
                 "rte_eth_rx_queue_setup:err=%d, port=%u, queueid: %d\n", ret,
                 (unsigned)portid, rxlcore_id);
    }

    /* init one TX queue on each port per CPU (this is redundant for this app)
     */
    fflush(stdout);
    for (rxlcore_id = 0; rxlcore_id < CONFIG.num_cores; rxlcore_id++) {
      ret = rte_eth_tx_queue_setup(portid, rxlcore_id, nb_txd,
                                   rte_eth_dev_socket_id(portid), &tx_conf);
      if (ret < 0)
        rte_exit(EXIT_FAILURE,
                 "rte_eth_tx_queue_setup:err=%d, port=%u, queueid: %d\n", ret,
                 (unsigned)portid, rxlcore_id);
    }

    /* Start device */
    ret = rte_eth_dev_start(portid);
    if (ret < 0)
      rte_exit(EXIT_FAILURE, "rte_eth_dev_start:err=%d, port=%u\n", ret,
               (unsigned)portid);

    rte_eth_promiscuous_disable(portid);

    promiscuous_mode = rte_eth_promiscuous_get(portid);

    if (promiscuous_mode) {
      fprintf(stdout, "NIC %d port promiscuous enable\n", portid);
      fflush(stdout);
    } else {
      fprintf(stdout, "NIC %d port promiscuous disable\n", portid);
      fflush(stdout);
    }

    ret = fdir_flush_init(portid);
    if (ret < 0)
      rte_exit(EXIT_FAILURE, "Failed to init fDir\n");
    else {
      fprintf(stdout, "Flushing fDir\n");
      fflush(stdout);
    }

    /* retrieve current flow control settings per port */
    memset(&fc_conf, 0, sizeof(fc_conf));
    ret = rte_eth_dev_flow_ctrl_get(portid, &fc_conf);
    if (ret != 0) TRACE_INFO("Failed to get flow control info!\n");

    /* and just disable the rx/tx flow control */
    fc_conf.mode = RTE_FC_NONE;
    ret = rte_eth_dev_flow_ctrl_set(portid, &fc_conf);
    if (ret != 0)
      TRACE_INFO("Failed to set flow control info!: errno: %d\n", ret);

#ifdef DEBUG
    printf("Port %u, MAC address: %02X:%02X:%02X:%02X:%02X:%02X\n\n",
           (unsigned)portid, ports_eth_addr[portid].addr_bytes[0],
           ports_eth_addr[portid].addr_bytes[1],
           ports_eth_addr[portid].addr_bytes[2],
           ports_eth_addr[portid].addr_bytes[3],
           ports_eth_addr[portid].addr_bytes[4],
           ports_eth_addr[portid].addr_bytes[5]);
#endif
  }
  /* only check for link status if the thread is master */
  check_all_ports_link_status(num_devices_attached, 0xFFFFFFFF);
}
else { /* CONFIG.multi_process && !CONFIG.multi_process_is_master */
  for (rxlcore_id = 0; rxlcore_id < CONFIG.num_cores; rxlcore_id++) {
    char name[RTE_MEMPOOL_NAMESIZE];
    sprintf(name, "mbuf_pool-%d", rxlcore_id);
    /* initialize the mbuf pools */
    pktmbuf_pool[rxlcore_id] = rte_mempool_lookup(name);
    if (pktmbuf_pool[rxlcore_id] == NULL)
      rte_exit(EXIT_FAILURE, "Cannot init mbuf pool\n");
  }

  int i;
  /* initializing dev_info struct */
  for (i = 0; i < num_devices_attached; i++) {
    /* get portid form the index of attached devices */
    portid = devices_attached[i];
    /* check port capabilities */
    rte_eth_dev_info_get(i, &dev_info[portid]);
  }
}
}
/*----------------------------------------------------------------------------*/
int32_t dpdk_dev_ioctl(struct mtcp_thread_context *ctx, int nif, int cmd,
                       void *argp) {
  struct dpdk_private_context *dpc;
  struct rte_mbuf *m;
  int len_of_mbuf;
  struct iphdr *iph;
  struct tcphdr *tcph;
  void **argpptr = (void **)argp;
#ifdef ENABLELRO
  uint8_t *payload, *to;
  int seg_off;
#endif

  if (cmd == DRV_NAME) {
    *argpptr = (void *)dev_info[nif].driver_name;
    return 0;
  }

  int eidx = CONFIG.nif_to_eidx[nif];

  iph = (struct iphdr *)argp;
  dpc = (struct dpdk_private_context *)ctx->io_private_context;
  len_of_mbuf = dpc->wmbufs[eidx].len;

  switch (cmd) {
    case PKT_TX_IP_CSUM:
      if ((dev_info[nif].tx_offload_capa & DEV_TX_OFFLOAD_IPV4_CKSUM) == 0)
        goto dev_ioctl_err;
      m = dpc->wmbufs[eidx].m_table[len_of_mbuf - 1];
      m->ol_flags = PKT_TX_IP_CKSUM | PKT_TX_IPV4;
#if RTE_VERSION < RTE_VERSION_NUM(19, 8, 0, 0)
      m->l2_len = sizeof(struct ether_hdr);
#else
      m->l2_len = sizeof(struct rte_ether_hdr);
#endif
      m->l3_len = (iph->ihl << 2);
      break;
    case PKT_TX_TCP_CSUM:
      if ((dev_info[nif].tx_offload_capa & DEV_TX_OFFLOAD_TCP_CKSUM) == 0)
        goto dev_ioctl_err;
      m = dpc->wmbufs[eidx].m_table[len_of_mbuf - 1];
      tcph = (struct tcphdr *)((unsigned char *)iph + (iph->ihl << 2));
      m->ol_flags |= PKT_TX_TCP_CKSUM;
#if RTE_VERSION < RTE_VERSION_NUM(19, 8, 0, 0)
      tcph->check = rte_ipv4_phdr_cksum((struct ipv4_hdr *)iph, m->ol_flags);
#else
      tcph->check =
          rte_ipv4_phdr_cksum((struct rte_ipv4_hdr *)iph, m->ol_flags);
#endif
      break;
#ifdef ENABLELRO
    case PKT_RX_TCP_LROSEG:
      m = dpc->cur_rx_m;
      // if (m->next != NULL)
      //	rte_prefetch0(rte_pktmbuf_mtod(m->next, void *));
      iph =
          rte_pktmbuf_mtod_offset(m, struct iphdr *, sizeof(struct ether_hdr));
      tcph = (struct tcphdr *)((u_char *)iph + (iph->ihl << 2));
      payload = (uint8_t *)tcph + (tcph->doff << 2);

      seg_off = m->data_len - sizeof(struct ether_hdr) - (iph->ihl << 2) -
                (tcph->doff << 2);

      to = (uint8_t *)argp;
      m = m->next;
      memcpy(to, payload, seg_off);
      while (m != NULL) {
        // if (m->next != NULL)
        //	rte_prefetch0(rte_pktmbuf_mtod(m->next, void *));
        memcpy(to + seg_off, rte_pktmbuf_mtod(m, uint8_t *), m->data_len);
        seg_off += m->data_len;
        m = m->next;
      }
      break;
#endif
    case PKT_TX_TCPIP_CSUM:
      if ((dev_info[nif].tx_offload_capa & DEV_TX_OFFLOAD_IPV4_CKSUM) == 0)
        goto dev_ioctl_err;
      if ((dev_info[nif].tx_offload_capa & DEV_TX_OFFLOAD_TCP_CKSUM) == 0)
        goto dev_ioctl_err;
      m = dpc->wmbufs[eidx].m_table[len_of_mbuf - 1];
#if RTE_VERSION < RTE_VERSION_NUM(19, 8, 0, 0)
      iph =
          rte_pktmbuf_mtod_offset(m, struct iphdr *, sizeof(struct ether_hdr));
#else
      iph = rte_pktmbuf_mtod_offset(m, struct iphdr *,
                                    sizeof(struct rte_ether_hdr));
#endif
      tcph = (struct tcphdr *)((uint8_t *)iph + (iph->ihl << 2));
#if RTE_VERSION < RTE_VERSION_NUM(19, 8, 0, 0)
      m->l2_len = sizeof(struct ether_hdr);
#else
      m->l2_len = sizeof(struct rte_ether_hdr);
#endif
      m->l3_len = (iph->ihl << 2);
      m->l4_len = (tcph->doff << 2);
      m->ol_flags = PKT_TX_TCP_CKSUM | PKT_TX_IP_CKSUM | PKT_TX_IPV4;
#if RTE_VERSION < RTE_VERSION_NUM(19, 8, 0, 0)
      tcph->check = rte_ipv4_phdr_cksum((struct ipv4_hdr *)iph, m->ol_flags);
#else
      tcph->check =
          rte_ipv4_phdr_cksum((struct rte_ipv4_hdr *)iph, m->ol_flags);
#endif
      break;
    case PKT_RX_IP_CSUM:
      if ((dev_info[nif].rx_offload_capa & DEV_RX_OFFLOAD_IPV4_CKSUM) == 0)
        goto dev_ioctl_err;
      break;
    case PKT_RX_TCP_CSUM:
      if ((dev_info[nif].rx_offload_capa & DEV_RX_OFFLOAD_TCP_CKSUM) == 0)
        goto dev_ioctl_err;
      break;
    case PKT_TX_TCPIP_CSUM_PEEK:
      if ((dev_info[nif].tx_offload_capa & DEV_TX_OFFLOAD_IPV4_CKSUM) == 0)
        goto dev_ioctl_err;
      if ((dev_info[nif].tx_offload_capa & DEV_TX_OFFLOAD_TCP_CKSUM) == 0)
        goto dev_ioctl_err;
      break;
    default:
      goto dev_ioctl_err;
  }
  return 0;
dev_ioctl_err:
  return -1;
}
/*----------------------------------------------------------------------------*/
io_module_func dpdk_module_func = {
    .load_module = dpdk_load_module,
    .init_handle = dpdk_init_handle,
    .link_devices = dpdk_link_devices,
    .release_pkt = dpdk_release_pkt,
    .send_pkts = dpdk_send_pkts,
    .get_wptr = dpdk_get_wptr,
    .recv_pkts = dpdk_recv_pkts,
    .get_rptr = dpdk_get_rptr,
    //                                   .get_rptrv = dpdk_get_rptrv,
    .get_mbuf = dpdk_get_mbuf,
    .select = dpdk_select,
    .destroy_handle = dpdk_destroy_handle,
    .dev_ioctl = dpdk_dev_ioctl};
/*----------------------------------------------------------------------------*/
#else
io_module_func dpdk_module_func = {.load_module = NULL,
                                   .init_handle = NULL,
                                   .link_devices = NULL,
                                   .release_pkt = NULL,
                                   .send_pkts = NULL,
                                   .get_wptr = NULL,
                                   .recv_pkts = NULL,
                                   .get_rptr = NULL,
                                   .select = NULL,
                                   .destroy_handle = NULL,
                                   .dev_ioctl = NULL};
/*----------------------------------------------------------------------------*/
#endif /* !DISABLE_DPDK */
