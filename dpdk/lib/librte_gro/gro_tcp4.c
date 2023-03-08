/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2017 Intel Corporation
 */

#include "gro_tcp4.h"

#include <assert.h>
#include <rte_cycles.h>
#include <rte_ethdev.h>
#include <rte_ether.h>
#include <rte_malloc.h>
#include <rte_mbuf.h>

void *gro_tcp4_tbl_create(uint16_t socket_id, uint16_t max_flow_num,
                          uint16_t max_item_per_flow) {
  struct gro_tcp4_tbl *tbl;
  size_t size;
  uint32_t entries_num, i;

  entries_num = max_flow_num * max_item_per_flow;
  entries_num = RTE_MIN(entries_num, GRO_TCP4_TBL_MAX_ITEM_NUM);

  if (entries_num == 0) return NULL;

  tbl = rte_zmalloc_socket(__func__, sizeof(struct gro_tcp4_tbl),
                           RTE_CACHE_LINE_SIZE, socket_id);
  if (tbl == NULL) return NULL;

  size = sizeof(struct gro_tcp4_item) * entries_num;
  tbl->items =
      rte_zmalloc_socket(__func__, size, RTE_CACHE_LINE_SIZE, socket_id);
  if (tbl->items == NULL) {
    rte_free(tbl);
    return NULL;
  }
  tbl->max_item_num = entries_num;

  size = sizeof(struct gro_tcp4_flow) * entries_num;
  tbl->flows =
      rte_zmalloc_socket(__func__, size, RTE_CACHE_LINE_SIZE, socket_id);
  if (tbl->flows == NULL) {
    rte_free(tbl->items);
    rte_free(tbl);
    return NULL;
  }
  /* INVALID_ARRAY_INDEX indicates an empty flow */
  for (i = 0; i < entries_num; i++)
    tbl->flows[i].start_index = INVALID_ARRAY_INDEX;
  tbl->max_flow_num = entries_num;

  return tbl;
}

void gro_tcp4_tbl_destroy(void *tbl) {
  struct gro_tcp4_tbl *tcp_tbl = tbl;

  if (tcp_tbl) {
    rte_free(tcp_tbl->items);
    rte_free(tcp_tbl->flows);
  }
  rte_free(tcp_tbl);
}

static inline uint32_t find_an_empty_item(struct gro_tcp4_tbl *tbl) {
  uint32_t i;
  uint32_t max_item_num = tbl->max_item_num;

  for (i = 0; i < max_item_num; i++)
    if (tbl->items[i].firstseg == NULL) return i;
  return INVALID_ARRAY_INDEX;
}

static inline uint32_t find_an_empty_flow(struct gro_tcp4_tbl *tbl) {
  uint32_t i;
  uint32_t max_flow_num = tbl->max_flow_num;

  for (i = 0; i < max_flow_num; i++)
    if (tbl->flows[i].start_index == INVALID_ARRAY_INDEX) return i;
  return INVALID_ARRAY_INDEX;
}

static inline uint32_t insert_new_item(struct gro_tcp4_tbl *tbl,
                                       struct rte_mbuf *pkt,
                                       uint64_t start_time, uint32_t cur_idx,
                                       uint32_t sent_seq, uint16_t ip_id,
                                       uint8_t is_atomic) {
  uint32_t item_idx;
  uint32_t pre_idx;

  item_idx = find_an_empty_item(tbl);

  if (item_idx == INVALID_ARRAY_INDEX) {
    return INVALID_ARRAY_INDEX;
  }

  tbl->items[item_idx].firstseg = pkt;
  tbl->items[item_idx].lastseg = rte_pktmbuf_lastseg(pkt);
  tbl->items[item_idx].start_time = start_time;
  tbl->items[item_idx].next_pkt_idx = INVALID_ARRAY_INDEX;
  tbl->items[item_idx].pre_pkt_idx = INVALID_ARRAY_INDEX;
  tbl->items[item_idx].sent_seq = sent_seq;
  tbl->items[item_idx].ip_id = ip_id;
  tbl->items[item_idx].nb_merged = 1;
  tbl->items[item_idx].is_atomic = is_atomic;
  tbl->item_num++;

  /* if the previous packet exists, chain them together. */
  if (cur_idx != INVALID_ARRAY_INDEX) {
    #if 0
    printf("after insert, update pointer, cur_idx:%u\n", cur_idx);
    #endif
  //  printf("connect two pkts\n");
    pre_idx = tbl->items[cur_idx].pre_pkt_idx;
    if (sent_seq < tbl->items[cur_idx].sent_seq) {
      tbl->items[item_idx].pre_pkt_idx = tbl->items[cur_idx].pre_pkt_idx;
      tbl->items[item_idx].next_pkt_idx = cur_idx;
      tbl->items[cur_idx].pre_pkt_idx = item_idx;
      if (pre_idx != INVALID_ARRAY_INDEX)
      tbl->items[pre_idx].next_pkt_idx = item_idx;
    } else {
      //      tbl->items[pos_idx].pre_pkt_idx =
      //      tbl->items[item_idx].pre_pkt_idx;
      tbl->items[item_idx].pre_pkt_idx = cur_idx;
      tbl->items[cur_idx].next_pkt_idx = item_idx;
    }
  } 
  
  #if 0
  else
    printf("connot connect two pkts due to cur_idx == INVALID_ARRAY_INDEX\n");
  #endif

  return item_idx;
}

#if 0
static inline uint32_t insert_new_item(struct gro_tcp4_tbl *tbl,
                                       struct rte_mbuf *pkt,
                                       uint64_t start_time, uint32_t prev_idx,
                                       uint32_t sent_seq, uint16_t ip_id,
                                       uint8_t is_atomic) {
  uint32_t item_idx;

  item_idx = find_an_empty_item(tbl);
  if (item_idx == INVALID_ARRAY_INDEX) return INVALID_ARRAY_INDEX;

  tbl->items[item_idx].firstseg = pkt;
  tbl->items[item_idx].lastseg = rte_pktmbuf_lastseg(pkt);
  tbl->items[item_idx].start_time = start_time;
  tbl->items[item_idx].next_pkt_idx = INVALID_ARRAY_INDEX;
  tbl->items[item_idx].pre_pkt_idx = INVALID_ARRAY_INDEX;
  tbl->items[item_idx].sent_seq = sent_seq;
  tbl->items[item_idx].ip_id = ip_id;
  tbl->items[item_idx].nb_merged = 1;
  tbl->items[item_idx].is_atomic = is_atomic;
  tbl->item_num++;

  /* if the previous packet exists, chain them together. */
  if (prev_idx != INVALID_ARRAY_INDEX) {
    if (sent_seq > tbl->items[prev_idx].sent_seq) {
      tbl->items[item_idx].next_pkt_idx = tbl->items[prev_idx].next_pkt_idx;
      tbl->items[prev_idx].next_pkt_idx = item_idx;
    } else {
      tbl->items[prev_idx].next_pkt_idx = tbl->items[item_idx].next_pkt_idx;
      tbl->items[item_idx].next_pkt_idx = prev_idx;
    }
  }

  return item_idx;
}
#endif

static inline uint32_t delete_item(struct gro_tcp4_tbl *tbl, uint32_t item_idx) {
                                 //  uint32_t prev_item_idx) {
  uint32_t next_idx = tbl->items[item_idx].next_pkt_idx;

  /* NULL indicates an empty item */
  tbl->items[item_idx].firstseg = NULL;
  tbl->item_num--;
  #if 0
  if (prev_item_idx != INVALID_ARRAY_INDEX)
    tbl->items[prev_item_idx].next_pkt_idx = next_idx;
  #endif
  if (next_idx != INVALID_ARRAY_INDEX)
    tbl->items[next_idx].pre_pkt_idx = INVALID_ARRAY_INDEX;

  return next_idx;
}

static inline uint32_t insert_new_flow(struct gro_tcp4_tbl *tbl,
                                       struct tcp4_flow_key *src,
                                       uint32_t item_idx) {
  struct tcp4_flow_key *dst;
  uint32_t flow_idx;

  flow_idx = find_an_empty_flow(tbl);
  if (unlikely(flow_idx == INVALID_ARRAY_INDEX)) return INVALID_ARRAY_INDEX;

  dst = &(tbl->flows[flow_idx].key);

  ether_addr_copy(&(src->eth_saddr), &(dst->eth_saddr));
  ether_addr_copy(&(src->eth_daddr), &(dst->eth_daddr));
  dst->ip_src_addr = src->ip_src_addr;
  dst->ip_dst_addr = src->ip_dst_addr;
  dst->recv_ack = src->recv_ack;
  dst->src_port = src->src_port;
  dst->dst_port = src->dst_port;

  tbl->flows[flow_idx].start_index = item_idx;
  tbl->flows[flow_idx].start_time = rte_rdtsc();
  tbl->flow_num++;

  return flow_idx;
}

/*
 * update the packet length for the flushed packet.
 */
static inline void update_header(struct gro_tcp4_item *item) {
  struct ipv4_hdr *ipv4_hdr;
  struct rte_mbuf *pkt = item->firstseg;

  ipv4_hdr = (struct ipv4_hdr *)(rte_pktmbuf_mtod(pkt, char *) + pkt->l2_len);
  ipv4_hdr->total_length = rte_cpu_to_be_16(pkt->pkt_len - pkt->l2_len);
}

int32_t gro_tcp4_gdp_reassemble(struct rte_mbuf *pkt, uint64_t start_time, struct fdir_hashtable *htable, fdir_flow **flow) {
  (*flow) = FDirStreamHTSearch(htable, pkt->hash.fdir.hi);
  (*flow) = FDirStreamHTInsert(htable, pkt->hash.fdir.hi);
    if (NULL == *flow) {
      /*
       * Fail to insert a new flow, so delete the
       * stored packet.
       */
      //delete_item(tbl, item_idx, INVALID_ARRAY_INDEX);
      delete_item(tbl, item_idx);
      return -1;
    }
}


int32_t gro_tcp4_dplane_reassemble(struct rte_mbuf *pkt,
                                   struct gro_tcp4_tbl *tbl,
                                   uint64_t start_time,
                                   struct fdir_hashtable *htable,
                                   fdir_flow **flow) {
  struct ether_hdr *eth_hdr;
  struct ipv4_hdr *ipv4_hdr;
  struct tcp_hdr *tcp_hdr;
  uint32_t sent_seq;
  uint16_t tcp_dl, ip_id, hdr_len, frag_off;
  uint8_t is_atomic;

  // struct tcp4_flow_key key;
  // uint32_t pos_idx;
  uint32_t cur_idx = INVALID_ARRAY_INDEX, item_idx;
  uint32_t i = 0;
  // uint32_t i, max_flow_num, remaining_flow_num;
  int cmp;

  struct rte_mbuf *p;
  //  tcp_stream stream;
  // data_off of some packets becomes 194, not 128
  rte_pktmbuf_reset_headroom(pkt);

  eth_hdr = rte_pktmbuf_mtod(pkt, struct ether_hdr *);
// eth_hdr = rte_pktmbuf_mtod(pkt, struct ether_hdr *);
#if 0
  printf("dst:%d %d %d %d %d %d src:%d %d %d %d %d %d type:%d\n",
         eth_hdr->d_addr.addr_bytes[0], eth_hdr->d_addr.addr_bytes[1],
         eth_hdr->d_addr.addr_bytes[2], eth_hdr->d_addr.addr_bytes[3],
         eth_hdr->d_addr.addr_bytes[4], eth_hdr->d_addr.addr_bytes[5],
         eth_hdr->s_addr.addr_bytes[0], eth_hdr->s_addr.addr_bytes[1],
         eth_hdr->s_addr.addr_bytes[2], eth_hdr->s_addr.addr_bytes[3],
         eth_hdr->s_addr.addr_bytes[4], eth_hdr->s_addr.addr_bytes[5],
         eth_hdr->ether_type);
#endif

  /*
   * Don't process non-IP packet
   *
   */
  p = pkt->next;
  while (p != NULL) {
    rte_pktmbuf_reset_headroom(p);
    struct ether_hdr *tmp_eth_hdr = rte_pktmbuf_mtod(p, struct ether_hdr *);
    struct ipv4_hdr *ip_hdr = (struct ipv4_hdr *)(tmp_eth_hdr + 1);
    struct tcp_hdr *thdr = (struct tcp_hdr *)(ip_hdr + 1);
    printf("pkt->next not null, eth_type: %u, data_len:%u src_port:%u\n",
           tmp_eth_hdr->ether_type, rte_pktmbuf_data_len(p),
           ntohs(thdr->src_port));
    p = p->next;
  }

  if (eth_hdr->ether_type != rte_cpu_to_be_16(ETHER_TYPE_IPv4)) {
    printf("noe ip packet:0x%x IPv4:0x%x, data_off:%u mbuf_datalen:%u\n",
           eth_hdr->ether_type, rte_cpu_to_be_16(ETHER_TYPE_IPv4),
           pkt->data_off, pkt->data_len);
    return -1;
  }

  ipv4_hdr = (struct ipv4_hdr *)((char *)eth_hdr + pkt->l2_len);
  tcp_hdr = (struct tcp_hdr *)((char *)ipv4_hdr + pkt->l3_len);
  hdr_len = pkt->l2_len + pkt->l3_len + pkt->l4_len;

  /*
   * Don't process the packet which has FIN, SYN, RST, PSH, URG, ECE
   * or CWR set.
   */
  if (tcp_hdr->tcp_flags != TCP_ACK_FLAG &&
      tcp_hdr->tcp_flags != (TCP_ACK_FLAG | TCP_PSH_FLAG)) {
    printf("cannot merge due to tcp_flag:%d != %d\n", tcp_hdr->tcp_flags,
           TCP_ACK_FLAG);
    return -1;
  }
  /*
   * Don't process the packet whose payload length is less than or
   * equal to 0.
   */

  tcp_dl = pkt->pkt_len - hdr_len;

  if (tcp_dl <= 0) {
#if 0
    printf("cannot merge due to len (pkt_len:%u hdr_len:%u tc_dl:%d)\n",\
    pkt->pkt_len, hdr_len, tcp_dl);
#endif
    return -1;
  }

  /*
   * Save IPv4 ID for the packet whose DF bit is 0. For the packet
   * whose DF bit is 1, IPv4 ID is ignored.
   */
  frag_off = rte_be_to_cpu_16(ipv4_hdr->fragment_offset);
  is_atomic = (frag_off & IPV4_HDR_DF_FLAG) == IPV4_HDR_DF_FLAG;
  ip_id = is_atomic ? 0 : rte_be_to_cpu_16(ipv4_hdr->packet_id);
  sent_seq = rte_be_to_cpu_32(tcp_hdr->sent_seq);

  //  flow->fdir_id = pkt->hash.fdir.hi;
  (*flow) = FDirStreamHTSearch(htable, pkt->hash.fdir.hi);
  /*
   * Fail to find a matched flow. Insert a new flow and store the
   * packet into the flow.
   */

  if (NULL == (*flow)) {
   // printf("no err before insert new item when flow is null\n");
    item_idx = insert_new_item(tbl, pkt, start_time, INVALID_ARRAY_INDEX,
                               sent_seq, ip_id, is_atomic);
    if (item_idx == INVALID_ARRAY_INDEX) {
      return -1;
    }

    tbl->items[item_idx].size = tcp_dl;

    (*flow) = FDirStreamHTInsert(htable, pkt->hash.fdir.hi, tcp_dl, item_idx);
    if (NULL == *flow) {
      /*
       * Fail to insert a new flow, so delete the
       * stored packet.
       */
      //delete_item(tbl, item_idx, INVALID_ARRAY_INDEX);
      delete_item(tbl, item_idx);
      return -1;
    }

    // return 0;

    //printf("no err before end of inserting new item when flow is null\n");
    return 2;
  }

  /*
   * Check all packets in the flow and try to find a neighbor for
   * the input packet.
   */
  // if (INVALID_ARRAY_INDEX == (*flow)->start_idx && (*flow)->state == RECV)
  cur_idx = (*flow)->tmp_end_idx;

  (*flow)->size += tcp_dl;

 // printf("in dplane_reassemble, before while cur_idx:%u\n", cur_idx);
  //fflush(stdout);

  while (i < (*flow)->num && cur_idx != INVALID_ARRAY_INDEX &&
         (sent_seq + tcp_dl >= tbl->items[cur_idx].sent_seq)) {
   // printf("in dplane_reassemble, cur_idx:%u\n", cur_idx);
    //fflush(stdout);
    cmp = check_seq_option(&(tbl->items[cur_idx]), tcp_hdr, sent_seq, ip_id,
                           pkt->l4_len, tcp_dl, 0, is_atomic);
    if (cmp) {
      if (merge_two_tcp4_packets(&(tbl->items[cur_idx]), pkt, cmp, sent_seq,
                                 ip_id, 0)) {
        return 1;
      } else {
        /*
         * Fail to merge the two packets, as the packet
         * length is greater than the max value. Store
         * the packet into the flow.
         */
        item_idx = insert_new_item(tbl, pkt, start_time, cur_idx, sent_seq,
                                   ip_id, is_atomic);

        if (item_idx == INVALID_ARRAY_INDEX) {
          // return -1;
          (*flow)->size -= tcp_dl;
          (*flow)->state = RECV;
          return -2;
        }

        tbl->items[item_idx].size = tcp_dl;

        if (sent_seq < tbl->items[(*flow)->start_idx].sent_seq) {
          #if 0
          printf("change fdir_id:%u start_idx from %u to %u\n", (*flow)->fdir_id, \
          (*flow)->tmp_end_idx, item_idx);
          #endif

          (*flow)->start_idx = item_idx;
        }
        else if (sent_seq > tbl->items[(*flow)->tmp_end_idx].sent_seq) {
          #if 0
          printf("change fdir_id:%u tmp_end_idx from %u to %u\n", (*flow)->fdir_id, \
          (*flow)->tmp_end_idx, item_idx);
          #endif

          (*flow)->tmp_end_idx = item_idx;
        }

        (*flow)->num++;
      //  printf("insert one pkt, num:%u\n", (*flow)->num);

        return 0;
      }
    } else if (sent_seq > tbl->items[(*flow)->tmp_end_idx].sent_seq)
      break;
    //  pos_idx = cur_idx;
    // prev_idx = cur_idx;
    cur_idx = tbl->items[cur_idx].pre_pkt_idx;
    i++;
    // cur_idx = tbl->items[cur_idx].next_pkt_idx;
  }
  /* Fail to find a neighbor, so store the packet into the flow. */

  item_idx = insert_new_item(tbl, pkt, start_time, cur_idx, sent_seq, ip_id,
                             is_atomic);
  if (item_idx == INVALID_ARRAY_INDEX) {
    //   printf("line:%d cannot find a neighbor\n", __LINE__);
    (*flow)->size -= tcp_dl;
    (*flow)->state = RECV;
    return -1;
  }

  tbl->items[item_idx].size = tcp_dl;

  (*flow)->num++;
  //printf("insert one pkt, num:%u\n", (*flow)->num);

   #if 0
  printf("flow %u start_idx:%u tmp_end_idx:%u\n", pkt->hash.fdir.hi,
         (*flow)->start_idx, (*flow)->tmp_end_idx);
  fflush(stdout);
  #endif

  if (INVALID_ARRAY_INDEX == (*flow)->start_idx ||
      (INVALID_ARRAY_INDEX != (*flow)->start_idx && sent_seq < tbl->items[(*flow)->start_idx].sent_seq)) {
    #if 0
    printf("change start_idx from %u to %u\n", (*flow)->start_idx,
           item_idx);
    #endif
    (*flow)->start_idx = item_idx;
      }

  if (INVALID_ARRAY_INDEX == (*flow)->tmp_end_idx ||
       (INVALID_ARRAY_INDEX != (*flow)->tmp_end_idx && sent_seq > tbl->items[(*flow)->tmp_end_idx].sent_seq)) {
    #if 0
    printf("change tmp_end_idx from %u to %u\n", (*flow)->tmp_end_idx,
           item_idx);
    #endif
    (*flow)->tmp_end_idx = item_idx;
  } 


return 0;
}

int32_t gro_tcp4_reassemble(struct rte_mbuf *pkt, struct gro_tcp4_tbl *tbl,
                            uint64_t start_time) {
  struct ether_hdr *eth_hdr;
  struct ipv4_hdr *ipv4_hdr;
  struct tcp_hdr *tcp_hdr;
  uint32_t sent_seq;
  uint16_t tcp_dl, ip_id, hdr_len, frag_off;
  uint8_t is_atomic;

  struct tcp4_flow_key key;
  uint32_t cur_idx, prev_idx, item_idx;
  uint32_t i, max_flow_num, remaining_flow_num;
  int cmp;
  uint8_t find;

  // qi
  // uint32_t flow_idx=0;

  eth_hdr = rte_pktmbuf_mtod(pkt, struct ether_hdr *);

  /*
   * Don't process non-IP packet
   *
   */
  if (eth_hdr->ether_type != ETHER_TYPE_IPv4) {
    //  printf("noe ip packet\n");
    return -1;
  }

  ipv4_hdr = (struct ipv4_hdr *)((char *)eth_hdr + pkt->l2_len);
  tcp_hdr = (struct tcp_hdr *)((char *)ipv4_hdr + pkt->l3_len);
  hdr_len = pkt->l2_len + pkt->l3_len + pkt->l4_len;

  /*
   * Don't process the packet which has FIN, SYN, RST, PSH, URG, ECE
   * or CWR set.
   */
  if (tcp_hdr->tcp_flags != TCP_ACK_FLAG) {
    //   printf("cannot merge due to tcp_flag:%d != %d\n", tcp_hdr->tcp_flags,
    //   TCP_ACK_FLAG);
    return -1;
  }
  /*
   * Don't process the packet whose payload length is less than or
   * equal to 0.
   */

  tcp_dl = pkt->pkt_len - hdr_len;
  // printf("pkt_len:%u l2_len:%u l3_len:%u l4_len:%u tcp_dl:%u\n",
  // pkt->pkt_len, pkt->l2_len, pkt->l3_len, pkt->l4_len, tcp_dl);
  if (tcp_dl <= 0) {
#if 0
    printf("cannot merge due to len (pkt_len:%u hdr_len:%u tc_dl:%d)\n",\
    pkt->pkt_len, hdr_len, tcp_dl);
#endif
    return -1;
  }

  /*
   * Save IPv4 ID for the packet whose DF bit is 0. For the packet
   * whose DF bit is 1, IPv4 ID is ignored.
   */
  frag_off = rte_be_to_cpu_16(ipv4_hdr->fragment_offset);
  is_atomic = (frag_off & IPV4_HDR_DF_FLAG) == IPV4_HDR_DF_FLAG;
  ip_id = is_atomic ? 0 : rte_be_to_cpu_16(ipv4_hdr->packet_id);
  sent_seq = rte_be_to_cpu_32(tcp_hdr->sent_seq);

  ether_addr_copy(&(eth_hdr->s_addr), &(key.eth_saddr));
  ether_addr_copy(&(eth_hdr->d_addr), &(key.eth_daddr));
  key.ip_src_addr = ipv4_hdr->src_addr;
  key.ip_dst_addr = ipv4_hdr->dst_addr;
  key.src_port = tcp_hdr->src_port;
  key.dst_port = tcp_hdr->dst_port;
  key.recv_ack = tcp_hdr->recv_ack;

  /* Search for a matched flow. */
  max_flow_num = tbl->max_flow_num;
  remaining_flow_num = tbl->flow_num;
  find = 0;

  for (i = 0; i < max_flow_num && remaining_flow_num; i++) {
    if (tbl->flows[i].start_index != INVALID_ARRAY_INDEX) {
      if (is_same_tcp4_flow(tbl->flows[i].key, key)) {
        find = 1;
        break;
      }
      remaining_flow_num--;
    }
  }

  /*
   * Fail to find a matched flow. Insert a new flow and store the
   * packet into the flow.
   */
  if (find == 0) {
    item_idx = insert_new_item(tbl, pkt, start_time, INVALID_ARRAY_INDEX,
                               sent_seq, ip_id, is_atomic);
    if (item_idx == INVALID_ARRAY_INDEX) {
      return -1;
    }

    if (insert_new_flow(tbl, &key, item_idx) == INVALID_ARRAY_INDEX) {
      /*
       * Fail to insert a new flow, so delete the
       * stored packet.
       */
      printf("cannot insert flow\n");
      //delete_item(tbl, item_idx, INVALID_ARRAY_INDEX);
      delete_item(tbl, item_idx);
      return -1;
    }

    return 0;
  }

  /*
   * Check all packets in the flow and try to find a neighbor for
   * the input packet.
   */
  cur_idx = tbl->flows[i].start_index;
  prev_idx = cur_idx;
  if (cur_idx == INVALID_ARRAY_INDEX)
    printf("cur_idx is INVALID_ARRAY_INDEX\n");

  do {
#if 0
    struct rte_mbuf *pkt_org = (&(tbl->items[cur_idx]))->firstseg;

    printf("tcp_hl:%u tcp_hl_org:%u, atomic:%d atomic2:%d\n", pkt->l4_len, \
    pkt_org->l4_len, (&(tbl->items[cur_idx]))->is_atomic, is_atomic);

    printf("seq1:%u len1: %u seq2:%u len2:%u diff:%d\n", sent_seq, tcp_dl, (&(tbl->items[cur_idx]))->sent_seq, \
    pkt_org->pkt_len - pkt_org->l2_len - pkt_org->l3_len - pkt_org->l4_len, (sent_seq - (&(tbl->items[cur_idx]))->sent_seq));
#endif

    cmp = check_seq_option(&(tbl->items[cur_idx]), tcp_hdr, sent_seq, ip_id,
                           pkt->l4_len, tcp_dl, 0, is_atomic);

    // printf("check_seq_option result:%d\n", cmp);

    if (cmp) {
      if (merge_two_tcp4_packets(&(tbl->items[cur_idx]), pkt, cmp, sent_seq,
                                 ip_id, 0))
        return 1;
      else {
        /*
         * Fail to merge the two packets, as the packet
         * length is greater than the max value. Store
         * the packet into the flow.
         */
        item_idx = insert_new_item(tbl, pkt, start_time, prev_idx, sent_seq,
                                   ip_id, is_atomic);
        if (item_idx == INVALID_ARRAY_INDEX) {
          //  printf("cannot merge pkts due to length larger than max\n");
          // return -1;
          return -2;
        }

// qi-add
#if 0
       if (cur_idx == tbl->flows[i].start_index) {
          return 2;
       }
#endif
        if (tbl->flows[i].size == 602112) return 2;
        return 0;
      }
    }
    prev_idx = cur_idx;
    cur_idx = tbl->items[cur_idx].next_pkt_idx;
  } while (cur_idx != INVALID_ARRAY_INDEX);

  /* Fail to find a neighbor, so store the packet into the flow. */
  if (insert_new_item(tbl, pkt, start_time, prev_idx, sent_seq, ip_id,
                      is_atomic) == INVALID_ARRAY_INDEX) {
    //   printf("line:%d cannot find a neighbor\n", __LINE__);
    return -1;
  }

  return 0;
}

uint16_t gro_tcp4_tbl_dplane_flush(struct gro_tcp4_tbl *tbl,
                                   struct rte_mbuf **out, uint16_t nb_out,
                                   struct fdir_hashtable *ftable,
                                   uint32_t fdir_id) {
  uint16_t k = 0;
  // uint32_t j,idx;
  uint32_t j;
  fdir_flow *item;

  item = FDirStreamHTSearch(ftable, fdir_id);

  assert(NULL != item);

  j = item->start_idx;


  #if 0
  if (j != INVALID_ARRAY_INDEX)
    printf("fdir_id:%u num:%u curr_idx:%u nex_idx:%u\n", item->fdir_id,
           item->num, j, tbl->items[j].next_pkt_idx);
  else
    printf("fdir_id:%u num:%u curr_idx:%u nex_idx:%lu\n", item->fdir_id,
           item->num, j, INVALID_ARRAY_INDEX);
  #endif
  // while (j != INVALID_ARRAY_INDEX && j!= item->tmp_end_idx && k < nb_out) {
  while (j != INVALID_ARRAY_INDEX && k < nb_out) {
    item->size -= tbl->items[j].size;

    #if 0
    printf("fdir:%u idx:%u flush pkt size:%u\n", fdir_id, j,
           tbl->items[j].size);
    #endif

    out[k++] = tbl->items[j].firstseg;
    if (tbl->items[j].nb_merged > 1) {
      update_header(&(tbl->items[j]));
    }

    //j = delete_item(tbl, j, INVALID_ARRAY_INDEX);
    j = delete_item(tbl, j);
    // tbl->flows[flow_idx].start_index = j;
    item->start_idx = j;
    item->state = RECV;
    item->num--;
    //printf("num pkts in flow:%u\n", item->num);
    #if 0
    if (j == INVALID_ARRAY_INDEX) {
      printf("j==INVALID_ARRAY_INDEX, item->tmp_end_idx:%u\n",
             item->tmp_end_idx);
      //    ftable->flow_num--;
    }
    #endif
    // tbl->flow_num--;
  }

  item->start_idx = j;
  if (j > item->tmp_end_idx) {
    item->tmp_end_idx = j;
  }
  return k;
}

uint16_t gro_tcp4_tbl_timeout_flush(struct gro_tcp4_tbl *tbl,
                                    uint64_t flush_timestamp,
                                    struct rte_mbuf **out, uint16_t nb_out) {
  uint16_t k = 0;
  uint32_t i, j;
  uint32_t max_flow_num = tbl->max_flow_num;

  for (i = 0; i < max_flow_num; i++) {
    if (unlikely(tbl->flow_num == 0)) return k;

    j = tbl->flows[i].start_index;
    while (j != INVALID_ARRAY_INDEX) {
      if (tbl->items[j].start_time <= flush_timestamp) {
        out[k++] = tbl->items[j].firstseg;
        if (tbl->items[j].nb_merged > 1) update_header(&(tbl->items[j]));
        /*
         * Delete the packet and get the next
         * packet in the flow.
         */
        //j = delete_item(tbl, j, INVALID_ARRAY_INDEX);
        j = delete_item(tbl, j);
        tbl->flows[i].start_index = j;
        if (j == INVALID_ARRAY_INDEX) tbl->flow_num--;

        if (unlikely(k == nb_out)) return k;
      } else
        /*
         * The left packets in this flow won't be
         * timeout. Go to check other flows.
         */
        break;
    }
  }
  return k;
}

#if 0
uint16_t gro_tcp4_tbl_timeout_flush(struct gro_tcp4_tbl *tbl,
                                    uint64_t flush_timestamp,
                                    struct rte_mbuf **out, uint16_t nb_out) {
  uint16_t k = 0;
  uint32_t i, j;
  uint32_t max_flow_num = tbl->max_flow_num;

  for (i = 0; i < max_flow_num; i++) {
    if (unlikely(tbl->flow_num == 0)) return k;

    j = tbl->flows[i].start_index;
    while (j != INVALID_ARRAY_INDEX) {
      if (tbl->items[j].start_time <= flush_timestamp) {
        out[k++] = tbl->items[j].firstseg;
        if (tbl->items[j].nb_merged > 1) update_header(&(tbl->items[j]));
        /*
         * Delete the packet and get the next
         * packet in the flow.
         */
        j = delete_item(tbl, j, INVALID_ARRAY_INDEX);
        tbl->flows[i].start_index = j;
        if (j == INVALID_ARRAY_INDEX) tbl->flow_num--;

        if (unlikely(k == nb_out)) return k;
      } else
        /*
         * The left packets in this flow won't be
         * timeout. Go to check other flows.
         */
        break;
    }
  }
  return k;
}
#endif

uint32_t gro_tcp4_tbl_pkt_count(void *tbl) {
  struct gro_tcp4_tbl *gro_tbl = tbl;

  if (gro_tbl) return gro_tbl->item_num;

  return 0;
}
