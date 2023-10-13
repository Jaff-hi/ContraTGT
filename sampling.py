from collections import defaultdict
import numpy as np



#spatial sampling
def get_adj_list(edge):
    adj_list = defaultdict(list)
    for i,edg in enumerate(edge['idx']):
        st = int(edg[0])#边的起点
        en = int(edg[1])#边的终点
        ts = int(edg[2])#边的时间戳
        idx = int(edg[3])#边的索引
        adj_list[st].append((en,ts,idx))#在st节点的邻接节点集合中添加en节点
        adj_list[en].append((st,ts,idx)) #若为有向图，这一步可以去掉
    return adj_list

def init_offset(adj_list):
    node_l = []
    ts_l = []
    idx_l = []
    offset_l = [0]
    for i in range(len(adj_list)):
        curr = adj_list[i]  # curr记录为当前i节点
        curr = sorted(curr, key=lambda x: x[2])  # 按照时间戳排序
        node_l.extend([x[0] for x in curr])  # 将相应的邻居节点append到node_l中
        ts_l.extend([x[1] for x in curr])  # 将邻居节点对应的时间戳保存
        idx_l.extend([x[2] for x in curr])  # 保存idx
        offset_l.append(len(node_l))  # 记录每个节点的邻节点个数

    node_l = np.array(node_l)
    ts_l = np.array(ts_l)
    idx_l = np.array(idx_l)
    offset_l = np.array(offset_l)

    return node_l, ts_l, idx_l, offset_l

def find_before(node_l, ts_l, idx_l, offset_l, node, ts):
    ngh_node_l = node_l[offset_l[node]:offset_l[node + 1]]  # 对应u节点的邻居
    ngh_ts_l = ts_l[offset_l[node]:offset_l[node + 1]]  # 对应u节点的时间戳
    ngh_idx_l = idx_l[offset_l[node]:offset_l[node + 1]]  # 对应u节点交互的顺序idx

    if len(ngh_node_l) == 0 or len(ngh_ts_l) == 0:
        return ngh_node_l, ngh_ts_l, ngh_idx_l

    left = 0
    right = len(ngh_node_l) - 1
    while left + 1 < right:  # 二分法找到对应的时间戳
        mid = (left + right) // 2
        curr_t = ngh_ts_l[mid]
        if curr_t < ts:
            left = mid
        else:
            right = mid

    if ngh_ts_l[right] < ts:  # 返回对应时间戳之前的所有交互
        return ngh_node_l[:right], ngh_idx_l[:right], ngh_ts_l[:right]
    else:
        return ngh_node_l[:left], ngh_idx_l[:left], ngh_ts_l[:left]


def get_neighbor_list(node_l, ts_l, idx_l, offset_l, node, timestamp, num_sample):
    #     print(timestamp)
    ngh_node = np.zeros((len(node), num_sample)).astype(np.int32)  # 建一个【len，num_sample】的0矩阵保存序列
    ngh_ts = np.zeros((len(node), num_sample)).astype(np.int32)  # 保存时间戳
    ngh_idx = np.zeros((len(node), num_sample)).astype(np.int32)  # 保存idx
    ngh_mask = np.zeros((len(node), num_sample)).astype(np.int32)  # 保存mask矩阵
    num_sample = num_sample - 1
    for i, edg in enumerate(zip(node, timestamp)):
        node, idx, ts = find_before(node_l, ts_l, idx_l, offset_l, edg[0], edg[1])  # 根据给定的源节点和时间戳，找时间戳之前发生的邻居节点
        node = node[::-1]
        idx = idx[::-1]
        ts = ts[::-1]
        #         print(ts)
        #         print('*********************')
        #         print(int(edg[1])-ts)
        if len(node) > 0:
            if len(node) > num_sample:  # 如果个数大于采样需求的数量，只保留最近的num_sample条数据
                node = node[:num_sample]
                idx = idx[:num_sample]
                ts = ts[:num_sample]
            ngh_node[i, 1:len(node) + 1] = node
            ngh_ts[i, 1:len(node) + 1] = int(edg[1]) - ts
            ngh_idx[i, 1:len(node) + 1] = idx
            ngh_mask[i, 1:len(node) + 1] = 1

        ngh_node[i, 0] = edg[0]
        ngh_mask[i, 0] = 1
    return ngh_node, ngh_ts, ngh_idx, ngh_mask

#temporal sampling
def get_interaction_list(edge):
    idx_list = defaultdict(list)
    for i,edg in enumerate(edge['idx']):
        st = int(edg[0])#边的起点
        en = int(edg[1])#边的终点
        idx = int(edg[3])#边的索引
        idx_list[st].append(idx)#在st节点的邻接节点集合中添加en节点
        idx_list[en].append(idx) #若为有向图，这一步可以去掉
    return idx_list


def get_unique_node_sequence(b_edge, f_edge, k, idx_list, flag):
    num_nodes = len(b_edge['idx'])  # 确定节点个数
    node_sequence = np.zeros((num_nodes, k)).astype(np.int32)  # 用于保存节点序列
    node_timestamp = np.zeros((num_nodes, k)).astype(np.int32)  # 用于保存时间差
    node_seq_mask = np.zeros((num_nodes, k)).astype(np.int32)  # 用于保存mask矩阵
    row = int(k / 2)  # 5
    for i, edg in enumerate(b_edge['idx']):  # 遍历每一行数据
        ts = int(edg[2])
        idx = int(edg[3])
        # node序列
        if flag:  # 目标节点
            from_node = int(edg[0])
        else:  # 目的节点
            from_node = int(edg[1])
        if idx in idx_list[from_node]:
            index = idx_list[from_node].index(idx)
        else:
            left = 0
            right = len(idx_list[from_node]) - 1
            while left + 1 < right:  # 二分法找到对应的时间戳
                mid = (left + right) // 2
                curr_idx = idx_list[from_node][mid]
                if curr_idx < idx:
                    left = mid
                else:
                    right = mid
            index = left
        if index > 0:
            last_event = idx_list[from_node][index - 1]
            if last_event < row:
                node = np.zeros(2 * last_event).astype(np.int32)  # 创建一个0矩阵
                time = np.zeros(2 * last_event).astype(np.int32)
                for j, ed in enumerate(f_edge['idx'][0:last_event]):
                    node[2 * j] = ed[1]
                    node[2 * j + 1] = ed[0]
                    time[2 * j] = ts - int(ed[2])
                    time[2 * j + 1] = ts - int(ed[2])
                node = node[::-1]  # 排列倒置
                time = time[::-1]
                # print(node)
                node_timestamp[i, 1:2 * last_event + 1] = time
                node_sequence[i, 1:2 * last_event + 1] = node  # 存入节点序列中
                node_seq_mask[i, 1:2 * last_event + 1] = 1  # 保存mask矩阵
            else:
                node = np.zeros(2 * row).astype(np.int32)  # 创建一个0矩阵
                time = np.zeros(2 * row).astype(np.int32)
                for j, ed in enumerate(f_edge['idx'][last_event - row:last_event]):
                    node[2 * j] = ed[1]
                    node[2 * j + 1] = ed[0]
                    time[2 * j] = ts - int(ed[2])
                    time[2 * j + 1] = ts - int(ed[2])
                node = node[::-1]  # 排列倒置
                time = time[::-1]
                node_timestamp[i, 1:] = time
                node_sequence[i, 1:] = node  # 存入节点序列中
                node_seq_mask[i, 1:] = 1  # 保存mask矩阵
        node_timestamp[i, 0] = 0
        node_sequence[i, 0] = from_node  # 存入节点序列中
        node_seq_mask[i, 0] = 1  # 保存mask矩阵

    return node_sequence, node_seq_mask, node_timestamp