import time

from CSRGraph4 import read_edge_txt, edgelist_to_CSR
from torch_scatter import segment_csr
from truss_class import TrussFile
from utils import device, calculate_time

import torch

'''
第六版：优化后的存三角形ktruss

去除调试信息
'''

@calculate_time
def calculate_support3(edge_starts: torch.Tensor, edge_ends: torch.Tensor, row_ptr: torch.Tensor,
                       columns: torch.Tensor):
    """
    以节点为中心，求交集
    """
    off = 10000000
    # batch = 20000000
    batch = 4000000  # 一次允许这么多元素求交集

    def get_all_nbr(starts: torch.Tensor, nexts: torch.Tensor):
        """
        :param starts:[1,1,1,5,5]
        :param nexts: [4,4,4,8,8]
        :return:
        """
        sizes = nexts - starts
        nbr_ptr = torch.cat((torch.tensor([0], device=device), torch.cumsum(sizes, dim=0)))  # [0,3,6,9,12,15]
        # starts_indices = nbr_ptr[:-1]  # [0,3,6,9,12]

        # 从索引到点
        nbr = torch.arange(nbr_ptr[-1], device=device) - torch.repeat_interleave(nbr_ptr[:-1] - starts, sizes)
        return nbr, nbr_ptr, sizes

    # edge_supports = torch.tensor([], device=device)
    triangle_source_edge = torch.tensor([], device=device)
    triangle_left_edge = torch.tensor([], device=device)
    triangle_right_edge = torch.tensor([], device=device)
    triangle_id_once = torch.tensor([], dtype=torch.long, device=device)

    sizes = (row_ptr[1:] - row_ptr[:-1])

    # columns中每个点的度数，其中columns即为每轮batch的限制，columns为uvw中的中间点v
    max_nbr_count = segment_csr(sizes[columns], row_ptr)  # 长度与row_ptr相同
    max_nbr_count = max_nbr_count.to(torch.int64).cumsum(0)  # 求和的值太大了，要检查是否溢出了
    # 获取插入的索引
    group = torch.searchsorted(max_nbr_count, torch.arange(0, max_nbr_count[-1] + batch, step=batch, dtype=torch.int64,
                                                           device=device), side='right')

    # 给出arg2在arg1中的索引，max_nbr_count的head，tail索引即为columns中该批次的起始结束位置??错误
    # 给出arg2在arg1中的索引，max_nbr_count的head，tail索引即为row_ptr中该批次的起始结束位置，代表点的编号

    for head, tail in zip(group[0:-1], group[1:]):
        if head == tail:
            # 说明这个点的邻居太多，很可能难以算完，但是没办法，只能在一次batch中计算
            # 所以continue直到tail后移一位，此时head还是不变，不会影响结果
            continue
        u_vertices = torch.arange(head, tail, device=device)

        # 获取u的邻居v
        u_nbr_indices = torch.arange(row_ptr[head], row_ptr[tail], device=device)
        v_vertices = columns[row_ptr[head]: row_ptr[tail]]

        # 计算u.v即直连的u.w
        u_repeat = torch.repeat_interleave(u_vertices, row_ptr[head + 1: tail + 1] - row_ptr[head: tail])
        u_w = u_repeat.type(torch.long) * off + v_vertices

        # 获取v的邻居w
        v_nbr_indices, v_nbr_ptr, v_nbr_sizes = get_all_nbr(row_ptr[v_vertices], row_ptr[v_vertices + 1])
        u_v_nbr = columns[v_nbr_indices]

        # 计算u.w，经过v相连的u.w
        u_v_repeat = torch.repeat_interleave(u_repeat, v_nbr_sizes)
        u_v_w = u_v_repeat.type(torch.long) * off + u_v_nbr

        uvw_mask = torch.isin(u_v_w, u_w)

        # # 来算uv
        batch_support = segment_csr(uvw_mask.int(), v_nbr_ptr)  # 算的是uv的支持度（一个uv下边有几个共同的w点）
        # source_edge = torch.repeat_interleave(column_to_edge[v_nbr_indices][uvw_mask], batch_support)
        # 总长度即为uvw_mask中true的数量

        # 去除column_to_edge之后，原column格式的索引即为边id
        source_edge = torch.repeat_interleave(u_nbr_indices, batch_support)

        # 列出vw
        left_edge = v_nbr_indices[uvw_mask]

        # # 列出uv
        # uv_nbr_indices = torch.repeat_interleave(u_nbr_indices, v_nbr_sizes)
        # right_edge = column_to_edge[uv_nbr_indices][uvw_mask]

        # 列出uw，组成三角形的这些w点，在v中的位置分布, 这些indices从o开始分布，需要加上初始偏移
        uw_indices = torch.bucketize(u_v_w[uvw_mask], u_w) + row_ptr[head]
        right_edge = uw_indices

        # 有向图三角形合成
        triangle_source_edge = torch.cat((triangle_source_edge, source_edge))
        triangle_left_edge = torch.cat((triangle_left_edge, left_edge))
        triangle_right_edge = torch.cat((triangle_right_edge, right_edge))
        # 三角形从零开始编号
        triangle_id_once = torch.cat((triangle_id_once, torch.arange(triangle_id_once.size(0),
                                                                     triangle_id_once.size(0) + right_edge.size(0),
                                                                     device=device)))
        # end += batch

    # print(triangle_id_once)
    # 三角形交叉拼接

    triangle_id = torch.repeat_interleave(triangle_id_once, 3)
    triangle_source = torch.zeros(triangle_id.size(0), device=device)

    triangle_id_once *= 3
    triangle_source[triangle_id_once] = triangle_source_edge
    triangle_source[triangle_id_once + 1] = triangle_left_edge
    triangle_source[triangle_id_once + 2] = triangle_right_edge

    source_edge = torch.cat((triangle_source_edge, triangle_left_edge, triangle_right_edge))
    left_edge = torch.cat((triangle_left_edge, triangle_right_edge, triangle_source_edge))
    right_edge = torch.cat((triangle_right_edge, triangle_source_edge, triangle_left_edge))

    return triangle_source.type(torch.long), triangle_id.type(torch.long) \
        , source_edge.type(torch.long), left_edge.type(torch.long), right_edge.type(torch.long)


@calculate_time
# @profile
def truss_decomposition(triangle_source_sort: torch.Tensor, triangle_id_sort: torch.Tensor, edge_num: int):
    """
    :param triangle_source: 源边id
    :param triangle_id:
    :param edge_num:
    """
    # 既然有三角形，k就得从3开始
    k = 3
    i = 0

    triangle_source, indices_mask = torch.sort(triangle_source_sort, stable=True)
    triangle_id = triangle_id_sort[indices_mask]

    # 计算一次支持度
    edge_id, edge_supports = torch.unique_consecutive(triangle_source, return_inverse=False,
                                                      return_counts=True)

    truss_results = torch.full((edge_num,), 2, dtype=torch.long, device=device)

    all_edge_supports = torch.zeros((edge_num,), dtype=torch.long, device=device)
    all_edge_supports[edge_id] = edge_supports
    all_edge_id = torch.arange(0, edge_num, device=device)

    triangle_ptr = torch.cat((torch.tensor([0], device=device), torch.cumsum(all_edge_supports, dim=0)))

    leave_edge_mask = torch.full((all_edge_id.size(0),), True, device=device)

    # 查找第一轮被删除的边
    selected_edge_mask = all_edge_supports == k - 2
    selected_edge_id = all_edge_id[selected_edge_mask]

    def edge_select_triangle(starts: torch.Tensor, nexts: torch.Tensor):
        """
        :param starts:[1,1,1,5,5]
        :param nexts: [4,4,4,8,8]
        :return:
        """
        sizes = nexts - starts
        nbr_ptr = torch.cat((torch.tensor([0], device=device), torch.cumsum(sizes, dim=0)))  # [0,3,6,9,12,15]
        # starts_indices = nbr_ptr[:-1]  # [0,3,6,9,12]

        # 从索引到点
        triangle_indices = torch.arange(nbr_ptr[-1], device=device) - torch.repeat_interleave(nbr_ptr[:-1] - starts, sizes)
        return triangle_indices, nbr_ptr

    # 对于每个ktruss，都获取support==k-2的，直至不存在support==k-2的边
    while True:
        i += 1

        if selected_edge_id.size(0) == 0:
            k += 1
            if k > 300:
                break
            selected_edge_mask = all_edge_supports == k - 2
            selected_edge_id = all_edge_id[selected_edge_mask]
            continue

        # 被删除的边，标记为false
        leave_edge_mask[selected_edge_id] = False

        # 这一轮剩下的triangle，即全都留下
        leave_triangle_mask = torch.full((triangle_id.size(0),), True, device=device)

        # 由edgeid获得的三角形索引
        triangle_indices, nbr_ptr = edge_select_triangle(triangle_ptr[selected_edge_id], triangle_ptr[selected_edge_id+1])

        # 索引位置设置为false，记为该位置三角形被删除
        leave_triangle_mask[triangle_indices] = False

        # 获得三角形的唯一值
        selected_triangle_id, counts = torch.unique(triangle_id[triangle_indices], return_counts=True)  # 获得三角形id的唯一值

        # 找到三角形计数少于三次的三角形id，这样的三角形会影响到其他边
        selected_triangle_id = selected_triangle_id[counts <= 2]

        # 这里乘3了
        no_all_triangle_id = 3*selected_triangle_id

        selected_triangle_indices = torch.cat(
            (no_all_triangle_id, no_all_triangle_id + 1, no_all_triangle_id + 2))

        # 查找所有的被影响的边
        # 还是包含一部分被选择的边
        all_affected_edge = triangle_source_sort[selected_triangle_indices]  # 去sort的source中查找所有受影响的边

        all_affected_edge, counts = torch.unique(all_affected_edge, return_counts=True)

        # 查找受影响的边（不包含被选择的边）
        affected_edge_mask = leave_edge_mask[all_affected_edge]

        # 排除在这轮被删除的边
        affected_edge = all_affected_edge[affected_edge_mask]
        counts = counts[affected_edge_mask]

        # 更新这些边的支持度
        all_edge_supports[selected_edge_id] = 0
        all_edge_supports[affected_edge] -= counts

        # 临时support
        temp_supports = all_edge_supports[affected_edge]

        # 支持度莫名奇妙被减到0的边，已经不需要在下一轮被选择了
        # 支持度被动减到0，说明这条边所参与的三角形已经全删了
        selected_edge_mask = temp_supports <= k-2
        selected_edge_mask2 = temp_supports > 0

        # 写入结果
        '''
        被删除的边的位置写入trussnumber值
        '''
        truss_results[selected_edge_id] = k
        # 支持度减为0的边，不让他去下一轮，直接在这轮写入结果
        truss_results[affected_edge[~selected_edge_mask2]] = k

        # 某些边的支持度也转为0了，需要被计入leavemask
        leave_edge_mask[affected_edge[~selected_edge_mask2]] = False

        # 在这之间的边为下一轮被删除的边
        selected_edge_mask = selected_edge_mask & selected_edge_mask2
        selected_edge_id = affected_edge[selected_edge_mask]

        # 查找受影响的边的三角形， 与原始三角形列表求交集
        affected_triangle_indices, nbr_ptr = edge_select_triangle(triangle_ptr[affected_edge], triangle_ptr[affected_edge+1])

        affected_triangle = triangle_id[affected_triangle_indices]

        affected_triangle_mask = torch.isin(affected_triangle, selected_triangle_id)

        leave_triangle_mask[affected_triangle_indices[affected_triangle_mask]] = False

        # 更新三角形结构
        triangle_id = triangle_id[leave_triangle_mask]
        triangle_ptr = torch.cat((torch.tensor([0], device=device), torch.cumsum(segment_csr(leave_triangle_mask.int(), triangle_ptr), dim=0)))


        # 所有三角形均已被删除
        if triangle_id.size(0) == 0:
            break

    return truss_results
    # k, num = torch.unique(truss_results, return_counts=True)
    # num = torch.flip(torch.cumsum(torch.flip(num, dims=(0,)), dim=0), dims=(0,))
    # print(k, num)


def serialization_truss_clsss_save():
    edge_starts, edge_ends, old_vertices_hash = read_edge_txt(r"C:\Users\Administrator\Desktop\graph\data\facebook_combined.txt", 0)
    row_ptr, columns = edgelist_to_CSR(edge_starts, edge_ends, direct=True)

    row_ptr = torch.tensor(row_ptr, device=device, dtype=torch.long)
    columns = torch.tensor(columns, device=device, dtype=torch.long)
    print("finish...")

    # 想办法处理掉支持度为0的边，即孤立边，即某些点只有一个邻居
    edge_num = columns.size(0)

    time1 = time.time()
    triangle_source, triangle_id, source_edge, left_edge, right_edge = calculate_support3(edge_starts, edge_ends,
                                                                                          row_ptr, columns)
    
    # print(triangle_source.size(0))
    truss_result = truss_decomposition(triangle_source, triangle_id, edge_num)

    truss_file = TrussFile(row_ptr, columns, truss_result, old_vertices_hash, source_edge, left_edge, right_edge)

    truss_file.save("catster.pt")

    time2 = time.time()
    print("truss分解总时间：" + str(time2 - time1))


if __name__ == '__main__':
    serialization_truss_clsss_save()
    print(torch.__version__)

