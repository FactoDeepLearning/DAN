#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#
#  This software is a computer program written in Python  whose purpose is to
#  provide public implementation of deep learning works, in pytorch.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.


from Datasets.dataset_formatters.rimes_formatter import SEM_MATCHING_TOKENS as RIMES_MATCHING_TOKENS
from Datasets.dataset_formatters.read2016_formatter import SEM_MATCHING_TOKENS as READ_MATCHING_TOKENS
import re
import networkx as nx
import editdistance
import numpy as np
from basic.post_pocessing_layout import PostProcessingModuleREAD, PostProcessingModuleRIMES


class MetricManager:

    def __init__(self, metric_names, dataset_name):
        self.dataset_name = dataset_name
        if "READ" in dataset_name and "page" in dataset_name:
            self.post_processing_module = PostProcessingModuleREAD
            self.matching_tokens = READ_MATCHING_TOKENS
            self.edit_and_num_edge_nodes = edit_and_num_items_for_ged_from_str_read
        elif "RIMES" in dataset_name and "page" in dataset_name:
            self.post_processing_module = PostProcessingModuleRIMES
            self.matching_tokens = RIMES_MATCHING_TOKENS
            self.edit_and_num_edge_nodes = edit_and_num_items_for_ged_from_str_rimes
        else:
            self.matching_tokens = dict()

        self.layout_tokens = "".join(list(self.matching_tokens.keys()) + list(self.matching_tokens.values()))
        if len(self.layout_tokens) == 0:
            self.layout_tokens = None
        self.metric_names = metric_names
        self.epoch_metrics = None

        self.linked_metrics = {
            "cer": ["edit_chars", "nb_chars"],
            "wer": ["edit_words", "nb_words"],
            "ger": ["edit_graph", "nb_nodes_and_edges", "nb_pp_op_layout", "nb_gt_layout_token"],
            "precision": ["precision", "weights"],
            "layout_mAP_per_class": ["layout_mAP", ],
            "layout_precision_per_class_per_threshold": ["layout_mAP", ],
        }

        self.init_metrics()

    def init_metrics(self):
        """
        Initialization of the metrics specified in metrics_name
        """
        self.epoch_metrics = {
            "nb_samples": list(),
            "names": list(),
            "ids": list(),
        }

        for metric_name in self.metric_names:
            if metric_name in self.linked_metrics:
                for linked_metric_name in self.linked_metrics[metric_name]:
                    if linked_metric_name not in self.epoch_metrics.keys():
                        self.epoch_metrics[linked_metric_name] = list()
            else:
                self.epoch_metrics[metric_name] = list()

    def update_metrics(self, batch_metrics):
        """
        Add batch metrics to the metrics
        """
        for key in batch_metrics.keys():
            if key in self.epoch_metrics:
                self.epoch_metrics[key] += batch_metrics[key]

    def get_display_values(self, output=False):
        """
        format metrics values for shell display purposes
        """
        metric_names = self.metric_names.copy()
        if output:
            metric_names.extend(["nb_samples"])
        display_values = dict()
        for metric_name in metric_names:
            value = None
            if output:
                if metric_name in ["nb_samples", "weights"]:
                    value = np.sum(self.epoch_metrics[metric_name])
                elif metric_name in ["time", ]:
                    total_time = np.sum(self.epoch_metrics[metric_name])
                    sample_time = total_time / np.sum(self.epoch_metrics["nb_samples"])
                    display_values["sample_time"] = round(sample_time, 4)
                    value = total_time
                elif metric_name == "layout_mAP_per_class":
                    value = compute_global_mAP_per_class(self.epoch_metrics["layout_mAP"])
                    for key in value.keys():
                        display_values["mAP_" + key] = round(value[key], 4)
                    continue
                elif metric_name == "layout_precision_per_class_per_threshold":
                    value = compute_global_precision_per_class_per_threshold(self.epoch_metrics["layout_mAP"])
                    for key_class in value.keys():
                        for threshold in value[key_class].keys():
                            display_values["mAP_{}_{}".format(key_class, threshold)] = round(
                                value[key_class][threshold], 4)
                    continue
            if metric_name == "cer":
                value = np.sum(self.epoch_metrics["edit_chars"]) / np.sum(self.epoch_metrics["nb_chars"])
                if output:
                    display_values["nb_chars"] = np.sum(self.epoch_metrics["nb_chars"])
            elif metric_name == "wer":
                value = np.sum(self.epoch_metrics["edit_words"]) / np.sum(self.epoch_metrics["nb_words"])
                if output:
                    display_values["nb_words"] = np.sum(self.epoch_metrics["nb_words"])
            elif metric_name in ["loss", "loss_ctc", "loss_ce", "syn_max_lines"]:
                value = np.average(self.epoch_metrics[metric_name], weights=np.array(self.epoch_metrics["nb_samples"]))
            elif metric_name == "layout_mAP":
                value = compute_global_mAP(self.epoch_metrics[metric_name])
            elif metric_name == "ger":
                value = np.sum(self.epoch_metrics["edit_graph"]) / np.sum(self.epoch_metrics["nb_nodes_and_edges"])
                display_values["percent_pp_op"] = round(np.sum(self.epoch_metrics["nb_pp_op_layout"]) / np.sum(self.epoch_metrics["nb_gt_layout_token"]), 4)
            elif value is None:
                continue

            display_values[metric_name] = round(value, 4)
        return display_values

    def compute_metrics(self, values, metric_names):
        metrics = {
            "nb_samples": [values["nb_samples"], ],
        }
        for v in ["weights", "time"]:
            if v in values:
                metrics[v] = [values[v]]
        for metric_name in metric_names:
            if metric_name == "cer":
                metrics["edit_chars"] = [edit_cer_from_string(u, v, self.layout_tokens) for u, v in zip(values["str_y"], values["str_x"])]
                metrics["nb_chars"] = [nb_chars_cer_from_string(gt, self.layout_tokens) for gt in values["str_y"]]
            elif metric_name == "wer":
                split_gt = [format_string_for_wer(gt, self.layout_tokens) for gt in values["str_y"]]
                split_pred = [format_string_for_wer(pred, self.layout_tokens) for pred in values["str_x"]]
                metrics["edit_words"] = [edit_wer_from_formatted_split_text(gt, pred) for (gt, pred) in zip(split_gt, split_pred)]
                metrics["nb_words"] = [len(gt) for gt in split_gt]
            elif metric_name in ["loss_ctc", "loss_ce", "loss", "syn_max_lines", ]:
                metrics[metric_name] = [values[metric_name], ]
            elif metric_name == "layout_mAP":
                pp_pred = list()
                pp_score = list()
                for pred, score in zip(values["str_x"], values["confidence_score"]):
                    pred_score = self.post_processing_module().post_process(pred, score)
                    pp_pred.append(pred_score[0])
                    pp_score.append(pred_score[1])
                metrics[metric_name] = [compute_layout_mAP_per_class(y, x, conf, self.matching_tokens) for x, conf, y in zip(pp_pred, pp_score, values["str_y"])]
            elif metric_name == "ger":
                pp_pred = list()
                metrics["nb_pp_op_layout"] = list()
                for pred in values["str_x"]:
                    pp_module = self.post_processing_module()
                    pp_pred.append(pp_module.post_process(pred))
                    metrics["nb_pp_op_layout"].append(pp_module.num_op)
                metrics["nb_gt_layout_token"] = [len(keep_only_tokens(str_x, self.layout_tokens)) for str_x in values["str_x"]]
                edit_and_num_items = [self.edit_and_num_edge_nodes(y, x) for x, y in zip(pp_pred, values["str_y"])]
                metrics["edit_graph"], metrics["nb_nodes_and_edges"] = [ei[0] for ei in edit_and_num_items], [ei[1] for ei in edit_and_num_items]
        return metrics

    def get(self, name):
        return self.epoch_metrics[name]


def keep_only_tokens(str, tokens):
    """
    Remove all but layout tokens from string
    """
    return re.sub('([^' + tokens + '])', '', str)


def keep_all_but_tokens(str, tokens):
    """
    Remove all layout tokens from string
    """
    return re.sub('([' + tokens + '])', '', str)


def edit_cer_from_string(gt, pred, layout_tokens=None):
    """
    Format and compute edit distance between two strings at character level
    """
    gt = format_string_for_cer(gt, layout_tokens)
    pred = format_string_for_cer(pred, layout_tokens)
    return editdistance.eval(gt, pred)


def nb_chars_cer_from_string(gt, layout_tokens=None):
    """
    Compute length after formatting of ground truth string
    """
    return len(format_string_for_cer(gt, layout_tokens))


def edit_wer_from_string(gt, pred, layout_tokens=None):
    """
    Format and compute edit distance between two strings at word level
    """
    split_gt = format_string_for_wer(gt, layout_tokens)
    split_pred = format_string_for_wer(pred, layout_tokens)
    return edit_wer_from_formatted_split_text(split_gt, split_pred)


def format_string_for_wer(str, layout_tokens):
    """
    Format string for WER computation: remove layout tokens, treat punctuation as word, replace line break by space
    """
    str = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', str)  # punctuation processed as word
    if layout_tokens is not None:
        str = keep_all_but_tokens(str, layout_tokens)  # remove layout tokens from metric
    str = re.sub('([ \n])+', " ", str).strip()  # keep only one space character
    return str.split(" ")


def format_string_for_cer(str, layout_tokens):
    """
    Format string for CER computation: remove layout tokens and extra spaces
    """
    if layout_tokens is not None:
        str = keep_all_but_tokens(str, layout_tokens)  # remove layout tokens from metric
    str = re.sub('([\n])+', "\n", str)  # remove consecutive line breaks
    str = re.sub('([ ])+', " ", str).strip()  # remove consecutive spaces
    return str


def edit_wer_from_formatted_split_text(gt, pred):
    """
    Compute edit distance at word level from formatted string as list
    """
    return editdistance.eval(gt, pred)


def extract_by_tokens(input_str, begin_token, end_token, associated_score=None, order_by_score=False):
    """
    Extract list of text regions by begin and end tokens
    Order the list by confidence score
    """
    if order_by_score:
        assert associated_score is not None
    res = list()
    for match in re.finditer("{}[^{}]*{}".format(begin_token, end_token, end_token), input_str):
        begin, end = match.regs[0]
        if order_by_score:
            res.append({
                "confidence": np.mean([associated_score[begin], associated_score[end-1]]),
                "content": input_str[begin+1:end-1]
            })
        else:
            res.append(input_str[begin+1:end-1])
    if order_by_score:
        res = sorted(res, key=lambda x: x["confidence"], reverse=True)
        res = [r["content"] for r in res]
    return res


def compute_layout_precision_per_threshold(gt, pred, score, begin_token, end_token, layout_tokens, return_weight=True):
    """
    Compute average precision of a given class for CER threshold from 5% to 50% with a step of 5%
    """
    pred_list = extract_by_tokens(pred, begin_token, end_token, associated_score=score, order_by_score=True)
    gt_list = extract_by_tokens(gt, begin_token, end_token)
    pred_list = [keep_all_but_tokens(p, layout_tokens) for p in pred_list]
    gt_list = [keep_all_but_tokens(gt, layout_tokens) for gt in gt_list]
    precision_per_threshold = [compute_layout_AP_for_given_threshold(gt_list, pred_list, threshold/100) for threshold in range(5, 51, 5)]
    if return_weight:
        return precision_per_threshold, len(gt_list)
    return precision_per_threshold


def compute_layout_AP_for_given_threshold(gt_list, pred_list, threshold):
    """
    Compute average precision of a given class for a given CER threshold
    """
    remaining_gt_list = gt_list.copy()
    num_true = len(gt_list)
    correct = np.zeros((len(pred_list)), dtype=np.bool)
    for i, pred in enumerate(pred_list):
        if len(remaining_gt_list) == 0:
            break
        cer_with_gt = [edit_cer_from_string(gt, pred)/nb_chars_cer_from_string(gt) for gt in remaining_gt_list]
        cer, ind = np.min(cer_with_gt), np.argmin(cer_with_gt)
        if cer <= threshold:
            correct[i] = True
            del remaining_gt_list[ind]
    precision = np.cumsum(correct, dtype=np.int) / np.arange(1, len(pred_list)+1)
    recall = np.cumsum(correct, dtype=np.int) / num_true
    max_precision_from_recall = np.maximum.accumulate(precision[::-1])[::-1]
    recall_diff = (recall - np.concatenate([np.array([0, ]), recall[:-1]]))
    P = np.sum(recall_diff * max_precision_from_recall)
    return P


def compute_layout_mAP_per_class(gt, pred, score, tokens):
    """
    Compute the mAP_cer for each class for a given sample
    """
    layout_tokens = "".join(list(tokens.keys()))
    AP_per_class = dict()
    for token in tokens.keys():
        if token in gt:
            AP_per_class[token] = compute_layout_precision_per_threshold(gt, pred, score, token, tokens[token], layout_tokens=layout_tokens)
    return AP_per_class


def compute_global_mAP(list_AP_per_class):
    """
    Compute the global mAP_cer for several samples
    """
    weights_per_doc = list()
    mAP_per_doc = list()
    for doc_AP_per_class in list_AP_per_class:
        APs = np.array([np.mean(doc_AP_per_class[key][0]) for key in doc_AP_per_class.keys()])
        weights = np.array([doc_AP_per_class[key][1] for key in doc_AP_per_class.keys()])
        if np.sum(weights) == 0:
            mAP_per_doc.append(0)
        else:
            mAP_per_doc.append(np.average(APs, weights=weights))
        weights_per_doc.append(np.sum(weights))
    if np.sum(weights_per_doc) == 0:
        return 0
    return np.average(mAP_per_doc, weights=weights_per_doc)


def compute_global_mAP_per_class(list_AP_per_class):
    """
    Compute the mAP_cer per class for several samples
    """
    mAP_per_class = dict()
    for doc_AP_per_class in list_AP_per_class:
        for key in doc_AP_per_class.keys():
            if key not in mAP_per_class:
                mAP_per_class[key] = {
                    "AP": list(),
                    "weights": list()
                }
            mAP_per_class[key]["AP"].append(np.mean(doc_AP_per_class[key][0]))
            mAP_per_class[key]["weights"].append(doc_AP_per_class[key][1])
    for key in mAP_per_class.keys():
        mAP_per_class[key] = np.average(mAP_per_class[key]["AP"], weights=mAP_per_class[key]["weights"])
    return mAP_per_class


def compute_global_precision_per_class_per_threshold(list_AP_per_class):
    """
    Compute the mAP_cer per class and per threshold for several samples
    """
    mAP_per_class = dict()
    for doc_AP_per_class in list_AP_per_class:
        for key in doc_AP_per_class.keys():
            if key not in mAP_per_class:
                mAP_per_class[key] = dict()
                for threshold in range(5, 51, 5):
                    mAP_per_class[key][threshold] = {
                        "precision": list(),
                        "weights": list()
                    }
            for i, threshold in enumerate(range(5, 51, 5)):
                mAP_per_class[key][threshold]["precision"].append(np.mean(doc_AP_per_class[key][0][i]))
                mAP_per_class[key][threshold]["weights"].append(doc_AP_per_class[key][1])
    for key_class in mAP_per_class.keys():
        for threshold in mAP_per_class[key_class]:
            mAP_per_class[key_class][threshold] = np.average(mAP_per_class[key_class][threshold]["precision"], weights=mAP_per_class[key_class][threshold]["weights"])
    return mAP_per_class


def str_to_graph_read(str):
    """
    Compute graph from string of layout tokens for the READ 2016 dataset at single-page and double-page levels
    """
    begin_layout_tokens = "".join(list(READ_MATCHING_TOKENS.keys()))
    layout_token_sequence = keep_only_tokens(str, begin_layout_tokens)
    g = nx.DiGraph()
    g.add_node("D", type="document", level=4, page=0)
    num = {
        "ⓟ": 0,
        "ⓐ": 0,
        "ⓑ": 0,
        "ⓝ": 0,
        "ⓢ": 0
    }
    previous_top_level_node = None
    previous_middle_level_node = None
    previous_low_level_node = None
    for ind, c in enumerate(layout_token_sequence):
        num[c] += 1
        if c == "ⓟ":
            node_name = "P_{}".format(num[c])
            g.add_node(node_name, type="page", level=3, page=num["ⓟ"])
            g.add_edge("D", node_name)
            if previous_top_level_node:
                g.add_edge(previous_top_level_node, node_name)
            previous_top_level_node = node_name
            previous_middle_level_node = None
            previous_low_level_node = None
        if c in "ⓝⓢ":
            node_name = "{}_{}".format("N" if c == "ⓝ" else "S", num[c])
            g.add_node(node_name, type="number" if c == "ⓝ" else "section", level=2, page=num["ⓟ"])
            g.add_edge(previous_top_level_node, node_name)
            if previous_middle_level_node:
                g.add_edge(previous_middle_level_node, node_name)
            previous_middle_level_node = node_name
            previous_low_level_node = None
        if c in "ⓐⓑ":
            node_name = "{}_{}".format("A" if c == "ⓐ" else "B", num[c])
            g.add_node(node_name, type="annotation" if c == "ⓐ" else "body", level=1, page=num["ⓟ"])
            g.add_edge(previous_middle_level_node, node_name)
            if previous_low_level_node:
                g.add_edge(previous_low_level_node, node_name)
            previous_low_level_node = node_name
    return g


def str_to_graph_rimes(str):
    """
    Compute graph from string of layout tokens for the RIMES dataset at page level
    """
    begin_layout_tokens = "".join(list(RIMES_MATCHING_TOKENS.keys()))
    layout_token_sequence = keep_only_tokens(str, begin_layout_tokens)
    g = nx.DiGraph()
    g.add_node("D", type="document", level=2, page=0)
    token_name_dict = {
        "ⓑ": "B",
        "ⓞ": "O",
        "ⓡ": "R",
        "ⓢ": "S",
        "ⓦ": "W",
        "ⓨ": "Y",
        "ⓟ": "P"
    }
    num = dict()
    previous_node = None
    for token in begin_layout_tokens:
        num[token] = 0
    for ind, c in enumerate(layout_token_sequence):
        num[c] += 1
        node_name = "{}_{}".format(token_name_dict[c], num[c])
        g.add_node(node_name, type=token_name_dict[c], level=1, page=0)
        g.add_edge("D", node_name)
        if previous_node:
            g.add_edge(previous_node, node_name)
        previous_node = node_name
    return g


def graph_edit_distance_by_page_read(g1, g2):
    """
    Compute graph edit distance page by page for the READ 2016 dataset
    """
    num_pages_g1 = len([n for n in g1.nodes().items() if n[1]["level"] == 3])
    num_pages_g2 = len([n for n in g2.nodes().items() if n[1]["level"] == 3])
    page_graphs_1 = [g1.subgraph([n[0] for n in g1.nodes().items() if n[1]["page"] == num_page]) for num_page in range(1, num_pages_g1+1)]
    page_graphs_2 = [g2.subgraph([n[0] for n in g2.nodes().items() if n[1]["page"] == num_page]) for num_page in range(1, num_pages_g2+1)]
    edit = 0
    for i in range(max(len(page_graphs_1), len(page_graphs_2))):
        page_1 = page_graphs_1[i] if i < len(page_graphs_1) else nx.DiGraph()
        page_2 = page_graphs_2[i] if i < len(page_graphs_2) else nx.DiGraph()
        edit += graph_edit_distance(page_1, page_2)
    return edit


def graph_edit_distance(g1, g2):
    """
    Compute graph edit distance between two graphs
    """
    for v in nx.optimize_graph_edit_distance(g1, g2,
                                             node_ins_cost=lambda node: 1,
                                             node_del_cost=lambda node: 1,
                                             node_subst_cost=lambda node1, node2: 0 if node1["type"] == node2["type"] else 1,
                                             edge_ins_cost=lambda edge: 1,
                                             edge_del_cost=lambda edge: 1,
                                             edge_subst_cost=lambda edge1, edge2: 0 if edge1 == edge2 else 1
                                             ):
        new_edit = v
    return new_edit


def edit_and_num_items_for_ged_from_str_read(str_gt, str_pred):
    """
    Compute graph edit distance and num nodes/edges for normalized graph edit distance
    For the READ 2016 dataset
    """
    g_gt = str_to_graph_read(str_gt)
    g_pred = str_to_graph_read(str_pred)
    return graph_edit_distance_by_page_read(g_gt, g_pred), g_gt.number_of_nodes() + g_gt.number_of_edges()


def edit_and_num_items_for_ged_from_str_rimes(str_gt, str_pred):
    """
    Compute graph edit distance and num nodes/edges for normalized graph edit distance
    For the RIMES dataset
    """
    g_gt = str_to_graph_rimes(str_gt)
    g_pred = str_to_graph_rimes(str_pred)
    return graph_edit_distance(g_gt, g_pred), g_gt.number_of_nodes() + g_gt.number_of_edges()