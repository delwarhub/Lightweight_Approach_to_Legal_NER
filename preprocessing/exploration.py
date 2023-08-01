"""Dataset exploration."""

import numpy as np
from data.data_util import load_train_part, get_instance_by_id


def check_num_labels(part: str = '', min_lbls: int = 0, sort_reverse: bool = True) -> list:
    """
    Check how many labels each data instance has.

    :param part: String specifying part; 'pre'=preambles, 'jdg'=judgements; pre+jdg if omitted.
    :param min_lbls: Minimum number of labels - only instances with at least this many labels are returned.
    :param sort_reverse: Bool value for the reverse argument of the sort function.
    :return: Sorted list of (number of labels, instance id) tuples.
    """
    data = load_train_part(part)

    return sorted([(len(instance['annotations'][0]['result']), instance['id']) for instance in data
                   if len(instance['annotations'][0]['result']) >= min_lbls], reverse=sort_reverse)


def check_text_lengths(part: str = '', threshold: int = 2048) -> list:
    """
    Check lengths of instance texts.

    :param part: String specifying part; 'pre'=preambles, 'jdg'=judgements; pre+jdg if omitted.
    :param threshold: Text length threshold.
    :return: List of text lengths, same indexes as dataset.
    """
    data = load_train_part(part)

    txt_lens = [len(instance['data']['text']) for instance in data]

    max_len = np.max(txt_lens)
    max_id = np.argmax(txt_lens)

    min_len = np.min(txt_lens)

    mean_len = np.mean(txt_lens)
    med_len = np.median(txt_lens)
    std_len = np.std(txt_lens)

    print(f"Text length:\nMax: {max_len}, Min: {min_len}, Mean: {mean_len}, Median: {med_len}, STD: {std_len}")

    # len_dist = {length: txt_lens.count(length) for length in set(txt_lens)}
    # print(len_dist[min_len])

    below_thr = len([count for count in txt_lens if count < threshold])

    print(f"{below_thr} out of {len(txt_lens)} ({(below_thr/len(txt_lens))*100}%) below length {threshold}.")

    return txt_lens


def get_gaps(part: str = '') -> list:
    """
    Get character gaps between labeled spans.

    :param part: String specifying part; 'pre'=preambles, 'jdg'=judgements; pre+jdg if omitted.
    :return: List of gaps with their adjacent labels and lengths.
    """
    data = load_train_part(part)

    all_gaps = []
    for instance in data:
        spans = []
        for annotation in instance['annotations'][0]['result']:
            start = annotation['value']['start']
            end = annotation['value']['end']
            label = annotation['value']['labels'][0]
            lbl_span = (start, label, end)
            spans.append(lbl_span)
        span_gaps = []
        for span_i in range(len(spans) - 1):
            first_span_end = spans[span_i][2]
            first_span_lbl = spans[span_i][1]
            scnd_span_start = spans[span_i + 1][0]
            scnd_span_lbl = spans[span_i + 1][1]
            lbl_span_gap = (first_span_lbl,
                            scnd_span_start - first_span_end,
                            first_span_end, scnd_span_start,
                            scnd_span_lbl)
            span_gaps.append(lbl_span_gap)
        all_gaps.append(span_gaps)

    return all_gaps


def check_span_adjacency(threshold: int = 2, part: str = '') -> set:
    """
    Check if any labeled spans are adjacent/how close they are by checking character gaps.

    :param threshold: Upper limit for gap lengths returned.
    :param part: String specifying part; 'pre'=preambles, 'jdg'=judgements; pre+jdg if omitted.
    :return: Set of instance indices with gaps of or below threshold.
    """
    gaps = get_gaps(part)

    inst_ids = set()
    # collect all indices of instances with fitting gaps:
    for instance_i in range(len(gaps)):
        if len(gaps[instance_i]) > 0:
            for gap in gaps[instance_i]:
                if gap[1] <= threshold:
                    inst_ids.add(instance_i)

    return inst_ids


def show_adjacent_instances(threshold: int = 2, gap_txt: bool = False, part: str = '') -> None:
    """
    Shows total number of instances with gaps within threshold, total number of single-space gaps and optionally all
    found gaps marked as they appear in the instance text.

    :param threshold: Upper limit for gap lengths.
    :param gap_txt: If True, marked instance texts for each found gap are printed.
    :param part: String specifying part; 'pre'=preambles, 'jdg'=judgements; pre+jdg if omitted.
    :return: Console output.
    """
    # needed objects:
    data = load_train_part(part)
    inst_ids = check_span_adjacency(threshold, part)
    gaps = get_gaps(part)
    # counter for single-space gaps:
    one_space_count = 0
    # go over all instances that have fitting gaps:
    for inst_id in inst_ids:
        # get the text of instance:
        inst_txt = data[inst_id]['data']['text']
        # go through fitting gaps of instance:
        for gap in gaps[inst_id]:
            if gap[1] <= threshold:
                gap_content = inst_txt[gap[2]:gap[3]]
                # print text formatted for checking for each fitting gap:
                if gap_txt:
                    print(f"ID {inst_id} '{gap_content}' "
                          f"{inst_txt[:gap[2]]}[{gap[0]}]{gap_content}[{gap[4]}]{inst_txt[gap[3]:]}'\n")
                # count single-space gaps:
                if gap_content == " ":
                    one_space_count += 1

    print(f"{len(inst_ids)} instances with gaps <= {threshold} found.")
    print(f"{one_space_count} single-space gaps found.")


def check_id_unique() -> list:
    """
    Check for dataset instance ID uniqueness. Returns list of non-unique IDs.

    :return: List of non-unique IDs.
    """
    data = load_train_part()

    found_ids = set()
    duplicate_ids = []
    for instance in data:
        if instance['id'] not in found_ids:
            found_ids.add(instance['id'])
        else:
            duplicate_ids.append(instance['id'])

    return duplicate_ids


if __name__ == "__main__":
    # print(check_id_unique())

    # print(check_num_labels())
    print(check_num_labels('jdg', 10))
    # print(check_num_labels('jdg', 2, sort_reverse=False))
    print(get_instance_by_id('00212ec9f8a345d3a12c1022e2deab1e'))
    # print(get_instance_by_id(check_num_labels('jdg', 10)[0][1]))
    # print(get_instance_by_id(check_num_labels('jdg', 2, sort_reverse=False)[1][1]))


    check_text_lengths('pre')
    check_text_lengths('jdg')
    """Length threshold of 2048 is informative as a (incredibly rough) heuristic to determine if texts might be too long
    for a model: With average luck, a BPE token is roughly four characters long. Based on this and a context of 512 
    tokens, a very rough estimate of the amount of texts that will have to be discarded or truncated can be made."""

    show_adjacent_instances(1)
    """Finding these is highly informative before tokenization and label mapping, as certain tokenizers (most BPE) have 
    single-space-led tokens. This is useful as it preserves word boundary information, but could lead to complications 
    with labeled spans after tokenization. The high number of single-space gaps highly suggests that BIO labeling of 
    tokens will be necessary."""
