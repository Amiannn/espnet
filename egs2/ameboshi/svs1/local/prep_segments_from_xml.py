#!/usr/bin/env python3
import argparse
import math
import os
import sys

import music21 as m21

from espnet2.fileio.score_scp import SingingScoreWriter, XMLReader

"""Generate segments according to structured musicXML."""
"""Transfer music score (from musicXML) into 'score.json' format."""


class SegInfo(object):
    def __init__(self):
        self.segs = []
        self.start = -1
        self.end = -1

    def add(self, start, end, label, midi):
        start = float(start)
        end = float(end)
        if self.start < 0 or self.start > start:
            self.start = start
        if self.end < end:
            self.end = end
        self.segs.append((start, end, label, midi))

    def split(self, threshold=30):
        seg_num = math.ceil((self.end - self.start) / threshold)
        if seg_num == 1:
            return [self.segs]
        avg = (self.end - self.start) / seg_num
        return_seg = []

        start_time = self.start
        cache_seg = []
        for seg in self.segs:
            cache_time = seg[1] - start_time
            if cache_time > avg:
                return_seg.append(cache_seg)
                start_time = seg[0]
                cache_seg = [seg]
            else:
                cache_seg.append(seg)

        return_seg.append(cache_seg)
        return return_seg


def pack_zero(file_id, number, length=4):
    number = str(number)
    return file_id + "_" + "0" * (length - len(number)) + number


def get_parser():
    parser = argparse.ArgumentParser(
        description="Prepare segments from MUSICXML files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scp", type=str, help="scp folder")
    parser.add_argument(
        "--threshold",
        type=int,
        help="threshold for silence identification.",
        default=30000,
    )
    parser.add_argument(
        "--silence", action="append", help="silence_phone", default=["P"]
    )
    parser.add_argument(
        "--score_dump", type=str, default="score_dump", help="score dump directory"
    )
    args = parser.parse_args()
    return parser


def make_segment(file_id, tempo, notes, threshold, sil=["P", "B"]):
    segments = []
    segment = SegInfo()
    for i in range(len(notes)):
        # Divide songs by 'P' (pause) or 'B' (breath) or GlottalStop
        note = notes[i]
        # fix errors in dataset
        if ("i_vow_to_thee_my_country" in file_id and note.lyric in sil and notes[i + 1].lyric == "ろ") or ("shabondama" in file_id and note.lyric in sil and notes[i + 1].lyric == "だ") or ("kusakeiba" in file_id and note.lyric in sil and notes[i - 3].lyric == "’い"):
            notes[i + 1].st = note.st
            continue
        if note.lyric in sil:
            if len(segment.segs) > 0:
                segments.extend(segment.split(threshold=threshold))
                segment = SegInfo()
            continue
        elif (note.lyric is not None and note.lyric[0] in ["・", "’"]) or ("hana" in file_id and note.lyric == "あ" and notes[i - 1].lyric == "や") or ("hana" in file_id and note.lyric == "も" and notes[i - 1].lyric == "に") or ("santa_lucia" in file_id and note.lyric == "な" and notes[i - 1].lyric == "わ") or ("santa_lucia" in file_id and note.lyric == "そ" and notes[i - 8].lyric == "ぜ") or ("furusato" in file_id and note.lyric == "か" and notes[i + 2].lyric == "か") or ("aria" in file_id and note.lyric == "でぃ" and notes[i - 1].lyric == "い") or ("aria" in file_id and note.lyric == "ぷ" and notes[i - 1].lyric == "じ") or ("antagata_dokosa" in file_id and notes[i - 1].lyric == "さ" and notes[i - 2].lyric != "こ") or ("kusakeiba" in file_id and note.lyric == "どぅ" and notes[i - 1].lyric == "ぎ"):
            if len(segment.segs) > 0:
                segments.extend(segment.split(threshold=threshold))
                segment = SegInfo()
            if note.lyric[0] in ["・", "’"]:
                segment.add(note.st, note.et, note.lyric[1:], note.midi)
            else:
                segment.add(note.st, note.et, note.lyric, note.midi)
            continue
        segment.add(note.st, note.et, note.lyric, note.midi)
    if len(segment.segs) > 0:
        segments.extend(segment.split(threshold=threshold))

    segments_w_id = {}
    id = 0
    for seg in segments:
        if len(seg) == 0:
            continue
        segments_w_id[pack_zero(file_id, id)] = tempo, seg
        id += 1
    return segments_w_id


if __name__ == "__main__":
    args = get_parser().parse_args()
    args.threshold *= 1e-3
    segments = []
    scorescp = open(os.path.join(args.scp, "score.scp"), "r", encoding="utf-8")
    update_segments = open(
        os.path.join(args.scp, "segments_from_xml.tmp"), "w", encoding="utf-8"
    )
    update_text = open(os.path.join(args.scp, "text.tmp"), "w", encoding="utf-8")
    reader = XMLReader(os.path.join(args.scp, "score.scp"))
    for xml_line in scorescp:
        xmlline = xml_line.strip().split(" ")
        recording_id = xmlline[0]
        path = xmlline[1]
        tempo, temp_info = reader[recording_id]
        segments.append(
            make_segment(recording_id, tempo, temp_info, args.threshold, args.silence)
        )
    writer = SingingScoreWriter(
        args.score_dump, os.path.join(args.scp, "score.scp.tmp")
    )
    for file in segments:
        for key, (tempo, val) in file.items():
            segment_begin = "{:.3f}".format(val[0][0])
            segment_end = "{:.3f}".format(val[-1][1])
            update_segments.write(
                "{} {} {} {}\n".format(
                    key, "_".join(key.split("_")[:-1]), segment_begin, segment_end
                )
            )
            update_text.write("{} ".format(key))
            score = dict(tempo=tempo, item_list=["st", "et", "lyric", "midi"], note=val)
            writer[key] = score
            for v in val:
                update_text.write(" {}".format(v[2]))
            update_text.write("\n")
