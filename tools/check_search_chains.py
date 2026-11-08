#!/usr/bin/env python3
"""Offline check (0 tokens) of the multi-hop kb questions in det_solver.solve_search.

Uses the organisers' REAL knowledge bases from the sample tasks (search_*/workspace/kb) and an independent ground truth:
this file parses the markdown itself and walks the graph itself; it shares no code with det_solver. For every person,
every mentor depth 0..MAXH, every target (born / mentor name / employer / headquarters / founded), several paraphrases of each question and three
ways of writing the chain
("X's mentor's mentor", "the mentor of the mentor of X", a mix) it asks the solver and compares.

    python3 tools/check_search_chains.py <directory with the sample task dirs> [--maxh 5]

Exit status 1 on any WRONG answer. NOMATCH (question goes to the LLM) is reported but is not an error for the
phrasings the solver does not claim to know.
"""
from __future__ import annotations

import argparse
import collections
import os
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import det_solver  # noqa: E402

FRAME = ("The directory kb/ contains a small knowledge base of markdown files. {q} Use the file tools or bash (ls, grep, "
         "cat) to find it. Work only inside the current directory. When you are done, write ONLY the final answer to the "
         "file answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")


def read_graph(kb: str):
    people, comps = {}, {}
    for fn in sorted(os.listdir(kb)):
        t = open(os.path.join(kb, fn), encoding="utf-8").read()
        name = re.match(r"# (.+)", t).group(1).strip()
        f = {k: v.strip() for k, v in re.findall(r"^(\w+): (.+)$", t, re.M)}
        (people if fn.startswith("person_") else comps)[name] = f
    return people, comps


def chain_phrases(name: str, k: int):
    """Different ways to write 'the person k mentor-hops after `name`'."""
    if k == 0:
        return [name]
    out = [name + "'s mentor" * k, "the mentor of " * k + name]
    if k >= 2:
        out.append("the mentor of " + name + "'s mentor" * (k - 1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("--maxh", type=int, default=5)
    args = ap.parse_args()
    kbs = [os.path.join(args.samples, d, "workspace") for d in sorted(os.listdir(args.samples))
           if d.startswith("search_") and os.path.isdir(os.path.join(args.samples, d, "workspace", "kb"))]
    if not kbs:
        print("no search_* samples with a kb/ found in", args.samples)
        return 2
    stat = collections.Counter()
    wrong = []
    for ws in kbs:
        people, comps = read_graph(os.path.join(ws, "kb"))
        for name in sorted(people):
            cur, ok_chain = name, True
            for k in range(args.maxh + 1):
                if k:
                    cur = people[cur].get("Mentor")
                    if cur not in people:
                        ok_chain = False
                        break
                p = people[cur]
                emp = p["Employer"]
                # target -> (frames asking for it, how the company is written (None: the person is the subject), expected)
                person_frames = [
                    ("born", ["In which year was {x} born?", "In what year was {x} born?", "What year was {x} born?",
                              "What is the birth year of {x}?"], p["Born"]),
                    ("mentor", ["Who is the mentor of {x}?", "Who is {x}'s mentor?"], p["Mentor"]),
                    ("employer", ["Which company employs {x}?", "Who is the employer of {x}?"], emp),
                ]
                company_frames = [
                    ("hq", ["In which city is the headquarters of {x}?", "Where is the headquarters of {x}?",
                            "In which city is {x} headquartered?"], comps[emp]["Headquarters"]),
                    ("founded", ["In which year was {x} founded?", "In what year was {x} founded?",
                                 "What is the founding year of {x}?"], comps[emp]["Founded"]),
                ]
                for phrase in chain_phrases(name, k):
                    for tname, frames, expected in person_frames:
                        for fr in frames:
                            q = fr.format(x=phrase)
                            r = det_solver.solve_search(det_solver._norm(FRAME.format(q=q)), ws)
                            stat[(tname, k, r.status)] += 1
                            if r.status != "answer" or r.answer != expected:
                                if r.status in ("answer", "unsolvable"):
                                    wrong.append((q, expected, r.status, r.answer))
                    for tname, frames, expected in company_frames:
                        for ce in (phrase + "'s employer", "the employer of " + phrase, "the company where " + phrase + " works",
                                   "the company that employs " + phrase, "the company " + phrase + " works for"):
                            for fr in frames:
                                q = fr.format(x=ce)
                                r = det_solver.solve_search(det_solver._norm(FRAME.format(q=q)), ws)
                                stat[(tname, k, r.status)] += 1
                                if r.status != "answer" or r.answer != expected:
                                    if r.status in ("answer", "unsolvable"):
                                        wrong.append((q, expected, r.status, r.answer))
            _ = ok_chain
    total = sum(stat.values())
    by = collections.Counter()
    for (t, k, st), n in stat.items():
        by[st] += n
    print(f"KBs: {len(kbs)}; questions: {total}; answered: {by['answer']}; nomatch (to LLM): {by['nomatch']}; "
          f"unsolvable-claims: {by['unsolvable']}; WRONG: {len(wrong)}")
    print("by hop depth (answered / asked):")
    for k in range(args.maxh + 1):
        a = sum(n for (t, kk, st), n in stat.items() if kk == k and st == "answer")
        b = sum(n for (t, kk, st), n in stat.items() if kk == k)
        print(f"  hops={k}: {a}/{b}")
    for w in wrong[:10]:
        print("WRONG:", w)

    # negative / safety checks on a throw-away copy of the first kb
    with tempfile.TemporaryDirectory() as tmp:
        ws = os.path.join(tmp, "w")
        shutil.copytree(kbs[0], ws)
        people, comps = read_graph(os.path.join(ws, "kb"))
        start = sorted(people)[0]
        m1 = people[start]["Mentor"]
        # (1) the mentor's record is gone but still named in the start person's file: must NOT produce an answer
        os.remove(os.path.join(ws, "kb", "person_" + m1.lower().replace(" ", "_") + ".md"))
        q = f"In which year was {start}'s mentor born?"
        r = det_solver.solve_search(det_solver._norm(FRAME.format(q=q)), ws)
        bad = r.status == "answer"
        print(f"dangling mentor -> {r.status} ({'BAD: invented an answer' if bad else 'ok, no invented answer'})")
        wrong += [("dangling",)] if bad else []
        # (2) phrasings the grammar must not claim
        for q in ("In which year was the employer of " + start + " born?",
                  "Which company employs the employer of " + start + "?",
                  "In which city is the headquarters of the mentor of " + start + "?",
                  "In which year was " + start + "'s mentor's employer born?",
                  "In which year was the company where the company where " + start + " works works founded?",
                  "What is the birth year of the company " + start + " works for?"):
            r = det_solver.solve_search(det_solver._norm(FRAME.format(q=q)), kbs[0])
            bad = r.status != "nomatch"
            print(f"nonsense question {q[:62]!r:66s} -> {r.status} {'BAD' if bad else 'ok'}")
            wrong += [("nonsense", q)] if bad else []
    print("RESULT:", "OK" if not wrong else f"{len(wrong)} problem(s)")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
