#!/usr/bin/env python3
"""Record four distinct-key long-context retrieval probes against an already idle HP server."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import pathlib
import statistics
import threading
import time
import urllib.request

ROOT = pathlib.Path(__file__).parent


def get_stats(url):
    with urllib.request.urlopen(url, timeout=10) as f:
        return json.load(f)


def fixtures():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file('/home/lxe/volta-publish-20261024/source/base/tokenizer.json')
    source = '\n'.join(
        f'def record_{i}(values):\n    total = sum(v * {i + 1} for v in values)\n'
        f'    return {{"record": {i}, "total": total, "count": len(values)}}\n'
        for i in range(40000))
    ids = tok.encode(source, add_special_tokens=False).ids
    result = {}
    for n in [512, 4096, 8192, 126000, 250000]:
        text = tok.decode(ids[:n], skip_special_tokens=False)
        result[str(n)] = {'text': text, 'content_tokens': len(tok.encode(text, add_special_tokens=False).ids),
                          'sha256': hashlib.sha256(text.encode()).hexdigest()}
    p = ROOT / 'screen-fixtures.json'
    p.write_text(json.dumps(result))
    print(json.dumps({k: {a: b for a, b in v.items() if a != 'text'} for k, v in result.items()}, indent=2))


def run_request(url, body, start, first, deadline):
    start.wait()
    t0 = time.monotonic()
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    usage, text, ttft, finish = {}, [], None, None
    content_events = []
    try:
        with urllib.request.urlopen(req, timeout=max(1, deadline - time.time())) as response:
            for line in response:
                if time.time() > deadline:
                    raise TimeoutError('screen deadline')
                if not line.startswith(b'data: '):
                    continue
                data = line[6:].strip()
                if data == b'[DONE]':
                    break
                chunk = json.loads(data)
                if chunk.get('usage'):
                    usage = chunk['usage']
                for choice in chunk.get('choices', []):
                    delta = choice.get('delta', {})
                    piece = delta.get('content') or delta.get('reasoning_content') or ''
                    if piece:
                        content_events.append(time.monotonic())
                    if piece or delta.get('tool_calls'):
                        if ttft is None:
                            ttft = time.monotonic() - t0
                            first.set()
                        text.append(piece)
                    if choice.get('finish_reason'):
                        finish = choice['finish_reason']
        wall = time.monotonic() - t0
        assert usage.get('completion_tokens', 0) > 0, ('missing completion usage', usage)
        assert ttft is not None, 'no streamed tokens'
        return {'wall_s': wall, 'start_monotonic': t0, 'end_monotonic': t0 + wall,
                'ttft_s': ttft, 'usage': usage, 'finish_reason': finish,
                'content_event_times_monotonic': content_events,
                'visible_chunk_gap_scope': 'client-observed SSE content chunks, which may contain multiple model tokens',
                'output_sha256': hashlib.sha256(''.join(text).encode()).hexdigest(), 'output': ''.join(text),
                'decode_tokens_per_s': max(0, usage['completion_tokens'] - 1) / max(.000001, wall - ttft)}
    except Exception as e:
        first.set()
        end = time.monotonic()
        return {'error': repr(e), 'wall_s': end - t0, 'start_monotonic': t0,
                'end_monotonic': end, 'usage': usage, 'ttft_s': ttft}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--out', type=pathlib.Path)
    ap.add_argument('--label', default='unlabelled')
    ap.add_argument('--clients', type=int, choices=[1, 4, 8], default=4)
    ap.add_argument('--input', type=int, choices=[512, 4096, 8192, 126000, 250000], default=4096)
    ap.add_argument('--output', type=int, default=512)
    ap.add_argument('--mixed', action='store_true', help='three decoders; admit an 8K prefiller after all have emitted a token')
    ap.add_argument('--needle', action='store_true', help='greedy long-context key-retrieval check; separate from throughput screens')
    ap.add_argument('--url', default='http://127.0.0.1:8090/v1/chat/completions')
    ap.add_argument('--stats', default='http://127.0.0.1:8091/stats')
    ap.add_argument('--timeout', type=int, default=300)
    ap.add_argument('--nonce', default='matched-screen-001')
    args = ap.parse_args()
    if args.prepare:
        fixtures()
        return
    assert args.out and not args.out.exists(), 'use a fresh receipt directory'
    inference_lock = (ROOT / 'screen.lock').open('w')
    fcntl.flock(inference_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    args.out.mkdir(parents=True)
    data = json.loads((ROOT / 'screen-fixtures.json').read_text())
    initial = get_stats(args.stats)
    assert initial['active'] == 0 and initial['queued'] == 0, 'server has other work'
    assert max(args.input, 8192 if args.mixed else 0) + args.output + 1024 <= initial['ctx'], 'input does not fit configured context'
    assert not args.mixed or args.clients == 4
    assert not args.needle or (args.clients == 4 and not args.mixed and args.input in [126000, 250000])
    count = args.clients
    starts = [threading.Event() for _ in range(count)]
    firsts = [threading.Event() for _ in range(count)]
    bodies = []
    for i in range(count):
        n = 8192 if args.mixed and i == 3 else args.input
        prompt = (f'Work item {args.nonce}, client {i}. Review this Python module.\n'
                  + data[str(n)]['text']
                  + '\nWrite a detailed code review with numbered findings and then a revised implementation. '
                  'Keep writing until your output budget is exhausted. Do not summarize early.')
        if args.needle:
            prompt = (f'Remember this deployment verification key: VOLTA_LONG_FOUR_{i:02d}_8a271d60b7fc.\n'
                      + data[str(n)]['text']
                      + '\nReturn the deployment verification key stated at the beginning, exactly, with no other text.')
        bodies.append({'model': 'deepseek-v4.1-flash', 'messages': [{'role': 'user', 'content': prompt}],
                       'temperature': 0 if args.needle else .7, 'top_p': 1, 'seed': 12000 + i, 'max_tokens': args.output,
                       'reasoning_effort': 'none', 'stream': True, 'stream_options': {'include_usage': True}})
    (args.out / 'requests.json').write_text(json.dumps(bodies))
    (args.out / 'initial-stats.json').write_text(json.dumps(initial, indent=2))
    deadline = min(time.time() + args.timeout, 1792908986)
    assert deadline > time.time() + 10, 'rental deadline too close'
    rows, results = [], []
    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(run_request, args.url, body, starts[i], firsts[i], deadline) for i, body in enumerate(bodies)]
        for event in starts[:3 if args.mixed else count]:
            event.set()
        admitted = not args.mixed
        while not all(f.done() for f in futures):
            if not admitted and all(e.is_set() for e in firsts[:3]):
                starts[3].set()
                admitted = True
            if time.time() > deadline:
                for event in starts:
                    event.set()
            try:
                s = get_stats(args.stats)
                s.pop('history', None)
                s['elapsed_s'] = time.monotonic() - t0
                rows.append(s)
            except Exception as e:
                rows.append({'error': repr(e), 'elapsed_s': time.monotonic() - t0})
            time.sleep(.25)
        results = [f.result() for f in futures]
    monitor_wall = time.monotonic() - t0
    wall = max(x['end_monotonic'] for x in results) - min(x['start_monotonic'] for x in results)
    final = get_stats(args.stats)
    (args.out / 'results.json').write_text(json.dumps(results, indent=2))
    (args.out / 'stats.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
    (args.out / 'final-stats.json').write_text(json.dumps(final, indent=2))
    good = [x for x in results if not x.get('error')]
    tokens = sum(x['usage']['completion_tokens'] for x in good)
    summary = {'label': args.label, 'clients': count, 'mixed': args.mixed, 'wall_s': wall,
               'monitor_wall_s': monitor_wall,
               'completed': len(good), 'requested_output_tokens': count * args.output,
               'output_tokens': tokens, 'all_hit_output_budget': len(good) == count and all(x['usage']['completion_tokens'] == args.output for x in good),
               'output_tokens_per_wall_s': tokens / wall,
               'prompt_tokens': sum(x['usage']['prompt_tokens'] for x in good),
               'cached_tokens_delta': final['tokens_cached'] - initial['tokens_cached'],
               'server_tokens_out_delta': final['tokens_out'] - initial['tokens_out'],
               'ttft_s': [x['ttft_s'] for x in good], 'ctx': initial['ctx'], 'heads': initial['heads'],
               'acceptance_median': statistics.median([x['acceptance'] for x in rows if 'acceptance' in x]) if rows else None}
    if args.needle:
        summary['workload'] = 'four concurrent distinct-key greedy long-context retrieval checks on a source-bound HP server; not a throughput comparison'
        summary['expected_keys'] = [f'VOLTA_LONG_FOUR_{i:02d}_8a271d60b7fc' for i in range(4)]
        summary['needle_verified'] = len(good) == 4 and all(x['output'].strip() == key for x, key in zip(good, summary['expected_keys']))
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    assert len(good) == count, 'request errors; inspect results'
    assert tokens == final['tokens_out'] - initial['tokens_out'], 'client/server token counters differ'
    if args.needle:
        assert summary['needle_verified'], 'input capacity exercised, but a distinct key retrieval failed'
        assert summary['cached_tokens_delta'] == 0, 'unexpected prefix reuse; retain result but do not call this a cold probe'


if __name__ == '__main__':
    main()
