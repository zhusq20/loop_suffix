"""Public-API adapter for ArcticInference with completion-only history publication."""

from __future__ import annotations

import numpy as np


class CompletionSuffixCache:
    """Separate request-local updates from cross-request output publication.

    `local` must disable global caching (max_cached_requests=0). `history` keeps
    empty local prompts for active requests and only receives their output at
    completion. Both use the ordinary ArcticInference cache public interface.
    No speculative tokens or active responses become visible across requests.
    """

    def __init__(self, local, history):
        if local.max_cached_requests != 0:
            raise ValueError("local cache must have max_cached_requests=0")
        if local.max_tree_depth != history.max_tree_depth:
            raise ValueError("suffix cache depths must match")
        self.local, self.history = local, history
        self.max_tree_depth = local.max_tree_depth
        self.responses = {}
        self.publications = []

    @classmethod
    def arctic(cls, max_tree_depth=64, max_cached_requests=100000):
        from arctic_inference.suffix_decoding import SuffixDecodingCache

        return cls(
            SuffixDecodingCache(max_tree_depth, 0),
            SuffixDecodingCache(max_tree_depth, max_cached_requests),
        )

    def start_request(self, req_id, prompt_token_ids):
        if req_id in self.responses:
            raise ValueError(f"request {req_id} already active")
        self.local.start_request(req_id, prompt_token_ids)
        self.history.start_request(req_id, np.empty(0, dtype=np.int32))
        self.responses[req_id] = []

    def add_active_response(self, req_id, token_ids):
        self.local.add_active_response(req_id, token_ids)
        self.responses[req_id].extend(int(t) for t in token_ids)

    def speculate(self, req_id, context, **kwargs):
        local = self.local.speculate(req_id, context, **kwargs)
        global_ = self.history.speculate(req_id, context, **kwargs)
        return local if local.score >= global_.score else global_

    def stop_request(self, req_id):
        output = self.responses[req_id]
        self.history.add_active_response(req_id, np.asarray(output, dtype=np.int32))
        self.history.stop_request(req_id)
        self.local.stop_request(req_id)
        self.publications.append(dict(request_id=req_id, output_tokens=len(output)))
        del self.responses[req_id]

    def abort_request(self, req_id):
        if req_id not in self.responses:
            return
        self.local.stop_request(req_id)
        self.history.stop_request(req_id)
        if req_id in self.history.cached_requests:
            self.history.evict_cached_response(req_id)
        del self.responses[req_id]
