"""Dev-only path probabilities and the paper's call/keep expected-utility rule."""
import json
from pathlib import Path


def confidence_bin(value):
    return min(3,max(0,int(float(value)*4)))


class AdmissionPolicy:
    def __init__(self, record):
        self.record = record
        assert record['split'] == 'development_only'
        self.save_time = record['costs']['t_save_s']
        self.head_time = record['costs']['t_head_s']
        self.node_time = record['costs']['t_node_s']
        if min(self.save_time,self.head_time,self.node_time) < 0 or self.save_time == 0:
            raise ValueError('Costs must be measured nonnegative seconds')

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def probability(self, distance, offset, confidence=None):
        table = self.record['quality']
        pooled = table['offset'].get(str(offset), dict(success=0,count=0))
        base = (pooled['success']+1)/(pooled['count']+2)
        cell = table['distance_offset'].get(f'{distance}/{offset}', dict(success=0,count=0))
        alpha = self.record.get('smoothing_strength',4.)
        before = (cell['success']+alpha*base)/(cell['count']+alpha)
        if confidence is None:
            return before
        cell = table['distance_offset_confidence'].get(f'{distance}/{offset}/{confidence_bin(confidence)}',dict(success=0,count=0))
        return (cell['success']+alpha*before)/(cell['count']+alpha)

    def call_length(self, distance, cap):
        best, best_utility, expected = 0, 0., 0.
        for length in range(1,cap+1):
            expected += self.probability(distance,length)
            utility = self.save_time*expected-self.head_time-length*self.node_time
            # Strict improvement preserves zero/shorter lengths in ties.
            if utility > best_utility:
                best, best_utility = length, utility
        return best

    def keep_length(self, distance, confidence):
        best, best_utility, expected = 0, 0., 0.
        for length,value in enumerate(confidence,1):
            expected += self.probability(distance,length,value)
            utility = self.save_time*expected-length*self.node_time
            if utility > best_utility:
                best, best_utility = length, utility
        return best

    def quality(self, node):
        if hasattr(node,'lad_offset'):
            return self.probability(node.lad_root_distance,node.lad_offset,node.lad_confidence)
        cell = self.record['quality']['native_distance'].get(str(node.depth),dict(success=0,count=0))
        return (cell['success']+1)/(cell['count']+2)


def quality_counts(requests):
    tables = {name:{} for name in ['offset','distance_offset','distance_offset_confidence','native_distance']}
    def add(table,key,success):
        entry = table.setdefault(str(key),dict(success=0,count=0))
        entry['count'] += 1
        entry['success'] += int(success)
    for request in requests:
        output = tuple(request['tokens'])
        for event in request['candidates']:
            path = tuple(event['path'])
            success = len(path) <= len(output) and output[:len(path)] == path
            if event['type'] == 'native':
                add(tables['native_distance'],event['distance'],success)
            else:
                d,j,c = event['distance'],event['offset'],confidence_bin(event['confidence'])
                add(tables['offset'],j,success)
                add(tables['distance_offset'],f'{d}/{j}',success)
                add(tables['distance_offset_confidence'],f'{d}/{j}/{c}',success)
    return tables
