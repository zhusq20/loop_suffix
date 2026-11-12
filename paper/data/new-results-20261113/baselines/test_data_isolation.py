import unittest

from data_isolation import IsolationIndex, normalize, token_grams


class IsolationChecks(unittest.TestCase):
    def setUp(self):
        self.question = 'A shop sells seven boxes of apples every Monday and nine boxes every Tuesday. How many boxes does it sell in four weeks?'
        self.index = IsolationIndex([dict(id='test/1', bucket='test', text=self.question)])

    def test_unicode_case_and_whitespace_exact(self):
        text = self.question.upper().replace(' ', '\u2003\u2003')
        self.assertEqual(self.index.match(text)['reason'], 'normalized_exact')
        self.assertEqual(normalize('Ａ + Ｂ = ３'), 'a + b = 3')

    def test_embedded_question_is_rejected(self):
        text = 'Please solve the following and show your working.\n' + self.question + '\nExplain each step.'
        self.assertEqual(self.index.match(text)['protected_id'], 'test/1')

    def test_near_duplicate_and_different_problem(self):
        # A final punctuation edit changes one 5-gram, so this case exceeds the declared .9 threshold.
        self.assertIsNotNone(self.index.match(self.question[:-1] + '!'))
        self.assertIsNone(self.index.match('Implement a stable sorting algorithm in Python.'))

    def test_short_question_containment_and_operators(self):
        index = IsolationIndex([dict(id='short', bucket='test', text='Compute the sum of 27 and 35.')])
        self.assertIsNotNone(index.match('Question: Compute the sum of 27 and 35. Answer:'))
        self.assertNotEqual(token_grams('Calculate a + b = c.'), token_grams('Calculate a - b = c.'))

    def test_bbh_boilerplate_does_not_make_different_instances_duplicates(self):
        rows = [dict(id=str(i), bucket='bbh/example', text='Solve the following problem and provide an exact numeric answer. '
                     + ('A shop sells seven boxes per week. Find its total after nine weeks.' if i == 0 else
                        f'Object {i} travels {i + 20} miles for {i + 4} hours. Find the speed.')) for i in range(20)]
        index = IsolationIndex(rows, bbh_boilerplate=True)
        self.assertIsNotNone(index.match(rows[0]['text']))
        self.assertIsNone(index.match('Solve the following problem and provide an exact numeric answer. '
                                      'What is the square root of 144?'))


if __name__ == '__main__':
    unittest.main()
