from pop.runs import Run

test = Run('test','.')
test.add_result('result1=1')
test.add_result('result2=2')
test.add_result('third result')

x = test.read_last_n_result(1)