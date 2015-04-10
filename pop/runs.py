"""
defines a "Run" class that stores stuff about the experiment.
"""
import os


class Run(object):
	"""
	A Run object keeps track of experiments that are running and makes them easy to see. The Run object is defined by three files: one which keeps track of the progress as the experiment is running, another which includes final results, dumped 
	parameters (i.e the model), and the last simply dumps any output that the run gives.
	"""
	def __init__(self, name, directory, split_token="|||+|||_||+||"):
		self.name = name
		# initialize files
		self.track_file = open(os.path.join(directory, name + '_track'), 'w')
		self.output_file = open(os.path.join(directory, name+'_output'), 'w')

		self.read_track_file = open(os.path.join(directory,name+'_track'))
		self.split_token = split_token


	def add_parameters(self, params):
		"""
		Sets the self.parameters value to params.

		Params is expected to be a dict mapping parameter name -> setting. Used for hyperparameters.
		"""
		self.parameters = params
		for k in params:
			self.track_file.write('%s : %s' % (str(k), str(params[k])))
		self.track_file.flush()

	def add_result(self, result):
		"""
		Adds a result to the track file. Result is expected to be a string.
		"""
		self.track_file.write(result + self.split_token)
		self.track_file.flush()

	def read_last_n_result(self,n):
		"""
		Prints the last result in the track file
		"""
		self.read_track_file.seek(0)
		content = self.read_track_file.read()
		splits = content.split(self.split_token)
		return '\n'.join(splits[:-1][-n:]) # (n+1) because printint always puts a split token at the end.

	def finalize(self, final_result, Pop):
		"""
		Finalizes the Run object

		TODO: better serialization of Pops
		"""
		dump_file = os.path.join(self.directory, self.name + '_dump_model')
		Pop.serialize(dump_file)


