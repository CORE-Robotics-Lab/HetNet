import time
from utils import *
import torch
import torch.multiprocessing as mp


class MultiProcessWorker(mp.Process):
    # TODO: Make environment init threadsafe
    def __init__(self, id, trainer_maker, comm, seed, *args, **kwargs):
        self.id = id
        self.seed = seed
        super(MultiProcessWorker, self).__init__()
        self.trainer = trainer_maker()
        self.comm = comm

    def run(self):
        torch.manual_seed(self.seed + self.id + 1)
        np.random.seed(self.seed + self.id + 1)

        while True:
            task = self.comm.recv()
            if type(task) == list:
                task, epoch = task

            if task == 'quit':
                return
            elif task == 'run_batch':
                batch, stat = self.trainer.run_batch(epoch)
                self.trainer.optimizer.zero_grad()
                s = self.trainer.compute_grad(batch)
                merge_stat(s, stat)

                self.comm.send(stat)
            elif task == 'send_grads':
                grads = []
                for p in self.trainer.params:
                    if p._grad is not None:
                        grads.append(p._grad.data)

                self.comm.send(grads)
            elif task == 'get_gpu_mem':
                self.comm.send((self.id, self.trainer.get_memory_peak()))
            elif task == 'reset_gpu_mem':
                self.trainer.reset_memory_peak()


class MultiProcessTrainer(object):
    def __init__(self, args, trainer_maker):
        self.comms = []
        self.trainer = trainer_maker()
        # itself will do the same job as workers
        self.nworkers = args.nprocesses - 1

        for i in range(self.nworkers):
            comm, comm_remote = mp.Pipe()
            self.comms.append(comm)
            worker = MultiProcessWorker(i, trainer_maker, comm_remote, seed=args.seed)
            worker.start()

        self.grads = None
        self.worker_grads = None
        self.is_random = args.random
        self.reset_mem_peak()

    def quit(self):
        for comm in self.comms:
            comm.send('quit')

    def obtain_grad_pointers(self):
        # only need perform this once
        if self.grads is None:
            self.grads = []
            for p in self.trainer.params:
                if p._grad is not None:
                    self.grads.append(p._grad.data)

        if self.worker_grads is None:
            self.worker_grads = []
            for comm in self.comms:
                comm.send('send_grads')
                grad = comm.recv()

                self.worker_grads.append(grad)

    def train_batch(self, epoch):
        self.cpu_memory_peak = np.zeros((self.nworkers + 1))
        self.gpu_memory_peak = np.zeros((self.nworkers + 1))

        # run workers in parallel
        for comm in self.comms:
            comm.send('reset_gpu_mem')
            comm.send(['run_batch', epoch])

        # run its own trainer
        batch, stat = self.trainer.run_batch(epoch)
        self.trainer.optimizer.zero_grad()
        s = self.trainer.compute_grad(batch)
        merge_stat(s, stat)

        # check if workers are finished
        for comm in self.comms:
            s = comm.recv()
            merge_stat(s, stat)

        # add gradients of workers
        self.obtain_grad_pointers()
        for i in range(len(self.grads)):
            for g in self.worker_grads:
                self.grads[i] += g[i]
            self.grads[i] /= stat['num_steps']

        self.trainer.optimizer.step()

        for comm in self.comms:
            comm.send('get_gpu_mem')
            id, mem = comm.recv()
            self.update_mem_peak(id + 1, *mem)

        return stat, np.array(self.cpu_memory_peak), np.array(self.gpu_memory_peak)

    def state_dict(self):
        return self.trainer.state_dict()

    def load_state_dict(self, state):
        self.trainer.load_state_dict(state)

    def reset_mem_peak(self):
        self.cpu_memory_peak = np.zeros((self.nworkers + 1))
        self.gpu_memory_peak = np.zeros((self.nworkers + 1))

        self.trainer.reset_memory_peak()

    def update_mem_peak(self, id, cpu_mem, gpu_mem):
        if cpu_mem > self.cpu_memory_peak[id]:
            self.cpu_memory_peak[id] = cpu_mem

        if gpu_mem > self.gpu_memory_peak[id]:
            self.gpu_memory_peak[id] = gpu_mem
