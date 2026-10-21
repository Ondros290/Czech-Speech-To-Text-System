
class HypothesisBuffer:

    def __init__(self):
        self.commited_in_buffer = []
        self.buffer = []
        self.new = []
        self.last_commited_word = None
        self.last_commited_frame = 0

    def insert(self, new, times):
        # compare self.commited_in_buffer and new. It inserts only the words in new that extend the commited_in_buffer, it means they are roughly behind last_commited_time and new in content
        # the new tail is added to self.new
        
        new = [(t, times[i]) for i, t in enumerate(new) if times[i] > self.last_commited_frame]
        self.new = new
        
        print(f"{new=}")
        print(f"{self.commited_in_buffer=}")
        print(f"{self.last_commited_frame=}")
        print("-------------------------------------------")

        if len(self.new) >= 1:
            if self.commited_in_buffer:
                # it's going to search for 1, 2, ..., 5 consecutive words (n-grams) that are identical in commited and new. If they are, they're dropped.
                cn = len(self.commited_in_buffer)
                nn = len(self.new)
                for i in range(1,min(min(cn,nn),5)+1):  # 5 is the maximum 
                    c = " ".join([self.commited_in_buffer[-j][0] for j in range(1,i+1)][::-1])
                    tail = " ".join(self.new[j-1][0] for j in range(1,i+1))
                    if c == tail:
                        words = []
                        for j in range(i):
                            words.append(repr(self.new.pop(0)))
                        words_msg = " ".join(words)
                        print(f"Dropping repeated words from buffer: {words_msg}")
                        break

    def flush(self):
        # returns commited chunk = the longest common prefix of 2 last inserts. 

        commit = []
        print(f"{self.new=}\n{self.buffer=}\n-------------------------------------------")
        while self.new:
            nt, frame = self.new[0]

            if len(self.buffer) == 0:
                break

            if nt == self.buffer[0][0]: # check if current token in new is exact in buffer
                commit.append(nt)
                self.last_commited_word = nt
                self.last_commited_frame = frame
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
            
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return self.commited_in_buffer

    def pop_commited(self, time):
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        return self.buffer