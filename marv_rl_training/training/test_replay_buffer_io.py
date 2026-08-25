import sys, os, tempfile, torch
sys.path.insert(0,'/home/robot/workspaces/MARV_RL/src/flipper_training')
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import RandomSampler
from tensordict import TensorDict
from marv_rl_training.training.replay_buffer_io import save_replay_subset, load_replay_subset, FILENAME

fails=[]
def check(n,c,d=""):
    print(("PASS " if c else "FAIL ")+n+("  "+d if d else "")); 
    if not c: fails.append(n)

CAP=3000
def mkbuf(n=0):
    b=TensorDictReplayBuffer(storage=LazyTensorStorage(max_size=CAP,ndim=1,device="cpu"),
                             sampler=RandomSampler(), batch_size=32)
    if n: b.extend(TensorDict({"obs":torch.arange(n*18,dtype=torch.float32).reshape(n,18),
                               "action":torch.randn(n,4)}, batch_size=[n]))
    return b

d=tempfile.mkdtemp()
# --- full buffer -> saves exactly 1/3 of CAPACITY
b=mkbuf(CAP)
n=save_replay_subset(b,CAP,d,1/3.)
check("full buffer saves 1/3 of capacity", n==CAP//3, f"{n} of cap {CAP}")
b2=mkbuf(); got=load_replay_subset(b2,[d])
check("reload restores that many", got==n==len(b2), f"loaded {got}, len {len(b2)}")
blob=torch.load(os.path.join(d,FILENAME),weights_only=False)
check("stale 'index' is not written to disk", "index" not in blob["data"].keys(), str(sorted(blob["data"].keys())))
check("reloaded transitions get fresh storage slots", b2[torch.arange(5)]["index"].flatten().tolist()==[0,1,2,3,4])

# --- partially filled (< 1/3 cap) -> saved whole
d2=tempfile.mkdtemp(); small=CAP//5
b=mkbuf(small); n=save_replay_subset(b,CAP,d2,1/3.)
check("under-filled buffer is saved whole", n==small, f"{n} of {small} held")

# --- subset is a random sample without replacement
d3=tempfile.mkdtemp()
b=mkbuf(CAP); save_replay_subset(b,CAP,d3,1/3.)
b3=mkbuf(); load_replay_subset(b3,[d3])
vals=b3[torch.arange(len(b3))]["obs"][:,0]
check("subset has no duplicates (sampled without replacement)", len(set(vals.tolist()))==len(vals))

# --- overwrite in place, single file
d4=tempfile.mkdtemp()
b=mkbuf(CAP)
for _ in range(3): save_replay_subset(b,CAP,d4,1/3.)
files=sorted(os.listdir(d4))
check("one file, overwritten in place", files==[FILENAME], str(files))

# --- CRASH SAFETY -------------------------------------------------------------
# truncated file (job died mid-write in a world without atomic rename)
d5=tempfile.mkdtemp()
b=mkbuf(CAP); save_replay_subset(b,CAP,d5,1/3.)
p=os.path.join(d5,FILENAME); raw=open(p,'rb').read()
open(p,'wb').write(raw[:len(raw)//2])
b5=mkbuf(); got=load_replay_subset(b5,[d5])
check("truncated file -> clean empty buffer, no raise", got==0 and len(b5)==0)

# garbage file
d6=tempfile.mkdtemp(); open(os.path.join(d6,FILENAME),'wb').write(b"not a checkpoint")
b6=mkbuf(); got=load_replay_subset(b6,[d6])
check("garbage file -> clean empty buffer, no raise", got==0 and len(b6)==0)

# wrong schema
d7=tempfile.mkdtemp(); torch.save({"format":999,"data":None}, os.path.join(d7,FILENAME))
b7=mkbuf(); got=load_replay_subset(b7,[d7])
check("schema drift -> clean empty buffer, no raise", got==0 and len(b7)==0)

# missing file / missing dir
b8=mkbuf()
check("missing file -> 0", load_replay_subset(b8,[tempfile.mkdtemp()])==0)
check("missing dir -> 0, no raise", load_replay_subset(mkbuf(),["/nonexistent/xyz"])==0)

# a stale .tmp must not be picked up as the real file
d9=tempfile.mkdtemp()
open(os.path.join(d9,FILENAME+".tmp"),'wb').write(b"junk")
check("stale .tmp is ignored", load_replay_subset(mkbuf(),[d9])==0)

# save into an unwritable dir must not raise
check("unwritable save dir -> 0, no raise", save_replay_subset(mkbuf(CAP),CAP,"/proc/nope",1/3.)==0)
# empty buffer / disabled
check("empty buffer saves nothing", save_replay_subset(mkbuf(),CAP,tempfile.mkdtemp(),1/3.)==0)
check("fraction 0 saves nothing", save_replay_subset(mkbuf(CAP),CAP,tempfile.mkdtemp(),0.0)==0)

# first readable dir wins, earlier missing ones skipped
b10=mkbuf(); got=load_replay_subset(b10,[tempfile.mkdtemp(), d])
check("searches candidate dirs in order", got==CAP//3, f"{got}")

print(); print("FAILED:", fails if fails else "none")
sys.exit(1 if fails else 0)
