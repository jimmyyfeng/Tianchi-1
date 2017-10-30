ff=open("train_cls.txt", 'r').read().splitlines()
nodule_pos=[ r for r in ff if r[-1]=='1']
print nodule_pos[:10]