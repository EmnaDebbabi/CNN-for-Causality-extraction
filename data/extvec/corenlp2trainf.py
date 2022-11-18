import sys
from operator import itemgetter

def voc_add(token):
    if token=='root_inv_ROOT': return
    if token in voc:
        voc[token]+=1
    else:
        voc[token]=1

def write_voc_file(line_count,min_count):
    voc_file.write('<\s> {}\n'.format(line_count))
    sorted_voc=sorted(voc.items(),key=itemgetter(1),reverse=True)
    for entry in sorted_voc:
        if entry[1]<min_count: break
        voc_file.write('{} {}\n'.format(entry[0],entry[1]))

def write_train_file(sent):
    for i in range(len(sent)):
        if sent[i] is None: continue
        voc_add(sent[i][0])
        voc_add('_inv_'.join((sent[i][1],sent[i][2])))
        children_depc=[]
        children_words=[]
        for j in range(len(sent)):
            if sent[j] is None: continue
            if sent[j][3]==i:
                dep_context='_'.join((sent[j][1],sent[j][0]))
                voc_add(dep_context)
                children_depc.append(dep_context)
                children_words.append(sent[j][0])
        #word-dependency contexts
        contexts_file.write('{} {} {}\n'.format(sent[i][0],
        '_inv_'.join((sent[i][1],sent[i][2])),
        ' '.join(children_depc)))
        #word-word contexts
        contexts_file.write('{} {} {}\n'.format(sent[i][0],
        sent[i][2],' '.join(children_words)))
        
def corenlp_reader(fname):
    annotations=False
    text=False
    line_count=0
    with open(fname) as f:
        for line in f:
            if text:
                text=False
                continue
            if annotations:
                if line[0]=='[': continue
                sep=line.find(', ')
                if sep==-1:
                    write_train_file(sent)
                    annotations=False
                    continue
                #create sentence annotation list
                #word ids correspond to positions in the list
                #similar to CoNLL style
                dep_tag=line[:line.find('(')]
                head=line[line.find('(')+1:sep]
                try:
                    head_id=int(head[head.rfind('-')+1:].rstrip("'"))-1
                except ValueError:
                    continue
                if head_id==-1: head_id=None
                head_token=head[:head.rfind('-')]
                child=line.rstrip()[sep+2:]
                try:
                    child_id=int(child[child.rfind('-')+1:-1])-1
                except ValueError:
                    continue
                child_token=child[:child.rfind('-')]
                if dep_tag=='root':
                    sent[child_id]=[child_token.lower(),dep_tag,'ROOT',head_id]
                else:
                    sent[child_id]=[child_token.lower(),dep_tag,
                    head_token.lower(),head_id]
                line_count+=1
            if line[:10]=='Sentence #':
                try:
                    num_tokens=int(line[line.find('(')+1:line.rfind(' ')])
                except ValueError:
                    continue
                annotations=True
                text=True               
                sent=[None for k in range(num_tokens)]
    return line_count

if __name__=='__main__':
    if len(sys.argv)!=5:
        print 'corenlp2trainf.py input_fname contexts_output_fname voc_output_fname min_count'
        sys.exit(1)
    input_fname=sys.argv[1]
    contexts_fname=sys.argv[2]
    voc_fname=sys.argv[3]
    min_count=int(sys.argv[4])
    contexts_file=open(contexts_fname,'w+')
    voc_file=open(voc_fname,'w+')
    voc={}
    line_count=corenlp_reader(input_fname)
    write_voc_file(line_count,min_count)
    contexts_file.close()
    voc_file.close()
