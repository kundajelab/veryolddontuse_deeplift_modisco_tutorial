#!/usr/bin/env python
import simdna
import simdna.synthetic as sn
reload(sn)
from collections import OrderedDict
import numpy as np
import random

#background frequence
background_frequency = OrderedDict([('A', 0.27), ('C', 0.23), ('G', 0.23), ('T', 0.27)])

#read in the motifs file
loaded_motifs = sn.LoadedEncodeMotifs(
                    fileName=simdna.ENCODE_MOTIFS_PATH,
                    pseudocountProb=0.0001,
                    background=background_frequency)

sequenceSetGenerators = []
for motif_set,name_prefix in [
                   (('GATA_disc1', 'TAL1_known1'), 'gata_tal1'),
                   (('GATA_disc1',), 'gata_only'),
                   (('TAL1_known1',), 'tal1_only'),
                   ([], 'empty')]:
    embedInBackground = sn.EmbedInABackground(
        backgroundGenerator=sn.ZeroOrderBackgroundGenerator(seqLength=200),
        embedders=[
            sn.RepeatedEmbedder(
                sn.SubstringEmbedder(
                    #sn.ReverseComplementWrapper(
                        substringGenerator=sn.PwmSamplerFromLoadedMotifs(
                            loadedMotifs=loaded_motifs,
                            motifName=motif_name)
                    #)
                    ,
                    positionGenerator=sn.UniformPositionGenerator(),
                    name=motif_name),
                quantityGenerator=sn.MinMaxWrapper(
                    sn.PoissonQuantityGenerator(2),
                    theMax=3, theMin=1),
            )
            for motif_name in motif_set
        ],
       namePrefix=name_prefix+"_")
    sequenceSetGenerators.append(
        sn.GenerateSequenceNTimes(embedInBackground, N=2000))

label_names = ["task1", "task2", "task3"]
#map motifs to the set of tasks they are relevant for
motifs_to_tasks = {("GATA_disc1",):[1],
                   ("TAL1_known1",):[2],
                   ("GATA_disc1", "TAL1_known1"):[0]}

def labeling_function(label_generator, generated_sequence):
    labels = [0,0,0]
    for motif_set, tasks in motifs_to_tasks.items():
        if (all([generated_sequence.additionalInfo.isInTrace(motif)
             for motif in motif_set])):
            for task in tasks:
                labels[task] = 1
    return labels

np.random.seed(1234)
random.seed(1234)
sequenceSet = sn.ChainSequenceSetGenerators(*sequenceSetGenerators)
sn.printSequences("sequences.simdata", sequenceSet,       
                         includeFasta=False, includeEmbeddings=True,
                         labelGenerator=sn.LabelGenerator(
                            labelNames=label_names,
                            labelsFromGeneratedSequenceFunction=labeling_function))
