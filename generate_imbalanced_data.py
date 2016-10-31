#!/usr/bin/env python
import simdna
import simdna.synthetic as sn
reload(sn)
from collections import OrderedDict
import numpy as np
import random

#background frequence
background_frequency = OrderedDict([('A', 0.3), ('C', 0.2), ('G', 0.2), ('T', 0.3)])

#read in the motifs file
loaded_motifs = sn.LoadedEncodeMotifs(
                    fileName=simdna.ENCODE_MOTIFS_PATH,
                    pseudocountProb=0.0001,
                    background=background_frequency)

for motif in ['GATA_disc1', 'TAL1_known1']:
    sequenceSetGenerators = []
    for motif_set, num_seqs, name_prefix in [
      ((motif,), 2000, 'gata_only'), ([], 40000, 'empty')]:
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
            sn.GenerateSequenceNTimes(embedInBackground, N=num_seqs))

    np.random.seed(2345)
    random.seed(2345)
    sequenceSet = sn.ChainSequenceSetGenerators(*sequenceSetGenerators)
    sn.printSequences(motif+"_"+"imbalanced.simdata",
                      sequenceSet,       
                      includeFasta=False, includeEmbeddings=True)
