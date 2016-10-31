from deeplift.visualization import viz_sequence

def viz_sequence_highlight_motifs(scores, one_hot, embeddings):
    highlight = {'blue':[
                (embedding.startPos, embedding.startPos+len(embedding.what))
                for embedding in embeddings if 'GATA_disc1' in embedding.what.getDescription()],
                'green':[
                (embedding.startPos, embedding.startPos+len(embedding.what))
                for embedding in embeddings if 'TAL1_known1' in embedding.what.getDescription()]}
    scores = one_hot*scores[None,None,:]
    viz_sequence.plot_weights(scores, subticks_frequency=10, highlight=highlight)
