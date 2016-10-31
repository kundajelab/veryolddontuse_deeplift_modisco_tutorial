from deeplift.visualization import viz_sequence
from avutils.perf_stats import recall_at_fdr_single_task
import deeplift
import numpy as np

def viz_sequence_highlight_motifs(scores, one_hot, embeddings):
    highlight = {'blue':[
                (embedding.startPos, embedding.startPos+len(embedding.what))
                for embedding in embeddings if 'GATA_disc1' in embedding.what.getDescription()],
                'green':[
                (embedding.startPos, embedding.startPos+len(embedding.what))
                for embedding in embeddings if 'TAL1_known1' in embedding.what.getDescription()]}
    scores = one_hot*scores[None,None,:]
    viz_sequence.plot_weights(scores, subticks_frequency=10, highlight=highlight)


def get_labels_on_predicted_hits(pred_hits_per_region_tuples,
                                 motif_locations,
                                 min_overlap=0.5):
    #for each region, will annotate a hit as a positive if it lies within min_overlap (a fraction) of a motif, and
    #as a negative otherwise. And put in any missed motifs as misses.
    #hits per region should be pairs of (start_index, end_index), one list per region
    hits_labels = []
    unmatched_hits = []
    for pred_hits_this_region, motif_locations_this_region in zip(pred_hits_per_region_tuples, motif_locations):        
        hits_labels_this_region = []
        unmatched_motifs_set = set(range(len(motif_locations_this_region)))
        for hit_location in pred_hits_this_region:
            #check each motif to see if it overlaps
            found_match=False
            for motif_idx, motif_location in enumerate(motif_locations_this_region):
                #let's be conservative and allow only one true hit per motif.
                if (motif_idx in unmatched_motifs_set):
                    if (min(max(0,hit_location[1]-motif_location[0]),
                            max(0,motif_location[1]-hit_location[0]))
                        >= min_overlap*(
                            min(motif_location[1]-motif_location[0],
                                hit_location[1]-hit_location[0]))):
                        hits_labels_this_region.append(1)
                        found_match=True
                        unmatched_motifs_set.remove(motif_idx)
                        break
            if (not found_match):
                hits_labels_this_region.append(0)
        hits_labels.append(hits_labels_this_region)
        unmatched_hits.append(len(unmatched_motifs_set))
    return np.array(hits_labels), unmatched_hits


def get_top_n_scores(scores, top_n, window_size, batch_size=20):
    #get a function that will smooth over the scores using a
    #sliding window of size window_size.
    #same_size_return=False means that we aren't going to pad the
    #flanks to make the returned values the same size as the input.
    smoothen_function = deeplift.util.get_smoothen_function(
                        window_size, same_size_return=False)

    #use this smoothening function to average the scores in sliding windows.
    averaged_scores = np.array(smoothen_function(scores, batch_size=batch_size))
    
    #for reach region, retain top n non-overlapping scores.
    #top_n_scores returns the scores, top_n_indices returns the index of them
    top_n_scores, top_n_indices =\
        deeplift.util.get_top_n_scores_per_region(
            averaged_scores, n=top_n,
            exclude_hits_within_window=int(window_size/2.0))
    
    return top_n_scores, top_n_indices


def compute_recall_at_fdrs(scores, window_size,
                         motif_locations,
                         fdr_thresholds):
    
    top_n_scores, top_n_indices = get_top_n_scores(
        scores, top_n=6, window_size=window_size)
    
    #use the top_n_indices to define windows that may overlap a motif
    pred_hits_per_region_tuples = [
        [(start_idx, start_idx+window_size) for start_idx in indices_for_region]
        for indices_for_region in top_n_indices]
    
    #annotate the windows as 1 if they have at least 50% overlap
    #with a motif, 0 otherwise. Also record the number of motifs
    #per region that don't overlap any window in unmatched_hits
    pred_hit_labels, unmatched_hits = get_labels_on_predicted_hits(
                pred_hits_per_region_tuples=pred_hits_per_region_tuples,
                motif_locations=motif_locations,
                min_overlap=0.5)
    total_unmatched_hits = sum(unmatched_hits)
    
    #zip up the top_n_scores and the corresponding labels
    scores_and_labels = np.array(zip(top_n_scores.ravel(), pred_hit_labels.ravel()))
    
    #find the total number of motifs
    total_motifs = np.sum(scores_and_labels[:,1]) + total_unmatched_hits
    
    #augment scores and labels with the false negatives of totally-missed motifs
    arr_to_augment_with = np.array([[-100, 1] for x in range(total_unmatched_hits)])
    if (len(arr_to_augment_with) > 0):
        augmented_scores_and_labels=np.concatenate([scores_and_labels, arr_to_augment_with], axis=0)
    else:
        augmented_scores_and_labels=scores_and_labels
    
    #find the recall at different FDRs
    recalls_at_fdrs = recall_at_fdr_single_task(
        predicted_scores=augmented_scores_and_labels[:,0],
        true_y=augmented_scores_and_labels[:,1],
        fdr_thresholds=fdr_thresholds)
   
    return recalls_at_fdrs 


def compile_motif_locations(motif_name, embeddings):
    motif_locations = []
    for idx,embeddings_this_region in enumerate(embeddings):
        motif_locations.append([])
        for embedding in embeddings_this_region:
            if (motif_name in embedding.what.getDescription()):
                motif_locations[-1].append((embedding.startPos, embedding.startPos+len(embedding.what)))
    return motif_locations
